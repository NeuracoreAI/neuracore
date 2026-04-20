"""Auto-tuner for finding the optimal batch size for model training."""

import gc
import logging
import multiprocessing
import queue as queue_module
import time
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split

from neuracore.ml import BatchedTrainingOutputs, NeuracoreModel
from neuracore.ml.datasets.pytorch_synchronized_dataset import (
    PytorchSynchronizedDataset,
)
from neuracore.ml.utils.device_utils import cpu_count
from neuracore.ml.utils.memory_monitor import MemoryMonitor, OutOfMemoryError

logger = logging.getLogger(__name__)


# Helpers for validator subprocess worker.
_WORKER_RESULT_SUCCESS = "ok"
_WORKER_RESULT_FAILURE = "fail"
_SUBPROCESS_TERMINATE_TIMEOUT_S = 5.0


class BatchSizeValidator:
    """Validator for batch size given a model and dataset.

    Each test constructs train and validation dataloaders, performs a
    brief training pass, then a short validation pass in a spawned subprocess.
    This approach ensures that CUDA out-of-memory errors (or any fatal state the
    CUDA allocator cannot recover from) do not affect the parent process.
    """

    def __init__(
        self,
        model: NeuracoreModel,
        train_dataset: Dataset,
        val_dataset: Dataset,
        train_dataloader_kwargs: dict[str, Any],
        val_dataloader_kwargs: dict[str, Any],
        num_iterations: int = 2,
    ):
        """Initialize a batch-size validator for a specific model and datasets."""
        self.device = model.device

        if not torch.cuda.is_available() or "cuda" not in self.device.type:
            raise ValueError("Batch size testing is only supported on GPUs.")

        # Keep the parent-side copy on CPU to avoid unnecessary VRAM usage.
        self.model = model.to("cpu")
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_dataloader_kwargs = train_dataloader_kwargs
        self.val_dataloader_kwargs = val_dataloader_kwargs
        self.num_iterations = num_iterations

    def test_batch_size(self, batch_size: int) -> bool:
        """Test if a specific batch size works.

        The actual probing (dataloader construction, forward/backward, etc.) is
        executed in a subprocess. Anything that leaves the subprocess in a bad
        state due to memory pressure (CUDA OOM, RAM OOM, process killed by the
        OS) is surfaced here as ``False``. Unexpected probe errors are raised to
        the caller to avoid misclassifying logic/data bugs as batch-size issues.

        Args:
            batch_size: Batch size to test

        Returns:
            True if the batch size works, False if it causes an OOM-related
            failure.
        """
        logger.info(f"Testing batch size: {batch_size}")

        # Ensure the parent GPU state is clean so the child has maximum room.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return self._run_in_subprocess(batch_size)

    def _run_in_subprocess(self, batch_size: int) -> bool:
        """Spawn a subprocess that probes batch_size and return the result."""
        ctx = multiprocessing.get_context("spawn")
        result_queue: Any = ctx.Queue()

        proc = ctx.Process(
            target=_run_batch_size_test_worker,
            args=(
                result_queue,
                self.model,
                self.train_dataset,
                self.val_dataset,
                self.train_dataloader_kwargs,
                self.val_dataloader_kwargs,
                self.num_iterations,
                batch_size,
                str(self.device),
            ),
        )

        try:
            proc.start()
            proc.join()

            if proc.exitcode != 0:
                logger.info(
                    "Batch size %s subprocess exited with code %s; "
                    "treating as failure.",
                    batch_size,
                    proc.exitcode,
                )
                return False

            try:
                status, payload = result_queue.get_nowait()
            except queue_module.Empty:
                logger.warning(
                    "No result received from batch-size subprocess for "
                    "batch size %s; treating as failure.",
                    batch_size,
                )
                return False

            if status == _WORKER_RESULT_SUCCESS:
                success = bool(payload)
                if success:
                    logger.info("Batch size %s test succeeded", batch_size)
                else:
                    logger.info("Batch size %s test failed", batch_size)
                return success

            raise RuntimeError(
                f"Unexpected failure while probing batch size {batch_size}: {payload}"
            )
        finally:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=_SUBPROCESS_TERMINATE_TIMEOUT_S)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
            try:
                result_queue.close()
                result_queue.join_thread()
            except Exception:
                pass


def _run_batch_size_test_worker(
    result_queue: Any,
    model: NeuracoreModel,
    train_dataset: Dataset,
    val_dataset: Dataset,
    train_dataloader_kwargs: dict[str, Any],
    val_dataloader_kwargs: dict[str, Any],
    num_iterations: int,
    batch_size: int,
    device_str: str,
) -> None:
    """Subprocess entrypoint that probes a single batch size."""
    logging.basicConfig(level=logging.INFO)
    worker_logger = logging.getLogger(__name__)

    try:
        device = torch.device(device_str)
        model = model.to(device)

        success = _probe_batch_size(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_dataloader_kwargs=train_dataloader_kwargs,
            val_dataloader_kwargs=val_dataloader_kwargs,
            num_iterations=num_iterations,
            batch_size=batch_size,
            device=device,
        )
        result_queue.put((_WORKER_RESULT_SUCCESS, success))
    except BaseException as exc:  # noqa: BLE001 - forward anything to parent
        worker_logger.error(
            "Unhandled exception while probing batch size %s: %s",
            batch_size,
            exc,
            exc_info=True,
        )
        try:
            result_queue.put((_WORKER_RESULT_FAILURE, repr(exc)))
        except Exception:
            pass


def _probe_batch_size(
    model: NeuracoreModel,
    train_dataset: Dataset,
    val_dataset: Dataset,
    train_dataloader_kwargs: dict[str, Any],
    val_dataloader_kwargs: dict[str, Any],
    num_iterations: int,
    batch_size: int,
    device: torch.device,
) -> bool:
    """Run the actual batch-size probe (executed inside the subprocess)."""
    try:
        memory_monitor = MemoryMonitor(
            max_ram_utilization=0.8, max_gpu_utilization=0.95
        )

        train_loader = DataLoader(
            train_dataset,
            **{
                **train_dataloader_kwargs,
                "batch_size": batch_size,
                "shuffle": False,
                "drop_last": False,  # make sure at least one batch is loaded
            },
        )
        val_loader = DataLoader(
            val_dataset,
            **{
                **val_dataloader_kwargs,
                "batch_size": batch_size,
                "shuffle": False,
                "drop_last": False,  # make sure at least one batch is loaded
            },
        )

        optimizers = model.configure_optimizers()
        torch.cuda.reset_peak_memory_stats(device)

        _train_probe(
            model, train_loader, optimizers, memory_monitor, num_iterations, device
        )
        with torch.no_grad():
            _validate_probe(model, val_loader, memory_monitor, num_iterations, device)

        peak_mem_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_gb = peak_mem_bytes / (1024**3)
        logger.info(
            "Batch size %s succeeded (peak GPU memory: %.2f GB)",
            batch_size,
            peak_memory_gb,
        )
        return True

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if (
            isinstance(e, torch.cuda.OutOfMemoryError)
            or "out of memory" in str(e).lower()
        ):
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            logger.error("Batch size %s failed due to OOM error", batch_size)
            return False

        logger.error(
            "RuntimeError while probing batch size %s: %s",
            batch_size,
            e,
            exc_info=True,
        )
        raise

    except OutOfMemoryError as e:
        logger.error("Batch size %s failed due to RAM OOM error: %s", batch_size, e)
        return False

    except Exception as e:  # noqa: BLE001
        logger.error(
            "Unexpected exception while probing batch size %s: %s",
            batch_size,
            e,
            exc_info=True,
        )
        raise

    finally:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
            torch.cuda.empty_cache()
        gc.collect()


def _train_probe(
    model: NeuracoreModel,
    data_loader: DataLoader,
    optimizers: list[torch.optim.Optimizer],
    memory_monitor: MemoryMonitor,
    num_iterations: int,
    device: torch.device,
) -> None:
    """Run a short training loop for memory profiling."""
    model.train()

    for optimizer in optimizers:
        optimizer.zero_grad()

    i = 0
    while i < num_iterations:
        for batch in data_loader:
            memory_monitor.check_memory(log=True)

            batch = batch.to(device)

            outputs: BatchedTrainingOutputs = model.training_step(batch)
            loss = sum(outputs.losses.values()).mean()

            loss.backward()

            for optimizer in optimizers:
                optimizer.step()

            # Check again before freeing up gradients
            memory_monitor.check_memory(log=True)

            # Free-up GPU during validation or before next forward pass
            for optimizer in optimizers:
                optimizer.zero_grad()

            del batch, outputs, loss

            i += 1
            if i >= num_iterations:
                break


def _validate_probe(
    model: NeuracoreModel,
    val_loader: DataLoader,
    memory_monitor: MemoryMonitor,
    num_iterations: int,
    device: torch.device,
) -> None:
    """Run a short validation loop for memory profiling."""
    assert len(val_loader) > 0, "Validation loader must have at least one batch"
    model.train()  # Keep in train mode to get losses

    j = 0
    while j < num_iterations:
        for v_batch in val_loader:
            memory_monitor.check_memory(log=True)

            v_batch = v_batch.to(device)

            outputs: BatchedTrainingOutputs = model.training_step(v_batch)
            _ = outputs  # load outputs in memory to force GPU usage

            # Check again after forward pass
            memory_monitor.check_memory(log=True)

            del v_batch, outputs

            j += 1
            if j >= num_iterations:
                break


class BatchSizeAutotuner:
    """Auto-tuner for finding the optimal batch size for model training."""

    def __init__(
        self,
        model: NeuracoreModel,
        train_dataset: Dataset,
        val_dataset: Dataset,
        train_dataloader_kwargs: dict[str, Any] | None = None,
        val_dataloader_kwargs: dict[str, Any] | None = None,
        min_batch_size: int = 8,
        max_batch_size: int = 512,
        num_iterations: int = 2,
        safety_factor: float = 0.7,
    ):
        """Initialize the batch size auto-tuner.

        Args:
            model: Model to use for testing
            train_dataset: Dataset to use for training
            val_dataset: Dataset to use for validation
            train_dataloader_kwargs: Additional arguments for the train DataLoader
            val_dataloader_kwargs: Additional arguments for the val DataLoader
            min_batch_size: Minimum batch size to try
            max_batch_size: Maximum batch size to try
            num_iterations: Number of iterations to run for each batch size
            safety_factor: Reduce optimal batch size by a factor to be conservative.
        """
        assert num_iterations >= 2, "At least two consecutive batches must be loaded"

        self.model = model
        self.train_dataset = train_dataset
        self.train_dataloader_kwargs = train_dataloader_kwargs or {}
        self.val_dataset = val_dataset
        self.val_dataloader_kwargs = val_dataloader_kwargs or {}
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.num_iterations = num_iterations
        self.safety_factor = safety_factor
        self.device = model.device

        if not torch.cuda.is_available() or "cuda" not in self.device.type:
            raise ValueError("Autotuning batch size is only supported on GPUs.")

        if safety_factor < 0.0 or safety_factor > 1.0:
            raise ValueError("safety_factor must be between 0.0 and 1.0")

        # Validate batch size ranges
        if min_batch_size > max_batch_size:
            raise ValueError(
                f"min_batch_size ({min_batch_size}) must be "
                f"<= max_batch_size ({max_batch_size})"
            )

        # Validate dataset size
        if len(train_dataset) < min_batch_size:
            raise ValueError(
                f"Dataset size ({len(train_dataset)}) is smaller "
                f"than min_batch_size ({min_batch_size})"
            )

        self.validator = BatchSizeValidator(
            model=self.model,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            train_dataloader_kwargs=self.train_dataloader_kwargs,
            val_dataloader_kwargs=self.val_dataloader_kwargs,
            num_iterations=self.num_iterations,
        )

    def find_optimal_batch_size(self) -> int:
        """Find the optimal batch size using binary search.

        Returns:
            The optimal batch size
        """
        # Initialize binary search range
        low = self.min_batch_size
        high = self.max_batch_size

        # Granularity: stop searching when the range is sufficiently small
        granularity = int((high - low) / 20)  # 5% of the search range
        granularity = max(1, min(granularity, 50))  # clip to [1, 50]

        optimal_batch_size = low  # Start conservative
        search_step = 0

        while low + granularity - 1 <= high:
            mid = (low + high) // 2
            search_step += 1

            success = self.validator.test_batch_size(mid)

            if success:
                # This batch size works, enter the upper half of the search range
                optimal_batch_size = mid
                low = mid + 1
            else:
                # This batch size failed, enter the lower half of the search range
                high = mid - 1

        # Reduce by self.safety_factor to be safe (e.g. 0.7 for 30% reduction)
        reduced_batch_size = int(optimal_batch_size * self.safety_factor)
        msg = (
            f"Optimal batch size found: {optimal_batch_size}, "
            f"Reducing by {(1 - self.safety_factor) * 100:.1f}% to {reduced_batch_size}"
        )
        logger.info(msg)

        # Re-test the reduced size and, if it fails, keep shrinking until it fits
        candidate = reduced_batch_size
        while candidate >= self.min_batch_size:
            if self.validator.test_batch_size(candidate):
                return candidate

            logger.info(
                "Reduced batch size %s failed on re-test; trying %s",
                candidate,
                candidate - 1,
            )
            candidate -= 1

        raise OutOfMemoryError(
            "Unable to find a valid batch size after safety reduction.",
            device=str(self.device),
        )


def find_optimal_batch_size(
    cfg: DictConfig,
    model: NeuracoreModel,
    dataset: PytorchSynchronizedDataset,
    device: torch.device,
) -> int:
    """Tune the batch size automatically via binary search."""
    train_dataset, val_dataset = _split_train_val_dataset(cfg, dataset)
    model = model.to(device)

    max_batch_size = (
        cfg.max_batch_size if "max_batch_size" in cfg else len(train_dataset)
    )
    max_batch_size = min(max_batch_size, len(train_dataset))  # Clamp to train len
    min_batch_size = cfg.min_batch_size if "min_batch_size" in cfg else 2

    num_train_workers = min(cfg.num_train_workers, cpu_count())
    num_val_workers = min(cfg.num_val_workers, cpu_count())

    logger.info(
        f"Autotuning batch size with max_batch_size: {max_batch_size}, "
        f"min_batch_size: {min_batch_size}, "
        f"num_train_workers: {num_train_workers}, "
        f"num_val_workers: {num_val_workers}"
    )

    start_time = time.perf_counter()

    autotuner = BatchSizeAutotuner(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_dataloader_kwargs={
            "collate_fn": dataset.collate_fn,
            "num_workers": num_train_workers,
            "persistent_workers": num_train_workers > 0,
            "pin_memory": True,
        },
        val_dataloader_kwargs={
            "collate_fn": dataset.collate_fn,
            "num_workers": num_val_workers,
            "persistent_workers": num_val_workers > 0,
            "pin_memory": True,
        },
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
    )

    # Perform binary search to find the optimal batch size
    optimal_batch_size = autotuner.find_optimal_batch_size()

    elapsed_time = time.perf_counter() - start_time
    logger.info("Autotune batch_size took %.3fs", elapsed_time)

    return optimal_batch_size


def is_valid_batch_size(
    cfg: DictConfig,
    model: NeuracoreModel,
    dataset: PytorchSynchronizedDataset,
    batch_size: int,
    device: torch.device,
) -> bool:
    """Check whether a specific batch size fits in RAM and GPU memory."""
    train_dataset, val_dataset = _split_train_val_dataset(cfg, dataset)
    model = model.to(device)

    if batch_size > len(train_dataset):
        batch_size = len(train_dataset)
        logger.info(
            f"Batch size {batch_size} exceeds train dataset size {len(train_dataset)}; "
            "clamping to train dataset size"
        )

    num_train_workers = min(cfg.num_train_workers, cpu_count())
    num_val_workers = min(cfg.num_val_workers, cpu_count())

    logger.info(
        f"Validating batch_size: {batch_size}, "
        f"num_train_workers: {num_train_workers}, "
        f"num_val_workers: {num_val_workers}"
    )

    validator = BatchSizeValidator(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_dataloader_kwargs={
            "collate_fn": dataset.collate_fn,
            "num_workers": num_train_workers,
            "persistent_workers": num_train_workers > 0,
            "pin_memory": True,
        },
        val_dataloader_kwargs={
            "collate_fn": dataset.collate_fn,
            "num_workers": num_val_workers,
            "persistent_workers": num_val_workers > 0,
            "pin_memory": True,
        },
    )

    valid = validator.test_batch_size(batch_size)

    return valid


def _split_train_val_dataset(
    cfg: DictConfig,
    dataset: PytorchSynchronizedDataset,
) -> tuple[Dataset, Dataset]:
    """Split dataset into deterministic train and validation subsets."""
    dataset_size = len(dataset)
    train_split = 1 - cfg.validation_split
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    return train_dataset, val_dataset
