"""Auto-tuner for finding the optimal batch size for model training."""

import gc
import logging
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
        self.last_peak_memory_gb: float | None = None
        self.last_gpu_memory_gb: float | None = None

        if not torch.cuda.is_available() or "cuda" not in self.device.type:
            raise ValueError("Autotuning batch size is only supported on GPUs.")

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

    def find_optimal_batch_size(self) -> int:
        """Find the optimal batch size using binary search.

        Returns:
            The optimal batch size
        """
        logger.info(
            "Finding optimal batch size between "
            f"{self.min_batch_size} and {self.max_batch_size}"
        )

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

            success = self._test_batch_size(mid)

            if success:
                # This batch size works, enter the upper half of the search range
                optimal_batch_size = mid
                low = mid + 1
            else:
                # This batch size failed, enter the lower half of the search range
                high = mid - 1

            self._cleanup()

        # Reduce by self.safety_factor to be safe (e.g. 0.7 for 30% reduction)
        reduced_batch_size = int(optimal_batch_size * self.safety_factor)
        msg = (
            f"Optimal batch size found {optimal_batch_size}, "
            f"Reducing it by {self.safety_factor * 100}% to {reduced_batch_size}"
        )
        logger.info(msg)

        # Re-test the reduced size and, if it fails, keep shrinking until it fits
        candidate = reduced_batch_size
        while candidate >= self.min_batch_size:
            if self._test_batch_size(candidate):
                self._cleanup()
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

    def is_valid_batch_size(self, batch_size: int) -> bool:
        """Check if a specific batch size fits in GPU memory without OOM.

        Args:
            batch_size: Batch size to validate.

        Returns:
            True if the batch size fits in GPU memory, False if it causes OOM.
        """
        is_valid = self._test_batch_size(batch_size)
        self._cleanup()
        return is_valid

    def _test_batch_size(self, batch_size: int) -> bool:
        """Test if a specific batch size works.

        Args:
            batch_size: Batch size to test

        Returns:
            True if the batch size works, False if it causes OOM error
        """
        logger.info(f"Testing batch size: {batch_size}")

        if not torch.cuda.is_available() or "cuda" not in self.device.type:
            raise ValueError("Batch size testing is only supported on GPUs.")

        base_state = None
        train_loader: DataLoader | None = None
        val_loader: DataLoader | None = None
        try:
            memory_monitor = MemoryMonitor(
                max_ram_utilization=0.8, max_gpu_utilization=0.95
            )

            # Snapshot model weights for next tuning iteration
            base_state = {
                k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            }

            # Create train dataloader
            train_dataloader_kwargs = {
                **self.train_dataloader_kwargs,
                "batch_size": batch_size,
                "shuffle": False,
                "drop_last": False,  # make sure at least one batch is loaded
            }
            train_loader = DataLoader(self.train_dataset, **train_dataloader_kwargs)

            # Create val dataloader
            val_dataloader_kwargs = {
                **self.val_dataloader_kwargs,
                "batch_size": batch_size,
                "shuffle": False,
                "drop_last": False,  # make sure at least one batch is loaded
            }
            val_loader = DataLoader(self.val_dataset, **val_dataloader_kwargs)

            # Fresh optimizers each trial
            optimizers = self.model.configure_optimizers()

            # Track peak memory for this test
            torch.cuda.reset_peak_memory_stats(self.device)

            # Train the model for "self.num_iterations" batches
            self._train(
                train_loader,
                optimizers,
                memory_monitor,
            )

            # Validate the model for "self.num_iterations" batches
            with torch.no_grad():
                self._validate(
                    val_loader,
                    memory_monitor,
                )

            peak_mem_bytes = torch.cuda.max_memory_allocated(self.device)
            self.last_peak_memory_gb = peak_mem_bytes / (1024**3)
            self.last_gpu_memory_gb = self.last_peak_memory_gb
            msg = (
                f"Batch size {batch_size} succeeded (peak GPU memory: "
                f"{self.last_peak_memory_gb:.2f} GB)"
            )
            logger.info(msg)

            return True

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if (
                isinstance(e, torch.cuda.OutOfMemoryError)
                or "out of memory" in str(e).lower()
            ):
                torch.cuda.synchronize(self.device)
                logger.info(f"Batch size {batch_size} failed due to OOM error")
                return False

            logger.error(f"_test_batch_size RuntimeError: {e}", exc_info=True)
            return False

        except OutOfMemoryError as e:
            logger.info(f"Batch size {batch_size} failed due to RAM OOM error: {e}")
            return False

        except Exception as e:
            logger.error(f"_test_batch_size unexpected exception: {e}", exc_info=True)
            return False

        finally:
            # Restore model weights for next tuning iteration
            if base_state is not None:
                try:
                    self.model.load_state_dict(base_state)
                except Exception:
                    logger.exception(
                        "Failed to restore model weights after tuning trial."
                    )
            self.model.zero_grad()

            # Drop references and clean CUDA allocator
            try:
                del optimizers
            except Exception:
                pass
            try:
                del train_loader, val_loader
            except Exception:
                pass

            self._cleanup()

    def _train(
        self,
        data_loader: DataLoader,
        optimizers: list[torch.optim.Optimizer],
        memory_monitor: MemoryMonitor,
    ) -> None:
        """Run a short training loop for memory profiling during autotuning.

        Args:
            data_loader: DataLoader to use for training
            optimizers: List of optimizers to use for training
            memory_monitor: MemoryMonitor to use for monitoring memory usage
        """
        self.model.train()

        for optimizer in optimizers:
            optimizer.zero_grad()

        i = 0
        while i < self.num_iterations:
            for batch in data_loader:
                memory_monitor.check_memory(log=True)

                batch = batch.to(self.device)

                # Forward pass
                outputs: BatchedTrainingOutputs = self.model.training_step(batch)
                loss = sum(outputs.losses.values()).mean()

                # Backward pass
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
                if i >= self.num_iterations:
                    break

    def _validate(
        self,
        val_loader: DataLoader,
        memory_monitor: MemoryMonitor,
    ) -> None:
        """Run a short validation loop for memory profiling during autotuning.

        Args:
            val_loader: DataLoader to use for validation
            memory_monitor: MemoryMonitor to use for monitoring memory usage
        """
        assert len(val_loader) > 0, "Validation loader must have at least one batch"
        self.model.train()  # Keep in train mode to get losses

        j = 0
        while j < self.num_iterations:
            for v_batch in val_loader:
                memory_monitor.check_memory(log=True)

                v_batch = v_batch.to(self.device)

                outputs: BatchedTrainingOutputs = self.model.training_step(v_batch)
                _ = outputs  # load outputs in memory to force GPU usage

                # Check again after forward pass
                memory_monitor.check_memory(log=True)

                del v_batch, outputs

                j += 1
                if j >= self.num_iterations:
                    break

    def _cleanup(self) -> None:
        """Clean up the autotuner."""
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        gc.collect()


def find_optimal_batch_size(
    cfg: DictConfig,
    model: NeuracoreModel,
    dataset: PytorchSynchronizedDataset,
    device: torch.device,
) -> int:
    """Tune the batch size automatically via binary search."""
    # Split dataset into train and validation sets
    dataset_size = len(dataset)
    train_split = 1 - cfg.validation_split
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    try:
        model = model.to(device)

        max_batch_size = (
            cfg.max_batch_size if "max_batch_size" in cfg else len(train_dataset)
        )
        max_batch_size = min(max_batch_size, len(train_dataset))  # Clamp to train len
        min_batch_size = cfg.min_batch_size if "min_batch_size" in cfg else 2

        num_train_workers = min(cfg.num_train_workers, cpu_count())
        num_val_workers = min(cfg.num_val_workers, cpu_count())

        logger.info(
            f"using max_batch_size: {max_batch_size}, "
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

    except Exception:
        logger.error("Batch size autotuning failed", exc_info=True)
        raise


def is_valid_batch_size(
    cfg: DictConfig,
    model: NeuracoreModel,
    dataset: PytorchSynchronizedDataset,
    batch_size: int,
    device: torch.device,
) -> bool:
    """Check whether a specific batch size fits in RAM and GPU memory."""
    # Split dataset into train and validation sets
    dataset_size = len(dataset)
    train_split = 1 - cfg.validation_split
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    try:
        if batch_size > len(train_dataset):
            logger.info(
                "Batch size %d exceeds train dataset size %d; treating as invalid.",
                batch_size,
                len(train_dataset),
            )
            return False

        model = model.to(device)

        num_train_workers = min(cfg.num_train_workers, cpu_count())
        num_val_workers = min(cfg.num_val_workers, cpu_count())

        logger.info(
            f"Validating batch_size: {batch_size}, "
            f"num_train_workers: {num_train_workers}, "
            f"num_val_workers: {num_val_workers}"
        )

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
            min_batch_size=batch_size,
            max_batch_size=batch_size,
        )

        valid = autotuner.is_valid_batch_size(batch_size)

        return valid

    except Exception:
        logger.error("Batch size validation failed", exc_info=True)
        raise
