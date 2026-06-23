"""Auto-tuner for finding the optimal batch size for model training."""

import gc
import logging
import multiprocessing
import queue as queue_module
import time
from collections.abc import Callable
from dataclasses import dataclass
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

# Number of warmup iterations run before memory is measured. Warmup absorbs
# one-off allocations so the measured peak reflects steady-state training memory.
_NUM_WARMUP_ITERATIONS = 1

# When the affine prediction (or a probe) OOMs, the batch size is scaled down by
# this factor and re-probed until it fits -- a cheap alternative to bisection.
_DOWNSCALE_FACTOR = 0.9


@dataclass
class BatchProbeResult:
    """Result of probing a single batch size in the isolated subprocess.

    Attributes:
        fitted: True if the batch size ran without an OOM-related failure.
        peak_reserved_bytes: Steady-state peak *reserved* GPU memory measured
            after warmup. Zero when fitted is False.
        total_bytes: Total VRAM of the probe device. Zero when fitted is
            False.
    """

    fitted: bool
    peak_reserved_bytes: int = 0
    total_bytes: int = 0


class _BatchSizeEstimationError(Exception):
    """Raised when the affine estimate cannot be produced or confirmed.

    Caught by estimate_optimal_batch_size to fall back to the largest
    validated probe instead of extrapolating from an unusable fit.
    """


class BatchSizeValidator:
    """Validator for batch size given a model and dataset.

    Each test constructs train and validation dataloaders, performs a
    brief training pass, then a short validation pass in a spawned subprocess.
    This approach ensures that CUDA out-of-memory errors (or any fatal state the
    CUDA allocator cannot recover from) do not affect the parent process.
    """

    def __init__(
        self,
        model_factory: Callable[[], NeuracoreModel],
        device: torch.device,
        train_dataset: Dataset,
        train_dataloader_kwargs: dict[str, Any],
        num_iterations: int = 2,
    ):
        """Initialize a batch-size validator."""
        self.device = device

        if not torch.cuda.is_available() or "cuda" not in self.device.type:
            raise ValueError("Batch size testing is only supported on GPUs.")

        self.model_factory = model_factory
        self.train_dataset = train_dataset
        self.train_dataloader_kwargs = train_dataloader_kwargs
        self.num_iterations = num_iterations

    def probe_batch_size(self, batch_size: int) -> BatchProbeResult:
        """Probe a batch size and return its measured memory.

        The actual probing runs in a subprocess so that a CUDA OOM (or any state
        the allocator cannot recover from) cannot poison the parent process.

        Args:
            batch_size: Batch size to probe.

        Returns:
            BatchProbeResult with fitted. False on OOM-related failure.
        """
        logger.info("Probing batch size: %s", batch_size)

        # Ensure the parent GPU state is clean so the child has maximum room.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        start_time = time.perf_counter()
        result = self._run_in_subprocess(batch_size)
        logger.info(
            "Probing batch size %s took %.2fs (%s)",
            batch_size,
            time.perf_counter() - start_time,
            "fit" if result.fitted else "OOM",
        )
        return result

    def test_batch_size(self, batch_size: int) -> bool:
        """Return True if batch_size runs without an OOM-related failure."""
        return self.probe_batch_size(batch_size).fitted

    def _run_in_subprocess(self, batch_size: int) -> BatchProbeResult:
        """Spawn a subprocess that probes batch_size and return the result."""
        ctx = multiprocessing.get_context("spawn")
        result_queue: Any = ctx.Queue()

        proc = ctx.Process(
            target=_run_batch_size_probe_worker,
            args=(
                result_queue,
                self.model_factory,
                self.train_dataset,
                self.train_dataloader_kwargs,
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
                return BatchProbeResult(fitted=False)

            try:
                status, payload = result_queue.get_nowait()
            except queue_module.Empty:
                logger.warning(
                    "No result received from batch-size subprocess for "
                    "batch size %s; treating as failure.",
                    batch_size,
                )
                return BatchProbeResult(fitted=False)

            if status == _WORKER_RESULT_SUCCESS:
                result: BatchProbeResult = payload
                if result.fitted:
                    logger.info("Batch size %s test succeeded", batch_size)
                else:
                    logger.info("Batch size %s test failed", batch_size)
                return result

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


def _run_batch_size_probe_worker(
    result_queue: Any,
    model_factory: Callable[[], NeuracoreModel],
    train_dataset: Dataset,
    train_dataloader_kwargs: dict[str, Any],
    num_iterations: int,
    batch_size: int,
    device_str: str,
) -> None:
    """Subprocess entrypoint that probes a single batch size."""
    logging.basicConfig(level=logging.INFO)
    worker_logger = logging.getLogger(__name__)

    try:
        device = torch.device(device_str)
        model = model_factory().to(device)

        result = _probe_batch_size(
            model=model,
            train_dataset=train_dataset,
            train_dataloader_kwargs=train_dataloader_kwargs,
            num_iterations=num_iterations,
            batch_size=batch_size,
            device=device,
        )
        result_queue.put((_WORKER_RESULT_SUCCESS, result))
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
    train_dataloader_kwargs: dict[str, Any],
    num_iterations: int,
    batch_size: int,
    device: torch.device,
) -> BatchProbeResult:
    """Run the batch-size probe inside the subprocess and measure peak memory.

    Returns a BatchProbeResult. fitted is False for OOM-related failures.
    On success peak_reserved_bytes is the steady-state peak reserved memory,
    measured AFTER warmup so one-off torch.compile / cuDNN-autotune / lazy
    optimizer-state allocations are excluded from the value we extrapolate from.
    """
    train_loader = None
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
                "num_workers": 0,
                "persistent_workers": False,
                "pin_memory": False,
                "prefetch_factor": None,
            },
        )

        optimizers = model.configure_optimizers()

        # Warm up first; this triggers any one-time compilation / autotuning and
        # allocates lazy optimizer state, none of which should count toward the
        # steady-state peak used for extrapolation.
        if _NUM_WARMUP_ITERATIONS > 0:
            _train_probe(
                model,
                train_loader,
                optimizers,
                memory_monitor,
                _NUM_WARMUP_ITERATIONS,
                device,
            )

        # Reset AFTER warmup so the peak reflects steady-state training memory.
        torch.cuda.reset_peak_memory_stats(device)

        # Only the training step is measured. Validation runs under no_grad with
        # no backward and no optimizer step, so its peak memory is strictly below
        # the training-step peak that sets the OOM point.
        _train_probe(
            model, train_loader, optimizers, memory_monitor, num_iterations, device
        )

        # Reserved (not allocated) is what the caching allocator holds and what
        # actually triggers OOM, so the estimate extrapolates against reserved.
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
        total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
        logger.info(
            "Batch size %s succeeded (peak reserved: %.2f GB / %.2f GB)",
            batch_size,
            peak_reserved_bytes / (1024**3),
            total_bytes / (1024**3),
        )
        return BatchProbeResult(
            fitted=True,
            peak_reserved_bytes=peak_reserved_bytes,
            total_bytes=total_bytes,
        )

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if (
            isinstance(e, torch.cuda.OutOfMemoryError)
            or "out of memory" in str(e).lower()
        ):
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            logger.error("Batch size %s failed due to OOM error", batch_size)
            return BatchProbeResult(fitted=False)

        logger.error(
            "RuntimeError while probing batch size %s: %s",
            batch_size,
            e,
            exc_info=True,
        )
        raise

    except OutOfMemoryError as e:
        logger.error("Batch size %s failed due to RAM OOM error: %s", batch_size, e)
        return BatchProbeResult(fitted=False)

    except ValueError as e:
        # BatchNorm in training mode needs more than 1 value per channel; a
        # single-sample batch raises this. Surface an actionable message instead
        # of a cryptic "Unexpected failure" so users know to raise min_batch_size.
        if "more than 1 value per channel" in str(e):
            raise ValueError(
                f"Batch size {batch_size} is incompatible with BatchNorm layers "
                "(they require more than 1 sample per channel in training mode). "
                "Set min_batch_size >= 2 in your training config."
            ) from e
        raise

    except Exception as e:  # noqa: BLE001
        logger.error(
            "Unexpected exception while probing batch size %s: %s",
            batch_size,
            e,
            exc_info=True,
        )
        raise

    finally:
        # Drop the loader (and any cached batches) before reclaiming CUDA memory.
        del train_loader
        gc.collect()
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


class BatchSizeAutotuner:
    """Auto-tuner for finding the optimal batch size for model training."""

    def __init__(
        self,
        model_factory: Callable[[], NeuracoreModel],
        device: torch.device,
        train_dataset: Dataset,
        train_dataloader_kwargs: dict[str, Any] | None = None,
        min_batch_size: int = 2,
        max_batch_size: int = 512,
        num_iterations: int = 2,
        safety_factor: float = 0.7,
    ):
        """Initialize the batch size auto-tuner.

        Args:
            model_factory: Callable that constructs a fresh model for testing
            device: CUDA device to test on
            train_dataset: Dataset to use for training
            train_dataloader_kwargs: Additional arguments for the train DataLoader
            min_batch_size: Minimum batch size to try
            max_batch_size: Maximum batch size to try
            num_iterations: Number of iterations to run for each batch size
            safety_factor: Reduce optimal batch size by a factor to be conservative.
        """
        assert num_iterations >= 2, "At least two consecutive batches must be loaded"

        self.train_dataset = train_dataset
        self.train_dataloader_kwargs = train_dataloader_kwargs or {}
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.num_iterations = num_iterations
        self.safety_factor = safety_factor
        self.device = device

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
            model_factory=model_factory,
            device=self.device,
            train_dataset=self.train_dataset,
            train_dataloader_kwargs=self.train_dataloader_kwargs,
            num_iterations=self.num_iterations,
        )

    def _downscale_to_fit(self, start: int, floor_fit: int) -> int:
        """Scale start down by _DOWNSCALE_FACTOR until a probe fits.

        floor_fit is a batch already known to fit (the search floor). Returns
        the largest fitting batch found, reduced by safety_factor for headroom.
        """
        candidate = start
        while candidate > floor_fit:
            if self.validator.test_batch_size(candidate):
                return self._apply_safety(candidate)
            nxt = int(candidate * _DOWNSCALE_FACTOR)
            nxt = nxt if nxt < candidate else candidate - 1  # guarantee progress
            logger.info(
                "[autotune:estimate] batch %s OOM'd; downscaling to %s",
                candidate,
                nxt,
            )
            candidate = nxt

        return self._apply_safety(floor_fit)

    def _apply_safety(self, batch_size: int) -> int:
        """Reduce a fitting batch size by safety_factor for training headroom."""
        reduced = max(self.min_batch_size, int(batch_size * self.safety_factor))
        logger.info(
            "[autotune:estimate] largest fitting batch %s; reducing by %.1f%% "
            "(safety) to %s",
            batch_size,
            (1 - self.safety_factor) * 100,
            reduced,
        )
        return reduced

    def estimate_optimal_batch_size(self, max_gpu_utilization: float = 0.9) -> int:
        """Find the optimal batch size from an affine memory model.

        Probes min, min*8, and a third, larger batch (shrunk toward
        min*8 until it fits, so even strongly convex models yield a real
        large-batch measurement). The fit uses the slope between the *second and
        third* probes -- the local slope in the larger regime, which captures the
        convex growth (fragmentation / workspace) that a tiny-batch slope misses
        -- and solves for the batch that fills the VRAM budget. There is no
        separate search: whenever a probe (or the prediction) OOMs, the batch size
        is scaled down by _DOWNSCALE_FACTOR
        until it fits, then reduced by safety_factor.

        Args:
            max_gpu_utilization: Fraction of VRAM used as the budget.

        Returns:
            The estimated optimal batch size.

        Raises:
            OutOfMemoryError: If even min_batch_size does not fit.
        """
        low = self.min_batch_size
        high = min(self.min_batch_size * 8, self.max_batch_size)
        logger.info(
            "[autotune:estimate] starting affine memory-model estimation | "
            "probe points=(%s, %s) | range=[%s, %s] | "
            "max_gpu_utilization=%.2f | safety_factor=%.2f",
            low,
            high,
            self.min_batch_size,
            self.max_batch_size,
            max_gpu_utilization,
            self.safety_factor,
        )

        # Probe the smallest batch first. If even that OOMs, nothing fits.
        low_result = self.validator.probe_batch_size(low)
        if not low_result.fitted:
            raise OutOfMemoryError(
                f"Smallest batch size {low} does not fit.",
                device=str(self.device),
            )
        _log_probe(low, low_result)

        # Degenerate range (e.g. max_batch_size <= min): nothing more to try.
        if high <= low:
            return self._apply_safety(low)

        # Probe the second anchor. If it OOMs (low fits), scale down from it.
        high_result = self.validator.probe_batch_size(high)
        if not high_result.fitted:
            logger.info("[autotune:estimate] second probe %s OOM'd; downscaling", high)
            return self._downscale_to_fit(int(high * _DOWNSCALE_FACTOR), low)
        _log_probe(high, high_result)

        budget_bytes = high_result.total_bytes * max_gpu_utilization

        # A rough (no-safety) 2-point prediction locates a larger third probe in
        # the relevant regime.
        try:
            rough_max = _estimate_max_batch_size(
                measurements=[
                    (low, low_result.peak_reserved_bytes),
                    (high, high_result.peak_reserved_bytes),
                ],
                total_bytes=high_result.total_bytes,
                max_gpu_utilization=max_gpu_utilization,
                safety_factor=1.0,
                min_batch_size=self.min_batch_size,
                max_batch_size=self.max_batch_size,
            )
        except _BatchSizeEstimationError as exc:
            # Non-positive / unusable slope: cannot extrapolate. Fall back to the
            # largest validated probe (flat memory is rare for real models).
            logger.warning(
                "[autotune:estimate] unusable affine fit (%s); using largest "
                "probed batch size %s",
                exc,
                high,
            )
            return self._apply_safety(high)

        # Find a fitting third anchor for the slope(2nd, 3rd) refit. Start at half
        # the (overshooting) rough max; if it OOMs -- which happens for large,
        # strongly convex models where the tiny-batch slope is unreliable -- shrink
        # the anchor geometrically toward `high` until a probe fits, so we still
        # get a real large-batch measurement instead of crawling down step by step.
        third = min(rough_max // 2, self.max_batch_size)
        third_result = None
        while third > high:
            logger.info("[autotune:estimate] probing third anchor at %s", third)
            result = self.validator.probe_batch_size(third)
            if result.fitted:
                _log_probe(third, result)
                third_result = result
                break
            logger.info(
                "[autotune:estimate] third anchor %s OOM'd; shrinking toward %s",
                third,
                high,
            )
            third = int((high * third) ** 0.5)  # geometric mean toward high

        if third_result is None:
            # Could not place an anchor above `high`; predict from the two probes.
            return self._downscale_to_fit(rough_max, high)

        # Slope between the SECOND and THIRD probes -- a better extrapolation
        # basis than averaging in the tiny-batch points, because memory grows
        # faster at larger batches.
        slope = (third_result.peak_reserved_bytes - high_result.peak_reserved_bytes) / (
            third - high
        )
        if slope <= 0:
            logger.warning(
                "[autotune:estimate] non-positive slope between %s and %s; " "using %s",
                high,
                third,
                third,
            )
            return self._apply_safety(third)

        predicted = int(
            third + (budget_bytes - third_result.peak_reserved_bytes) / slope
        )
        predicted = max(self.min_batch_size, min(predicted, self.max_batch_size))
        logger.info(
            "[autotune:estimate] slope(%s,%s)=%.1f MB/sample -> predicted %s",
            high,
            third,
            slope / (1024**2),
            predicted,
        )

        # Confirm; on OOM, downscale (no bisection). `third` is the known floor.
        return self._downscale_to_fit(predicted, third)


def _log_probe(batch_size: int, result: BatchProbeResult) -> None:
    """Log the measured peak reserved memory for an affine probe."""
    pct = (
        100.0 * result.peak_reserved_bytes / result.total_bytes
        if result.total_bytes
        else 0.0
    )
    logger.info(
        "[autotune:estimate] probe batch=%s -> peak reserved %.3f GB "
        "(%.1f%% of %.3f GB total)",
        batch_size,
        result.peak_reserved_bytes / (1024**3),
        pct,
        result.total_bytes / (1024**3),
    )


def _estimate_max_batch_size(
    measurements: list[tuple[int, int]],
    total_bytes: int,
    max_gpu_utilization: float,
    safety_factor: float,
    min_batch_size: int,
    max_batch_size: int,
) -> int:
    """Predict the largest batch size that fits, from an affine memory model.

    GPU training memory is approximately peak = intercept + slope * batch.
    We fit that line by least squares over the measured (batch, peak) points,
    then solve for the batch that fills the VRAM budget.

    Args:
        measurements: At least two (batch_size, peak_reserved_bytes) points
            measured at distinct batch sizes.
        total_bytes: Total VRAM of the device.
        max_gpu_utilization: Fraction of VRAM treated as the usable budget.
        safety_factor: Extra headroom multiplier applied to the prediction
            (absorbs batch-content variance and allocator fragmentation).
        min_batch_size: Lower clamp for the result.
        max_batch_size: Upper clamp for the result.

    Returns:
        Predicted batch size, clamped to [min_batch_size, max_batch_size].

    Raises:
        _BatchSizeEstimationError: If there are fewer than two distinct points
            or the fitted slope is non-positive (memory not increasing with
            batch size means the linear model is unusable; caller falls back to
            binary search).
    """
    if len(measurements) < 2:
        raise _BatchSizeEstimationError(
            f"need >= 2 measurements to fit a line, got {len(measurements)}"
        )

    batch_sizes = [float(b) for b, _ in measurements]
    peaks = [float(p) for _, p in measurements]
    n = len(measurements)
    mean_batch = sum(batch_sizes) / n
    mean_peak = sum(peaks) / n

    variance_batch = sum((b - mean_batch) ** 2 for b in batch_sizes)
    if variance_batch == 0:
        raise _BatchSizeEstimationError("probe batch sizes are identical")

    covariance = sum(
        (b - mean_batch) * (p - mean_peak) for b, p in zip(batch_sizes, peaks)
    )
    slope = covariance / variance_batch
    if slope <= 0:
        raise _BatchSizeEstimationError(
            f"non-positive memory slope ({slope:.0f} bytes/sample); "
            "linear model unusable"
        )
    intercept = mean_peak - slope * mean_batch

    budget_bytes = total_bytes * max_gpu_utilization
    raw_predicted = (budget_bytes - intercept) / slope
    safe_predicted = int(raw_predicted * safety_factor)
    clamped = max(min_batch_size, min(safe_predicted, max_batch_size))

    logger.info(
        "[autotune:fit] affine model: peak = %.3f GB + %.4f MB/sample * batch | "
        "budget=%.3f GB (%.0f%% of %.3f GB)",
        intercept / (1024**3),
        slope / (1024**2),
        budget_bytes / (1024**3),
        100.0 * max_gpu_utilization,
        total_bytes / (1024**3),
    )
    logger.info(
        "[autotune:fit] raw predicted=%.1f -> after safety_factor %.2f=%s "
        "-> clamped to [%s, %s]=%s",
        raw_predicted,
        safety_factor,
        safe_predicted,
        min_batch_size,
        max_batch_size,
        clamped,
    )

    return clamped


def find_optimal_batch_size(
    cfg: DictConfig,
    model_factory: Callable[[], NeuracoreModel],
    dataset: PytorchSynchronizedDataset,
    device: torch.device,
) -> int:
    """Tune the batch size automatically via an affine memory model."""
    # Only the train split is probed; the val split is discarded since its
    # no_grad pass stays below the training peak).
    train_dataset, _ = _split_train_val_dataset(cfg, dataset)

    max_batch_size = (
        cfg.max_batch_size if "max_batch_size" in cfg else len(train_dataset)
    )
    if max_batch_size > len(train_dataset):
        logger.info(
            "max_batch_size (%d) exceeds train dataset size (%d); clamping to %d",
            max_batch_size,
            len(train_dataset),
            len(train_dataset),
        )
        max_batch_size = len(train_dataset)

    # Default to 2, not 1: a single-sample batch breaks BatchNorm layers in
    # training mode (they need >1 value per channel) and is a poor memory anchor
    # anyway. Users can still force 1 via cfg for norm-free models.
    min_batch_size = cfg.min_batch_size if "min_batch_size" in cfg else 2
    if min_batch_size > len(train_dataset):
        logger.info(
            "min_batch_size (%d) exceeds train dataset size (%d); clamping to %d",
            min_batch_size,
            len(train_dataset),
            len(train_dataset),
        )
        min_batch_size = len(train_dataset)

    num_train_workers = min(cfg.num_train_workers, cpu_count())

    device_desc = str(device)
    if "cuda" in device.type and torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(device)
            device_desc = f"{props.name} ({props.total_memory / (1024**3):.1f} GB)"
        except (RuntimeError, AssertionError):
            # Fetching device properties is best-effort for the log line only;
            # fall back to str(device) if CUDA is not available.
            pass

    logger.info(
        "[autotune] ===== starting batch-size autotune =====\n"
        "  device:        %s\n"
        "  train dataset: %s samples\n"
        "  batch range:   [%s, %s]\n"
        "  dataloader workers: train=%s",
        device_desc,
        len(train_dataset),
        min_batch_size,
        max_batch_size,
        num_train_workers,
    )

    start_time = time.perf_counter()

    autotuner = BatchSizeAutotuner(
        model_factory=model_factory,
        device=device,
        train_dataset=train_dataset,
        train_dataloader_kwargs={
            "collate_fn": dataset.collate_fn,
            "num_workers": num_train_workers,
            "persistent_workers": num_train_workers > 0,
            "pin_memory": True,
        },
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
    )

    # Find the batch size from the affine memory model (self-sufficient: it
    # handles OOM and overshoot internally by downscaling).
    optimal_batch_size = autotuner.estimate_optimal_batch_size()

    elapsed_time = time.perf_counter() - start_time
    logger.info(
        "[autotune] ===== done in %.2fs | chosen batch size=%s =====",
        elapsed_time,
        optimal_batch_size,
    )

    return optimal_batch_size


def is_valid_batch_size(
    cfg: DictConfig,
    model_factory: Callable[[], NeuracoreModel],
    dataset: PytorchSynchronizedDataset,
    batch_size: int,
    device: torch.device,
) -> bool:
    """Check whether a specific batch size fits in RAM and GPU memory."""
    train_dataset, _ = _split_train_val_dataset(cfg, dataset)

    if batch_size > len(train_dataset):
        batch_size = len(train_dataset)
        logger.info(
            f"Batch size {batch_size} exceeds train dataset size {len(train_dataset)}; "
            "clamping to train dataset size"
        )

    num_train_workers = min(cfg.num_train_workers, cpu_count())

    logger.info(
        f"Validating batch_size: {batch_size}, "
        f"num_train_workers: {num_train_workers}"
    )

    validator = BatchSizeValidator(
        model_factory=model_factory,
        device=device,
        train_dataset=train_dataset,
        train_dataloader_kwargs={
            "collate_fn": dataset.collate_fn,
            "num_workers": num_train_workers,
            "persistent_workers": num_train_workers > 0,
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
