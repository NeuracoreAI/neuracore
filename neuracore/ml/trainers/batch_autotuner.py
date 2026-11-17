"""Find the largest training batch size that fits in GPU memory."""

import gc
import logging
import multiprocessing
import os
import queue as queue_module
import socket
import time
from collections.abc import Callable, Sized
from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.distributed as torch_distributed
from omegaconf import DictConfig
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from neuracore.ml import NeuracoreModel
from neuracore.ml.datasets.pytorch_synchronized_dataset import (
    PytorchSynchronizedDataset,
)
from neuracore.ml.logging.stream_handlers import build_stream_handlers
from neuracore.ml.preprocessing.base import PreprocessingConfiguration
from neuracore.ml.utils.dataset_utils import split_train_val_datasets
from neuracore.ml.utils.memory_monitor import (
    TRAINING_MAX_GPU_UTILIZATION,
    TRAINING_MAX_RAM_UTILIZATION,
    MemoryMonitor,
    OutOfMemoryError,
)
from neuracore.ml.utils.preprocessing import (
    apply_device_preprocessing,
    resolve_input_output_preprocessing,
)

logger = logging.getLogger(__name__)

_SLOPE_BATCH_SIZES = (8, 16)
_MEASURED_STEPS_PER_TRIAL = 2
_WARM_UP_BATCH_SIZE = 2
_BISECTION_TOLERANCE_FRACTION = 0.02
_STEP_DOWN_AFTER_FAILED_ESTIMATE = 0.95
_MAX_TUNING_PROCESS_STARTS = 3
_SUBPROCESS_TERMINATE_TIMEOUT_S = 5.0
_RESULT_POLL_INTERVAL_S = 1.0
_HEALTH_CHECK_ELEMENT_COUNT = 1024

_RESULT_COMPLETED = "completed"
_RESULT_RESTART_REQUIRED = "restart_required"
_RESULT_ERROR = "error"


@dataclass
class BatchSizeTrialMeasurement:
    """GPU memory measured while training at one batch size.

    Attributes:
        batch_size: Batch size of the trial.
        backward_peak_bytes: Highest live tensor memory during forward and
            backward passes.
        optimizer_step_peak_bytes: Highest live tensor memory during gradient
            clipping and the optimizer step.
        peak_reserved_bytes: Highest memory the caching allocator held.
        steps_with_allocator_retries: Measured steps in which the allocator had
            to free its cache and retry an allocation.
    """

    batch_size: int
    backward_peak_bytes: int
    optimizer_step_peak_bytes: int
    peak_reserved_bytes: int
    steps_with_allocator_retries: int


@dataclass
class BatchSizeSearchSettings:
    """Settings the tuning process needs to run its trials.

    Attributes:
        min_batch_size: Smallest batch size the search may return.
        max_batch_size: Largest batch size the search may return.
        batch_size_to_validate: Batch size to check instead of searching.
        largest_fitting_batch_size: Largest batch size an earlier tuning
            process confirmed, used when a process restarts the search.
        smallest_failing_batch_size: Smallest batch size an earlier tuning
            process saw fail, used when a process restarts the search.
        trial_under_ddp: Whether trials wrap the model in single process DDP.
        ddp_master_port: Free local port for the single process DDP group.
        clip_grad_norm: Gradient clipping norm the trainer applies.
        train_device_preprocessing: Device side preprocessing the trainer
            applies to inputs and outputs.
    """

    min_batch_size: int
    max_batch_size: int
    batch_size_to_validate: int | None = None
    largest_fitting_batch_size: int | None = None
    smallest_failing_batch_size: int | None = None
    trial_under_ddp: bool = False
    ddp_master_port: int = 0
    clip_grad_norm: float | None = None
    train_device_preprocessing: tuple[
        PreprocessingConfiguration, PreprocessingConfiguration
    ] = field(
        default_factory=lambda: (
            PreprocessingConfiguration(),
            PreprocessingConfiguration(),
        )
    )


@dataclass
class BatchSizeSearchResult:
    """Outcome reported by the tuning process.

    Attributes:
        status: One of completed, restart_required or error.
        chosen_batch_size: Largest batch size that fit, when completed.
        largest_fitting_batch_size: Largest batch size confirmed so far.
        smallest_failing_batch_size: Smallest batch size that failed so far.
        target_reserved_bytes: Reserved memory limit the trials ran under.
        trial_count: Number of trials, including those that ran out of memory.
        trials: Measurements of the trials that completed, in order.
        error_message: Description of an error, when status is error.
    """

    status: str
    chosen_batch_size: int | None = None
    largest_fitting_batch_size: int | None = None
    smallest_failing_batch_size: int | None = None
    target_reserved_bytes: int = 0
    trial_count: int = 0
    trials: list[BatchSizeTrialMeasurement] = field(default_factory=list)
    error_message: str = ""


class _GpuUnusableAfterOutOfMemoryError(Exception):
    """Raised when the CUDA context stops working after an out of memory error."""


def _dataset_length(dataset: Dataset) -> int:
    """Return the number of samples in a map style dataset."""
    return len(cast(Sized, dataset))


def _is_gpu_out_of_memory_error(error: BaseException) -> bool:
    """Return whether an exception reports exhausted GPU memory."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    message = str(error)
    return isinstance(error, RuntimeError) and (
        "out of memory" in message.lower()
        or "CUDNN_STATUS_NOT_SUPPORTED" in message
        or "CUDNN_STATUS_INTERNAL_ERROR" in message
        or "CUBLAS_STATUS_ALLOC_FAILED" in message
    )


def _gpu_is_healthy(device: torch.device) -> bool:
    """Allocate a small tensor and synchronise to confirm the CUDA context works."""
    try:
        health_check_tensor = torch.ones(_HEALTH_CHECK_ELEMENT_COUNT, device=device)
        health_check_tensor.add_(1)
        torch.cuda.synchronize(device)
        return True
    except Exception:
        return False


def _gpu_memory_outside_pytorch_allocator(device: torch.device) -> tuple[int, int]:
    """Return this process's own memory outside the allocator and other processes' memory.

    NVML reports the memory the driver assigns to each process. When NVML
    cannot see this process, as inside some containers, all memory outside
    the allocator counts as this process's own.

    Args:
        device: CUDA device to inspect.

    Returns:
        Bytes this process holds outside PyTorch's allocator, and bytes that
        other processes hold on the same device.
    """
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    device_used_bytes = total_bytes - free_bytes
    reserved_bytes = torch.cuda.memory_reserved(device)
    try:
        import pynvml

        pynvml.nvmlInit()
        device_handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)
        for process in pynvml.nvmlDeviceGetComputeRunningProcesses(device_handle):
            if process.pid == os.getpid() and process.usedGpuMemory:
                own_bytes = int(process.usedGpuMemory)
                return (
                    max(0, own_bytes - reserved_bytes),
                    max(0, device_used_bytes - own_bytes),
                )
    except Exception:  # noqa: BLE001
        logger.debug("NVML per process memory unavailable", exc_info=True)
    return max(0, device_used_bytes - reserved_bytes), 0


class _TrialRunner:
    """Run training trials at different batch sizes on one model in one process."""

    def __init__(
        self,
        model: NeuracoreModel,
        trainable_module: nn.Module,
        optimizers: list[torch.optim.Optimizer],
        train_dataset: Dataset,
        collate_function: Callable,
        settings: BatchSizeSearchSettings,
        device: torch.device,
    ) -> None:
        """Store the model, data and settings every trial uses.

        Args:
            model: Model under test.
            trainable_module: Module the forward pass calls, the model itself
                or its DDP wrapper.
            optimizers: Optimizers returned by the model.
            train_dataset: Training split to draw batches from.
            collate_function: Function that batches dataset samples.
            settings: Search settings.
            device: CUDA device the trials run on.
        """
        self.model = model
        self.trainable_module = trainable_module
        self.optimizers = optimizers
        self.train_dataset = train_dataset
        self.collate_function = collate_function
        self.settings = settings
        self.device = device
        self.host_memory_monitor = MemoryMonitor(
            max_ram_utilization=TRAINING_MAX_RAM_UTILIZATION, gpu_id=None
        )
        self.trials: list[BatchSizeTrialMeasurement] = []
        self.trial_count = 0

    def run_training_steps(self, batch_size: int, step_count: int) -> list[dict]:
        """Run full training steps and return per step memory readings.

        Args:
            batch_size: Batch size of each step.
            step_count: Number of steps to run.

        Returns:
            One dictionary of peak readings per step.
        """
        data_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=self.collate_function,
        )
        batch_iterator = iter(data_loader)
        step_readings = []
        for _ in range(step_count):
            try:
                batch = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(data_loader)
                batch = next(batch_iterator)
            self.host_memory_monitor.check_memory()
            retries_before_step = torch.cuda.memory_stats(self.device).get(
                "num_alloc_retries", 0
            )
            batch = batch.to(self.device)
            apply_device_preprocessing(batch, *self.settings.train_device_preprocessing)
            torch.cuda.reset_peak_memory_stats(self.device)
            if self.settings.trial_under_ddp:
                outputs = self.trainable_module(batch)
            else:
                outputs = self.model.training_step(batch)
            loss = torch.stack(list(outputs.losses.values()), dim=0).sum(dim=0).mean()
            loss.backward()
            torch.cuda.synchronize(self.device)
            backward_peak_bytes = torch.cuda.max_memory_allocated(self.device)
            reserved_peak_bytes = torch.cuda.max_memory_reserved(self.device)
            del outputs, loss, batch
            torch.cuda.reset_peak_memory_stats(self.device)
            if self.settings.clip_grad_norm:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.settings.clip_grad_norm
                )
            for optimizer in self.optimizers:
                optimizer.step()
            torch.cuda.synchronize(self.device)
            step_readings.append(
                {
                    "backward_peak_bytes": backward_peak_bytes,
                    "optimizer_step_peak_bytes": torch.cuda.max_memory_allocated(
                        self.device
                    ),
                    "peak_reserved_bytes": max(
                        reserved_peak_bytes,
                        torch.cuda.max_memory_reserved(self.device),
                    ),
                    "had_allocator_retries": torch.cuda.memory_stats(self.device).get(
                        "num_alloc_retries", 0
                    )
                    > retries_before_step,
                }
            )
            for optimizer in self.optimizers:
                optimizer.zero_grad(set_to_none=True)
        return step_readings

    def trial(self, batch_size: int) -> BatchSizeTrialMeasurement | None:
        """Train at one batch size and return its measurement if it fits.

        A batch size fits when its steps neither run out of memory nor need
        allocator retries.

        Args:
            batch_size: Batch size to try.

        Returns:
            The measurement when the batch size fits, otherwise None.

        Raises:
            _GpuUnusableAfterOutOfMemoryError: If the CUDA context fails its
                health check after an out of memory error.
        """
        self.trial_count += 1
        gc.collect()
        torch.cuda.empty_cache()
        failure: BaseException | None = None
        try:
            step_readings = self.run_training_steps(
                batch_size, _MEASURED_STEPS_PER_TRIAL
            )
        except (RuntimeError, OutOfMemoryError) as error:
            if not (
                isinstance(error, OutOfMemoryError)
                or _is_gpu_out_of_memory_error(error)
            ):
                raise
            failure = error
        if failure is not None:
            failure_message = str(failure)[:200]
            del failure
            for optimizer in self.optimizers:
                optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(
                "[autotune] batch %s ran out of memory: %s", batch_size, failure_message
            )
            if not _gpu_is_healthy(self.device):
                raise _GpuUnusableAfterOutOfMemoryError(failure_message)
            return None
        measurement = BatchSizeTrialMeasurement(
            batch_size=batch_size,
            backward_peak_bytes=max(r["backward_peak_bytes"] for r in step_readings),
            optimizer_step_peak_bytes=max(
                r["optimizer_step_peak_bytes"] for r in step_readings
            ),
            peak_reserved_bytes=max(r["peak_reserved_bytes"] for r in step_readings),
            steps_with_allocator_retries=sum(
                1 for r in step_readings if r["had_allocator_retries"]
            ),
        )
        self.trials.append(measurement)
        fits = measurement.steps_with_allocator_retries == 0
        logger.info(
            "[autotune] batch %s %s | backward peak %.0f MB | optimizer step peak "
            "%.0f MB | reserved %.0f MB | steps with allocator retries %s",
            batch_size,
            "fits" if fits else "needs allocator retries",
            measurement.backward_peak_bytes / 2**20,
            measurement.optimizer_step_peak_bytes / 2**20,
            measurement.peak_reserved_bytes / 2**20,
            measurement.steps_with_allocator_retries,
        )
        return measurement if fits else None


def _search_largest_fitting_batch_size(
    trial_runner: _TrialRunner,
    settings: BatchSizeSearchSettings,
    target_reserved_bytes: int,
    search_result: BatchSizeSearchResult,
) -> int:
    """Estimate the largest batch size from a two batch memory slope, then verify it.

    Args:
        trial_runner: Runner that trains the model at a batch size.
        settings: Search settings with the batch size range and known bounds.
        target_reserved_bytes: Reserved memory limit for training.
        search_result: Result object that records the known bounds as they move.

    Returns:
        The largest batch size that fits.
    """
    min_batch_size = settings.min_batch_size
    max_batch_size = settings.max_batch_size

    def record_fit(batch_size: int) -> None:
        search_result.largest_fitting_batch_size = max(
            batch_size, search_result.largest_fitting_batch_size or 0
        )

    def record_failure(batch_size: int) -> None:
        search_result.smallest_failing_batch_size = min(
            batch_size, search_result.smallest_failing_batch_size or batch_size
        )

    def bisect(largest_fit: int, smallest_failure: int) -> int:
        while smallest_failure - largest_fit > max(
            1, int(_BISECTION_TOLERANCE_FRACTION * largest_fit)
        ):
            middle = (largest_fit + smallest_failure) // 2
            if trial_runner.trial(middle) is not None:
                largest_fit = middle
                record_fit(middle)
            else:
                smallest_failure = middle
                record_failure(middle)
        return largest_fit

    if (
        settings.largest_fitting_batch_size is not None
        and settings.smallest_failing_batch_size is not None
    ):
        return bisect(
            settings.largest_fitting_batch_size, settings.smallest_failing_batch_size
        )

    small_slope_batch_size, large_slope_batch_size = (
        max(min_batch_size, min(batch_size, max_batch_size))
        for batch_size in _SLOPE_BATCH_SIZES
    )
    small_measurement = trial_runner.trial(small_slope_batch_size)
    if small_measurement is None:
        record_failure(small_slope_batch_size)
        if small_slope_batch_size == min_batch_size:
            raise OutOfMemoryError(
                f"Smallest batch size {min_batch_size} does not fit.", device="GPU"
            )
        record_fit(min_batch_size)
        return bisect(min_batch_size, small_slope_batch_size)
    record_fit(small_slope_batch_size)
    if large_slope_batch_size <= small_slope_batch_size:
        return small_slope_batch_size
    large_measurement = trial_runner.trial(large_slope_batch_size)
    if large_measurement is None:
        record_failure(large_slope_batch_size)
        return bisect(small_slope_batch_size, large_slope_batch_size)
    record_fit(large_slope_batch_size)
    if large_slope_batch_size >= max_batch_size:
        return large_slope_batch_size

    live_bytes_per_sample = (
        large_measurement.backward_peak_bytes - small_measurement.backward_peak_bytes
    ) / (large_slope_batch_size - small_slope_batch_size)
    reserved_per_live_byte = large_measurement.peak_reserved_bytes / max(
        large_measurement.backward_peak_bytes,
        large_measurement.optimizer_step_peak_bytes,
    )
    live_budget_bytes = target_reserved_bytes / reserved_per_live_byte
    if live_bytes_per_sample <= 0:
        logger.warning(
            "[autotune] memory did not grow from batch %s to %s; bisecting up to %s",
            small_slope_batch_size,
            large_slope_batch_size,
            max_batch_size,
        )
        if trial_runner.trial(max_batch_size) is not None:
            return max_batch_size
        record_failure(max_batch_size)
        return bisect(large_slope_batch_size, max_batch_size)
    estimated_batch_size = int(
        large_slope_batch_size
        + (live_budget_bytes - large_measurement.backward_peak_bytes)
        / live_bytes_per_sample
    )
    estimated_batch_size = max(
        large_slope_batch_size, min(estimated_batch_size, max_batch_size)
    )
    logger.info(
        "[autotune] %.1f MB per sample from batches %s and %s | reserved per live "
        "byte %.3f | estimate %s",
        live_bytes_per_sample / 2**20,
        small_slope_batch_size,
        large_slope_batch_size,
        reserved_per_live_byte,
        estimated_batch_size,
    )
    if estimated_batch_size == large_slope_batch_size:
        return large_slope_batch_size

    estimate_measurement = trial_runner.trial(estimated_batch_size)
    if estimate_measurement is None:
        record_failure(estimated_batch_size)
        stepped_down_batch_size = max(
            large_slope_batch_size,
            int(estimated_batch_size * _STEP_DOWN_AFTER_FAILED_ESTIMATE),
        )
        if stepped_down_batch_size == large_slope_batch_size:
            return large_slope_batch_size
        if trial_runner.trial(stepped_down_batch_size) is not None:
            record_fit(stepped_down_batch_size)
            return stepped_down_batch_size
        record_failure(stepped_down_batch_size)
        return bisect(large_slope_batch_size, stepped_down_batch_size)
    record_fit(estimated_batch_size)

    reserved_bytes_per_sample = (
        estimate_measurement.peak_reserved_bytes
        - large_measurement.peak_reserved_bytes
    ) / (estimated_batch_size - large_slope_batch_size)
    if reserved_bytes_per_sample <= 0:
        return estimated_batch_size
    refined_batch_size = min(
        max_batch_size,
        int(
            estimated_batch_size
            + (target_reserved_bytes - estimate_measurement.peak_reserved_bytes)
            / reserved_bytes_per_sample
        ),
    )
    if refined_batch_size <= estimated_batch_size:
        return estimated_batch_size
    logger.info("[autotune] refined estimate %s", refined_batch_size)
    if trial_runner.trial(refined_batch_size) is not None:
        record_fit(refined_batch_size)
        return refined_batch_size
    record_failure(refined_batch_size)
    return bisect(estimated_batch_size, refined_batch_size)


def _run_batch_size_search_worker(
    result_queue: Any,
    model_factory: Callable[[], NeuracoreModel],
    train_dataset: Dataset,
    collate_function: Callable,
    settings: BatchSizeSearchSettings,
    device_string: str,
) -> None:
    """Build the model once and run every batch size trial in this process."""
    logging.basicConfig(level=logging.INFO, handlers=build_stream_handlers())
    device = torch.device(device_string)
    if device.index is None:
        device = torch.device(device.type, torch.cuda.current_device())
    search_result = BatchSizeSearchResult(
        status=_RESULT_COMPLETED,
        largest_fitting_batch_size=settings.largest_fitting_batch_size,
        smallest_failing_batch_size=settings.smallest_failing_batch_size,
    )
    ddp_group_started = False
    try:
        torch.cuda.set_device(device)
        model = model_factory().to(device)
        model.train()
        optimizers = model.configure_optimizers()
        trainable_module: nn.Module = model
        if settings.trial_under_ddp:
            from neuracore.ml.trainers.distributed_trainer import NestedModule

            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(settings.ddp_master_port)
            torch_distributed.init_process_group("nccl", rank=0, world_size=1)
            ddp_group_started = True
            trainable_module = DistributedDataParallel(
                NestedModule(model).to(device),
                device_ids=[device.index if device.index is not None else 0],
                find_unused_parameters=False,
            )
        trial_runner = _TrialRunner(
            model=model,
            trainable_module=trainable_module,
            optimizers=optimizers,
            train_dataset=train_dataset,
            collate_function=collate_function,
            settings=settings,
            device=device,
        )
        trial_runner.run_training_steps(
            min(max(_WARM_UP_BATCH_SIZE, settings.min_batch_size), _dataset_length(train_dataset)),
            1,
        )
        own_bytes_outside_allocator, other_process_bytes = (
            _gpu_memory_outside_pytorch_allocator(device)
        )
        total_bytes = torch.cuda.get_device_properties(device).total_memory
        target_reserved_bytes = int(
            min(
                total_bytes - own_bytes_outside_allocator - other_process_bytes,
                TRAINING_MAX_GPU_UTILIZATION * total_bytes,
            )
        )
        search_result.target_reserved_bytes = target_reserved_bytes
        torch.cuda.set_per_process_memory_fraction(
            target_reserved_bytes / total_bytes, device
        )
        logger.info(
            "[autotune] reserved memory target %.0f MB of %.0f MB | outside the "
            "allocator: this process %.0f MB, other processes %.0f MB",
            target_reserved_bytes / 2**20,
            total_bytes / 2**20,
            own_bytes_outside_allocator / 2**20,
            other_process_bytes / 2**20,
        )
        if other_process_bytes > 0:
            logger.warning(
                "[autotune] other processes hold %.0f MB on this GPU; the result "
                "assumes they keep the same amount during training",
                other_process_bytes / 2**20,
            )
        if settings.batch_size_to_validate is not None:
            fits = trial_runner.trial(settings.batch_size_to_validate) is not None
            search_result.chosen_batch_size = (
                settings.batch_size_to_validate if fits else None
            )
        else:
            search_result.chosen_batch_size = _search_largest_fitting_batch_size(
                trial_runner, settings, target_reserved_bytes, search_result
            )
        search_result.trials = trial_runner.trials
        search_result.trial_count = trial_runner.trial_count
    except _GpuUnusableAfterOutOfMemoryError as error:
        search_result.status = _RESULT_RESTART_REQUIRED
        search_result.error_message = str(error)
    except OutOfMemoryError as error:
        search_result.status = _RESULT_ERROR
        search_result.error_message = str(error)
    except BaseException as error:  # noqa: BLE001
        logger.exception("[autotune] tuning process failed")
        search_result.status = _RESULT_ERROR
        search_result.error_message = repr(error)
    finally:
        if ddp_group_started:
            torch_distributed.destroy_process_group()
    result_queue.put(search_result)


def _find_free_local_port() -> int:
    """Return a free TCP port on the local machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        return int(port_socket.getsockname()[1])


def _run_tuning_process(
    model_factory: Callable[[], NeuracoreModel],
    train_dataset: Dataset,
    collate_function: Callable,
    settings: BatchSizeSearchSettings,
    device: torch.device,
) -> BatchSizeSearchResult:
    """Run the trials in a spawned process and restart it if CUDA becomes unusable.

    Args:
        model_factory: Callable that builds a fresh model.
        train_dataset: Training split for the trials.
        collate_function: Function that batches dataset samples.
        settings: Search settings.
        device: CUDA device for the trials.

    Returns:
        The completed search result.

    Raises:
        OutOfMemoryError: If the smallest batch size does not fit.
        RuntimeError: If the tuning process crashes or keeps losing its CUDA
            context.
    """
    spawn_context = multiprocessing.get_context("spawn")
    for process_start in range(_MAX_TUNING_PROCESS_STARTS):
        if settings.trial_under_ddp:
            settings.ddp_master_port = _find_free_local_port()
        result_queue: Any = spawn_context.Queue()
        tuning_process = spawn_context.Process(
            target=_run_batch_size_search_worker,
            args=(
                result_queue,
                model_factory,
                train_dataset,
                collate_function,
                settings,
                str(device),
            ),
        )
        tuning_process.start()
        search_result: BatchSizeSearchResult | None = None
        try:
            while search_result is None:
                try:
                    search_result = result_queue.get(timeout=_RESULT_POLL_INTERVAL_S)
                except queue_module.Empty:
                    if not tuning_process.is_alive():
                        break
        finally:
            tuning_process.join(timeout=_SUBPROCESS_TERMINATE_TIMEOUT_S)
            if tuning_process.is_alive():
                tuning_process.kill()
                tuning_process.join()
        if search_result is None:
            raise RuntimeError(
                "Batch size tuning process exited with code "
                f"{tuning_process.exitcode} before reporting a result."
            )
        if search_result.status == _RESULT_COMPLETED:
            return search_result
        if search_result.status == _RESULT_ERROR:
            if "does not fit" in search_result.error_message:
                raise OutOfMemoryError(search_result.error_message, device=str(device))
            raise RuntimeError(
                f"Batch size tuning failed: {search_result.error_message}"
            )
        logger.warning(
            "[autotune] CUDA context unusable after an out of memory error; "
            "restarting the tuning process (start %s of %s)",
            process_start + 2,
            _MAX_TUNING_PROCESS_STARTS,
        )
        settings.largest_fitting_batch_size = search_result.largest_fitting_batch_size
        settings.smallest_failing_batch_size = (
            search_result.smallest_failing_batch_size
        )
    raise RuntimeError(
        "Batch size tuning lost its CUDA context "
        f"{_MAX_TUNING_PROCESS_STARTS} times after out of memory errors."
    )


def _search_settings_from_config(
    cfg: DictConfig,
    train_dataset: Dataset,
    trial_under_ddp: bool,
) -> BatchSizeSearchSettings:
    """Build search settings from the training config and the training split.

    Args:
        cfg: Training configuration.
        train_dataset: Training split.
        trial_under_ddp: Whether trials wrap the model in single process DDP.

    Returns:
        Search settings with the batch size range clamped to the dataset.
    """
    dataset_length = _dataset_length(train_dataset)
    max_batch_size = min(
        cfg.max_batch_size if "max_batch_size" in cfg else dataset_length,
        dataset_length,
    )
    min_batch_size = min(
        cfg.min_batch_size if "min_batch_size" in cfg else 2, dataset_length
    )
    if min_batch_size > max_batch_size:
        raise ValueError(
            f"min_batch_size ({min_batch_size}) must be "
            f"<= max_batch_size ({max_batch_size})"
        )
    train_input_preprocessing, train_output_preprocessing = (
        resolve_input_output_preprocessing(
            cfg.get("train_preprocessing"), role_name="train_preprocessing"
        )
    )
    algorithm_config = cfg.get("algorithm") or {}
    return BatchSizeSearchSettings(
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        trial_under_ddp=trial_under_ddp,
        clip_grad_norm=algorithm_config.get("clip_grad_norm"),
        train_device_preprocessing=(
            train_input_preprocessing.split_by_stage()[1],
            train_output_preprocessing.split_by_stage()[1],
        ),
    )


def find_optimal_batch_size(
    cfg: DictConfig,
    model_factory: Callable[[], NeuracoreModel],
    dataset: PytorchSynchronizedDataset,
    device: torch.device,
    trial_under_ddp: bool = False,
) -> int:
    """Return the largest batch size that trains within the GPU memory target.

    Args:
        cfg: Training configuration.
        model_factory: Callable that builds a fresh model.
        dataset: Full synchronized dataset with train preprocessing.
        device: CUDA device to tune on.
        trial_under_ddp: Whether trials wrap the model in single process DDP,
            as multi GPU training does.

    Returns:
        The chosen batch size.
    """
    if not torch.cuda.is_available() or "cuda" not in device.type:
        raise ValueError("Autotuning batch size is only supported on GPUs.")
    train_dataset, _ = _split_train_val_dataset(cfg, dataset)
    settings = _search_settings_from_config(cfg, train_dataset, trial_under_ddp)
    logger.info(
        "[autotune] starting | %s training samples | batch range [%s, %s] | DDP "
        "trials %s",
        _dataset_length(train_dataset),
        settings.min_batch_size,
        settings.max_batch_size,
        trial_under_ddp,
    )
    start_time = time.perf_counter()
    search_result = _run_tuning_process(
        model_factory=model_factory,
        train_dataset=train_dataset,
        collate_function=dataset.collate_fn,
        settings=settings,
        device=device,
    )
    chosen_batch_size = int(search_result.chosen_batch_size or settings.min_batch_size)
    logger.info(
        "[autotune] done in %.1f s | %s trials | chosen batch size %s | reserved "
        "memory target %.0f MB",
        time.perf_counter() - start_time,
        search_result.trial_count,
        chosen_batch_size,
        search_result.target_reserved_bytes / 2**20,
    )
    return chosen_batch_size


def is_valid_batch_size(
    cfg: DictConfig,
    model_factory: Callable[[], NeuracoreModel],
    dataset: PytorchSynchronizedDataset,
    batch_size: int,
    device: torch.device,
    trial_under_ddp: bool = False,
) -> bool:
    """Return whether a batch size trains within the same memory target the tuner uses.

    Args:
        cfg: Training configuration.
        model_factory: Callable that builds a fresh model.
        dataset: Full synchronized dataset with train preprocessing.
        batch_size: Batch size to check.
        device: CUDA device to check on.
        trial_under_ddp: Whether the trial wraps the model in single process DDP.

    Returns:
        True if a trial at the batch size fits without allocator retries.
    """
    if not torch.cuda.is_available() or "cuda" not in device.type:
        raise ValueError("Batch size testing is only supported on GPUs.")
    train_dataset, _ = _split_train_val_dataset(cfg, dataset)
    settings = _search_settings_from_config(cfg, train_dataset, trial_under_ddp)
    settings.batch_size_to_validate = min(batch_size, _dataset_length(train_dataset))
    search_result = _run_tuning_process(
        model_factory=model_factory,
        train_dataset=train_dataset,
        collate_function=dataset.collate_fn,
        settings=settings,
        device=device,
    )
    return search_result.chosen_batch_size is not None


def _split_train_val_dataset(
    cfg: DictConfig,
    dataset: PytorchSynchronizedDataset,
) -> tuple[Dataset, Dataset]:
    """Split dataset into deterministic train and validation subsets.

    Args:
        cfg: Training config providing validation_split, seed and
            inference_preprocessing.
        dataset: Full synchronized dataset with train preprocessing.

    Returns:
        The train and validation subsets from split_train_val_datasets.
    """
    (
        inference_input_preprocessing_config,
        inference_output_preprocessing_config,
    ) = resolve_input_output_preprocessing(
        cfg.get("inference_preprocessing"),
        role_name="inference_preprocessing",
    )
    return split_train_val_datasets(
        dataset,
        validation_split=cfg.validation_split,
        seed=cfg.seed,
        inference_input_preprocessing_config=inference_input_preprocessing_config,
        inference_output_preprocessing_config=inference_output_preprocessing_config,
    )
