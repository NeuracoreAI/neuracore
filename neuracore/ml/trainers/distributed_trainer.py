"""Distributed Trainer."""

import logging
import os
import statistics
import time
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from neuracore.ml import BatchedTrainingOutputs, NeuracoreModel
from neuracore.ml.core.ml_types import BatchedTrainingSamples
from neuracore.ml.logging.training_logger import TrainingLogger
from neuracore.ml.utils.device_utils import get_default_device
from neuracore.ml.utils.memory_monitor import MemoryMonitor, OutOfMemoryError
from neuracore.ml.utils.training_storage_handler import TrainingStorageHandler

logger = logging.getLogger(__name__)

# Only update the training metadata every N steps to avoid excessive API calls
UPDATE_TRAINING_METADATA_EVERY = 20

# Checking RAM/VRAM headroom reads /proc/meminfo and the CUDA allocator, which
# is far more often than a slow-moving quantity needs. Matches the interval the
# dataset uses for the same check (see PytorchSynchronizedDataset).
CHECK_MEMORY_INTERVAL = 100


class NestedModule(nn.Module):
    """A special case to allow NeuracoreModel to be used in DDP."""

    def __init__(self, neuracore_model: NeuracoreModel):
        """Initialize the nested module.

        Args:
            neuracore_model: The NeuracoreModel instance to wrap
        """
        super().__init__()
        self.neuracore_model = neuracore_model

    def forward(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Forward pass for the nested module.

        Args:
            batch: A batch of training samples
        """
        return self.neuracore_model.training_step(batch)


class DistributedTrainer:
    """Trainer for distributed multi-GPU training with TensorBoard logging."""

    def __init__(
        self,
        model: NeuracoreModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        training_logger: TrainingLogger,
        storage_handler: TrainingStorageHandler,
        output_dir: Path,
        num_epochs: int,
        log_freq: int = 50,
        histogram_log_freq: int = 0,
        timing_sample_interval: int = 25,
        save_freq: int = 1,
        save_checkpoints: bool = True,
        keep_last_n_checkpoints: int = 5,
        clip_grad_norm: float | None = None,
        rank: int = 0,
        world_size: int = 1,
        device: torch.device | None = None,
    ):
        """Initialize the distributed trainer.

        Args:
            model: The model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            training_logger: Logger for training metrics (TensorBoard, etc.)
            storage_handler: Handler for model storage
            output_dir: Directory for output files
            num_epochs: Number of epochs to train
            log_freq: Frequency to log metrics (in steps)
            histogram_log_freq: Frequency to log weight/gradient histograms
                (in steps). 0 disables them. Kept separate from log_freq
                because a histogram of every parameter and every gradient
                costs orders of magnitude more than a handful of scalars.
            timing_sample_interval: Collect a device-synchronised timing
                breakdown every N iterations, reported at the end of each
                epoch. 0 reports only total iteration times.
            save_freq: Frequency to save checkpoints (in epochs)
            save_checkpoints: Whether to save checkpoints
            keep_last_n_checkpoints: Number of checkpoints to keep
            clip_grad_norm: Maximum norm for gradient clipping
            rank: Rank of this process
            world_size: Total number of processes/GPUs
            device: Optional device to use for training
        """
        if keep_last_n_checkpoints <= 0:
            raise ValueError("keep_last_n_checkpoints must be greater than 0")

        self.device = device or get_default_device(gpu_index=rank)

        logger.info(f"Process {rank} using device: {self.device}")

        # Set up the model for distributed training
        self.model = model.to(self.device)

        if torch.cuda.is_available() and world_size > 1:
            self.model = NestedModule(self.model).to(self.device)
            self.model = DDP(
                self.model, device_ids=[rank], find_unused_parameters=False
            )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.training_logger = training_logger
        self.storage_handler = storage_handler
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.log_freq = log_freq
        self.histogram_log_freq = histogram_log_freq
        self.timing_sample_interval = timing_sample_interval
        self.save_freq = save_freq
        self.save_checkpoints = save_checkpoints
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.clip_grad_norm = clip_grad_norm
        self.rank = rank
        self.world_size = world_size
        self.global_train_step = 0
        self.global_val_step = 0

        # Progress bars are rendered only on rank 0, and never when logs are
        # being shipped to the cloud. Cached because it also gates whether the
        # loss is worth pulling off the device to display.
        self._pbar_enabled = rank == 0 and not storage_handler.log_to_cloud
        # Histograms are skipped entirely when the backend discards them
        # (CloudTrainingLogger), and on non-zero ranks, which would otherwise
        # race each other writing into one TensorBoard directory.
        self._histograms_enabled = (
            rank == 0 and histogram_log_freq > 0 and training_logger.supports_histograms
        )
        # Copies host memory into pinned staging buffers only pays for itself
        # when the destination is a real device.
        self._transfer_non_blocking = self.device.type == "cuda"

        num_training_steps = self.num_epochs * len(self.train_loader)
        self.optimizers = model.configure_optimizers()
        self.schedulers = model.configure_schedulers(
            self.optimizers,
            num_training_steps,
        )
        # Create checkpoint directory
        if rank == 0:
            self.checkpoint_dir = output_dir / "checkpoints"
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one epoch of training.

        Args:
            epoch: Current epoch number

        Returns:
            A dictionary of averaged metrics for the epoch
        """
        self.model.train()
        accumulator = _MetricAccumulator()

        memory_monitor = MemoryMonitor(
            max_ram_utilization=0.8,
            max_gpu_utilization=0.95,
            gpu_id=self.device.index if self.device.type == "cuda" else None,
        )
        timings = _EpochTimings(
            device=self.device,
            sample_interval=self.timing_sample_interval if self.rank == 0 else 0,
            batch_size=self.train_loader.batch_size,
        )

        # Progress bar only on rank 0
        pbar = tqdm(
            self.train_loader,
            desc=f"Training Epoch {epoch}",
            disable=not self._pbar_enabled,
        )

        for optimizer in self.optimizers:
            optimizer.zero_grad()

        timings.start_iteration(0)
        for batch_idx, batch in enumerate(pbar):
            timings.mark("data_wait")
            if batch_idx % CHECK_MEMORY_INTERVAL == 0:
                memory_monitor.check_memory()

            # Move tensors to device and format batch
            batch = batch.to(self.device, non_blocking=self._transfer_non_blocking)
            timings.mark("to_device")

            # Forward pass
            if self.world_size > 1:
                batch_output = self.model(batch)
            else:
                batch_output = cast(NeuracoreModel, self.model).training_step(batch)
            loss = (
                torch.stack(list(batch_output.losses.values()), dim=0).sum(dim=0).mean()
            )
            timings.mark("forward")

            # Backward pass
            loss.backward()
            timings.mark("backward")

            # Clip gradients if configured
            if self.clip_grad_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            for optimizer in self.optimizers:
                optimizer.step()

            if self.schedulers:
                for scheduler in self.schedulers:
                    scheduler.step()
            timings.mark("optimizer")

            is_log_step = self.log_freq > 0 and (
                self.global_train_step % self.log_freq == 0
            )
            if is_log_step:
                self._log_scalars(
                    batch_output.losses,
                    self.global_train_step,
                    prefix="train/step/loss",
                )
                self._log_scalars(
                    batch_output.metrics,
                    self.global_train_step,
                    prefix="train/step/metrics",
                )
            if (
                self._histograms_enabled
                and self.global_train_step % self.histogram_log_freq == 0
            ):
                self._log_gradients(self.global_train_step)
                self._log_weights(self.global_train_step)

            # loss.item() is a device sync. Only pay for it when a human is
            # actually watching the bar, and then only as often as we log.
            if self._pbar_enabled and is_log_step:
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "step": self.global_train_step}
                )
            accumulator.update(batch_output)
            self.global_train_step += 1
            if (
                self.rank == 0
                and self.global_train_step % UPDATE_TRAINING_METADATA_EVERY == 0
            ):
                self.storage_handler.update_training_progress(
                    epoch=epoch, step=self.global_train_step
                )

            # Free-up GPU during validation or before next forward pass
            for optimizer in self.optimizers:
                optimizer.zero_grad()

            del batch, batch_output, loss
            timings.mark("logging")
            timings.end_iteration()
            timings.start_iteration(batch_idx + 1)

        timings.log_summary("train", epoch)
        avg_epoch_losses, avg_epoch_metrics = accumulator.averages()
        self._log_scalars(avg_epoch_losses, epoch, prefix="train/epoch/loss")
        self._log_scalars(avg_epoch_metrics, epoch, prefix="train/epoch/metrics")
        return avg_epoch_losses

    def validate(self, epoch: int) -> dict[str, float]:
        """Run validation.

        Args:
            epoch: Current epoch number

        Returns:
            A dictionary of averaged validation metrics
        """
        self.model.train()  # Keep in train mode to get losses
        accumulator = _MetricAccumulator()
        timings = _EpochTimings(
            device=self.device,
            sample_interval=self.timing_sample_interval if self.rank == 0 else 0,
            batch_size=self.val_loader.batch_size,
        )

        pbar = tqdm(
            self.val_loader,
            desc=f"Validation Epoch {epoch}",
            disable=not self._pbar_enabled,
        )

        timings.start_iteration(0)
        for batch_idx, batch in enumerate(pbar):
            timings.mark("data_wait")
            batch = batch.to(self.device, non_blocking=self._transfer_non_blocking)
            timings.mark("to_device")

            # Forward pass
            if self.world_size > 1:
                batch_output = self.model(batch)
            else:
                batch_output = cast(NeuracoreModel, self.model).training_step(batch)
            timings.mark("forward")

            if self.log_freq > 0 and self.global_val_step % self.log_freq == 0:
                self._log_scalars(
                    batch_output.losses, self.global_val_step, prefix="val/step/loss"
                )
                self._log_scalars(
                    batch_output.metrics,
                    self.global_val_step,
                    prefix="val/step/metrics",
                )
            accumulator.update(batch_output)
            self.global_val_step += 1
            timings.mark("logging")
            timings.end_iteration()
            timings.start_iteration(batch_idx + 1)

        timings.log_summary("val", epoch)
        avg_losses, avg_metrics = accumulator.averages()
        self._log_scalars(avg_losses, epoch, prefix="val/epoch/loss")
        self._log_scalars(avg_metrics, epoch, prefix="val/epoch/metrics")
        return avg_losses

    def train(self, start_epoch: int = 0) -> None:
        """Run the training loop.

        Args:
            start_epoch: Epoch to start from (for resuming training)
        """
        if self.rank == 0:
            self.storage_handler.update_training_progress(
                epoch=start_epoch, step=self.global_train_step
            )

        try:
            start_epoch = max(start_epoch, 1)
            for epoch in range(start_epoch, self.num_epochs + 1):
                # Set epoch for distributed sampler to
                # ensure different shuffling each epoch
                if isinstance(self.train_loader.sampler, DistributedSampler):
                    self.train_loader.sampler.set_epoch(epoch)

                train_loss_metrics = self.train_epoch(epoch)

                # Save checkpoint and artifacts periodically (only from rank 0)
                if self.rank == 0 and epoch % self.save_freq == 0:
                    self.save_checkpoint(epoch, train_loss_metrics)

                    # Save model artifacts
                    self.storage_handler.save_model_artifacts(
                        model=self.get_model_without_ddp(),
                        output_dir=self.output_dir,
                    )

                with torch.no_grad():
                    self.validate(epoch)

                # Save metadata
                if self.rank == 0:
                    self.storage_handler.update_training_progress(
                        epoch=epoch,
                        step=self.global_train_step,
                    )
                    # Flush logger to ensure data is written
                    if hasattr(self.training_logger, "flush"):
                        self.training_logger.flush()

        except OutOfMemoryError:
            logger.error(
                "Batch size %s is too large. "
                "Try reducing batch size or using a more powerful machine.",
                self.train_loader.batch_size,
            )
            raise
        except Exception:
            logger.error("Error during training.", exc_info=True)
            raise
        finally:
            if self.rank == 0:
                # Checkpoint/artifact uploads run in the background (see
                # TrainingStorageHandler); block here until they've all
                # landed so the process can't exit — and the training
                # VM/container be torn down — while the final checkpoint is
                # still mid-upload.
                self.storage_handler.wait_for_pending_uploads()
                # Progress updates are also sent off-thread, so flush the last
                # one rather than letting it die with the worker.
                self.storage_handler.wait_for_pending_progress_updates()
                # Close the logger
                self.training_logger.close()

    def get_model_without_ddp(self) -> nn.Module:
        """Get the model without DDP wrapper.

        Returns:
            The underlying model if wrapped in DDP, or the model itself otherwise.
        """
        if isinstance(self.model, DDP):
            return self.model.module.neuracore_model
        return self.model

    def save_checkpoint(self, epoch: int, metrics: dict) -> None:
        """Save checkpoint with metadata.

        Args:
            epoch: Current epoch number
            metrics: Metrics to save in the checkpoint
        """
        if not self.save_checkpoints or self.rank != 0:
            return
        logger.info("Saving checkpoint...")

        # Get the model state dict (different for DDP vs non-DDP models)
        model_state = self.get_model_without_ddp().state_dict()

        checkpoint = {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_states": [opt.state_dict() for opt in self.optimizers],
            "scheduler_states": (
                [sch.state_dict() for sch in self.schedulers] if self.schedulers else []
            ),
            "metrics": metrics,
            "global_train_step": self.global_train_step,
            "global_val_step": self.global_val_step,
        }

        checkpoint_path = self.checkpoint_dir / f"checkpoint_{epoch}.pt"
        self.storage_handler.save_checkpoint(checkpoint, checkpoint_path)
        checkpoint_epoch_to_remove = epoch - self.keep_last_n_checkpoints
        if checkpoint_epoch_to_remove > 0:
            checkpoint_to_remove = (
                self.checkpoint_dir / f"checkpoint_{checkpoint_epoch_to_remove}.pt"
            )
            self.storage_handler.delete_checkpoint(checkpoint_to_remove)

        logger.info("... checkpoint saved!")

    def load_checkpoint(self, path: str) -> dict:
        """Load checkpoint and restore training state.

        Args:
            path: Path to the checkpoint file

        Returns:
            A dictionary containing the checkpoint data
        """
        checkpoint = self.storage_handler.load_checkpoint(path)

        # Handle model loading (different for DDP vs non-DDP models)
        self.get_model_without_ddp().load_state_dict(checkpoint["model_state"])
        for optimizer, opt_state in zip(
            self.optimizers, checkpoint["optimizer_states"]
        ):
            optimizer.load_state_dict(opt_state)
        if self.schedulers and checkpoint.get("scheduler_states"):
            for scheduler, sch_state in zip(
                self.schedulers, checkpoint["scheduler_states"]
            ):
                scheduler.load_state_dict(sch_state)
        # Restore step counters
        self.global_train_step = checkpoint.get("global_train_step", 0)
        self.global_val_step = checkpoint.get("global_val_step", 0)

        return checkpoint

    def _log_gradients(self, step: int) -> None:
        """Log gradient histograms for model parameters.

        Args:
            step: Training step.
        """
        if not self._histograms_enabled:
            return
        model = self.get_model_without_ddp()
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.training_logger.log_histogram(
                    f"gradients/{name}", param.grad, step
                )

    def _log_weights(self, step: int) -> None:
        """Log weight histograms for model parameters.

        Args:
            step: Training step.
        """
        if not self._histograms_enabled:
            return
        model = self.get_model_without_ddp()
        for name, param in model.named_parameters():
            self.training_logger.log_histogram(f"weights/{name}", param, step)

    def _log_scalars(
        self, scalars: dict[str, float], step: int, prefix: str = "train/"
    ) -> None:
        """Log batch outputs to TensorBoard.

        Args:
            scalars: Dictionary of scalar values to log
            step: Training step
            prefix: Prefix for the log names (e.g., "train/step" or "val/batch")
        """
        if self.rank != 0:
            return
        for scalar_name, scalar_value in scalars.items():
            scalar_value = (
                scalar_value.item()
                if isinstance(scalar_value, torch.Tensor)
                else scalar_value
            )
            self.training_logger.log_scalar(
                f"{prefix}/{scalar_name}", scalar_value, step
            )


class _MetricAccumulator:
    """Running sums of per-step losses and metrics for one epoch.

    Tensor values are summed on whichever device they arrive on and read back
    once, when the epoch ends. Summing on the host instead would mean a
    blocking device-to-host copy per loss and per metric on every single step.
    """

    def __init__(self) -> None:
        """Initialize empty accumulators."""
        self._loss_sums: dict[str, torch.Tensor | float] = {}
        self._metric_sums: dict[str, torch.Tensor | float] = {}
        self._loss_steps = 0
        self._metric_steps = 0

    @staticmethod
    def _add(
        sums: dict[str, torch.Tensor | float], values: dict[str, torch.Tensor]
    ) -> None:
        for key, value in values.items():
            contribution = value.detach() if isinstance(value, torch.Tensor) else value
            if key in sums:
                sums[key] = sums[key] + contribution
            else:
                sums[key] = contribution

    def update(self, batch_output: BatchedTrainingOutputs) -> None:
        """Add one step's losses and metrics to the running totals."""
        if batch_output.losses:
            self._add(self._loss_sums, batch_output.losses)
            self._loss_steps += 1
        if batch_output.metrics:
            self._add(self._metric_sums, batch_output.metrics)
            self._metric_steps += 1

    @staticmethod
    def _finalize(
        sums: dict[str, torch.Tensor | float], steps: int
    ) -> dict[str, float]:
        if steps == 0:
            return {}
        return {
            key: (total.item() if isinstance(total, torch.Tensor) else float(total))
            / steps
            for key, total in sums.items()
        }

    def averages(self) -> tuple[dict[str, float], dict[str, float]]:
        """Return the epoch-averaged losses and metrics as plain floats."""
        return (
            self._finalize(self._loss_sums, self._loss_steps),
            self._finalize(self._metric_sums, self._metric_steps),
        )


def _synchronize_device(device: torch.device) -> None:
    """Block until queued work on ``device`` has finished.

    Forward and backward passes enqueue kernels and return before they run, so
    host-side timestamps around them measure launch time, not compute time.
    Timing sections are only meaningful with a barrier between them — which is
    why timing is sampled rather than run on every iteration.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class _EpochTimings:
    """Collects a per-epoch breakdown of where iteration time went.

    Every iteration is wall-clock timed (two ``perf_counter`` calls, which is
    noise next to a training step). The finer per-section breakdown, which
    needs a device barrier to mean anything, is only collected every
    ``sample_interval`` iterations so the barriers do not serialise the run.
    """

    SECTIONS = ("data_wait", "to_device", "forward", "backward", "optimizer", "logging")

    def __init__(
        self,
        device: torch.device,
        sample_interval: int,
        batch_size: int | None,
    ) -> None:
        """Initialize the collector.

        Args:
            device: Device being trained on, used for the sampling barriers.
            sample_interval: Collect a detailed breakdown every N iterations.
                0 disables the breakdown; total iteration times are still kept.
            batch_size: Samples per batch, used to derive per-sample figures.
                None omits them.
        """
        self.device = device
        self.sample_interval = sample_interval
        self.batch_size = batch_size
        self._sections: dict[str, list[float]] = {}
        self._iteration_times: list[float] = []
        self._sampling = False
        self._last_mark = 0.0
        self._iteration_start = 0.0

    def start_iteration(self, index: int) -> None:
        """Begin an iteration, deciding whether to sample it in detail."""
        self._sampling = self.sample_interval > 0 and index % self.sample_interval == 0
        self._iteration_start = time.perf_counter()
        self._last_mark = self._iteration_start

    def mark(self, section: str) -> None:
        """Attribute the time since the previous mark to ``section``."""
        if not self._sampling:
            return
        _synchronize_device(self.device)
        now = time.perf_counter()
        self._sections.setdefault(section, []).append(now - self._last_mark)
        self._last_mark = now

    def end_iteration(self) -> None:
        """Close out an iteration and record its total wall time."""
        self._iteration_times.append(time.perf_counter() - self._iteration_start)
        self._sampling = False

    def log_summary(self, label: str, epoch: int) -> None:
        """Log the epoch's timing breakdown.

        Args:
            label: Phase name, e.g. ``"train"`` or ``"val"``.
            epoch: Epoch number the summary covers.
        """
        if not self._iteration_times:
            return

        iterations = len(self._iteration_times)
        total_s = sum(self._iteration_times)
        mean_iter_s = total_s / iterations

        lines = [
            f"[timing:{label}] epoch {epoch}: {iterations} iterations "
            f"in {total_s:.1f}s",
            f"  mean iteration        {mean_iter_s * 1000:9.2f} ms"
            f"   ({1 / mean_iter_s if mean_iter_s else 0:.2f} it/s)",
        ]
        if self.batch_size:
            samples_per_s = self.batch_size / mean_iter_s if mean_iter_s else 0.0
            lines.append(
                f"  mean per sample       "
                f"{mean_iter_s / self.batch_size * 1000:9.2f} ms"
                f"   ({samples_per_s:.1f} samples/s)"
            )

        if self._sections:
            means = {
                section: statistics.mean(self._sections[section])
                for section in self.SECTIONS
                if section in self._sections
            }
            sampled = len(next(iter(self._sections.values())))
            sampled_total = sum(means.values())
            lines.append(
                f"  breakdown over {sampled} sampled iterations "
                f"(sum {sampled_total * 1000:.2f} ms). Sampled iterations are "
                "barrier-separated so each section can be attributed, which "
                "removes host/device overlap - expect the sum to exceed the "
                "mean iteration above:"
            )
            for section, mean_s in means.items():
                share = mean_s / sampled_total * 100 if sampled_total else 0.0
                lines.append(
                    f"    {section:<18}{mean_s * 1000:9.2f} ms   ({share:4.1f}%)"
                )
            data_wait = means.get("data_wait")
            if data_wait is not None and self.batch_size:
                lines.append(
                    f"  dataloader stall per sample "
                    f"{data_wait / self.batch_size * 1000:.3f} ms "
                    f"(amortised over the batch; ~0 means workers keep up)"
                )

        logger.info("\n".join(lines))


def setup_distributed(rank: int, world_size: int) -> None:
    """Initialize the distributed process group.

    Args:
        rank: Rank of this process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Set device for this process
    torch.cuda.set_device(rank)

    # Initialize process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    logger.info(f"Initialized process group for rank {rank}/{world_size}")


def cleanup_distributed() -> None:
    """Clean up the distributed process group."""
    dist.destroy_process_group()
    logger.info("Destroyed process group")
