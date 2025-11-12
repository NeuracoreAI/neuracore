"""Distributed Trainer."""

import logging
import os
import traceback
from pathlib import Path
from typing import Optional

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
        save_freq: int = 1,
        save_checkpoints: bool = True,
        keep_last_n_checkpoints: int = 5,
        clip_grad_norm: Optional[float] = None,
        rank: int = 0,
        world_size: int = 1,
        device: Optional[torch.device] = None,
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
        self.save_freq = save_freq
        self.save_checkpoints = save_checkpoints
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.clip_grad_norm = clip_grad_norm
        self.rank = rank
        self.world_size = world_size
        self.optimizers = model.configure_optimizers()
        self.global_train_step = 0
        self.global_val_step = 0

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
        epoch_losses: list[dict[str, float]] = []
        epoch_metrics: list[dict[str, float]] = []

        memory_monitor = MemoryMonitor(
            max_ram_utilization=0.8, max_gpu_utilization=0.95
        )

        # Progress bar only on rank 0
        pbar = tqdm(
            self.train_loader,
            desc=f"Training Epoch {epoch}",
            disable=self.rank != 0 or self.storage_handler.log_to_cloud,
        )

        for batch_idx, batch in enumerate(pbar):
            memory_monitor.check_memory()

            # Move tensors to device and format batch
            batch = batch.to(self.device)

            # Forward pass
            if self.world_size > 1:
                batch_output = self.model(batch)
            else:
                batch_output = self.model.training_step(batch)
            loss = sum(batch_output.losses.values()).mean()

            # Backward pass
            for optimizer in self.optimizers:
                optimizer.zero_grad()
            loss.backward()

            # Clip gradients if configured
            if self.clip_grad_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            for optimizer in self.optimizers:
                optimizer.step()

            if self.log_freq > 0 and self.global_train_step % self.log_freq == 0:
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
                self._log_gradients(self.global_train_step)
                self._log_weights(self.global_train_step)
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "step": self.global_train_step}
            )
            epoch_losses, epoch_metrics = self._accumulate_epoch_metrics(
                batch_output, epoch_losses, epoch_metrics
            )
            self.global_train_step += 1
            if (
                self.rank == 0
                and self.global_train_step % UPDATE_TRAINING_METADATA_EVERY == 0
            ):
                self.storage_handler.update_training_metadata(
                    epoch=epoch, step=self.global_train_step
                )

        avg_epoch_losses, avg_epoch_metrics = self._average_epoch_metrics(
            epoch_losses, epoch_metrics, epoch
        )
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
        val_losses: list[dict[str, float]] = []
        val_metrics: list[dict[str, float]] = []

        pbar = tqdm(
            self.val_loader,
            desc=f"Validation Epoch {epoch}",
            disable=self.rank != 0 or self.storage_handler.log_to_cloud,
        )

        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)

            # Forward pass
            if self.world_size > 1:
                batch_output = self.model(batch)
            else:
                batch_output = self.model.training_step(batch)

            if self.log_freq > 0 and self.global_val_step % self.log_freq == 0:
                self._log_scalars(
                    batch_output.losses, self.global_val_step, prefix="val/step/loss"
                )
                self._log_scalars(
                    batch_output.metrics,
                    self.global_val_step,
                    prefix="val/step/metrics",
                )
            val_losses, val_metrics = self._accumulate_epoch_metrics(
                batch_output, val_losses, val_metrics
            )
            self.global_val_step += 1

        avg_losses, avg_metrics = self._average_epoch_metrics(
            val_losses, val_metrics, epoch
        )
        self._log_scalars(avg_losses, self.global_val_step, prefix="val/epoch/loss")
        self._log_scalars(avg_metrics, self.global_val_step, prefix="val/epoch/metrics")
        return avg_losses

    def train(self, start_epoch: int = 0) -> None:
        """Run the training loop.

        Args:
            start_epoch: Epoch to start from (for resuming training)
        """
        if self.rank == 0:
            self.storage_handler.update_training_metadata(
                epoch=start_epoch, step=self.global_train_step
            )

        err_msg: str = ""
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
                    self.storage_handler.update_training_metadata(
                        epoch=epoch,
                        step=self.global_train_step,
                    )
                    # Flush logger to ensure data is written
                    if hasattr(self.training_logger, "flush"):
                        self.training_logger.flush()

        except OutOfMemoryError:
            error_msg = (
                f"Batch size {self.train_loader.batch_size} is too large. "
                "Try reducing batch size or using a more powerful machine."
            )
            raise  # Re-raise to ensure proper exit code
        except Exception:
            error_msg = f"Error during training. \n{traceback.format_exc()}"
            raise  # Re-raise to ensure proper exit code
        finally:
            if err_msg:
                logger.error(error_msg)
                if self.rank == 0:
                    self.storage_handler.update_training_metadata(
                        epoch=epoch, step=self.global_train_step, error=err_msg
                    )
            # Close the logger
            if self.rank == 0:
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
            "metrics": metrics,
            "global_train_step": self.global_train_step,
            "global_val_step": self.global_val_step,
        }

        # Save regular checkpoint
        # TODO: remove latest in future PR and just keep numbered ones
        latest_checkpoint_path = self.checkpoint_dir / "checkpoint_latest.pt"
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{epoch}.pt"
        self.storage_handler.save_checkpoint(checkpoint, latest_checkpoint_path)
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

        # Restore step counters
        self.global_train_step = checkpoint.get("global_train_step", 0)
        self.global_val_step = checkpoint.get("global_val_step", 0)

        return checkpoint

    def _log_gradients(self, step: int) -> None:
        """Log gradient histograms for model parameters.

        Args:
            step: Training step.
        """
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

    def _accumulate_epoch_metrics(
        self,
        batch_output: BatchedTrainingOutputs,
        epoch_losses: list[dict[str, float]],
        epoch_metrics: list[dict[str, float]],
    ) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        """Accumulate metrics for the current epoch.

        Args:
            batch_output: Outputs from the training step
            epoch_losses: List of losses accumulated for the epoch
            epoch_metrics: List of metrics accumulated for the epoch

        Returns:
            Updated lists of losses and metrics for the epoch
        """
        # Accumulate losses
        if batch_output.losses:
            epoch_losses.append({
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in batch_output.losses.items()
            })

        # Accumulate metrics
        if batch_output.metrics:
            epoch_metrics.append({
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in batch_output.metrics.items()
            })

        return epoch_losses, epoch_metrics

    def _average_epoch_metrics(
        self,
        epoch_losses: list[dict[str, float]],
        epoch_metrics: list[dict[str, float]],
        epoch: int,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Average metrics across the epoch.

        Args:
            epoch_losses: List of losses accumulated for the epoch
            epoch_metrics: List of metrics accumulated for the epoch
            epoch: Current epoch number

        Returns:
            A dictionary of averaged losses and metrics for the epoch
        """
        avg_epoch_losses = {}
        avg_epoch_metrics = {}

        if epoch_losses:
            for key in epoch_losses[0].keys():
                avg_epoch_losses[key] = sum(x[key] for x in epoch_losses) / len(
                    epoch_losses
                )

        if epoch_metrics:
            for key in epoch_metrics[0].keys():
                avg_epoch_metrics[key] = sum(x[key] for x in epoch_metrics) / len(
                    epoch_metrics
                )

        return avg_epoch_losses, avg_epoch_metrics


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
