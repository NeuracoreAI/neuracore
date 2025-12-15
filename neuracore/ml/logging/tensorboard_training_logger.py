"""TensorBoard-based training logger for local training.

This module provides a unified logging interface similar to Weights & Biases
but using TensorBoard for visualization.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from neuracore.ml.logging.training_logger import TrainingLogger

logger = logging.getLogger(__name__)


class TensorboardTrainingLogger(TrainingLogger):
    """TensorBoard-based logger for local training."""

    def __init__(
        self,
        log_dir: Path,
        run_name: str | None = None,
        sync_interval: int = 60,
    ):
        """Initialize TensorBoard logger.

        Args:
            log_dir: Directory to save logs. If None, creates temp directory.
            run_name: Name of the run. Defaults to a timestamp-based name.
            sync_interval: Interval in seconds for cloud sync (if enabled).
        """
        self.sync_interval = sync_interval
        self.run_name = run_name or f"run_{int(time.time())}"
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        logger.info(f"TensorBoard logger initialized: {self.log_dir}")

    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log a scalar metric.

        Args:
            name: Name of the metric (e.g., "train/loss", "val/accuracy").
            value: Scalar value to log.
            step: Training step.
        """
        self.writer.add_scalar(name, value, step)

    def log_scalars(self, scalars: dict[str, float], step: int) -> None:
        """Log multiple scalar metrics at once.

        Args:
            scalars: Dictionary of metric name -> value.
            step: Training step.
        """
        for name, value in scalars.items():
            self.writer.add_scalar(name, value, step)

    def log_image(
        self,
        name: str,
        image: np.ndarray | torch.Tensor,
        step: int,
        dataformats: str = "CHW",
    ) -> None:
        """Log an image.

        Args:
            name: Name for the image.
            image: Image data as numpy array or torch tensor.
            step: Training step.
            dataformats: Format of the image data (e.g., "CHW", "HWC").
        """
        self.writer.add_image(name, image, step, dataformats=dataformats)

    def log_images(
        self,
        name: str,
        images: np.ndarray | torch.Tensor,
        step: int,
        dataformats: str = "NCHW",
    ) -> None:
        """Log multiple images as a grid.

        Args:
            name: Name for the image grid.
            images: Batch of images as numpy array or torch tensor.
            step: Training step.
            dataformats: Format of the image data (e.g., "NCHW").
        """
        self.writer.add_images(name, images, step, dataformats=dataformats)

    def log_histogram(
        self,
        name: str,
        values: np.ndarray | torch.Tensor,
        step: int,
    ) -> None:
        """Log a histogram of values.

        Args:
            name: Name for the histogram.
            values: Values to create histogram from.
            step: Training step.
            bins: Binning method ("tensorflow", "auto", "fd", etc.).
        """
        self.writer.add_histogram(name, values, step)

    def log_text(self, name: str, text: str, step: int) -> None:
        """Log text data.

        Args:
            name: Name for the text data.
            text: Text string to log.
            step: Training step.
        """
        self.writer.add_text(name, text, step)

    def log_hyperparameters(
        self,
        hparams: dict[str, Any],
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Log hyperparameters and optionally metrics.

        Args:
            hparams: Dictionary of hyperparameters.
            metrics: Optional dictionary of metrics to associate with hparams.
        """
        # Convert any non-primitive types to strings for TensorBoard compatibility
        clean_hparams = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float, str, bool)):
                clean_hparams[key] = value
            else:
                clean_hparams[key] = str(value)

        self.writer.add_hparams(clean_hparams, metrics or {})

        # Also save as JSON for easier access
        config_file = self.log_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(hparams, f, indent=2, default=str)

    def log_model_graph(
        self, model: torch.nn.Module, input_to_model: torch.Tensor | None = None
    ) -> None:
        """Log the model computational graph.

        Args:
            model: PyTorch model to log.
            input_to_model: Example input tensor for the model.
        """
        self.writer.add_graph(model, input_to_model)

    def flush(self) -> None:
        """Force flush any pending writes to disk."""
        self.writer.flush()

    def close(self) -> None:
        """Close the logger and clean up resources."""
        self.writer.close()

    def get_log_dir(self) -> str:
        """Get the log directory path."""
        return str(self.log_dir)
