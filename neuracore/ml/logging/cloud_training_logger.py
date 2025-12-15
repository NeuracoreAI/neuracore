"""Logging interface for cloud-based training."""

import logging
import threading
import time
from typing import Any

import numpy as np
import requests
import torch

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.ml.logging.training_logger import TrainingLogger

logger = logging.getLogger(__name__)


class CloudTrainingLogger(TrainingLogger):
    """Logger for cloud-based training metrics."""

    def __init__(
        self,
        training_id: str,
        sync_interval: int = 10,
    ):
        """Initialize logger.

        Args:
            training_id: Optional training ID for cloud sync.
            sync_interval: Interval in seconds for cloud sync (if enabled).
        """
        self.training_id = training_id
        self.sync_interval = sync_interval

        # Cloud sync setup
        self._stop_sync = threading.Event()
        self._sync_thread: threading.Thread | None = None
        self._last_sync_time: float = 0.0
        # Maps log name, to a dict of (step, value) pairs
        self._store: dict[str, dict[int, Any]] = {}
        self._start_cloud_sync()

        logger.info("Cloud logger initialized.")

    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log a scalar metric.

        Args:
            name: Name of the metric (e.g., "train/loss", "val/accuracy").
            value: Scalar value to log.
            step: Training step.
        """
        self._store.setdefault(name, {})[step] = value

    def log_scalars(self, scalars: dict[str, float], step: int) -> None:
        """Log multiple scalar metrics at once.

        Args:
            scalars: Dictionary of metric name -> value.
            step: Training step.
        """
        for name, value in scalars.items():
            self._store.setdefault(name, {})[step] = value

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
        pass  # TODO: "Not yet supported for cloud. Coming soon.

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
        pass  # TODO: "Not yet supported for cloud. Coming soon.

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
        pass  # TODO: "Not yet supported for cloud. Coming soon.

    def log_text(self, name: str, text: str, step: int) -> None:
        """Log text data.

        Args:
            name: Name for the text data.
            text: Text string to log.
            step: Training step.
        """
        pass  # TODO: "Not yet supported for cloud. Coming soon.

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
        pass  # TODO: "Not yet supported for cloud. Coming soon.

    def log_model_graph(
        self, model: torch.nn.Module, input_to_model: torch.Tensor | None = None
    ) -> None:
        """Log the model computational graph.

        Args:
            model: PyTorch model to log.
            input_to_model: Example input tensor for the model.
        """
        pass  # TODO: "Not yet supported for cloud. Coming soon.

    def _start_cloud_sync(self) -> None:
        """Start background thread for cloud synchronization."""
        if self._sync_thread is not None:
            return
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()

    def _sync_loop(self) -> None:
        """Background sync loop for cloud mode."""
        while not self._stop_sync.is_set():
            try:
                current_time = time.time()
                if current_time - self._last_sync_time >= self.sync_interval:
                    self._sync_to_cloud()
                    self._last_sync_time = current_time
            except Exception:
                logger.error("Error during cloud sync.", exc_info=True)

            # Wait with ability to be interrupted
            self._stop_sync.wait(min(10, self.sync_interval))

    def _sync_to_cloud(self) -> None:
        """Sync TensorBoard logs to cloud storage."""
        if len(self._store) == 0:
            return
        org_id = get_current_org()
        metricsData = {
            "metrics": {
                name: {
                    "data": {
                        int(step): float(value) for step, value in step_map.items()
                    }
                }
                for name, step_map in self._store.items()
            }
        }
        response = requests.put(
            f"{API_URL}/org/{org_id}/training/jobs/{self.training_id}/metrics",
            headers=get_auth().get_headers(),
            json=metricsData,
        )
        response.raise_for_status()
        self._store.clear()  # Clear local store after successful sync

    def close(self) -> None:
        """Close the logger and clean up resources."""
        if self._sync_thread is not None:
            self._stop_sync.set()
            self._sync_thread.join(timeout=30)

        # Final sync if in cloud mode
        if self.training_id is not None:
            self._sync_to_cloud()
