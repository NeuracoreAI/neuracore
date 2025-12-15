"""Abstract base class for training loggers."""

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TrainingLogger(ABC):
    """Abstract base class for training loggers."""

    @abstractmethod
    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log a scalar metric.

        Args:
            name: Name of the scalar metric.
            value: Value of the scalar metric.
            step: Training step.
        """
        pass

    @abstractmethod
    def log_scalars(self, scalars: dict[str, float], step: int) -> None:
        """Log multiple scalar metrics.

        Args:
            scalars: Dictionary of scalar metrics where keys are
                names and values are the metric values.
            step: Training step.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def log_histogram(
        self,
        name: str,
        values: np.ndarray | torch.Tensor,
        step: int,
    ) -> None:
        """Log a histogram of values.

        Args:
            name: Name for the histogram.
            values: Values to create the histogram from, as numpy array or torch tensor.
            step: Training step.
        """
        pass

    @abstractmethod
    def log_text(self, name: str, text: str, step: int) -> None:
        """Log text data.

        Args:
            name: Name for the text data.
            text: Text content to log.
            step: Training step.
        """
        pass

    @abstractmethod
    def log_hyperparameters(self, hparams: dict[str, Any]) -> None:
        """Log hyperparameters.

        Args:
            hparams: Dictionary of hyperparameters where keys are
                names and values are the hyperparameter values.
        """
        pass

    @abstractmethod
    def log_model_graph(
        self, model: torch.nn.Module, input_to_model: torch.Tensor | None = None
    ) -> None:
        """Log the model computational graph.

        Args:
            model: PyTorch model to log.
            input_to_model: Example input tensor for the model.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the logger and clean up resources."""
        pass

    def __enter__(self) -> "TrainingLogger":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
