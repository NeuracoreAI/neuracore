from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .batch_input import BatchInput
from .batch_output import BatchOutput
from .dataset_description import DatasetDescription


class NeuracoreModel(nn.Module, ABC):
    """Abstract base class for robot learning models."""

    def __init__(
        self,
        dataset_description: DatasetDescription,
    ):
        super().__init__()
        self.dataset_description = dataset_description
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def forward(self, batch: BatchInput) -> BatchOutput:
        """
        Forward pass of the model.
        Args:
            batch: Dictionary containing input tensors
        Returns:
            Dictionary containing output tensors and losses
        """
        pass

    @abstractmethod
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure and return optimizer for the model."""
        pass

    def process_batch(self, batch: BatchInput) -> BatchInput:
        """Called by dataloader to process the batch."""
        return batch
