from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .types import (
    BatchedInferenceOutputs,
    BatchedInferenceSamples,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    DatasetDescription,
)


class NeuracoreModel(nn.Module, ABC):
    """Abstract base class for robot learning models."""

    def __init__(
        self,
        dataset_description: DatasetDescription,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_description = dataset_description.to(self.device)

    @abstractmethod
    def forward(self, batch: BatchedInferenceSamples) -> BatchedInferenceOutputs:
        """Inference forward pass."""
        pass

    @abstractmethod
    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Inference forward pass."""
        pass

    @abstractmethod
    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Configure and return optimizer for the model."""
        pass
