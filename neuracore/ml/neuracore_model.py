from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..core.nc_types import DataType, ModelInitDescription
from .ml_types import (
    BatchedInferenceOutputs,
    BatchedInferenceSamples,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
)


class NeuracoreModel(nn.Module, ABC):
    """Abstract base class for robot learning models."""

    def __init__(
        self,
        model_init_description: ModelInitDescription,
    ):
        super().__init__()
        supported, input_data_types = set(self.get_supported_data_types()), set(
            model_init_description.dataset_description.get_data_types()
        )
        # input_data_types must be within support
        if not input_data_types.issubset(supported):
            raise ValueError(
                f"Model does not support data types: {input_data_types - supported}"
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_description = model_init_description.dataset_description
        self.action_prediction_horizon = (
            model_init_description.action_prediction_horizon
        )

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

    @abstractmethod
    def get_supported_data_types(self) -> list[DataType]:
        """Return the data types supported by the model."""
        pass
