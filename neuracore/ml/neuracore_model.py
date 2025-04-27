import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..core.nc_types import DataType, ModelInitDescription, ModelPrediction
from .ml_types import (
    BatchedInferenceSamples,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
)

logger = logging.getLogger(__name__)


class NeuracoreModel(nn.Module, ABC):
    """Abstract base class for robot learning models."""

    def __init__(
        self,
        model_init_description: ModelInitDescription,
    ):
        super().__init__()
        self.model_init_description = model_init_description
        self._validate_input_output_types()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_description = model_init_description.dataset_description
        self.output_prediction_horizon = (
            model_init_description.output_prediction_horizon
        )

    def _validate_input_output_types(self):
        req_input_data_types = set(self.model_init_description.input_data_types)
        types_in_dataset = set(
            self.model_init_description.dataset_description.get_data_types()
        )
        input_types_supported_by_model = set(self.get_supported_input_data_types())

        # Check if the requested input data types are in the dataset description
        if not req_input_data_types.issubset(types_in_dataset):
            raise ValueError(
                "Requested input data types not in dataset: "
                f"{req_input_data_types - types_in_dataset}"
            )

        # Check if the requested input data types are supported by the model
        if not req_input_data_types.issubset(input_types_supported_by_model):
            raise ValueError(
                "Requested input data types not supported by model: "
                f"{req_input_data_types - input_types_supported_by_model}"
            )

        req_output_data_types = set(self.model_init_description.output_data_types)
        outut_types_supported_by_model = set(self.get_supported_output_data_types())

        # Check if the requested output data types are supported by the model
        if not req_output_data_types.issubset(outut_types_supported_by_model):
            raise ValueError(
                "Requested output data types not supported by model: "
                f"{req_output_data_types - outut_types_supported_by_model}"
            )
        # Check if the requested output data types are in the dataset description
        if not req_output_data_types.issubset(types_in_dataset):
            raise ValueError(
                "Requested output data types not in dataset: "
                f"{req_output_data_types - types_in_dataset}"
            )

    @abstractmethod
    def forward(self, batch: BatchedInferenceSamples) -> ModelPrediction:
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

    @staticmethod
    def tokenize_text(self, text: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text."""
        raise NotImplementedError("User needs to implement this method")

    @staticmethod
    @abstractmethod
    def get_supported_input_data_types() -> list[DataType]:
        """Return the input data types supported by the model."""
        pass

    @staticmethod
    @abstractmethod
    def get_supported_output_data_types() -> list[DataType]:
        """Return the output data types supported by the model."""
        pass
