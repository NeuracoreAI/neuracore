"""Identity model supporting all Neuracore data types for validation testing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuracore_types import DataType, ModelInitDescription

from neuracore.ml import (
    BatchedInferenceInputs,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    NeuracoreModel,
)


class AllTypesModel(NeuracoreModel):
    """Minimal identity model (y = x) covering all 14 DataTypes.

    Exists solely to exercise the validation pipeline. Both inference and
    training treat every input as its own prediction.
    """

    def __init__(self, model_init_description: ModelInitDescription):
        super().__init__(model_init_description)
        # Learnable offset ensures a non-zero gradient during the backward pass
        # check in run_validation. Not part of the y = x model logic.
        self.offset = nn.Parameter(torch.ones(1))

    @staticmethod
    def get_supported_input_data_types() -> set[DataType]:
        return {
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.JOINT_TARGET_POSITIONS,
            DataType.VISUAL_JOINT_POSITIONS,
            DataType.END_EFFECTOR_POSES,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
            DataType.RGB_IMAGES,
            DataType.DEPTH_IMAGES,
            DataType.POINT_CLOUDS,
            DataType.POSES,
            DataType.LANGUAGE,
            DataType.CUSTOM_1D,
        }

    @staticmethod
    def get_supported_output_data_types() -> set[DataType]:
        return {
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.JOINT_TARGET_POSITIONS,
            DataType.VISUAL_JOINT_POSITIONS,
            DataType.END_EFFECTOR_POSES,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
            DataType.RGB_IMAGES,
            DataType.DEPTH_IMAGES,
            DataType.POINT_CLOUDS,
            DataType.POSES,
            DataType.LANGUAGE,
            DataType.CUSTOM_1D,
        }

    def forward(self, batch: BatchedInferenceInputs) -> dict:
        """y = x: return inputs unchanged."""
        return {
            data_type: batch.inputs[data_type] for data_type in self.output_data_types
        }

    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """y = x: compute MSE(x, target) plus a gradient anchor loss."""
        losses = [self.offset.square().squeeze()]
        for data_type in self.output_data_types:
            for input_item, target_item in zip(
                batch.inputs[data_type], batch.outputs[data_type]
            ):
                for field_name in type(input_item).model_fields:
                    predicted_value = getattr(input_item, field_name)
                    target_value = getattr(target_item, field_name)
                    if not isinstance(predicted_value, torch.Tensor):
                        continue
                    if not predicted_value.is_floating_point():
                        continue
                    predicted_value = predicted_value.repeat_interleave(
                        self.output_prediction_horizon, dim=1
                    )
                    losses.append(F.mse_loss(predicted_value, target_value.float()))

        return BatchedTrainingOutputs(
            losses={"mse": torch.stack(losses).mean()}, metrics={}
        )

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        return [torch.optim.Adam(self.parameters(), lr=1e-3)]
