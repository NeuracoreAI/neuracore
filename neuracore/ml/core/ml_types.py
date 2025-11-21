"""Machine learning data types for robot learning models.

This module provides data structures for handling batched robot sensor data
with support for masking, device placement, and multi-modal inputs including
joint states, images, point clouds, poses, end-effectors, and language tokens.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from neuracore_types import DataType


@dataclass(slots=True)
class MaskableData:
    """Container for tensor data with associated mask for variable-length sequences.

    Provides a unified interface for handling data that may have variable lengths
    or missing values, commonly used in robot learning for handling sequences
    of different lengths or optional sensor modalities.
    """

    data: torch.Tensor
    mask: torch.Tensor

    def to(self, device: torch.device) -> "MaskableData":
        """Move all tensors to the specified device.

        Args:
            device: Target device for tensor placement

        Returns:
            MaskableData: New instance with tensors moved to the specified device
        """
        return MaskableData(
            data=self.data.to(device),
            mask=self.mask.to(device),
        )


@dataclass(slots=True)
class BatchedTrainingSamples:
    """Container for batched training samples with inputs and target outputs.

    Provides structured access to training data including input features,
    target outputs, and prediction masks for supervised learning scenarios.
    """

    output_prediction_mask: torch.Tensor
    inputs: Dict[DataType, Dict[str, MaskableData]] = field(default_factory=dict)
    outputs: Dict[DataType, Dict[str, MaskableData]] = field(default_factory=dict)

    def to(self, device: torch.device) -> "BatchedTrainingSamples":
        """Move all tensors to the specified device.

        Args:
            device: Target device for tensor placement

        Returns:
            BatchedTrainingSamples: New instance with tensors moved to device
        """
        return BatchedTrainingSamples(
            output_prediction_mask=self.output_prediction_mask.to(device),
            inputs={
                key: {k: v.to(device) for k, v in value.items()}
                for key, value in self.inputs.items()
            },
            outputs={
                key: {k: v.to(device) for k, v in value.items()}
                for key, value in self.outputs.items()
            },
        )

    def combine_for_data_type(self, data_type: DataType) -> MaskableData:
        """Combine all named inputs for a given data type into a single MaskableData.

        Args:
            data_type: The DataType to combine inputs for

        Returns:
            MaskableData: Combined data and mask for the specified data type
        """
        data_list = []
        mask_list = []
        if data_type in self.inputs:
            for named_data in self.inputs[data_type].values():
                data_list.append(named_data.data)
                mask_list.append(named_data.mask)
            combined_data = torch.cat(data_list, dim=-1)
            combined_mask = torch.stack(mask_list, dim=0)
            return MaskableData(data=combined_data, mask=combined_mask)
        else:
            raise ValueError(f"No inputs found for data type: {data_type}")

    def __len__(self) -> int:
        """Get the batch size from the input data.

        Returns:
            int: Batch size
        """
        # Get the batch size from the first available tensor.
        for data_type_dict in self.inputs.values():
            for maskable_data in data_type_dict.values():
                if maskable_data.data is not None:
                    return maskable_data.data.size(0)
        raise ValueError("No tensor found in the batch input")


@dataclass(slots=True)
class BatchedTrainingOutputs:
    """Container for training step outputs including losses and metrics.

    Provides structured access to the results of a training step including
    computed losses and evaluation metrics.
    """

    losses: Dict[str, torch.Tensor]
    metrics: Dict[str, torch.Tensor]


@dataclass(slots=True)
class BatchedInferenceInputs:
    """Container for batched inference samples.

    Provides structured access to input data for model inference,
    supporting all robot sensor modalities with device placement.
    """

    inputs: Dict[DataType, Dict[str, MaskableData]] = field(default_factory=dict)

    def to(self, device: torch.device) -> "BatchedInferenceInputs":
        """Move all tensors to the specified device.

        Args:
            device: Target device for tensor placement

        Returns:
            The same BatchedInferenceSamples instance with tensors moved to device
        """
        self.inputs = {
            key: {k: v.to(device) for k, v in value.items()}
            for key, value in self.inputs.items()
        }
        return self

    def combine_for_data_type(self, data_type: DataType) -> MaskableData:
        """Combine all named inputs for a given data type into a single MaskableData.

        Args:
            data_type: The DataType to combine inputs for

        Returns:
            MaskableData: Combined data and mask for the specified data type
        """
        data_list = []
        mask_list = []
        if data_type in self.inputs:
            for named_data in self.inputs[data_type].values():
                data_list.append(named_data.data)
                mask_list.append(named_data.mask)
            combined_data = torch.cat(data_list, dim=-1)
            combined_mask = torch.stack(mask_list, dim=0)
            return MaskableData(data=combined_data, mask=combined_mask)
        else:
            raise ValueError(f"No inputs found for data type: {data_type}")

    def __len__(self) -> int:
        """Get the batch size from the first available tensor.

        Returns:
            int: Batch size (first dimension of available tensors)

        Raises:
            ValueError: If no tensors are found in the batch
        """
        # Get the batch size from the first available tensor.
        for data_type_dict in self.inputs.values():
            for maskable_data in data_type_dict.values():
                if maskable_data.data is not None:
                    return maskable_data.data.size(0)
        raise ValueError("No tensor found in the batch output")


@dataclass(slots=True)
class SynchronizedPointPrediction:
    """Model inference output containing predictions and timing information.

    Represents the results of model inference including predicted outputs
    for each configured data type and optional timing information for
    performance monitoring.
    """

    outputs: Dict[DataType, Dict[str, torch.Tensor]]
    prediction_time: Optional[float] = None
