"""Machine learning data types for robot learning models.

This module provides data structures for handling batched robot sensor data
with support for masking, device placement, and multi-modal inputs including
joint states, images, point clouds, poses, end-effectors, and language tokens.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from neuracore_types import BatchedNCData, DataType


def _map_tensors(
    batched_nc_data: BatchedNCData,
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> BatchedNCData:
    """Rebuild ``batched_nc_data`` with ``transform`` applied to its tensors.

    ``model_construct`` skips validation: the fields already came out of a
    validated instance, and only the tensors change. This runs once per sensor
    per data type per batch, so the difference matters.
    """
    return batched_nc_data.__class__.model_construct(**{
        name: transform(value) if isinstance(value, torch.Tensor) else value
        for name, value in batched_nc_data.__dict__.items()
    })


@dataclass
class BatchedTrainingSamples:
    """Container for batched training samples with inputs and target outputs.

    Provides structured access to training data including input features,
    target outputs, and prediction masks for supervised learning scenarios.
    """

    inputs: dict[DataType, list[BatchedNCData]]
    inputs_mask: dict[DataType, torch.Tensor]  # Dict[DataType, (B, MAX_LEN)]
    outputs: dict[DataType, list[BatchedNCData]]
    outputs_mask: dict[DataType, torch.Tensor]  # Dict[DataType, (B, MAX_LEN)]
    batch_size: int

    def _map(
        self, transform: Callable[[torch.Tensor], torch.Tensor]
    ) -> "BatchedTrainingSamples":
        """Return a new instance with ``transform`` applied to every tensor."""
        return BatchedTrainingSamples(
            inputs={
                key: [_map_tensors(item, transform) for item in value]
                for key, value in self.inputs.items()
            },
            inputs_mask={
                key: transform(value) for key, value in self.inputs_mask.items()
            },
            outputs={
                key: [_map_tensors(item, transform) for item in value]
                for key, value in self.outputs.items()
            },
            outputs_mask={
                key: transform(value) for key, value in self.outputs_mask.items()
            },
            batch_size=self.batch_size,
        )

    def to(
        self, device: torch.device, non_blocking: bool = False
    ) -> "BatchedTrainingSamples":
        """Move all tensors to the specified device.

        Args:
            device: Target device for tensor placement
            non_blocking: Issue asynchronous copies. Only has an effect when
                the source tensors are in pinned memory (see ``pin_memory``).

        Returns:
            BatchedTrainingSamples: New instance with tensors moved to device
        """
        return self._map(lambda t: t.to(device, non_blocking=non_blocking))

    def pin_memory(self) -> "BatchedTrainingSamples":
        """Move all tensors into pinned host memory.

        ``DataLoader(pin_memory=True)`` dispatches to this method. Without it
        this class is an unrecognised type to torch's pin_memory helper and is
        passed through untouched, which in turn makes ``non_blocking=True``
        transfers silently synchronous.
        """
        return self._map(lambda t: t.pin_memory())

    def __len__(self) -> int:
        """Get the batch size from the input data.

        Returns:
            int: Batch size
        """
        return self.batch_size


@dataclass
class BatchedTrainingOutputs:
    """Container for training step outputs including losses and metrics.

    Provides structured access to the results of a training step including
    computed losses and evaluation metrics.
    """

    losses: dict[str, torch.Tensor]
    metrics: dict[str, torch.Tensor]


@dataclass
class BatchedInferenceInputs:
    """Container for batched inference samples.

    Provides structured access to input data for model inference,
    supporting all robot sensor modalities with device placement.
    """

    inputs: dict[DataType, list[BatchedNCData]]
    inputs_mask: dict[DataType, torch.Tensor]  # Dict[DataType, (B, MAX_LEN)]
    batch_size: int

    def to(
        self, device: torch.device, non_blocking: bool = False
    ) -> "BatchedInferenceInputs":
        """Move all tensors to the specified device.

        Args:
            device: Target device for tensor placement
            non_blocking: Issue asynchronous copies. Only has an effect when
                the source tensors are in pinned memory (see ``pin_memory``).

        Returns:
            The same BatchedInferenceSamples instance with tensors moved to device
        """
        self.inputs = {
            key: [
                _map_tensors(item, lambda t: t.to(device, non_blocking=non_blocking))
                for item in value
            ]
            for key, value in self.inputs.items()
        }
        self.inputs_mask = {
            key: value.to(device, non_blocking=non_blocking)
            for key, value in self.inputs_mask.items()
        }
        return self

    def pin_memory(self) -> "BatchedInferenceInputs":
        """Move all tensors into pinned host memory.

        ``DataLoader(pin_memory=True)`` dispatches to this method.
        """
        self.inputs = {
            key: [_map_tensors(item, lambda t: t.pin_memory()) for item in value]
            for key, value in self.inputs.items()
        }
        self.inputs_mask = {
            key: value.pin_memory() for key, value in self.inputs_mask.items()
        }
        return self

    def __len__(self) -> int:
        """Get the batch size from the first available tensor.

        Returns:
            int: Batch size

        Raises:
            ValueError: If no tensors are found in the batch
        """
        return self.batch_size

    def __getitem__(self, key: DataType | str) -> list[BatchedNCData]:
        """Get item by DataType or field name."""
        # If key is a DataType enum, access the nested data dict
        if isinstance(key, DataType):
            return self.inputs[key]
        raise KeyError(f"Key {key} not found in BatchedInferenceInputs.")

    def __setitem__(self, key: DataType | str, value: list[BatchedNCData]) -> None:
        """Set item by DataType or field name."""
        # Same for setting
        if isinstance(key, DataType):
            self.inputs[key] = value
        raise KeyError(f"Key {key} not found in BatchedInferenceInputs.")


@dataclass
class SynchronizedPointPrediction:
    """Model inference output containing predictions and timing information.

    Represents the results of model inference including predicted outputs
    for each configured data type and optional timing information for
    performance monitoring.
    """

    outputs: dict[DataType, dict[str, torch.Tensor]]
    prediction_time: float | None = None
