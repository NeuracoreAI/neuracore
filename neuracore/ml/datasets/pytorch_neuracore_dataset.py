"""Abstract base class for Neuracore datasets with multi-modal data support.

This module provides the foundation for creating datasets that handle robot
demonstration data including images, joint states, depth images, point clouds,
poses, end-effectors, and language instructions.
"""

import logging
from abc import ABC, abstractmethod

import torch
from neuracore_types import BatchedNCData, DataType, NCDataStats, RobotDataSpec
from torch.utils.data import Dataset

from neuracore.ml import BatchedTrainingSamples
from neuracore.ml.utils.robot_data_spec_utils import merge_robot_data_spec

logger = logging.getLogger(__name__)

TrainingSample = BatchedTrainingSamples


class PytorchNeuracoreDataset(Dataset, ABC):
    """Abstract base class for Neuracore multi-modal robot datasets.

    This class provides a standardized interface for datasets containing robot
    demonstration data. It handles data type validation, preprocessing setup,
    batch collation, and error management for training machine learning models
    on robot data including images, joint states, depth images, point clouds,
    poses, end-effectors, and language instructions.
    """

    def __init__(
        self,
        num_recordings: int,
        input_robot_data_spec: RobotDataSpec,
        output_robot_data_spec: RobotDataSpec,
        output_prediction_horizon: int = 5,
    ):
        """Initialize the dataset with data type specifications and preprocessing.

        Args:
            input_robot_data_spec: List of data types to include as model inputs
                (e.g., RGB images, joint positions).
            output_robot_data_spec: List of data types to include as model outputs
                (e.g., joint target positions, actions).
            output_prediction_horizon: Number of future timesteps to predict
                for sequential output tasks.

        Raises:
            ValueError: If language data is requested but no tokenizer is provided.
        """
        if len(input_robot_data_spec) == 0 and len(output_robot_data_spec) == 0:
            raise ValueError(
                "Must supply both input and output data types for the dataset"
            )
        self.num_recordings = num_recordings
        self.input_robot_data_spec = input_robot_data_spec
        self.output_robot_data_spec = output_robot_data_spec
        self.output_prediction_horizon = output_prediction_horizon
        self.robot_data_spec = merge_robot_data_spec(
            self.input_robot_data_spec, self.output_robot_data_spec
        )

    @abstractmethod
    def load_sample(
        self, episode_idx: int, timestep: int | None = None
    ) -> TrainingSample:
        """Load a training sample from the dataset by episode index and timestep.

        This method must be implemented by concrete subclasses to define how
        individual samples are loaded and formatted.

        Args:
            episode_idx: Index of the episode to load data from.
            timestep: Optional specific timestep within the episode.
                If None, may load entire episode or use class-specific logic.

        Returns:
            A TrainingSample containing input and output data formatted
            for model training.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get the total number of samples in the dataset.

        Returns:
            The number of training samples available.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> TrainingSample:
        """Get a training sample from the dataset by index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A TrainingSample containing the requested data.

        Raises:
            Exception: If sample loading fails after exhausting retry attempts.
        """
        pass

    def _collate_input_outputs(
        self,
        samples: list[dict[DataType, list[BatchedNCData]]],
    ) -> dict[DataType, list[BatchedNCData]]:
        """Collate individual data samples into a batched format.

        Example samples looks like:
        [
            {
                DataType.JOINT_POSITIONS: [BatchedJointData, BatchedJointData, ...],
                DataType.RGB_IMAGES: [BatchedCameraData, ...],
                ...
            },
            ...
        ]

        Example output looks like:
        {
            DataType.JOINT_POSITIONS: [BatchedJointData, ...],
            DataType.RGB_IMAGES: [BatchedCameraData, ...],
            ...
        }


        Combines multiple samples into batched tensors with appropriate stacking
        for different data modalities. Handles masking for variable-length data.

        Args:
            samples: List of BatchedData objects to combine.
            data_types: List of data types to include in the batch.

        Returns:
            A single BatchedData object containing the stacked samples.
        """
        batched_data: dict[DataType, list[BatchedNCData]] = {}

        for data_type in samples[0].keys():
            # Get the number of items for this data type (e.g., number of joints)
            num_items = len(samples[0][data_type])
            batched_data[data_type] = []

            # Process each item index (e.g., joint_0, joint_1, camera_0, etc.)
            for item_idx in range(num_items):
                # Collect all samples for this specific item across the batch
                items_to_batch = [sample[data_type][item_idx] for sample in samples]

                # Get attribute names from the first item
                attrib_names_of_tensors = [
                    attr_name
                    for attr_name in vars(items_to_batch[0])
                    if isinstance(getattr(items_to_batch[0], attr_name), torch.Tensor)
                ]

                # Stack each attribute across samples
                batched_attributes = {}
                for attr_name in attrib_names_of_tensors:
                    attr_values = [getattr(item, attr_name) for item in items_to_batch]

                    # Stack tensors along batch dimension
                    if attr_values[0] is not None:
                        batched_attributes[attr_name] = torch.cat(attr_values, dim=0)
                    else:
                        batched_attributes[attr_name] = None

                # Create new batched object of the same type
                batched_item = type(items_to_batch[0])(**batched_attributes)
                batched_data[data_type].append(batched_item)

        return batched_data

    def _collate_masks(
        self,
        samples: list[dict[DataType, torch.Tensor]],
    ) -> dict[DataType, torch.Tensor]:
        """Collate individual data masks into a batched format.

        Args:
            samples: List of mask dictionaries to combine.

        Returns:
            A single dictionary containing the stacked masks.
        """
        batched_masks: dict[DataType, torch.Tensor] = {}

        for data_type in samples[0].keys():
            masks_to_batch = [sample[data_type] for sample in samples]
            batched_masks[data_type] = torch.stack(masks_to_batch, dim=0)

        return batched_masks

    def collate_fn(self, samples: list[TrainingSample]) -> BatchedTrainingSamples:
        """Collate training samples into a complete batch for model training.

        Combines individual training samples into batched inputs, outputs, and
        prediction masks suitable for model training. This function is typically
        used with PyTorch DataLoader.

        Args:
            samples: List of TrainingSample objects to batch together.

        Returns:
            A BatchedTrainingSamples object containing batched inputs, outputs,
            and prediction masks ready for model training.
        """
        return BatchedTrainingSamples(
            inputs=self._collate_input_outputs([sample.inputs for sample in samples]),
            inputs_mask=self._collate_masks([sample.inputs_mask for sample in samples]),
            outputs=self._collate_input_outputs([sample.outputs for sample in samples]),
            outputs_mask=self._collate_masks(
                [sample.outputs_mask for sample in samples]
            ),
            batch_size=len(samples),
        )

    @property
    @abstractmethod
    def dataset_statistics(self) -> dict[DataType, list[NCDataStats]]:
        """Return the dataset description.

        Returns:
            DatasetStatistics object describing the dataset.
        """
        pass
