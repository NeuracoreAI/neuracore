"""Dummy dataset for algorithm validation and testing without real data.

This module provides a synthetic dataset that generates random data matching
the structure of real Neuracore datasets. It's used for algorithm development,
testing, and validation without requiring actual robot demonstration data.
"""

import copy
import logging

import torch
from neuracore_types import (
    DATA_TYPE_TO_BATCHED_NC_DATA_CLASS,
    DATA_TYPE_TO_NC_DATA_CLASS,
    BatchedNCData,
    DataType,
    NCData,
    NCDataStats,
    RobotDataSpec,
)

from neuracore.core.robot import Robot
from neuracore.ml import BatchedTrainingSamples
from neuracore.ml.datasets.pytorch_neuracore_dataset import PytorchNeuracoreDataset

logger = logging.getLogger(__name__)

TrainingSample = BatchedTrainingSamples

T = 1
MAX_LEN_PER_DATA_TYPE = 2


class PytorchDummyDataset(PytorchNeuracoreDataset):
    """Synthetic dataset for algorithm validation and testing.

    This dataset generates random data with the same structure and dimensions
    as real Neuracore datasets, allowing for algorithm development and testing
    without requiring actual robot demonstration data. It supports all standard
    data types including images, joint data, depth images, point clouds,
    poses, end-effectors, and language instructions.
    """

    def __init__(
        self,
        input_robot_data_spec: RobotDataSpec,
        output_robot_data_spec: RobotDataSpec,
        num_samples: int = 50,
        num_episodes: int = 10,
        output_prediction_horizon: int = 5,
    ):
        """Initialize the dummy dataset with specified data types and dimensions.

        Args:
            input_robot_data_spec: Mapping from robot_id
                to data spec for model inputs.
            output_robot_data_spec: Mapping from robot_id
                to data spec for model outputs.
            num_samples: Total number of training samples to generate.
            num_episodes: Number of distinct episodes in the dataset.
            output_prediction_horizon: Length of output action sequences.
        """
        super().__init__(
            num_recordings=num_episodes,
            input_robot_data_spec=input_robot_data_spec,
            output_robot_data_spec=output_robot_data_spec,
            output_prediction_horizon=output_prediction_horizon,
        )
        self.num_samples = num_samples
        self.robot = Robot("dummy_robot", 0)
        self.robot.id = "dummy_robot_id"

        self._dataset_statistics: dict[DataType, list[NCDataStats]] = {}

        # Generate sample data once and reuse for all samples
        self._sample_to_return = self._generate_sample()

    def _generate_sample(self) -> TrainingSample:
        """Generate a single training sample template with random data.

        Creates synthetic data for all specified input and output data types,
        with appropriate dimensions and masking.

        Returns:
            A TrainingSample containing randomly generated input and output data.
        """
        inputs_and_outputs: dict[DataType, list[BatchedNCData]] = {}
        inputs_and_outputs_mask: dict[DataType, torch.Tensor] = {}

        # Generate data for all data types in the merged spec
        # robot_data_spec is dict[robot_id, dict[DataType, list[str]]]
        for robot_id, data_spec in self.robot_data_spec.items():
            for data_type, data_names in data_spec.items():
                num_items = (
                    len(data_names) if len(data_names) > 0 else MAX_LEN_PER_DATA_TYPE
                )

                if data_type not in inputs_and_outputs:
                    inputs_and_outputs[data_type] = []
                    inputs_and_outputs_mask[data_type] = torch.ones(
                        (num_items,), dtype=torch.float32
                    )
                    self._dataset_statistics[data_type] = []

                # Generate data for each item (e.g., each joint, each camera)
                for _ in range(num_items):
                    nc_data_class = DATA_TYPE_TO_NC_DATA_CLASS[data_type]
                    sampled_data: NCData = nc_data_class.sample()

                    # Store statistics for this data item
                    self._dataset_statistics[data_type].append(
                        sampled_data.calculate_statistics()
                    )

                    # Generate batched data
                    batched_nc_data_type = DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[data_type]
                    batched_nc_data = batched_nc_data_type.sample(
                        batch_size=1, time_steps=T
                    )

                    # Drop batch dimension for storage
                    for attr_name in vars(batched_nc_data):
                        attr_value = getattr(batched_nc_data, attr_name)
                        if (
                            isinstance(attr_value, torch.Tensor)
                            and attr_value.shape[0] == 1
                        ):
                            setattr(batched_nc_data, attr_name, attr_value)

                    inputs_and_outputs[data_type].append(batched_nc_data)

        # Separate inputs and outputs based on specs
        inputs: dict[DataType, list[BatchedNCData]] = {}
        inputs_mask: dict[DataType, torch.Tensor] = {}
        outputs: dict[DataType, list[BatchedNCData]] = {}
        outputs_mask: dict[DataType, torch.Tensor] = {}

        # Collect input data types from all robots
        for robot_id, data_spec in self.input_robot_data_spec.items():
            for data_type in data_spec.keys():
                if data_type not in inputs:
                    inputs[data_type] = inputs_and_outputs[data_type]
                    inputs_mask[data_type] = inputs_and_outputs_mask[data_type]

        # Collect output data types from all robots
        for robot_id, data_spec in self.output_robot_data_spec.items():
            for data_type in data_spec.keys():
                if data_type not in outputs:
                    # Extend output temporal dimension by prediction horizon
                    outputs[data_type] = []
                    for output in inputs_and_outputs[data_type]:
                        output = copy.deepcopy(output)
                        for attr_name in vars(output):
                            attr_value = getattr(output, attr_name)
                            if isinstance(attr_value, torch.Tensor):
                                # Repeat along time dimension for prediction horizon
                                attr_value = attr_value.repeat_interleave(
                                    self.output_prediction_horizon, dim=1
                                )
                                setattr(output, attr_name, attr_value)
                        outputs[data_type].append(output)
                    outputs_mask[data_type] = inputs_and_outputs_mask[data_type]

        return TrainingSample(
            inputs=inputs,
            inputs_mask=inputs_mask,
            outputs=outputs,
            outputs_mask=outputs_mask,
            batch_size=0,
        )

    def load_sample(
        self, episode_idx: int, timestep: int | None = None
    ) -> TrainingSample:
        """Generate a random training sample with realistic data structure.

        Creates synthetic data that matches the format and dimensions of real
        robot demonstration data, including appropriate masking and tensor shapes.

        Args:
            episode_idx: Index of the episode (used for reproducible randomness).
            timestep: Optional timestep within the episode (currently unused).

        Returns:
            A TrainingSample containing randomly generated input and output data
            matching the specified data types and dimensions.
        """
        return self._sample_to_return

    def __len__(self) -> int:
        """Get the total number of samples in the dataset.

        Returns:
            The number of training samples available in this dataset.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> BatchedTrainingSamples:
        """Get a training sample from the dataset by index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A TrainingSample containing the requested data.
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError("Index out of range for dataset.")
        return self.load_sample(0, 0)

    @property
    def dataset_statistics(self) -> dict[DataType, list[NCDataStats]]:
        """Return the dataset description.

        Returns:
            A dictionary mapping each DataType to a list of NCDataStats
            describing the statistics of that data type in the dataset.
        """
        return self._dataset_statistics
