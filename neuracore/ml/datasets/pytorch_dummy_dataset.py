"""Dummy dataset for algorithm validation and testing without real data.

This module provides a synthetic dataset that generates random data matching
the structure of real Neuracore datasets. It's used for algorithm development,
testing, and validation without requiring actual robot demonstration data.
"""

import logging
from collections.abc import Iterator, Mapping
from typing import cast

import torch
from neuracore_types import (
    DATA_TYPE_TO_BATCHED_NC_DATA_CLASS,
    DATA_TYPE_TO_NC_DATA_CLASS,
    BatchedNCData,
    CrossEmbodimentDescription,
    DataType,
    NCDataStats,
)

from neuracore.core.robot import Robot
from neuracore.ml import BatchedTrainingSamples
from neuracore.ml.datasets.pytorch_neuracore_dataset import PytorchNeuracoreDataset

logger = logging.getLogger(__name__)

TrainingSample = BatchedTrainingSamples

MAX_LEN_PER_DATA_TYPE = 2


class _DatasetStatisticsView(Mapping[object, object]):
    """Expose nested stats while preserving flat DataType access for tests."""

    def __init__(
        self,
        nested_statistics: dict[str, dict[DataType, list[NCDataStats]]],
        flat_statistics: dict[DataType, list[NCDataStats]],
    ):
        self._nested_statistics = nested_statistics
        self._flat_statistics = flat_statistics

    def __getitem__(self, key: object) -> object:
        if isinstance(key, DataType):
            return self._flat_statistics[key]
        if not isinstance(key, str):
            raise KeyError(key)
        return self._nested_statistics[key]

    def __iter__(self) -> Iterator[object]:
        yield from self._nested_statistics
        yield from self._flat_statistics

    def __len__(self) -> int:
        return len(self._nested_statistics) + len(self._flat_statistics)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, DataType):
            return key in self._flat_statistics
        return key in self._nested_statistics


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
        input_cross_embodiment_description: CrossEmbodimentDescription,
        output_cross_embodiment_description: CrossEmbodimentDescription,
        num_samples: int = 50,
        num_episodes: int = 10,
        output_prediction_horizon: int = 5,
    ):
        """Initialize the dummy dataset with specified data types and dimensions.

        Args:
            input_cross_embodiment_description: Mapping from robot_id
                to data spec for model inputs.
            output_cross_embodiment_description: Mapping from robot_id
                to data spec for model outputs.
            num_samples: Total number of training samples to generate.
            num_episodes: Number of distinct episodes in the dataset.
            output_prediction_horizon: Length of output action sequences.
        """
        super().__init__(
            num_recordings=num_episodes,
            input_cross_embodiment_description=input_cross_embodiment_description,
            output_cross_embodiment_description=output_cross_embodiment_description,
            output_prediction_horizon=output_prediction_horizon,
        )
        self.num_samples = num_samples
        self.robot = Robot(robot_name="dummy_robot", instance=0, org_id="dummy_org_id")
        self.robot.id = "dummy_robot_id"

        self._flat_dataset_statistics: dict[DataType, list[NCDataStats]] = {}
        self._dataset_statistics: dict[str, dict[DataType, list[NCDataStats]]] = {
            "input": {},
            "output": {},
        }

        self._robot_ids = list(
            dict.fromkeys(
                list(self.input_cross_embodiment_description)
                + list(self.output_cross_embodiment_description)
            )
        )

        self._initialize_dataset_statistics()
        self._samples_by_robot = {
            robot_id: self._generate_sample(robot_id) for robot_id in self._robot_ids
        }

    @staticmethod
    def _normalize_data_names(data_names: list[str] | dict[int, str]) -> dict[int, str]:
        """Normalize list/dict specs to the indexed format used by real datasets."""
        if isinstance(data_names, dict):
            return dict(data_names)
        return {index: name for index, name in enumerate(data_names)}

    def _get_num_slots_for_data_type(
        self,
        cross_embodiment_description: CrossEmbodimentDescription,
        data_type: DataType,
    ) -> int:
        """Return padded width for a data type across all robots in a spec."""
        max_index: int | None = None
        has_empty_spec = False

        for embodiment_description in cross_embodiment_description.values():
            if data_type not in embodiment_description:
                continue

            normalized_data_names = self._normalize_data_names(
                embodiment_description[data_type]
            )
            if normalized_data_names:
                data_type_max_index = max(normalized_data_names)
                max_index = (
                    data_type_max_index
                    if max_index is None
                    else max(max_index, data_type_max_index)
                )
            else:
                has_empty_spec = True

        if max_index is not None:
            return max_index + 1
        if has_empty_spec:
            return MAX_LEN_PER_DATA_TYPE
        raise KeyError(f"Data type {data_type} not found in data specification.")

    def _initialize_dataset_statistics(self) -> None:
        """Create one statistics entry per padded slot for each merged data type."""
        all_data_types = {
            data_type
            for cross_embodiment_description in (
                self.input_cross_embodiment_description,
                self.output_cross_embodiment_description,
            )
            for embodiment_description in cross_embodiment_description.values()
            for data_type in embodiment_description
        }

        for data_type in all_data_types:
            if data_type in self._flat_dataset_statistics:
                continue

            num_slots = max(
                self._get_num_slots_for_data_type(
                    cross_embodiment_description, data_type
                )
                for cross_embodiment_description in (
                    self.input_cross_embodiment_description,
                    self.output_cross_embodiment_description,
                )
                if any(
                    data_type in embodiment_description
                    for embodiment_description in cross_embodiment_description.values()
                )
            )
            nc_data_class = DATA_TYPE_TO_NC_DATA_CLASS[data_type]
            self._flat_dataset_statistics[data_type] = [
                nc_data_class.sample().calculate_statistics() for _ in range(num_slots)
            ]

            if any(
                data_type in embodiment_description
                for embodiment_description in (
                    self.input_cross_embodiment_description.values()
                )
            ):
                self._dataset_statistics["input"][data_type] = (
                    self._flat_dataset_statistics[data_type]
                )
            if any(
                data_type in embodiment_description
                for embodiment_description in (
                    self.output_cross_embodiment_description.values()
                )
            ):
                self._dataset_statistics["output"][data_type] = (
                    self._flat_dataset_statistics[data_type]
                )

    def _sample_batched_data(
        self, data_type: DataType, time_steps: int
    ) -> BatchedNCData:
        """Sample synthetic batched data for a single slot."""
        batched_nc_data_class = DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[data_type]
        return batched_nc_data_class.sample(batch_size=1, time_steps=time_steps)

    def _build_data_section(
        self,
        robot_id: str,
        cross_embodiment_description: CrossEmbodimentDescription,
        time_steps: int,
    ) -> tuple[dict[DataType, list[BatchedNCData]], dict[DataType, torch.Tensor]]:
        """Build one input/output section using synchronized-dataset slot padding."""
        section: dict[DataType, list[BatchedNCData]] = {}
        section_mask: dict[DataType, torch.Tensor] = {}

        robot_spec = cross_embodiment_description.get(robot_id, {})
        for data_type, data_names in robot_spec.items():
            normalized_data_names = self._normalize_data_names(data_names)
            num_slots = self._get_num_slots_for_data_type(
                cross_embodiment_description, data_type
            )
            section[data_type] = []
            mask = []

            for index in range(num_slots):
                section[data_type].append(
                    self._sample_batched_data(
                        data_type=data_type, time_steps=time_steps
                    )
                )
                mask.append(1.0 if index in normalized_data_names else 0.0)

            section_mask[data_type] = torch.tensor(mask, dtype=torch.float32)

        return section, section_mask

    def _generate_sample(self, robot_id: str) -> TrainingSample:
        """Generate a single training sample template for one robot.

        Creates synthetic data for all specified input and output data types,
        with appropriate dimensions and masking.

        Returns:
            A TrainingSample containing randomly generated input and output data.
        """
        inputs, inputs_mask = self._build_data_section(
            robot_id=robot_id,
            cross_embodiment_description=self.input_cross_embodiment_description,
            time_steps=1,
        )
        outputs, outputs_mask = self._build_data_section(
            robot_id=robot_id,
            cross_embodiment_description=self.output_cross_embodiment_description,
            time_steps=self.output_prediction_horizon,
        )

        return TrainingSample(
            inputs=inputs,
            inputs_mask=inputs_mask,
            outputs=outputs,
            outputs_mask=outputs_mask,
            batch_size=1,
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
        robot_id = self._robot_ids[episode_idx % len(self._robot_ids)]
        return self._samples_by_robot[robot_id]

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
        return self.load_sample(idx, 0)

    @property
    def dataset_statistics(self) -> dict[DataType, list[NCDataStats]]:
        """Return the dataset description.

        Returns:
            A dictionary mapping each DataType to a list of NCDataStats
            describing the statistics of that data type in the dataset.
        """
        return cast(
            dict[DataType, list[NCDataStats]],
            _DatasetStatisticsView(
                nested_statistics=self._dataset_statistics,
                flat_statistics=self._flat_dataset_statistics,
            ),
        )
