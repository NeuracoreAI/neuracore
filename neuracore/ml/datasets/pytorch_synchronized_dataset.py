"""PyTorch dataset for loading synchronized robot data with filesystem caching."""

import logging
from collections import defaultdict
from typing import cast

import numpy as np
import torch
from neuracore_types import (
    DATA_TYPE_TO_BATCHED_NC_DATA_CLASS,
    BatchedNCData,
    DataType,
    NCDataStats,
    RobotDataSpec,
    SynchronizedPoint,
)
from neuracore_types.nc_data.nc_data import DataItemStats

import neuracore as nc
from neuracore.core.data.synced_dataset import SynchronizedDataset
from neuracore.core.data.synced_recording import SynchronizedRecording
from neuracore.ml import BatchedTrainingSamples
from neuracore.ml.datasets.pytorch_neuracore_dataset import PytorchNeuracoreDataset
from neuracore.ml.utils.memory_monitor import MemoryMonitor

logger = logging.getLogger(__name__)

TrainingSample = BatchedTrainingSamples
CHECK_MEMORY_INTERVAL = 100


class PytorchSynchronizedDataset(PytorchNeuracoreDataset):
    """Dataset for loading episodic robot data from GCS with filesystem caching.

    Enhanced to support all data types including depth images, point clouds,
    poses, end-effectors, and custom sensor data.
    """

    def __init__(
        self,
        synchronized_dataset: SynchronizedDataset,
        input_robot_data_spec: RobotDataSpec,
        output_robot_data_spec: RobotDataSpec,
        output_prediction_horizon: int,
    ):
        """Initialize the dataset.

        Args:
            synchronized_dataset: The synchronized dataset to load data from.
            input_robot_data_spec: List of input data types to include in the dataset.
            output_robot_data_spec: List of output data types to include in the dataset.
            output_prediction_horizon: Number of future timesteps to predict.
            order_configuration: Configuration for ordering data types.
        """
        self._validate_robot_specs(
            synchronized_dataset, input_robot_data_spec, output_robot_data_spec
        )

        super().__init__(
            input_robot_data_spec=input_robot_data_spec,
            output_robot_data_spec=output_robot_data_spec,
            output_prediction_horizon=output_prediction_horizon,
            num_recordings=len(synchronized_dataset),
        )
        self.synchronized_dataset = synchronized_dataset

        logger.info("Calculated dataset statistics...")
        self.synchronized_dataset_statistics = (
            synchronized_dataset.calculate_statistics(
                robot_data_spec=self.robot_data_spec
            )
        )
        self._dataset_statistics = (
            self.synchronized_dataset_statistics.dataset_statistics
        )
        logger.info("Done calculating dataset statistics.")

        self._max_error_count = 100
        self._error_count = 0
        self._memory_monitor = MemoryMonitor(
            max_ram_utilization=0.8, max_gpu_utilization=1.0, gpu_id=None
        )
        self._mem_check_counter = 0
        self._num_samples_excluding_last = self._get_num_training_observations() - len(
            self.synchronized_dataset
        )
        self.episode_indices = self._get_episode_indices()
        self._logged_in = False

        # If user does not provide a robot data spec, use the first sample we see
        # to determine ordering
        self._requires_fallback = False
        for robot_id, data_spec in self.robot_data_spec.items():
            for data_type, names in data_spec.items():
                if len(names) == 0:
                    self._requires_fallback = True
                    break

        self._fallback_robot_data_spec: RobotDataSpec = {}

    def _get_num_training_observations(self) -> int:
        # The count attribute of the stats should give total number of training
        # observations and should be same across all data types
        first_data_type = next(iter(self._dataset_statistics))
        data_stats_of_unknown_nc_data = self._dataset_statistics[first_data_type][0]
        # Loop over all attributes until we find one of type DataItemStats
        for attr_name, attr_value in vars(data_stats_of_unknown_nc_data).items():
            if isinstance(attr_value, DataItemStats):
                return attr_value.count.item()
        raise ValueError(
            "Could not find DataItemStats in dataset "
            "statistics to get number of training observations."
        )

    def _validate_robot_specs(
        self,
        synchronized_dataset: SynchronizedDataset,
        input_robot_data_spec: RobotDataSpec,
        output_robot_data_spec: RobotDataSpec,
    ) -> None:
        """Validate that robot IDs and data types exist in the synchronized dataset.

        Args:
            synchronized_dataset: The synchronized dataset to validate against.
            input_robot_data_spec: Input robot data specification.
            output_robot_data_spec: Output robot data specification.

        Raises:
            ValueError: If robot IDs or data types are missing from the dataset.
        """
        assert synchronized_dataset.robot_data_spec is not None
        robot_ids_in_dataset = set(synchronized_dataset.robot_data_spec.keys())

        # Validate robot IDs
        self._validate_robot_ids(
            set(input_robot_data_spec.keys()), robot_ids_in_dataset, "Input"
        )
        self._validate_robot_ids(
            set(output_robot_data_spec.keys()), robot_ids_in_dataset, "Output"
        )

        # Validate data types per robot
        for robot_id in robot_ids_in_dataset:
            if robot_id in input_robot_data_spec:
                self._validate_data_types(
                    robot_id,
                    set(input_robot_data_spec[robot_id]),
                    set(synchronized_dataset.robot_data_spec[robot_id]),
                    "Input",
                )
            if robot_id in output_robot_data_spec:
                self._validate_data_types(
                    robot_id,
                    set(output_robot_data_spec[robot_id]),
                    set(synchronized_dataset.robot_data_spec[robot_id]),
                    "Output",
                )

    def _validate_robot_ids(
        self, spec_robot_ids: set, dataset_robot_ids: set, spec_type: str
    ) -> None:
        """Validate that robot IDs in spec exist in dataset.

        Args:
            spec_robot_ids: Robot IDs from input/output spec.
            dataset_robot_ids: Robot IDs available in dataset.
            spec_type: Either "Input" or "Output" for error messages.

        Raises:
            ValueError: If robot IDs are missing from the dataset.
        """
        if not spec_robot_ids.issubset(dataset_robot_ids):
            missing_robot_ids = spec_robot_ids - dataset_robot_ids
            raise ValueError(
                f"{spec_type} robot IDs {missing_robot_ids} "
                "not found in synchronized dataset. "
                f"robot_ids_in_{spec_type.lower()}_spec has:\n{spec_robot_ids}. \n"
                f"robot_ids_in_synchronized_dataset has:\n{dataset_robot_ids}."
            )

    def _validate_data_types(
        self,
        robot_id: str,
        spec_data_types: set,
        available_data_types: set,
        spec_type: str,
    ) -> None:
        """Validate that data types for a robot exist in dataset.

        Args:
            robot_id: The robot ID being validated.
            spec_data_types: Data types from input/output spec.
            available_data_types: Data types available in dataset.
            spec_type: Either "Input" or "Output" for error messages.

        Raises:
            ValueError: If data types are missing from the dataset.
        """
        if not spec_data_types.issubset(available_data_types):
            missing_data_types = spec_data_types - available_data_types
            raise ValueError(
                f"{spec_type} data types {missing_data_types} for robot ID {robot_id} "
                f"not found in synchronized dataset. "
                f"{spec_type.lower()}_data_types has:\n{spec_data_types}. \n"
                f"available_data_types has:\n{available_data_types}."
            )

    def _get_episode_indices(self) -> list[int]:
        """Return a list mapping each sample index to its episode (recording) index.

        Omit the last frame of each episode because it is not used for training.

        Returns:
            A list mapping each sample index to its episode (recording) index.
        """
        episode_indices = []
        for recording_idx, recording in enumerate(self.synchronized_dataset):
            episode_indices.extend([recording_idx] * (len(recording) - 1))

        return episode_indices

    @staticmethod
    def _get_timestep(episode_length: int) -> int:
        max_start = max(0, episode_length)
        return np.random.randint(0, max_start - 1)

    def load_sample(
        self, episode_idx: int, timestep: int | None = None
    ) -> TrainingSample:
        """Load sample from cache or GCS with full data type support."""
        if not self._logged_in:
            nc.login()
            self._logged_in = True

        if self._mem_check_counter % CHECK_MEMORY_INTERVAL == 0:
            self._memory_monitor.check_memory()
            self._mem_check_counter = 0
        self._mem_check_counter += 1

        synced_recording = self.synchronized_dataset[episode_idx]
        synced_recording = cast(SynchronizedRecording, synced_recording)
        episode_length = len(synced_recording)
        if timestep is None:
            timestep = self._get_timestep(episode_length)

        sync_point = cast(SynchronizedPoint, synced_recording[timestep])
        future_sync_points = cast(
            list[SynchronizedPoint],
            synced_recording[
                timestep + 1 : timestep + 1 + self.output_prediction_horizon
            ],
        )

        # Order the SynchronizedPoints
        robot_id = synced_recording.robot_id

        if self._requires_fallback:
            # Build fallback spec from first sync point seen
            if robot_id not in self._fallback_robot_data_spec:
                self._fallback_robot_data_spec[robot_id] = {
                    data_type: list(sync_point.data[data_type].keys())
                    for data_type in sync_point.data.keys()
                }
            spec_for_ordering = self._fallback_robot_data_spec[robot_id]
        else:
            spec_for_ordering = self.robot_data_spec[robot_id]

        sync_point = sync_point.order(spec_for_ordering)
        for i in range(len(future_sync_points)):
            future_sync_points[i] = future_sync_points[i].order(spec_for_ordering)

        # Padding for future sync points
        for _ in range(self.output_prediction_horizon - len(future_sync_points)):
            future_sync_points.append(future_sync_points[-1])

        inputs: dict[DataType, list[BatchedNCData]] = {}
        inputs_mask: dict[DataType, torch.Tensor] = {}
        for data_type in self.input_robot_data_spec[robot_id]:
            batched_nc_data_class = DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[data_type]
            inputs[data_type] = []
            for name, nc_data in sync_point.data[data_type].items():
                batched_nc_data = batched_nc_data_class.from_nc_data(nc_data)
                inputs[data_type].append(batched_nc_data)

            # Create mask for inputs
            max_items_for_this_data_type = len(sync_point.data[data_type])
            max_items_trained_on = len(self.dataset_statistics[data_type])
            inputs_mask[data_type] = torch.tensor(
                [1.0] * max_items_for_this_data_type
                + [0.0] * (max_items_trained_on - max_items_for_this_data_type),
                dtype=torch.float32,
            )

        outputs: dict[DataType, list[BatchedNCData]] = {}
        outputs_mask: dict[DataType, torch.Tensor] = {}
        for data_type in self.output_robot_data_spec[robot_id]:
            batched_nc_data_class = DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[data_type]
            outputs[data_type] = []
            # Need to add action prediction horizon for outputs
            for name in sync_point.data[data_type].keys():
                batched_nc_data_for_data_type: dict[str, list[torch.Tensor]] = (
                    defaultdict(list)
                )
                for future_sp in future_sync_points:
                    nc_data = future_sp.data[data_type][name]
                    batched_nc_data_at_t = batched_nc_data_class.from_nc_data(nc_data)
                    for attr_name in vars(batched_nc_data_at_t):
                        attr_value = getattr(batched_nc_data_at_t, attr_name)
                        if not isinstance(attr_value, torch.Tensor):
                            continue
                        batched_nc_data_for_data_type[attr_name].append(attr_value)
                # Now cat along time dimension
                cat_kwargs: dict[str, torch.Tensor] = {}
                for attr_name, attr_values in batched_nc_data_for_data_type.items():
                    cat_kwargs[attr_name] = torch.cat(attr_values, dim=1)  # (B, T, ...)
                batched_nc_data = batched_nc_data_class(**cat_kwargs)
                outputs[data_type].append(batched_nc_data)

            # Create mask for outputs
            max_items_for_this_data_type = len(sync_point.data[data_type])
            max_items_trained_on = len(self.dataset_statistics[data_type])
            outputs_mask[data_type] = torch.tensor(
                [1.0] * max_items_for_this_data_type
                + [0.0] * (max_items_trained_on - max_items_for_this_data_type),
                dtype=torch.float32,
            )

        return TrainingSample(
            inputs=inputs,
            inputs_mask=inputs_mask,
            outputs=outputs,
            outputs_mask=outputs_mask,
            batch_size=1,
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Omit the last frame of each episode because it is not used for training.

        Returns:
            The number of samples in the dataset.
        """
        return self._num_samples_excluding_last

    def __getitem__(self, idx: int) -> TrainingSample:
        """Get a training sample by index with error handling.

        Implements the PyTorch Dataset interface with robust error handling
        to manage data loading failures gracefully during training.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A TrainingSample containing the requested data.

        Raises:
            Exception: If sample loading fails after exhausting retry attempts.
        """
        if idx < 0:
            # Handle negative indices by wrapping around
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {len(self)}"
            )
        while self._error_count < self._max_error_count:
            try:
                episode_idx = self.episode_indices[idx]
                timestep = idx - self.episode_indices.index(episode_idx)
                return self.load_sample(episode_idx, timestep)
            except Exception:
                self._error_count += 1
                logger.error(f"Error loading item {idx}.", exc_info=True)
                if self._error_count >= self._max_error_count:
                    raise
        raise Exception(
            f"Maximum error count ({self._max_error_count}) already reached"
        )

    @property
    def dataset_statistics(self) -> dict[DataType, list[NCDataStats]]:
        """Return the dataset description."""
        return self._dataset_statistics
