"""PyTorch dataset for loading synchronized robot data with filesystem caching."""

import hashlib
import json
import logging
from typing import cast

import numpy as np
import torch
from neuracore_types import (
    DATA_TYPE_TO_BATCHED_NC_DATA_CLASS,
    BatchedNCData,
    CrossEmbodimentDescription,
    DataType,
    EmbodimentDescription,
    NCDataStats,
    PreProcessingConfiguration,
    RobotDataSpec,
    SynchronizedDatasetStatistics,
    SynchronizedPoint,
)
from neuracore_types.nc_data.nc_data import DataItemStats

import neuracore as nc
from neuracore.core.const import DEFAULT_CACHE_DIR
from neuracore.core.data.synced_dataset import SynchronizedDataset
from neuracore.core.data.synced_recording import SynchronizedRecording
from neuracore.core.utils.training_input_args_validation import (
    _validate_data_specs_against_dataset,
)
from neuracore.ml import BatchedTrainingSamples
from neuracore.ml.datasets.pytorch_neuracore_dataset import PytorchNeuracoreDataset
from neuracore.ml.preprocessing import apply_methods_for_slot
from neuracore.ml.utils.memory_monitor import MemoryMonitor

logger = logging.getLogger(__name__)

TrainingSample = BatchedTrainingSamples
CHECK_MEMORY_INTERVAL = 100


def _cacheable_cross_embodiment_description(
    description: CrossEmbodimentDescription,
) -> CrossEmbodimentDescription:
    """Return a JSON-serializable cross-embodiment description."""
    if hasattr(description, "model_dump"):
        return description.model_dump(mode="json")
    return description


class PytorchSynchronizedDataset(PytorchNeuracoreDataset):
    """Dataset for loading episodic robot data from GCS with filesystem caching.

    Enhanced to support all data types including depth images, point clouds,
    poses, end-effectors, and custom sensor data.
    """

    def __init__(
        self,
        synchronized_dataset: SynchronizedDataset,
        input_cross_embodiment_description: CrossEmbodimentDescription,
        output_cross_embodiment_description: CrossEmbodimentDescription,
        output_prediction_horizon: int,
        input_preprocessing_config: PreProcessingConfiguration | None = None,
        output_preprocessing_config: PreProcessingConfiguration | None = None,
    ):
        """Initialize the dataset.

        Args:
            synchronized_dataset: The synchronized dataset to load data from.
            input_cross_embodiment_description: List of input data types to
                include in the dataset.
            output_cross_embodiment_description: List of output data types to
                include in the dataset.
            output_prediction_horizon: Number of future timesteps to predict.
            input_preprocessing_config: Optional preprocessing configuration
                applied to input slots.
            output_preprocessing_config: Optional preprocessing configuration
                applied to output slots.
        """
        self._validate_cross_embodiment_specs(
            synchronized_dataset,
            input_cross_embodiment_description,
            output_cross_embodiment_description,
        )

        super().__init__(
            input_cross_embodiment_description=input_cross_embodiment_description,
            output_cross_embodiment_description=output_cross_embodiment_description,
            output_prediction_horizon=output_prediction_horizon,
            num_recordings=len(synchronized_dataset),
        )
        self.synchronized_dataset = synchronized_dataset

        # Try cached stats first; fall back to server computation if missing/unreadable.
        logger.info("Loading dataset statistics...")
        stats_request_payload = {
            "input_cross_embodiment_description": (
                _cacheable_cross_embodiment_description(
                    self.input_cross_embodiment_description
                )
            ),
            "output_cross_embodiment_description": (
                _cacheable_cross_embodiment_description(
                    self.output_cross_embodiment_description
                )
            ),
        }
        spec_key = json.dumps(
            stats_request_payload, sort_keys=True, separators=(",", ":")
        )
        spec_hash = hashlib.sha256(spec_key.encode("utf-8")).hexdigest()[:12]

        # Hash the full statistics request so different input/output roles do not
        # collide even when their merged sync union is identical.
        stats_cache_dir = DEFAULT_CACHE_DIR / "dataset_cache"
        stats_cache_path = (
            stats_cache_dir
            / f"{self.synchronized_dataset.id}_statistics_{spec_hash}.json"
        )

        self.synchronized_dataset_statistics = None
        # Read cached stats if present; ignore and recompute on parse errors.
        if stats_cache_path.exists():
            try:
                with stats_cache_path.open("r", encoding="utf-8") as handle:
                    cached = json.load(handle)
                self.synchronized_dataset_statistics = (
                    SynchronizedDatasetStatistics.model_validate(cached)
                )
                logger.info("Loaded dataset statistics from cache.")
            except (OSError, ValueError) as exc:
                logger.warning(
                    "Failed to read cached statistics at %s: %s",
                    stats_cache_path,
                    exc,
                )

        # Cache miss: compute via API, then persist for next run.
        if self.synchronized_dataset_statistics is None:
            logger.info("Calculating dataset statistics...")
            calculate_statistics = synchronized_dataset.calculate_statistics
            self.synchronized_dataset_statistics = calculate_statistics(
                input_cross_embodiment_description=self.input_cross_embodiment_description,
                output_cross_embodiment_description=self.output_cross_embodiment_description,
            )

            stats_cache_dir.mkdir(parents=True, exist_ok=True)
            with stats_cache_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    self.synchronized_dataset_statistics.model_dump(mode="json"),
                    handle,
                )
            logger.info("Done calculating dataset statistics.")

        self._dataset_statistics = (
            self.synchronized_dataset_statistics.dataset_statistics
        )

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
        self._input_preprocessing_config = input_preprocessing_config
        self._output_preprocessing_config = output_preprocessing_config

    @staticmethod
    def _apply_slot_preprocessing(
        data_type: DataType,
        slot_idx: int,
        batched_nc_data: BatchedNCData,
        preprocessing_config: PreProcessingConfiguration | None,
    ) -> BatchedNCData:
        if preprocessing_config is None:
            return batched_nc_data
        methods = preprocessing_config.steps.get(data_type, {}).get(slot_idx, [])
        if not methods:
            return batched_nc_data
        return apply_methods_for_slot(
            data_type=data_type,
            batched_data=batched_nc_data,
            methods=methods,
        )

    def _get_num_training_observations(self) -> int:
        # The count attribute of the stats should give total number of training
        # observations and should be same across all data types
        first_data_type = next(iter(self._dataset_statistics["input"]))
        data_stats_of_unknown_nc_data = self._dataset_statistics["input"][
            first_data_type
        ][0]
        # Loop over all attributes until we find one of type DataItemStats
        for attr_name, attr_value in vars(data_stats_of_unknown_nc_data).items():
            if isinstance(attr_value, DataItemStats):
                return attr_value.count.item()
        raise ValueError(
            "Could not find DataItemStats in dataset "
            "statistics to get number of training observations."
        )

    def _validate_cross_embodiment_specs(
        self,
        synchronized_dataset: SynchronizedDataset,
        input_cross_embodiment_description: CrossEmbodimentDescription,
        output_cross_embodiment_description: CrossEmbodimentDescription,
    ) -> None:
        """Validate that robot IDs and data types exist in the synchronized dataset.

        Args:
            synchronized_dataset: The synchronized dataset to validate against.
            input_cross_embodiment_description: Input robot data specification.
            output_cross_embodiment_description: Output robot data specification.

        Raises:
            ValueError: If robot IDs or data types are missing from the dataset.
        """
        _validate_data_specs_against_dataset(
            dataset=synchronized_dataset.dataset,
            dataset_name=f"synchronized dataset {synchronized_dataset.id}",
            cross_embodiment_description=input_cross_embodiment_description,
            spec_kind="Input",
        )
        _validate_data_specs_against_dataset(
            dataset=synchronized_dataset.dataset,
            dataset_name=f"synchronized dataset {synchronized_dataset.id}",
            cross_embodiment_description=output_cross_embodiment_description,
            spec_kind="Output",
        )

    def _get_episode_indices(self) -> list[int]:
        """Return a list mapping each sample index to its episode (recording) index.

        Omit the last frame of each episode because it is not used for training.

        Returns:
            A list mapping each sample index to its episode (recording) index.
        """
        episode_indices = []
        for recording_idx, recording in enumerate(self.synchronized_dataset):
            # Each recording must have at least 2 timesteps because we drop the
            # last frame from training. Otherwise alignment with per-recording
            # metadata breaks (zero samples contributed).
            if len(recording) <= 1:
                raise ValueError(
                    "Synchronized recording "
                    f"'{recording.name}' has only {len(recording)} frame(s); "
                    "need >= 2 frames to generate training samples."
                )
            episode_indices.extend([recording_idx] * (len(recording) - 1))

        return episode_indices

    def convert_to_embodiment_description(
        self, value: dict[DataType, list[str]]
    ) -> EmbodimentDescription:
        """Normalize list-based sensor specs into indexed embodiment mappings.

        Converts:
            {
                DataType.JOINT_POSITIONS: ["joint1", "joint2"]
            }

        Into:
            {
                DataType.JOINT_POSITIONS: {
                    0: "joint1",
                    1: "joint2"
                }
            }

        Guarantees:
        - Order is preserved → index defines semantic position
        - Deterministic mapping
        - No mutation of input
        """
        if value is None:
            return {}

        embodiment_description: EmbodimentDescription = {}

        for data_type, items in value.items():
            if not isinstance(items, list):
                raise TypeError(
                    f"Expected list for {data_type}, got {type(items).__name__}"
                )

            # Optional: strict validation (useful for your pipeline)
            if any(not isinstance(x, str) for x in items):
                raise ValueError(f"All entries for {data_type} must be strings")

            embodiment_description[data_type] = {
                idx: name for idx, name in enumerate(items)
            }

        return embodiment_description

    @staticmethod
    def _project_sync_point_to_spec(
        sync_point: SynchronizedPoint,
        embodiment_description: EmbodimentDescription,
    ) -> SynchronizedPoint:
        """Project a sync point onto the requested spec in deterministic order.

        Extra data types or sensor names in the source sync point are ignored.
        Missing required data types or sensor names raise a ValueError.
        """
        projected_data: dict[DataType, dict[str, object]] = {}

        for data_type, indexed_names in embodiment_description.items():
            source_data_for_type = sync_point.data.get(data_type)
            if source_data_for_type is None:
                raise ValueError(
                    f"SynchronizedPoint is missing required data type: {data_type}"
                )

            projected_data[data_type] = {}
            for index in sorted(indexed_names):
                name = indexed_names[index]
                if name not in source_data_for_type:
                    raise ValueError(
                        "SynchronizedPoint is missing required sensor name "
                        f"'{name}' for data type {data_type}"
                    )
                projected_data[data_type][name] = source_data_for_type[name]

        return SynchronizedPoint.model_construct(
            timestamp=sync_point.timestamp,
            robot_id=sync_point.robot_id,
            data=projected_data,
        )

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

        spec_for_ordering: dict[DataType, list[str]] = (
            self.merged_cross_embodiment_description[robot_id]
        )
        converted_embodiment_description: EmbodimentDescription = (
            self.convert_to_embodiment_description(spec_for_ordering)
        )
        sync_point = self._project_sync_point_to_spec(
            sync_point, converted_embodiment_description
        )

        for i in range(len(future_sync_points)):
            future_sync_points[i] = self._project_sync_point_to_spec(
                future_sync_points[i], converted_embodiment_description
            )

        # Padding for future sync points
        for _ in range(self.output_prediction_horizon - len(future_sync_points)):
            future_sync_points.append(future_sync_points[-1])

        # Sort out Inputs
        inputs: dict[DataType, list[BatchedNCData]] = {}
        inputs_mask: dict[DataType, torch.Tensor] = {}

        for data_type in self.input_cross_embodiment_description[robot_id]:
            batched_nc_data_class = DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[data_type]
            inputs[data_type] = []
            mask = []
            for slot_idx, (name, nc_data) in enumerate(
                sync_point.data[data_type].items()
            ):
                batched_nc_data = batched_nc_data_class.from_nc_data(nc_data)
                batched_nc_data = self._apply_slot_preprocessing(
                    data_type=data_type,
                    slot_idx=slot_idx,
                    batched_nc_data=batched_nc_data,
                    preprocessing_config=self._input_preprocessing_config,
                )
                inputs[data_type].append(batched_nc_data)

            max_items_for_this_data_type = 0
            # Iterate through all robots and find the max index for this data
            # type across all robots to determine padding length.
            for other_robot_id in self.input_cross_embodiment_description:
                for index in self.input_cross_embodiment_description[other_robot_id][
                    data_type
                ].keys():
                    if index > max_items_for_this_data_type:
                        max_items_for_this_data_type = index

            for index in range(max_items_for_this_data_type + 1):
                name = self.input_cross_embodiment_description[robot_id][data_type].get(
                    index
                )

                if name is not None:
                    # If the current robot has a name for this index, use it to
                    # get the data. Otherwise, pad with zeros.
                    nc_data = sync_point.data[data_type][name]
                    batched_nc_data = batched_nc_data_class.from_nc_data(nc_data)
                    batched_nc_data.transform_nc_data()
                    inputs[data_type].append(batched_nc_data)
                    mask.append(1.0)

                else:
                    # Pad missing data with zeros
                    inputs[data_type].append(
                        batched_nc_data_class.sample(batch_size=1, time_steps=1)
                    )
                    mask.append(0.0)

            # Create mask for inputs
            inputs_mask[data_type] = torch.tensor(mask, dtype=torch.float32)

        outputs: dict[DataType, list[BatchedNCData]] = {}
        outputs_mask: dict[DataType, torch.Tensor] = {}
        for data_type in self.output_cross_embodiment_description[robot_id]:
            batched_nc_data_class = DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[data_type]
            outputs[data_type] = []
            mask = []
            # Need to add action prediction horizon for outputs
            for slot_idx, name in enumerate(sync_point.data[data_type].keys()):
                nc_data_list = [
                    future_sp.data[data_type][name] for future_sp in future_sync_points
                ]
                batched_nc_data = batched_nc_data_class.from_nc_data_list(nc_data_list)
                batched_nc_data = self._apply_slot_preprocessing(
                    data_type=data_type,
                    slot_idx=slot_idx,
                    batched_nc_data=batched_nc_data,
                    preprocessing_config=self._output_preprocessing_config,
                )
                outputs[data_type].append(batched_nc_data)

            max_items_for_this_data_type = 0
            # Iterate through all robots and find the max index for this data
            # type across all robots to determine padding length.
            for other_robot_id in self.output_cross_embodiment_description:
                for index in self.output_cross_embodiment_description[other_robot_id][
                    data_type
                ].keys():
                    if index > max_items_for_this_data_type:
                        max_items_for_this_data_type = index

            # Need to add action prediction horizon for outputs.
            for index in range(max_items_for_this_data_type + 1):
                name = self.output_cross_embodiment_description[robot_id][
                    data_type
                ].get(index)

                if name is not None:
                    # If the current robot has a name for this index, use it to
                    # get the data. Otherwise, pad with zeros.
                    nc_data_list = [
                        future_sp.data[data_type][name]
                        for future_sp in future_sync_points
                    ]
                    batched_nc_data = batched_nc_data_class.from_nc_data_list(
                        nc_data_list
                    )
                    batched_nc_data.transform_nc_data()
                    outputs[data_type].append(batched_nc_data)
                    mask.append(1.0)
                else:
                    # Pad missing data with zeros.
                    outputs[data_type].append(
                        batched_nc_data_class.sample(
                            batch_size=1,
                            time_steps=self.output_prediction_horizon,
                        )
                    )
                    mask.append(0.0)

            # Create mask for outputs.
            outputs_mask[data_type] = torch.tensor(mask, dtype=torch.float32)

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
    def dataset_statistics(self) -> dict[str, dict[DataType, list[NCDataStats]]]:
        """Return the dataset description."""
        return self._dataset_statistics
