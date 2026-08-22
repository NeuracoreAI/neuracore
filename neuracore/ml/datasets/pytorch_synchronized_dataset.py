"""PyTorch dataset for loading synchronized robot data with filesystem caching."""

import hashlib
import json
import logging
from typing import cast

import numpy as np
import torch
from neuracore_types import (
    DATA_TYPE_TO_BATCHED_NC_DATA_CLASS,
    TARGET_OUTPUT_DATA_TYPES,
    BatchedNCData,
    CrossEmbodimentDescription,
    DataType,
    EmbodimentDescription,
    EmbodimentUnion,
    NCDataStats,
    SynchronizedDatasetStatistics,
    SynchronizedPoint,
)
from neuracore_types.nc_data.nc_data import DataItemStats

import neuracore as nc
from neuracore.core.const import DEFAULT_CACHE_DIR
from neuracore.core.data.synced_dataset import SynchronizedDataset
from neuracore.core.data.synced_recording import SynchronizedRecording
from neuracore.core.utils.training_input_args_validation import (
    _validate_cross_embodiment_description_against_dataset,
)
from neuracore.ml import BatchedTrainingSamples
from neuracore.ml.datasets.batch_sample_cache import BatchSampleCache
from neuracore.ml.datasets.pytorch_neuracore_dataset import PytorchNeuracoreDataset
from neuracore.ml.preprocessing.base import PreprocessingConfiguration
from neuracore.ml.utils.json_serialization import JsonValue, to_json_serializable
from neuracore.ml.utils.memory_monitor import MemoryMonitor
from neuracore.ml.utils.preprocessing import apply_preprocessing_methods

logger = logging.getLogger(__name__)

TrainingSample = BatchedTrainingSamples
CHECK_MEMORY_INTERVAL = 100


def _cacheable_cross_embodiment_description(
    description: object,
) -> JsonValue:
    """Return a JSON-serializable cross-embodiment description."""
    return to_json_serializable(description)


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
        input_preprocessing_config: PreprocessingConfiguration,
        output_preprocessing_config: PreprocessingConfiguration,
        output_prediction_horizon: int,
        input_observation_horizon: int = 1,
        sample_cache: bool = True,
    ):
        """Initialize the dataset.

        Args:
            synchronized_dataset: The synchronized dataset to load data from.
            input_cross_embodiment_description: List of input data types to
                include in the dataset.
            output_cross_embodiment_description: List of output data types to
                include in the dataset.
            input_preprocessing_config: Preprocessing configuration applied
                to input slots.
            output_preprocessing_config: Preprocessing configuration applied
                to output slots.
            output_prediction_horizon: Number of future timesteps to predict.
            input_observation_horizon: Number of consecutive observations ending
                at the current timestep to supply as input. ``1`` supplies only
                the current observation.
            sample_cache: Reuse fully built samples from an on-disk cache
                across epochs and runs, rebuilding on a miss.
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
            input_observation_horizon=input_observation_horizon,
            num_recordings=len(synchronized_dataset),
        )
        self.synchronized_dataset = synchronized_dataset

        # Try cached stats first; fall back to server computation if missing/unreadable.
        logger.info("Loading dataset statistics...")
        recording_fingerprint = [
            {
                "id": recording.id,
                "total_bytes": recording.total_bytes,
                "robot_id": recording.robot_id,
                "instance": recording.instance,
                "start_time": recording.start_time,
                "end_time": recording.end_time,
            }
            for recording in self.synchronized_dataset.dataset
        ]
        stats_request_payload = {
            "recordings": recording_fingerprint,
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

        self._memory_monitor = MemoryMonitor(
            max_ram_utilization=0.8, max_gpu_utilization=1.0, gpu_id=None
        )
        self._mem_check_counter = 0
        self._num_samples_excluding_last = self._get_num_training_observations() - len(
            self.synchronized_dataset
        )

        (
            self.episode_indices,
            self.episode_start_offsets,
            # Keys the sample cache: an episode index is a position in a
            # server-ordered list and can point at a different recording once
            # the dataset changes, where an id cannot.
            self._episode_recording_ids,
        ) = self._get_sample_to_episode_mapping()
        self._logged_in = False

        # Only the worker-side half runs here. Device-side methods are applied
        # by the trainer once the batch is on the accelerator, where they run
        # batched rather than once per frame on a contended worker CPU.
        self.input_preprocessing_config, _ = input_preprocessing_config.split_by_stage()
        self.output_preprocessing_config, _ = (
            output_preprocessing_config.split_by_stage()
        )

        # Everything below is a pure function of the cross-embodiment
        # descriptions, which are fixed once the dataset exists. Deriving it
        # here keeps it out of load_sample, which runs once per sample.
        self._max_items_per_input_type = self._get_max_items_per_data_type(
            self.input_cross_embodiment_description
        )
        self._max_items_per_output_type = self._get_max_items_per_data_type(
            self.output_cross_embodiment_description
        )
        # Index-ordered (index, name) pairs per robot, so projecting a sync
        # point does not re-sort the same keys for every one of the
        # output_prediction_horizon + 1 sync points a sample touches.
        self._merged_ordered_items = {
            robot_id: self._order_embodiment_items(
                self._convert_to_embodiment_description(embodiment_union)
            )
            for robot_id, embodiment_union in (
                self.merged_cross_embodiment_description.items()
            )
        }

        self._sample_cache = self._build_sample_cache() if sample_cache else None

    def rebuild_sample_cache(self) -> None:
        """Re-key the sample cache against the current preprocessing.

        Call after swapping a preprocessing configuration on an existing
        dataset, so entries are stored under a key describing what is actually
        being built rather than what the dataset was constructed with.
        """
        if self._sample_cache is not None:
            self._sample_cache = self._build_sample_cache()

    def _build_sample_cache(self) -> BatchSampleCache:
        """Create the cache for the currently configured preprocessing."""
        cache = BatchSampleCache(
            synchronized_dataset_id=self.synchronized_dataset.id,
            input_cross_embodiment_description=(
                self.input_cross_embodiment_description
            ),
            output_cross_embodiment_description=(
                self.output_cross_embodiment_description
            ),
            output_prediction_horizon=self.output_prediction_horizon,
            input_observation_horizon=self.input_observation_horizon,
            # Worker-side halves only; the dataset already discarded the rest.
            input_preprocessing_config=self.input_preprocessing_config,
            output_preprocessing_config=self.output_preprocessing_config,
        )
        logger.info("Caching built samples under %s", cache.directory)
        return cache

    @staticmethod
    def _get_max_items_per_data_type(
        cross_embodiment_description: CrossEmbodimentDescription,
    ) -> dict[DataType, int]:
        """Return the padded slot count for each data type.

        The count is the highest index used for that data type across every
        robot, plus one, so samples from different embodiments pad out to a
        common width.
        """
        highest_index: dict[DataType, int] = {}
        for data_types in cross_embodiment_description.values():
            for data_type, indexed_names in data_types.items():
                # Floored at 0 so a data type declared with no sensors still
                # gets a single padded slot, matching the per-sample scan this
                # replaces.
                highest_index[data_type] = max(
                    highest_index.get(data_type, 0), *indexed_names, 0
                )
        return {data_type: highest + 1 for data_type, highest in highest_index.items()}

    @staticmethod
    def _order_embodiment_items(
        description: EmbodimentDescription,
    ) -> dict[DataType, list[tuple[int, str]]]:
        """Flatten an embodiment description into index-ordered (index, name) pairs."""
        return {
            data_type: [
                (index, indexed_names[index]) for index in sorted(indexed_names)
            ]
            for data_type, indexed_names in description.items()
        }

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
            input_cross_embodiment_description: Input cross-embodiment description.
            output_cross_embodiment_description: Output cross-embodiment description.

        Raises:
            ValueError: If robot IDs or data types are missing from the dataset.
        """
        _validate_cross_embodiment_description_against_dataset(
            dataset=synchronized_dataset.dataset,
            dataset_name=f"synchronized dataset {synchronized_dataset.id}",
            cross_embodiment_description=input_cross_embodiment_description,
            description_kind="Input",
        )
        _validate_cross_embodiment_description_against_dataset(
            dataset=synchronized_dataset.dataset,
            dataset_name=f"synchronized dataset {synchronized_dataset.id}",
            cross_embodiment_description=output_cross_embodiment_description,
            description_kind="Output",
        )

    def _get_sample_to_episode_mapping(self) -> tuple[list[int], list[int], list[str]]:
        """Map each sample index to its episode index, start offset, and recording ID.

        Omit the last frame of each episode because it is not used for training.

        Returns:
            ``(episode_indices, episode_start_offsets, episode_recording_ids)``
            where ``episode_indices[sample_idx]`` is the episode index,
            ``episode_start_offsets[episode_idx]`` is the sample index that
            episode starts at, and ``episode_recording_ids[episode_idx]``
            is its id. The offsets let ``__getitem__`` recover a timestep by
            subtraction rather than by scanning ``episode_indices`` for the
            episode's first occurrence. The ids let the sample cache be keyed
            without a recording lookup; they are gathered here because
            iterating the synchronized dataset is not guaranteed to be
            restartable, so it must be walked exactly once.
        """
        episode_indices: list[int] = []
        episode_start_offsets: list[int] = []
        episode_recording_ids: list[str] = []
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
            episode_start_offsets.append(len(episode_indices))
            episode_recording_ids.append(recording.id)
            episode_indices.extend([recording_idx] * (len(recording) - 1))

        return episode_indices, episode_start_offsets, episode_recording_ids

    def _convert_to_embodiment_description(
        self, value: EmbodimentUnion
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
    def _project_sync_point(
        sync_point: SynchronizedPoint,
        ordered_items: dict[DataType, list[tuple[int, str]]],
    ) -> SynchronizedPoint:
        """Project a sync point onto the requested spec in deterministic order.

        Extra data types or sensor names in the source sync point are ignored.
        Missing required data types or sensor names raise a ValueError.

        Args:
            sync_point: The sync point to project.
            ordered_items: Index-ordered ``{data_type: [(index, name), ...]}``
                built once at construction, so this does not re-sort the same
                keys for every sync point of every sample.
        """
        projected_data: dict[DataType, dict[str, object]] = {}

        for data_type, indexed_names in ordered_items.items():
            source_data_for_type = sync_point.data.get(data_type)
            if source_data_for_type is None:
                raise ValueError(
                    f"SynchronizedPoint is missing required data type: {data_type}"
                )

            projected_for_type: dict[str, object] = {}
            for _, name in indexed_names:
                if name not in source_data_for_type:
                    raise ValueError(
                        "SynchronizedPoint is missing required sensor name "
                        f"'{name}' for data type {data_type}"
                    )
                projected_for_type[name] = source_data_for_type[name]
            projected_data[data_type] = projected_for_type

        return SynchronizedPoint.model_construct(
            timestamp=sync_point.timestamp,
            robot_id=sync_point.robot_id,
            data=projected_data,
        )

    @staticmethod
    def _get_timestep(episode_length: int) -> int:
        max_start = max(0, episode_length)
        return np.random.randint(0, max_start - 1)

    def _load_projected_output_sync_points(
        self,
        synced_recording: SynchronizedRecording,
        timestep: int,
        ordered_items: dict[DataType, list[tuple[int, str]]],
    ) -> list[SynchronizedPoint]:
        """Load the superset window for all output data types.

        Fetches ``[timestep, timestep + 1 + horizon]`` once so target types
        (aligned to the input step) and non-target types (next step onward)
        can share the same loaded sync points.
        """
        output_sync_points = cast(
            list[SynchronizedPoint],
            synced_recording[timestep : timestep + 1 + self.output_prediction_horizon],
        )
        return [
            self._project_sync_point(sync_point, ordered_items)
            for sync_point in output_sync_points
        ]

    def _load_projected_input_sync_points(
        self,
        synced_recording: SynchronizedRecording,
        timestep: int,
        ordered_items: dict[DataType, list[tuple[int, str]]],
    ) -> list[SynchronizedPoint]:
        """Load the observation window ending at ``timestep``.

        Returns exactly ``input_observation_horizon`` points. Near the start of
        a recording there is less history than asked for, so the earliest
        available point is repeated to fill the gap — the mirror of the
        last-point repeat the output side uses at the end of a recording. The
        length has to be exact either way, because collation concatenates
        samples along the batch axis and cannot stack ragged time extents.

        Args:
            synced_recording: Recording to read from.
            timestep: The current timestep, which ends the window.
            ordered_items: Index-ordered ``{data_type: [(index, name), ...]}``.
        """
        horizon = self.input_observation_horizon
        # A negative slice start would be normalized against the recording
        # length and silently return the wrong window, so clamp it here.
        start = max(0, timestep - horizon + 1)
        window = cast(
            list[SynchronizedPoint],
            synced_recording[start : timestep + 1],
        )
        if not window:
            raise ValueError(
                f"No input sync points available at timestep {timestep} in "
                f"recording '{synced_recording.name}'"
            )
        padding = [window[0]] * (horizon - len(window))
        return [
            self._project_sync_point(sync_point, ordered_items)
            for sync_point in padding + window
        ]

    @staticmethod
    def _output_sync_points_for_data_type(
        output_sync_points: list[SynchronizedPoint],
        data_type: DataType,
        output_prediction_horizon: int,
        *,
        timestep: int,
        recording_name: str,
    ) -> list[SynchronizedPoint]:
        """Select the per-type output window from a preloaded sync-point slice.

        Target output types are aligned with the input timestep.
        Other outputs use the next timestep onward.
        """
        window_start = 0 if data_type in TARGET_OUTPUT_DATA_TYPES else 1
        aligned_output_sync_points = list(
            output_sync_points[window_start : window_start + output_prediction_horizon]
        )

        if not aligned_output_sync_points:
            raise ValueError(
                f"No output sync points available for data type {data_type.value} "
                f"at timestep {timestep} in recording '{recording_name}'"
            )

        for _ in range(output_prediction_horizon - len(aligned_output_sync_points)):
            aligned_output_sync_points.append(aligned_output_sync_points[-1])

        return aligned_output_sync_points

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

        # Check for a cached sample first, keyed by recording id and timestep.
        if self._sample_cache is not None and timestep is not None:
            cached = self._sample_cache.load(
                self._episode_recording_ids[episode_idx], timestep
            )
            if cached is not None:
                return cached

        synced_recording = self.synchronized_dataset[episode_idx]
        synced_recording = cast(SynchronizedRecording, synced_recording)
        episode_length = len(synced_recording)
        if timestep is None:
            timestep = self._get_timestep(episode_length)

        # Order the SynchronizedPoints to the merged embodiment description.
        robot_id = synced_recording.robot_id

        merged_ordered_items = self._merged_ordered_items[robot_id]
        input_sync_points = self._load_projected_input_sync_points(
            synced_recording=synced_recording,
            timestep=timestep,
            ordered_items=merged_ordered_items,
        )

        output_sync_points = self._load_projected_output_sync_points(
            synced_recording=synced_recording,
            timestep=timestep,
            ordered_items=merged_ordered_items,
        )
        recording_name = getattr(synced_recording, "name", "recording")

        # Sort out Inputs
        inputs: dict[DataType, list[BatchedNCData]] = {}
        inputs_mask: dict[DataType, torch.Tensor] = {}

        for data_type in self.input_cross_embodiment_description[robot_id]:
            batched_nc_data_class = DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[data_type]
            inputs[data_type] = []

            max_items_trained_on = self._max_items_per_input_type[data_type]
            input_mask_values: list[float] = [0.0] * max_items_trained_on
            for index in range(max_items_trained_on):
                name = self.input_cross_embodiment_description[robot_id][data_type].get(
                    index
                )

                if name is None:
                    # Pad missing data with zeros.
                    batched_nc_data = batched_nc_data_class.sample(
                        batch_size=1, time_steps=self.input_observation_horizon
                    )
                else:
                    # If the current robot has a name for this index, use it to
                    # get the data.
                    nc_data_list = [
                        input_sp.data[data_type][name] for input_sp in input_sync_points
                    ]
                    batched_nc_data = batched_nc_data_class.from_nc_data_list(
                        nc_data_list
                    )
                    input_mask_values[index] = 1.0

                batched_nc_data = apply_preprocessing_methods(
                    batched_data=batched_nc_data,
                    methods=self.input_preprocessing_config.get(data_type, []),
                )
                inputs[data_type].append(batched_nc_data)

            # Create mask for inputs
            inputs_mask[data_type] = torch.tensor(
                input_mask_values, dtype=torch.float32
            )

        outputs: dict[DataType, list[BatchedNCData]] = {}
        outputs_mask: dict[DataType, torch.Tensor] = {}
        for data_type in self.output_cross_embodiment_description[robot_id]:
            batched_nc_data_class = DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[data_type]
            outputs[data_type] = []

            max_items_trained_on = self._max_items_per_output_type[data_type]
            aligned_output_sync_points = self._output_sync_points_for_data_type(
                output_sync_points,
                data_type,
                self.output_prediction_horizon,
                timestep=timestep,
                recording_name=recording_name,
            )
            output_mask_values: list[float] = [0.0] * max_items_trained_on
            for index in range(max_items_trained_on):
                name = self.output_cross_embodiment_description[robot_id][
                    data_type
                ].get(index)

                if name is None:
                    # Pad missing data with zeros.
                    batched_nc_data = batched_nc_data_class.sample(
                        batch_size=1,
                        time_steps=self.output_prediction_horizon,
                    )
                else:
                    # If the current robot has a name for this index,
                    # use it to get the data.
                    nc_data_list = [
                        output_sp.data[data_type][name]
                        for output_sp in aligned_output_sync_points
                    ]
                    batched_nc_data = batched_nc_data_class.from_nc_data_list(
                        nc_data_list
                    )
                    output_mask_values[index] = 1.0

                batched_nc_data = apply_preprocessing_methods(
                    batched_data=batched_nc_data,
                    methods=self.output_preprocessing_config.get(data_type, []),
                )
                outputs[data_type].append(batched_nc_data)

            # Create mask for outputs.
            outputs_mask[data_type] = torch.tensor(
                output_mask_values, dtype=torch.float32
            )

        sample = TrainingSample(
            inputs=inputs,
            inputs_mask=inputs_mask,
            outputs=outputs,
            outputs_mask=outputs_mask,
            batch_size=1,
        )
        if self._sample_cache is not None:
            self._sample_cache.store(synced_recording.id, timestep, sample)
        return sample

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Omit the last frame of each episode because it is not used for training.

        Returns:
            The number of samples in the dataset.
        """
        return self._num_samples_excluding_last

    def __getitem__(self, idx: int) -> TrainingSample:
        """Get a training sample by index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A TrainingSample containing the requested data.
        """
        if idx < 0:
            # Handle negative indices by wrapping around
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {len(self)}"
            )

        episode_idx = self.episode_indices[idx]
        timestep = idx - self.episode_start_offsets[episode_idx]
        return self.load_sample(episode_idx, timestep)

    @property
    def dataset_statistics(self) -> dict[str, dict[DataType, list[NCDataStats]]]:
        """Return the dataset description."""
        return self._dataset_statistics
