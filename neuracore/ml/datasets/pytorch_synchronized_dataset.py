"""PyTorch dataset for loading synchronized robot data with filesystem caching."""

import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
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
from neuracore.core.data.cache_manager import CacheManager
from neuracore.core.data.synced_dataset import SynchronizedDataset
from neuracore.core.data.synced_recording import SynchronizedRecording
from neuracore.core.utils.training_input_args_validation import (
    _validate_cross_embodiment_description_against_dataset,
)
from neuracore.ml import BatchedTrainingSamples
from neuracore.ml.datasets.pytorch_neuracore_dataset import PytorchNeuracoreDataset
from neuracore.ml.preprocessing.base import (
    PreprocessingConfiguration,
    PreprocessingStage,
)
from neuracore.ml.utils.json_serialization import JsonValue, to_json_serializable
from neuracore.ml.utils.memory_monitor import MemoryMonitor
from neuracore.ml.utils.preprocessing import apply_preprocessing_methods

logger = logging.getLogger(__name__)

TrainingSample = BatchedTrainingSamples
CHECK_MEMORY_INTERVAL = 100

# How many timed samples go into one summary line. With the default sampling
# interval this is a handful of lines per worker per epoch.
TIMING_SAMPLES_PER_SUMMARY = 20


def _force_frame_decode(nc_data: object) -> None:
    """Force a lazily-opened image to decode now.

    ``_get_frame_from_disk_cache`` returns ``Image.open`` handles, which only
    read the PNG header; the pixels are inflated later, when the batched type
    calls ``np.asarray`` on them. That hides decode cost inside tensor
    construction. Calling ``load()`` first pulls it into its own timed section.
    Idempotent, and only the ordering changes, never the total work -- but it
    is still only invoked on timed samples so the common path is untouched.
    """
    frame = getattr(nc_data, "frame", None)
    load = getattr(frame, "load", None)
    if callable(load):
        load()


class _SampleCache:
    """On-disk cache of fully built training samples, shared across runs.

    Building a sample is dominated by inflating a full-resolution PNG and then
    discarding most of it during resize -- work that is identical every epoch
    and every run of the same configuration. Caching the finished sample skips
    all of it.

    This is only sound because augmentation runs on the device. Everything
    behind this cache is deterministic (projection, tensor construction, and
    worker-stage preprocessing, which is resize only), so a hit cannot freeze
    the random jitter and noise that a model actually needs to vary. Adding a
    non-deterministic worker-stage method would break that and require this
    cache to be disabled.

    Entries live under a hash of everything that changes what a sample
    contains, so a configuration change starts a fresh tree rather than
    silently serving stale tensors. It sits outside the per-run output
    directory so a repeat of the same run hits a warm cache immediately.
    """

    # Bump when the structure or dtype of a built sample changes, so existing
    # entries are ignored rather than loaded back into the wrong shape.
    FORMAT_VERSION = 1

    def __init__(self, root: Path, spec_hash: str) -> None:
        """Initialize the cache.

        Args:
            root: Directory holding every spec's entries.
            spec_hash: Digest of the configuration these entries belong to.
        """
        self.directory = root / spec_hash
        self.cache_manager = CacheManager(self.directory)
        self.hits = 0
        self.misses = 0

    def _path(self, recording_id: str, timestep: int) -> Path:
        # Sharded by recording to keep any one directory a reasonable size,
        # mirroring the layout of the decoded frame cache.
        return self.directory / recording_id / f"{timestep}.pt"

    def load(self, recording_id: str, timestep: int) -> TrainingSample | None:
        """Return the cached sample, or None if it is absent or unreadable."""
        path = self._path(recording_id, timestep)
        if not path.exists():
            self.misses += 1
            return None
        try:
            # weights_only=False is required to rebuild the batched pydantic
            # types. Safe here: these files are written by this process on the
            # local disk and are never fetched from anywhere.
            sample = torch.load(path, weights_only=False)
        except Exception:
            # A truncated or stale entry must cost a rebuild, never a crash.
            logger.warning("Discarding unreadable sample cache entry %s", path)
            path.unlink(missing_ok=True)
            self.misses += 1
            return None
        self.hits += 1
        return cast(TrainingSample, sample)

    def store(self, recording_id: str, timestep: int, sample: TrainingSample) -> None:
        """Write a built sample. A failure here costs a rebuild, nothing more."""
        path = self._path(recording_id, timestep)
        staging: Path | None = None
        try:
            self.cache_manager.ensure_space_available()
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write then rename: several workers populate this concurrently and
            # a reader must never see a half-written entry.
            with tempfile.NamedTemporaryFile(
                dir=path.parent, suffix=".tmp", delete=False
            ) as handle:
                staging = Path(handle.name)
                torch.save(sample, handle)
            os.replace(staging, path)
        except Exception:
            logger.warning("Could not write sample cache entry %s", path, exc_info=True)
            if staging is not None:
                staging.unlink(missing_ok=True)

    def summary(self) -> str:
        """One-line hit rate, reset for the next window."""
        total = self.hits + self.misses
        rate = self.hits / total * 100 if total else 0.0
        line = (
            f"  sample cache: {self.hits}/{total} hits ({rate:.0f}%) "
            f"at {self.directory}"
        )
        self.hits = 0
        self.misses = 0
        return line


class _SampleTimings:
    """Accumulates a breakdown of where ``load_sample`` spends its time.

    ``load_sample`` runs inside DataLoader worker processes, so there is no
    cheap way to ship measurements back to the trainer. Each worker instead
    logs its own summary, tagged with its worker id, and the lines interleave
    in the training log.

    Only every ``interval``-th call is timed, so the cost on the common path is
    a counter increment.
    """

    SECTIONS = (
        "recording_lookup",
        "sample_cache_load",
        "input_load",
        "output_load",
        "input_decode",
        "input_tensor",
        "input_pad",
        "input_preprocess",
        "output_build",
        "output_preprocess",
        "sample_cache_store",
    )

    def __init__(
        self, interval: int, label: str, cache: "_SampleCache | None" = None
    ) -> None:
        """Initialize the accumulator.

        Args:
            interval: Time every Nth sample. 0 disables timing entirely.
            cache: Sample cache whose hit rate is reported alongside the
                breakdown, if one is in use.
            label: Tag distinguishing e.g. the train split from the val split,
                which run different preprocessing pipelines.
        """
        self.interval = interval
        self.label = label
        self.cache = cache
        self._calls = 0
        self._timing = False
        self._section_start = 0.0
        self._current: dict[str, float] = {}
        self._totals: dict[str, float] = {}
        self._worst: dict[str, float] = {}
        self._timed_samples = 0
        self._sample_start = 0.0
        self._sample_total = 0.0
        self._sample_worst = 0.0

    @property
    def is_timing(self) -> bool:
        """Whether the call in progress is being timed."""
        return self._timing

    def begin_sample(self) -> None:
        """Decide whether to time this call, and start the clock if so."""
        self._timing = self.interval > 0 and self._calls % self.interval == 0
        self._calls += 1
        if self._timing:
            self._current.clear()
            self._sample_start = time.perf_counter()
            self._section_start = self._sample_start

    def mark(self, section: str) -> None:
        """Attribute time since the previous mark to ``section``.

        Sections inside the per-slot loops are marked several times per sample,
        so time lands in a per-sample bucket first and is only folded into the
        running totals at ``end_sample``. Otherwise the reported worst case
        would be the worst single slot rather than the worst whole sample, and
        could come out below the mean.
        """
        if not self._timing:
            return
        now = time.perf_counter()
        self._current[section] = (
            self._current.get(section, 0.0) + now - self._section_start
        )
        self._section_start = now

    def end_sample(self) -> None:
        """Close out a timed sample and log a summary once enough have run."""
        if not self._timing:
            return
        total = time.perf_counter() - self._sample_start
        for section, elapsed in self._current.items():
            self._totals[section] = self._totals.get(section, 0.0) + elapsed
            self._worst[section] = max(self._worst.get(section, 0.0), elapsed)
        self._sample_total += total
        self._sample_worst = max(self._sample_worst, total)
        self._timed_samples += 1
        self._timing = False
        if self._timed_samples >= TIMING_SAMPLES_PER_SUMMARY:
            self._log_summary()

    def _log_summary(self) -> None:
        from torch.utils.data import get_worker_info

        worker_info = get_worker_info()
        worker = "main" if worker_info is None else f"w{worker_info.id}"
        mean_total = self._sample_total / self._timed_samples

        lines = [
            f"[timing:dataset:{self.label}:{worker}] "
            f"mean {mean_total * 1000:.2f} ms/sample over {self._timed_samples} "
            f"timed samples (1 in {self.interval}), "
            f"slowest {self._sample_worst * 1000:.2f} ms"
        ]
        for section in self.SECTIONS:
            if section not in self._totals:
                continue
            mean_s = self._totals[section] / self._timed_samples
            share = mean_s / mean_total * 100 if mean_total else 0.0
            lines.append(
                f"    {section:<20}{mean_s * 1000:9.3f} ms   ({share:4.1f}%)"
                f"   worst {self._worst[section] * 1000:9.3f} ms"
            )
        if self.cache is not None:
            lines.append(self.cache.summary())
        logger.info("\n".join(lines))

        self._totals.clear()
        self._worst.clear()
        self._timed_samples = 0
        self._sample_total = 0.0
        self._sample_worst = 0.0


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
        timing_sample_interval: int = 0,
        sample_cache: bool = False,
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
            timing_sample_interval: Time every Nth ``load_sample`` call and log
                a breakdown periodically from each worker. 0 disables it.
            sample_cache: Reuse fully built samples from an on-disk cache
                across runs, rebuilding on a miss. See _SampleCache. Off by
                default so constructing a dataset never silently reads or
                writes the shared cache; training turns it on via config.
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

        self.episode_indices, self.episode_start_offsets = self._get_episode_indices()
        self._logged_in = False

        # Only the worker stage runs here. Device-stage methods are applied by
        # the trainer once the batch is on the accelerator, where they run
        # batched rather than once per frame on a contended worker CPU.
        self.input_preprocessing_config = input_preprocessing_config.for_stage(
            PreprocessingStage.WORKER
        )
        self.output_preprocessing_config = output_preprocessing_config.for_stage(
            PreprocessingStage.WORKER
        )

        # Built lazily, and per process: workers are forked after this point,
        # so each one accumulates and reports its own timings.
        self._timing_sample_interval = timing_sample_interval
        self._timing_label = "train"
        self._sample_timings: _SampleTimings | None = None

        # Everything below is a pure function of the cross-embodiment
        # descriptions, which never change after construction. Computing it
        # here keeps it out of load_sample, which runs once per sample.
        self._max_items_per_input_type = self._get_max_items_per_data_type(
            self.input_cross_embodiment_description
        )
        self._max_items_per_output_type = self._get_max_items_per_data_type(
            self.output_cross_embodiment_description
        )
        self._robot_embodiment_descriptions = {
            robot_id: self._convert_to_embodiment_description(embodiment_union)
            for robot_id, embodiment_union in (
                self.merged_cross_embodiment_description.items()
            )
        }
        self._sample_cache_enabled = sample_cache
        self._sample_cache: _SampleCache | None = None
        # Sample cache keys are recording ids, not episode indices, which are
        # positions in a server-ordered list and can point at a different
        # recording after the dataset changes. Resolved up front so a cache
        # read costs a list index rather than a recording lookup.
        self._episode_recording_ids: list[str] = (
            [recording.id for recording in self.synchronized_dataset]
            if sample_cache
            else []
        )

        # Pre-sorted index order per (robot, data type), so projecting a sync
        # point does not re-sort the same keys for every one of the
        # output_prediction_horizon + 1 sync points a sample touches.
        self._merged_ordered_items = {
            robot_id: self._order_embodiment_items(description)
            for robot_id, description in self._robot_embodiment_descriptions.items()
        }
        # The output window is projected onto the output description alone, not
        # the merged one. Only output data types are ever read back out of it,
        # and narrowing it here is what lets the recording skip decoding camera
        # frames for the whole prediction horizon.
        self._output_ordered_items = {
            robot_id: self._order_embodiment_items(description)
            for robot_id, description in (
                self.output_cross_embodiment_description.items()
            )
        }

    def _sample_spec_hash(self) -> str:
        """Digest everything that changes what a built sample contains.

        A cache entry is only reusable if all of this matches: the synchronized
        dataset it came from (which the server keys on the synchronization
        parameters), which sensors get projected in and how they are padded,
        how many future steps the outputs span, and the worker-stage
        preprocessing that shapes the tensors.

        Erring toward including a field costs a cache miss. Erring toward
        omitting one serves tensors that silently do not match the config, so
        anything doubtful belongs in here.
        """

        def preprocessing_spec(
            config: PreprocessingConfiguration,
        ) -> dict[str, list[dict[str, object]]]:
            return {
                data_type.value: [method.to_dict() for method in methods]
                for data_type, methods in sorted(
                    config.items(), key=lambda item: item[0].value
                )
            }

        spec = {
            "format_version": _SampleCache.FORMAT_VERSION,
            "synchronized_dataset_id": self.synchronized_dataset.id,
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
            "output_prediction_horizon": self.output_prediction_horizon,
            # Worker-stage only: the device stage runs after the cache and so
            # does not change what is stored.
            "input_preprocessing": preprocessing_spec(self.input_preprocessing_config),
            "output_preprocessing": preprocessing_spec(
                self.output_preprocessing_config
            ),
        }
        serialized = json.dumps(
            spec, sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

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

    @staticmethod
    def _get_max_items_per_data_type(
        cross_embodiment_description: CrossEmbodimentDescription,
    ) -> dict[DataType, int]:
        """Return the padded slot count for each data type.

        The count is the highest index used for that data type across every
        robot, plus one, so samples from different embodiments pad to a common
        width.
        """
        highest_index: dict[DataType, int] = {}
        for data_types in cross_embodiment_description.values():
            for data_type, indexed_names in data_types.items():
                # Floor at 0 so a data type declared with no sensors still gets
                # a single padded slot, matching the per-sample scan this
                # replaces.
                highest_index[data_type] = max(
                    highest_index.get(data_type, 0), *indexed_names, 0
                )
        return {data_type: highest + 1 for data_type, highest in highest_index.items()}

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

    def _get_episode_indices(self) -> tuple[list[int], list[int]]:
        """Map each sample index to its episode, and each episode to its first sample.

        Omit the last frame of each episode because it is not used for training.

        Returns:
            ``(episode_indices, episode_start_offsets)`` where
            ``episode_indices[sample_idx]`` is the recording index and
            ``episode_start_offsets[recording_idx]`` is the sample index that
            recording starts at. The offsets let ``__getitem__`` recover a
            timestep in constant time instead of scanning ``episode_indices``.
        """
        episode_indices: list[int] = []
        episode_start_offsets: list[int] = []
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
            episode_indices.extend([recording_idx] * (len(recording) - 1))

        return episode_indices, episode_start_offsets

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
            ordered_items: Pre-sorted ``{data_type: [(index, name), ...]}``
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
        projected_input_sync_point: SynchronizedPoint,
    ) -> list[SynchronizedPoint]:
        """Load the superset window for all output data types.

        Fetches ``[timestep, timestep + 1 + horizon]`` once so target types
        (aligned to the input step) and non-target types (next step onward)
        can share the same loaded sync points.

        Only the data types in ``ordered_items`` are materialised. That matters
        because loading a sync point decodes camera frames off disk, and the
        output window is usually joints-only — so without the filter every
        sample would pay for ``horizon + 1`` frames per camera and then discard
        them during projection.

        The sync point at ``timestep`` is reused from
        ``projected_input_sync_point`` rather than being loaded a second time.
        """
        output_sync_points = cast(
            list[SynchronizedPoint],
            synced_recording.get_range(
                timestep + 1,
                timestep + 1 + self.output_prediction_horizon,
                data_types=frozenset(ordered_items),
            ),
        )
        return [projected_input_sync_point] + [
            self._project_sync_point(sync_point, ordered_items)
            for sync_point in output_sync_points
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

    def _init_worker_state(self) -> "_SampleTimings":
        """Build the per-process timing and cache state on first use.

        Deferred rather than built in ``__init__`` because DataLoader workers
        are forked afterwards and each needs its own, and because the shallow
        copy taken for the validation split swaps in different preprocessing
        and so must key its cache differently.

        Returns:
            The timing accumulator for this process.
        """
        if self._sample_cache_enabled:
            self._sample_cache = _SampleCache(
                DEFAULT_CACHE_DIR / "sample_cache", self._sample_spec_hash()
            )
        self._sample_timings = _SampleTimings(
            self._timing_sample_interval, self._timing_label, self._sample_cache
        )
        return self._sample_timings

    def load_sample(
        self, episode_idx: int, timestep: int | None = None
    ) -> TrainingSample:
        """Load sample from cache or GCS with full data type support."""
        if not self._logged_in:
            nc.login()
            self._logged_in = True

        timings = self._sample_timings or self._init_worker_state()
        timings.begin_sample()

        if self._mem_check_counter % CHECK_MEMORY_INTERVAL == 0:
            self._memory_monitor.check_memory()
            self._mem_check_counter = 0
        self._mem_check_counter += 1

        # Served before the recording is touched at all. A hit needs only the
        # episode's recording id, which is resolved at construction, so the
        # whole load collapses to one file read. Skipped when the caller left
        # the timestep unset, since choosing one needs the episode length.
        if self._sample_cache is not None and timestep is not None:
            cached = self._sample_cache.load(
                self._episode_recording_ids[episode_idx], timestep
            )
            if cached is not None:
                timings.mark("sample_cache_load")
                timings.end_sample()
                return cached

        synced_recording = self.synchronized_dataset[episode_idx]
        synced_recording = cast(SynchronizedRecording, synced_recording)
        episode_length = len(synced_recording)
        if timestep is None:
            timestep = self._get_timestep(episode_length)

        # Order the SynchronizedPoints to the merged embodiment description.
        robot_id = synced_recording.robot_id
        timings.mark("recording_lookup")

        input_sync_point = self._project_sync_point(
            cast(SynchronizedPoint, synced_recording[timestep]),
            self._merged_ordered_items[robot_id],
        )
        timings.mark("input_load")

        output_sync_points = self._load_projected_output_sync_points(
            synced_recording=synced_recording,
            timestep=timestep,
            ordered_items=self._output_ordered_items[robot_id],
            projected_input_sync_point=input_sync_point,
        )
        recording_name = getattr(synced_recording, "name", "recording")
        timings.mark("output_load")

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
                        batch_size=1, time_steps=1
                    )
                    timings.mark("input_pad")
                else:
                    # If the current robot has a name for this index, use it to
                    # get the data.
                    nc_data = input_sync_point.data[data_type][name]
                    if timings.is_timing:
                        # Separate image decode from tensor construction; see
                        # _force_frame_decode for why they otherwise merge.
                        _force_frame_decode(nc_data)
                        timings.mark("input_decode")
                    batched_nc_data = batched_nc_data_class.from_nc_data(nc_data)
                    input_mask_values[index] = 1.0
                    timings.mark("input_tensor")

                batched_nc_data = apply_preprocessing_methods(
                    batched_data=batched_nc_data,
                    methods=self.input_preprocessing_config.get(data_type, []),
                )
                inputs[data_type].append(batched_nc_data)
                timings.mark("input_preprocess")

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
                timings.mark("output_build")

                batched_nc_data = apply_preprocessing_methods(
                    batched_data=batched_nc_data,
                    methods=self.output_preprocessing_config.get(data_type, []),
                )
                outputs[data_type].append(batched_nc_data)
                timings.mark("output_preprocess")

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
            timings.mark("sample_cache_store")
        timings.end_sample()
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
