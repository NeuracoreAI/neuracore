"""PyTorch dataset for loading synchronized robot data with filesystem caching."""

import hashlib
import json
import logging
import os
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


class _TensorCacheProbe:
    """Measures what a per-sample tensor cache would cost, without building one.

    A cache like this existed until commit 2813e45 and was removed on the
    grounds that decoded frames were already cached on disk. The per-section
    timings show that only covers download and video decode: PNG decode plus
    tensor construction still dominates every sample, every epoch.

    Caching is only sound now because augmentation moved to the device. The
    worker stage that would sit behind a cache is deterministic (resize only),
    so a cache hit cannot freeze the random jitter and noise the way it would
    have if reinstated before that change.

    This writes and re-reads a real sample so the numbers reflect actual
    serialisation cost and on-disk size, then deletes the file. It measures;
    it does not cache.
    """

    def __init__(self, directory: Path) -> None:
        """Initialize the probe.

        Args:
            directory: Where to write probe files. Should sit on the volume a
                real cache would use, so the IO cost is representative.
        """
        self.directory = directory
        self.save_seconds = 0.0
        self.load_seconds = 0.0
        self.total_bytes = 0
        self.samples = 0
        self.failed = False

    def measure(self, sample: TrainingSample) -> None:
        """Round-trip ``sample`` through disk and record the cost."""
        if self.failed:
            return
        path = self.directory / f"probe_{os.getpid()}.pt"
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            torch.save(sample, path)
            saved = time.perf_counter()
            torch.load(path, weights_only=False)
            loaded = time.perf_counter()

            self.save_seconds += saved - start
            self.load_seconds += loaded - saved
            self.total_bytes += path.stat().st_size
            self.samples += 1
        except Exception:
            # A probe must never take training down with it.
            logger.warning("Tensor cache probe failed; disabling.", exc_info=True)
            self.failed = True
        finally:
            path.unlink(missing_ok=True)

    def summary_line(self, mean_build_seconds: float) -> str | None:
        """Describe the trade against the measured cost of building a sample."""
        if not self.samples:
            return None
        save_ms = self.save_seconds / self.samples * 1000
        load_ms = self.load_seconds / self.samples * 1000
        kib = self.total_bytes / self.samples / 1024
        saved_ms = mean_build_seconds * 1000 - load_ms
        return (
            f"  tensor-cache probe over {self.samples} samples: "
            f"save {save_ms:.2f} ms | load {load_ms:.2f} ms | {kib:.1f} KiB/sample\n"
            f"    a warm cache would trade {mean_build_seconds * 1000:.2f} ms of "
            f"build for {load_ms:.2f} ms of load ({saved_ms:.2f} ms/sample saved)"
        )

    def reset(self) -> None:
        """Clear the accumulated measurements."""
        self.save_seconds = 0.0
        self.load_seconds = 0.0
        self.total_bytes = 0
        self.samples = 0


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
        "input_load",
        "output_load",
        "input_decode",
        "input_tensor",
        "input_pad",
        "input_preprocess",
        "output_build",
        "output_preprocess",
    )

    def __init__(
        self, interval: int, label: str, cache_probe: "_TensorCacheProbe | None" = None
    ) -> None:
        """Initialize the accumulator.

        Args:
            interval: Time every Nth sample. 0 disables timing entirely.
            label: Tag distinguishing e.g. the train split from the val split,
                which run different preprocessing pipelines.
            cache_probe: Optional probe measuring what a per-sample tensor
                cache would cost, reported alongside the breakdown.
        """
        self.interval = interval
        self.label = label
        self.cache_probe = cache_probe
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

    def end_sample(self, sample: TrainingSample | None = None) -> None:
        """Close out a timed sample and log a summary once enough have run.

        Args:
            sample: The finished sample, handed to the tensor-cache probe if
                one is configured. Probe time is excluded from the sample's
                own total, since it is measurement, not work the sample needs.
        """
        if not self._timing:
            return
        total = time.perf_counter() - self._sample_start
        if self.cache_probe is not None and sample is not None:
            self.cache_probe.measure(sample)
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
        if self.cache_probe is not None:
            probe_line = self.cache_probe.summary_line(mean_total)
            if probe_line is not None:
                lines.append(probe_line)
            self.cache_probe.reset()
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
        tensor_cache_probe: bool = False,
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
            tensor_cache_probe: Round-trip timed samples through disk to measure
                what a per-sample tensor cache would cost. Measurement only --
                nothing is cached. Requires timing to be enabled.
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
        self._tensor_cache_probe = tensor_cache_probe
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

    def load_sample(
        self, episode_idx: int, timestep: int | None = None
    ) -> TrainingSample:
        """Load sample from cache or GCS with full data type support."""
        if not self._logged_in:
            nc.login()
            self._logged_in = True

        if self._sample_timings is None:
            probe = (
                _TensorCacheProbe(DEFAULT_CACHE_DIR / "tensor_cache_probe")
                if self._tensor_cache_probe
                else None
            )
            self._sample_timings = _SampleTimings(
                self._timing_sample_interval, self._timing_label, probe
            )
        timings = self._sample_timings
        timings.begin_sample()

        if self._mem_check_counter % CHECK_MEMORY_INTERVAL == 0:
            self._memory_monitor.check_memory()
            self._mem_check_counter = 0
        self._mem_check_counter += 1

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
        timings.end_sample(sample)
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
