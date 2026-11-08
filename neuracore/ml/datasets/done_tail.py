"""Synthesize an episode done flag for datasets that do not log one.

A policy with a done head needs ``DataType.DONE`` labels, which older datasets
never logged. ``DoneTailDataset`` wraps a ``SynchronizedDataset`` after
synchronization and makes every episode look as if it ended with a short hold:
it appends ``K = round(tail_s * frequency)`` virtual frames that repeat the
last real frame, with zero joint velocities, and labels them ``done=1``. Real
frames are labelled ``done=0``.

The wrapper presents the same interface as the dataset it wraps, so the
training dataset reads DONE from it exactly as it would read a logged DONE.
Synchronization and the server never see DONE: it is removed from every
description passed to the wrapped dataset and added back locally.
"""

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Union, cast

import numpy as np
from neuracore_types import (
    CrossEmbodimentDescription,
    DataItemStats,
    DataType,
    DoneData,
    DoneDataStats,
    EmbodimentDescription,
    JointData,
    NCDataStats,
    SynchronizedDatasetStatistics,
    SynchronizedPoint,
)

from neuracore.core.data.synced_dataset import SynchronizedDataset
from neuracore.core.data.synced_recording import SynchronizedRecording

if TYPE_CHECKING:
    from neuracore.core.data.dataset import Dataset

logger = logging.getLogger(__name__)

DONE_SENSOR_NAME = "done"
"""Sensor name the synthesized done flag is stored under for every robot."""

_DONE_EMBODIMENT: dict[int, str] = {0: DONE_SENSOR_NAME}


def done_tail_length(tail_s: float, frequency: float) -> int:
    """Return the number of virtual done frames appended to each episode.

    Args:
        tail_s: Length of the done tail in seconds. Must be positive.
        frequency: Synchronization frequency in Hz.

    Returns:
        The tail length in frames, at least one.

    Raises:
        ValueError: If ``tail_s`` or ``frequency`` is not positive.
    """
    if tail_s <= 0:
        raise ValueError(f"Done tail length must be positive, got {tail_s}s.")
    if frequency <= 0:
        raise ValueError(f"Frequency must be positive, got {frequency}Hz.")
    return max(1, round(tail_s * frequency))


def should_synthesize_done(tail_s: float | None, dataset: "Dataset") -> bool:
    """Decide whether a training run should synthesize the done flag.

    Synthesis is requested by a positive ``tail_s``. It is disabled when the
    dataset already logs DONE, so logged labels are never replaced.

    Args:
        tail_s: Requested done tail length in seconds, or None when not set.
        dataset: The dataset being trained on.

    Returns:
        True when the done flag should be synthesized.
    """
    if not tail_s:
        return False
    if DataType.DONE in dataset.data_types:
        logger.warning(
            "Dataset '%s' already logs DONE; ignoring synthesize_done_tail_s "
            "and training on the logged done flags.",
            dataset.name,
        )
        return False
    return True


def add_done_to_cross_embodiment_description(
    cross_embodiment_description: CrossEmbodimentDescription,
) -> CrossEmbodimentDescription:
    """Return a copy of the description with the done flag added for every robot.

    Args:
        cross_embodiment_description: Description to add DONE to.

    Returns:
        A new description where each robot has DONE at index 0.
    """
    return {
        robot_id: {**embodiment, DataType.DONE: dict(_DONE_EMBODIMENT)}
        for robot_id, embodiment in cross_embodiment_description.items()
    }


def _without_done(embodiment: EmbodimentDescription) -> EmbodimentDescription:
    return {
        data_type: names
        for data_type, names in embodiment.items()
        if data_type != DataType.DONE
    }


def remove_done_from_cross_embodiment_description(
    cross_embodiment_description: CrossEmbodimentDescription,
) -> CrossEmbodimentDescription:
    """Return a copy of the description with DONE removed for every robot.

    Args:
        cross_embodiment_description: Description to remove DONE from.

    Returns:
        A new description without DONE.
    """
    return {
        robot_id: _without_done(embodiment)
        for robot_id, embodiment in cross_embodiment_description.items()
    }


class _DoneTailDatasetView:
    """Raw dataset view that reports the synthesized DONE as present.

    The training dataset validates its descriptions against the raw dataset's
    data types and embodiments. This view adds DONE to both and forwards
    everything else to the dataset it wraps.
    """

    def __init__(self, dataset: "Dataset"):
        self._dataset = dataset

    @property
    def data_types(self) -> list[DataType]:
        data_types = list(self._dataset.data_types)
        if DataType.DONE not in data_types:
            data_types.append(DataType.DONE)
        return data_types

    def get_full_embodiment_description(self, robot_id: str) -> EmbodimentDescription:
        embodiment = dict(self._dataset.get_full_embodiment_description(robot_id))
        embodiment[DataType.DONE] = dict(_DONE_EMBODIMENT)
        return embodiment

    def __iter__(self) -> Iterator[Any]:
        return iter(self._dataset)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: Any) -> Any:
        return self._dataset[idx]

    def __getattr__(self, name: str) -> Any:
        # Guard against recursion while unpickling, before _dataset is set.
        if name == "_dataset":
            raise AttributeError(name)
        return getattr(self._dataset, name)


class DoneTailRecording(SynchronizedRecording):
    """Synchronized recording extended by a tail of virtual done frames.

    Timesteps ``0 .. L-1`` are the real frames with ``done=0``. Timesteps
    ``L .. L+K-1`` repeat frame ``L-1`` with zero joint velocities and
    ``done=1``. The base class constructor is not called: every method the
    training dataset uses is overridden here, and other attributes are read
    from the wrapped recording.
    """

    def __init__(self, recording: SynchronizedRecording, tail_length: int):
        """Wrap a recording.

        Args:
            recording: The synchronized recording to extend.
            tail_length: Number of virtual done frames to append.
        """
        self._recording = recording
        self._tail_length = tail_length
        self._iter_idx = 0

    @property
    def num_real_frames(self) -> int:
        """Number of frames in the wrapped recording."""
        return len(self._recording)

    @property
    def tail_length(self) -> int:
        """Number of virtual done frames appended to the recording."""
        return self._tail_length

    @property
    def frequency(self) -> int:
        """Synchronization frequency of the wrapped recording in Hz."""
        return self._recording.frequency

    @property
    def cross_embodiment_union(self) -> Any:
        """Cross-embodiment union of the wrapped recording."""
        return self._recording.cross_embodiment_union

    def _with_done(
        self, sync_point: SynchronizedPoint, timestep: int
    ) -> SynchronizedPoint:
        is_tail = timestep >= self.num_real_frames
        data = dict(sync_point.data)
        if is_tail and DataType.JOINT_VELOCITIES in data:
            # The tail is a hold, so the robot is not moving.
            data[DataType.JOINT_VELOCITIES] = {
                name: cast(JointData, joint).model_copy(update={"value": 0.0})
                for name, joint in data[DataType.JOINT_VELOCITIES].items()
            }
        data[DataType.DONE] = {
            DONE_SENSOR_NAME: DoneData(done=is_tail, timestamp=sync_point.timestamp)
        }
        return sync_point.model_copy(update={"data": data})

    def get_sync_point(
        self, timestep: int, embodiment_description: EmbodimentDescription
    ) -> SynchronizedPoint:
        """Return the sync point at timestep with the done flag added.

        Args:
            timestep: Timestep counting from 0, up to ``L + K - 1``.
            embodiment_description: Data types and sensor names to load. DONE
                is synthesized here and is not requested from the recording.

        Returns:
            The sync point with DONE set for this timestep.

        Raises:
            IndexError: If the timestep is outside the extended recording.
        """
        if not 0 <= timestep < len(self):
            raise IndexError(
                f"Timestep {timestep} out of range for recording of length "
                f"{len(self)}"
            )
        real_timestep = min(timestep, self.num_real_frames - 1)
        sync_point = self._recording.get_sync_point(
            real_timestep, _without_done(embodiment_description)
        )
        return self._with_done(sync_point, timestep)

    def get_sync_points(
        self,
        start_timestep: int,
        end_timestep: int,
        embodiment_description: EmbodimentDescription,
    ) -> list[SynchronizedPoint]:
        """Return the sync points from start_timestep up to end_timestep.

        The range is clamped to the extended recording the same way a slice is.

        Args:
            start_timestep: Timestep of the first sync point.
            end_timestep: Timestep one past the last sync point.
            embodiment_description: Data types and sensor names to load.

        Returns:
            The sync points in the range, each with DONE set.
        """
        start_timestep, end_timestep, _ = slice(start_timestep, end_timestep).indices(
            len(self)
        )
        return [
            self.get_sync_point(timestep, embodiment_description)
            for timestep in range(start_timestep, end_timestep)
        ]

    def __len__(self) -> int:
        """Return the number of real frames plus the done tail."""
        return self.num_real_frames + self._tail_length

    def __getitem__(
        self, idx: int | slice
    ) -> SynchronizedPoint | list[SynchronizedPoint]:
        """Return a sync point, or a list of them for a slice, with DONE added.

        Args:
            idx: Integer index or slice over the extended recording.

        Returns:
            The sync point or sync points with DONE set.

        Raises:
            IndexError: If the index is out of range.
        """
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [cast(SynchronizedPoint, self[i]) for i in range(start, stop, step)]
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError("Index out of range")
        real_idx = min(idx, self.num_real_frames - 1)
        sync_point = cast(SynchronizedPoint, self._recording[real_idx])
        return self._with_done(sync_point, idx)

    def __iter__(self) -> "DoneTailRecording":
        """Start iterating over the extended recording."""
        self._iter_idx = 0
        return self

    def __next__(self) -> SynchronizedPoint:
        """Return the next sync point of the extended recording."""
        if self._iter_idx >= len(self):
            raise StopIteration
        sync_point = cast(SynchronizedPoint, self[self._iter_idx])
        self._iter_idx += 1
        return sync_point

    def __getattr__(self, name: str) -> Any:
        """Read any other attribute, such as id or robot_id, from the recording."""
        # Guard against recursion while unpickling, before _recording is set.
        if name == "_recording":
            raise AttributeError(name)
        return getattr(self._recording, name)


class DoneTailDataset(SynchronizedDataset):
    """Synchronized dataset whose episodes end in a synthesized done tail.

    The base class constructor is not called, so nothing is synchronized or
    prefetched again: every method the training dataset uses is overridden
    here, and other attributes are read from the wrapped dataset.
    """

    def __init__(self, synchronized_dataset: SynchronizedDataset, tail_s: float):
        """Wrap a synchronized dataset.

        Args:
            synchronized_dataset: The synchronized dataset to extend.
            tail_s: Length of the done tail in seconds.
        """
        self._synchronized_dataset = synchronized_dataset
        self.tail_s = tail_s
        self.tail_length = done_tail_length(tail_s, synchronized_dataset.frequency)
        # The id keys the statistics and sample caches, so a different tail must
        # not reuse what was cached without it.
        self.id = f"{synchronized_dataset.id}+done_tail_{tail_s}s"
        self.dataset = cast(
            "Dataset", _DoneTailDatasetView(synchronized_dataset.dataset)
        )
        self.frequency = synchronized_dataset.frequency
        self._recordings: dict[int, DoneTailRecording] = {}
        self._recording_idx = 0

    def _wrapped_recording(self, idx: int) -> DoneTailRecording:
        if idx not in self._recordings:
            recording = cast(SynchronizedRecording, self._synchronized_dataset[idx])
            self._recordings[idx] = DoneTailRecording(recording, self.tail_length)
        return self._recordings[idx]

    def __len__(self) -> int:
        """Return the number of episodes."""
        return len(self._synchronized_dataset)

    def __getitem__(
        self, idx: int | slice
    ) -> Union["SynchronizedRecording", "SynchronizedDataset"]:
        """Return an extended recording, or an extended dataset for a slice.

        Args:
            idx: Integer index or slice over the episodes.

        Returns:
            A DoneTailRecording, or a DoneTailDataset for a slice.

        Raises:
            IndexError: If the index is out of range.
            TypeError: If the index is not an integer or slice.
        """
        if isinstance(idx, slice):
            sliced = cast(SynchronizedDataset, self._synchronized_dataset[idx])
            return DoneTailDataset(sliced, self.tail_s)
        if not isinstance(idx, int):
            raise TypeError(
                f"Dataset indices must be integers or slices, not {type(idx)}"
            )
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError("Dataset index out of range")
        return self._wrapped_recording(idx)

    def __iter__(self) -> "DoneTailDataset":
        """Start iterating over the extended episodes."""
        self._recording_idx = 0
        return self

    def __next__(self) -> SynchronizedRecording:
        """Return the next extended episode."""
        if self._recording_idx >= len(self):
            raise StopIteration
        recording = self._wrapped_recording(self._recording_idx)
        self._recording_idx += 1
        return recording

    def calculate_statistics(
        self,
        input_cross_embodiment_description: CrossEmbodimentDescription,
        output_cross_embodiment_description: CrossEmbodimentDescription,
    ) -> SynchronizedDatasetStatistics:
        """Calculate statistics with the done tail accounted for.

        Statistics for everything except DONE come from the wrapped dataset,
        which never sees DONE. Their counts grow by the tail frames. DONE
        statistics are computed here from the episode lengths.

        Args:
            input_cross_embodiment_description: Input description.
            output_cross_embodiment_description: Output description, which may
                include DONE.

        Returns:
            Statistics for the extended dataset.
        """
        inner = self._synchronized_dataset.calculate_statistics(
            input_cross_embodiment_description=(
                remove_done_from_cross_embodiment_description(
                    input_cross_embodiment_description
                )
            ),
            output_cross_embodiment_description=(
                remove_done_from_cross_embodiment_description(
                    output_cross_embodiment_description
                )
            ),
        )

        recordings = [self._wrapped_recording(i) for i in range(len(self))]
        descriptions = {
            "input": input_cross_embodiment_description,
            "output": output_cross_embodiment_description,
        }

        dataset_statistics: dict[str, dict[DataType, list[NCDataStats]]] = {}
        for role, role_stats in inner.dataset_statistics.items():
            description = descriptions.get(role, {})
            dataset_statistics[role] = {
                data_type: [
                    _with_extra_count(
                        stats,
                        self.tail_length
                        * _num_recordings_with_slot(
                            recordings, description, data_type, index
                        ),
                    )
                    for index, stats in enumerate(stats_list)
                ]
                for data_type, stats_list in role_stats.items()
            }
            if any(DataType.DONE in embodiment for embodiment in description.values()):
                dataset_statistics[role][DataType.DONE] = [
                    _done_statistics(recordings, description)
                ]

        return SynchronizedDatasetStatistics(
            synchronized_dataset_id=self.id,
            input_cross_embodiment_description=input_cross_embodiment_description,
            output_cross_embodiment_description=output_cross_embodiment_description,
            dataset_statistics=cast(Any, dataset_statistics),
        )

    def __getattr__(self, name: str) -> Any:
        """Read any other attribute from the wrapped synchronized dataset."""
        # Guard against recursion while unpickling, before it is set.
        if name == "_synchronized_dataset":
            raise AttributeError(name)
        return getattr(self._synchronized_dataset, name)


def _num_recordings_with_slot(
    recordings: list[DoneTailRecording],
    description: CrossEmbodimentDescription,
    data_type: DataType,
    index: int,
) -> int:
    """Count recordings whose robot has data at this canonical index."""
    return sum(
        1
        for recording in recordings
        if index in description.get(recording.robot_id, {}).get(data_type, {})
    )


def _with_extra_count(stats: NCDataStats, extra: int) -> NCDataStats:
    """Return a copy of the statistics with every count raised by extra."""
    if extra == 0:
        return stats
    stats = stats.model_copy(deep=True)
    for name, value in vars(stats).items():
        if isinstance(value, DataItemStats):
            setattr(
                stats,
                name,
                value.model_copy(update={"count": value.count + extra}),
            )
    return stats


def _done_statistics(
    recordings: list[DoneTailRecording],
    description: CrossEmbodimentDescription,
) -> DoneDataStats:
    """Compute DONE statistics for the recordings whose robot outputs DONE."""
    num_frames = 0
    num_done = 0
    for recording in recordings:
        if DataType.DONE in description.get(recording.robot_id, {}):
            num_frames += len(recording)
            num_done += recording.tail_length
    rate = num_done / num_frames if num_frames else 0.0
    return DoneDataStats(
        done=DataItemStats(
            mean=np.array([rate], dtype=np.float32),
            std=np.array([np.sqrt(rate * (1.0 - rate))], dtype=np.float32),
            count=np.array([num_frames], dtype=np.int32),
            min=np.array([0.0 if num_done < num_frames else 1.0], dtype=np.float32),
            max=np.array([1.0 if num_done else 0.0], dtype=np.float32),
            q01=np.array([0.0 if rate < 0.99 else 1.0], dtype=np.float32),
            q99=np.array([1.0 if rate > 0.01 else 0.0], dtype=np.float32),
        )
    )
