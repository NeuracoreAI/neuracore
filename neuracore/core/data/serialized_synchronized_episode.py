"""Synchronized episodes with their points stored as JSON."""

from collections.abc import Iterable, Iterator

import numpy as np
from neuracore_types import SynchronizedEpisode, SynchronizedPoint


class SerializedSynchronizedPoints:
    """Synchronized points stored as JSON in one bytes object."""

    def __init__(self, sync_points: Iterable[SynchronizedPoint]) -> None:
        """Serialize every sync point into one bytes object.

        Args:
            sync_points: The sync points to store, in timestep order.
        """
        serialized = [
            sync_point.model_dump_json().encode() for sync_point in sync_points
        ]
        self._data = b"".join(serialized)
        self._offsets = np.cumsum([0, *map(len, serialized)])

    def __len__(self) -> int:
        """Return the number of stored sync points."""
        return len(self._offsets) - 1

    def __iter__(self) -> Iterator[SynchronizedPoint]:
        """Deserialize each stored sync point in order."""
        return (self[index] for index in range(len(self)))

    def __getitem__(self, index: int) -> SynchronizedPoint:
        """Deserialize the sync point at index.

        Args:
            index: Position of the sync point. Negative values count from the end.

        Returns:
            A new SynchronizedPoint.

        Raises:
            IndexError: If index is out of range.
        """
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"Sync point index {index} out of range")
        start, stop = self._offsets[index], self._offsets[index + 1]
        return SynchronizedPoint.model_validate_json(self._data[start:stop])


class SerializedSynchronizedEpisode:
    """Synchronized episode with its points stored as JSON."""

    def __init__(self, episode: SynchronizedEpisode) -> None:
        """Copy the episode metadata and serialize its points.

        Args:
            episode: The synchronized episode to store.
        """
        self.robot_id = episode.robot_id
        self.start_time = episode.start_time
        self.end_time = episode.end_time
        self.observations = SerializedSynchronizedPoints(episode.observations)
