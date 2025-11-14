"""SynchronizedDataset class for managing synchronized datasets."""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional, Union, cast

from neuracore_types import DatasetDescription, DataType
from tqdm import tqdm

from neuracore.core.data.synced_recording import SynchronizedRecording

if TYPE_CHECKING:
    from neuracore.core.data.dataset import Dataset


logger = logging.getLogger(__name__)


class SynchronizedDataset:
    """Class for managing synchronized datasets."""

    def __init__(
        self,
        dataset: "Dataset",
        frequency: int,
        data_types: Optional[list[DataType]],
        dataset_description: DatasetDescription,
        prefetch_videos: bool = False,
        max_workers: int = 4,
    ):
        """Initialize a dataset from server response data.

        Args:
            dataset: Dataset object containing recordings.
            frequency: Frequency of the dataset in Hz.
            data_types: List of data types to include in the dataset.
            dataset_description: Description of the dataset.
            prefetch_videos: Whether to prefetch video data to cache on initialization.
            max_workers: Number of threads to use for prefetching videos.
        """
        self.dataset = dataset
        self.frequency = frequency
        self.data_types = data_types or []
        self.dataset_description = dataset_description
        self._prefetch_videos = prefetch_videos
        self._recording_idx = 0
        self._synced_recording_cache: dict[int, SynchronizedRecording] = {}

        if prefetch_videos:
            prefetch_needed = False
            for rec in self.dataset.recordings:
                cache_dir = (
                    self.dataset.cache_dir / f"{rec['id']}" / f"{self.frequency}Hz"
                )
                if not cache_dir.exists():
                    prefetch_needed = True
                    break
            if prefetch_needed:
                self._perform_videos_prefetch(max_workers=max_workers)

    def _perform_videos_prefetch(self, max_workers: int) -> None:
        """Prefetch video data for all recordings using multiple threads.

        Args:
            max_workers: Number of threads to use for prefetching videos.
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(
                tqdm(
                    executor.map(
                        lambda idx: self[idx], range(len(self.dataset.recordings))
                    ),
                    total=len(self.dataset.recordings),
                    desc="Prefetching videos",
                    unit="Recording",
                )
            )

    @property
    def num_transitions(self) -> int:
        """Get the number of transitions in the dataset."""
        return self.dataset_description.total_num_transitions

    def __iter__(self) -> "SynchronizedDataset":
        """Initialize iterator over episodes in the dataset.

        Returns:
            Self for iteration over episodes.
        """
        self._recording_idx = 0
        return self

    def __len__(self) -> int:
        """Get the number of episodes in the dataset.

        Returns:
            Number of demonstration episodes in the dataset.
        """
        return len(self.dataset)

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union["SynchronizedRecording", "SynchronizedDataset"]:
        """Support for indexing and slicing dataset episodes.

        Args:
            idx: Integer index or slice object for accessing episodes.

        Returns:
            SynchronizedRecording for a single episode or
            SynchronizedDataset for a slice of episodes.

        Raises:
            IndexError: If the index is out of range.
            TypeError: If the index is not an integer or slice.
        """
        if isinstance(idx, slice):
            # Handle slice
            dataset = self.dataset[idx.start : idx.stop : idx.step]
            return SynchronizedDataset(
                dataset=cast("Dataset", dataset),
                frequency=self.frequency,
                data_types=self.data_types,
                dataset_description=self.dataset_description,
                prefetch_videos=False,  # Avoid prefetching again
            )
        else:
            # Handle single index
            if isinstance(idx, int):
                if idx < 0:  # Handle negative indices
                    idx += len(self.dataset.recordings)
                if not 0 <= idx < len(self.dataset.recordings):
                    raise IndexError("Dataset index out of range")
                if idx not in self._synced_recording_cache:
                    synced_recording = SynchronizedRecording(
                        recording_id=self.dataset.recordings[idx]["id"],
                        dataset=self.dataset,
                        robot_id=self.dataset.recordings[idx]["robot_id"],
                        instance=self.dataset.recordings[idx]["instance"],
                        frequency=self.frequency,
                        data_types=self.data_types,
                        prefetch_videos=self._prefetch_videos,
                    )
                    self._synced_recording_cache[idx] = synced_recording
                return self._synced_recording_cache[idx]
            raise TypeError(
                f"Dataset indices must be integers or slices, not {type(idx)}"
            )

    def __next__(self) -> SynchronizedRecording:
        """Get the next episode in the dataset iteration.

        Returns:
            SynchronizedRecording for the next episode.

        Raises:
            StopIteration: When all episodes have been processed.
        """
        if self._recording_idx >= len(self.dataset.recordings):
            raise StopIteration

        if self._recording_idx not in self._synced_recording_cache:
            recording = self.dataset.recordings[self._recording_idx]
            if self._recording_idx not in self._synced_recording_cache:
                s = SynchronizedRecording(
                    recording_id=recording["id"],
                    dataset=self.dataset,
                    robot_id=recording["robot_id"],
                    instance=recording["instance"],
                    frequency=self.frequency,
                    data_types=self.data_types,
                    prefetch_videos=self._prefetch_videos,
                )
                self._synced_recording_cache[self._recording_idx] = s

        to_return = self._synced_recording_cache[self._recording_idx]
        self._recording_idx += 1
        return to_return
