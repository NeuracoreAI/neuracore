"""SynchronizedDataset class for managing synchronized datasets."""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Union, cast

from neuracore_types import (
    CrossEmbodimentDescription,
    CrossEmbodimentUnion,
    SynchronizedDatasetStatistics,
)
from tqdm import tqdm

from neuracore.core.auth import get_auth
from neuracore.core.const import API_URL
from neuracore.core.data.recording import Recording
from neuracore.core.data.synced_recording import SynchronizedRecording
from neuracore.core.utils.http_session import thread_local_session

if TYPE_CHECKING:
    from neuracore.core.data.dataset import Dataset


logger = logging.getLogger(__name__)


class SynchronizedDataset:
    """Class for managing synchronized datasets."""

    def __init__(
        self,
        id: str,
        dataset: "Dataset",
        frequency: int,
        cross_embodiment_union: CrossEmbodimentUnion | None = None,
        prefetch_videos: bool = False,
        max_prefetch_workers: int = 1,
        synced_recording_cache: dict[int, SynchronizedRecording] | None = None,
    ):
        """Initialize a dataset from server response data.

        Args:
            id: Identifier for the synchronized dataset.
            dataset: Dataset object containing recordings.
            frequency: Frequency of the dataset in Hz.
            cross_embodiment_union: Cross-embodiment union for synchronization.
            prefetch_videos: Whether to prefetch video data to cache on initialization.
            max_prefetch_workers: Number of threads to use for prefetching videos.
            synced_recording_cache: Already-fetched synced recordings keyed by
                index, used when slicing to avoid re-fetching data the parent
                dataset already loaded.
        """
        self.id = id
        self.dataset = dataset
        self.frequency = frequency
        self.cross_embodiment_union = cross_embodiment_union
        self._prefetch_videos = prefetch_videos
        self._max_prefetch_workers = max_prefetch_workers
        self._recording_idx = 0
        self._synced_recording_cache: dict[int, SynchronizedRecording] = (
            dict(synced_recording_cache) if synced_recording_cache else {}
        )

        self._prefetch_videos_needed = False
        if prefetch_videos:
            for rec in self.dataset:
                cache_dir = self.dataset.cache_dir / rec.id

                lock_file = cache_dir / ".recording.lock"

                # Check if cache directory exists
                if not cache_dir.exists() or lock_file.exists():
                    # NOTE: we check if the directly exists to avoid re downloading
                    #  if the lock file exists it keeps a worker waiting in case the
                    #  other download is in progress fails, we can retry
                    self._prefetch_videos_needed = True
                    break
        if not self._is_synced_recording_cache_complete():
            self._perform_synced_data_prefetch(
                max_prefetch_workers=max_prefetch_workers
            )

    def _is_synced_recording_cache_complete(self) -> bool:
        """Check whether every recording is already in the synced cache."""
        return all(
            idx in self._synced_recording_cache for idx in range(len(self.dataset))
        )

    def _perform_synced_data_prefetch(self, max_prefetch_workers: int) -> None:
        """Prefetch synced data for all recordings using multiple threads.

        Args:
            max_prefetch_workers: Number of threads to use for prefetching synced data.
        """
        # Indexing the last recording pages in all metadata up front, so the
        # threaded prefetch below just reads cache instead of paging concurrently.
        num_recordings = len(self.dataset)
        if num_recordings > 0:
            self.dataset[num_recordings - 1]

        desc = "Prefetching synced data"
        if self._prefetch_videos_needed:
            desc += " and videos"
        desc += (
            f" with {max_prefetch_workers}"
            f"{' workers' if max_prefetch_workers > 1 else ' worker'}"
        )
        with ThreadPoolExecutor(max_workers=max_prefetch_workers) as executor:
            list(
                tqdm(
                    executor.map(lambda idx: self[idx], range(len(self.dataset))),
                    total=len(self.dataset),
                    desc=desc,
                    unit="Recording",
                )
            )

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
        self, idx: int | slice
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
            # Hand already-fetched recordings to the slice (re-indexed) so it
            # does not prefetch data this instance already loaded.
            start, stop, step = idx.indices(len(self.dataset))
            sliced_cache = {
                new_idx: self._synced_recording_cache[old_idx]
                for new_idx, old_idx in enumerate(range(start, stop, step))
                if old_idx in self._synced_recording_cache
            }
            return SynchronizedDataset(
                id=self.id,
                dataset=cast("Dataset", dataset),
                frequency=self.frequency,
                cross_embodiment_union=self.cross_embodiment_union,
                prefetch_videos=False,  # Avoid prefetching again
                max_prefetch_workers=self._max_prefetch_workers,
                synced_recording_cache=sliced_cache,
            )
        else:
            # Handle single index
            if isinstance(idx, int):
                if idx < 0:  # Handle negative indices
                    idx += len(self.dataset)
                if not 0 <= idx < len(self.dataset):
                    raise IndexError("Dataset index out of range")
                if idx not in self._synced_recording_cache:
                    rec = cast(Recording, self.dataset[idx])
                    synced_recording = SynchronizedRecording(
                        recording_id=rec.id,
                        recording_name=rec.name,
                        dataset=self.dataset,
                        robot_id=rec.robot_id,
                        instance=rec.instance,
                        frequency=self.frequency,
                        cross_embodiment_union=self.cross_embodiment_union,
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
        if self._recording_idx >= len(self.dataset):
            raise StopIteration

        if self._recording_idx not in self._synced_recording_cache:
            recording: Recording = cast(Recording, self.dataset[self._recording_idx])
            if self._recording_idx not in self._synced_recording_cache:
                s = SynchronizedRecording(
                    recording_id=recording.id,
                    recording_name=recording.name,
                    dataset=self.dataset,
                    robot_id=recording.robot_id,
                    instance=recording.instance,
                    frequency=self.frequency,
                    cross_embodiment_union=self.cross_embodiment_union,
                    prefetch_videos=self._prefetch_videos,
                )
                self._synced_recording_cache[self._recording_idx] = s

        to_return = self._synced_recording_cache[self._recording_idx]
        self._recording_idx += 1
        return to_return

    def calculate_statistics(
        self,
        input_cross_embodiment_description: CrossEmbodimentDescription,
        output_cross_embodiment_description: CrossEmbodimentDescription,
    ) -> SynchronizedDatasetStatistics:
        """Calculate statistics for each data type in the synchronized dataset.

        Args:
            input_cross_embodiment_description: Cross-embodiment
            description for input data types.
            output_cross_embodiment_description: Cross-embodiment
            description for output data types.

        Returns:
            SynchronizedDatasetStatistics containing the calculated statistics.
        """
        session = thread_local_session()
        response = session.post(
            f"{API_URL}/org/{self.dataset.org_id}/synchronized-dataset/calculate-dataset-statistics",
            json=SynchronizedDatasetStatistics(
                synchronized_dataset_id=self.id,
                input_cross_embodiment_description=input_cross_embodiment_description,
                output_cross_embodiment_description=output_cross_embodiment_description,
            ).model_dump(mode="json"),
            headers=get_auth().get_headers(),
        )
        response.raise_for_status()
        return SynchronizedDatasetStatistics.model_validate(response.json())
