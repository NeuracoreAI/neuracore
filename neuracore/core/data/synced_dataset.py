"""SynchronizedDataset class for managing synchronized datasets."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Union, cast

import requests
from neuracore_types import (
    CalculateDatasetStatisticsRequest,
    CrossEmbodimentDescription,
    CrossEmbodimentUnion,
    DatasetStatisticsJob,
    DatasetStatisticsJobStatus,
    SynchronizedDatasetStatistics,
)
from tqdm import tqdm

from neuracore.core.auth import get_auth
from neuracore.core.const import API_URL
from neuracore.core.data.recording import Recording
from neuracore.core.data.synced_recording import SynchronizedRecording
from neuracore.core.exceptions import DatasetError
from neuracore.core.utils.http_errors import extract_error_detail
from neuracore.core.utils.http_session import thread_local_session

if TYPE_CHECKING:
    from neuracore.core.data.dataset import Dataset


logger = logging.getLogger(__name__)

DATASET_STATISTICS_POLL_INTERVAL_S = 5.0
DATASET_STATISTICS_TIMEOUT_S = 1800.0
DATASET_STATISTICS_RESULT_TIMEOUT_S = (5.0, 120.0)
_FATAL_PROGRESS_STATUS_CODES = frozenset({400, 401, 403, 404, 409, 422})


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

        The calculation runs server-side, so this starts it and then blocks,
        reporting progress, until it finishes. Repeat calls for the same
        recordings and cross-embodiment descriptions join the running calculation
        or reuse its result, so an interrupted call resumes rather than restarts.

        Args:
            input_cross_embodiment_description: Cross-embodiment
            description for input data types.
            output_cross_embodiment_description: Cross-embodiment
            description for output data types.

        Returns:
            SynchronizedDatasetStatistics containing the calculated statistics.
        """
        job = self._start_statistics_job(
            input_cross_embodiment_description=input_cross_embodiment_description,
            output_cross_embodiment_description=output_cross_embodiment_description,
        )
        job = self._await_statistics_job(job)
        return self._fetch_statistics_result(job.job_id)

    def _start_statistics_job(
        self,
        input_cross_embodiment_description: CrossEmbodimentDescription,
        output_cross_embodiment_description: CrossEmbodimentDescription,
    ) -> DatasetStatisticsJob:
        """Start or join the statistics calculation for this dataset and spec.

        Args:
            input_cross_embodiment_description: Cross-embodiment
            description for input data types.
            output_cross_embodiment_description: Cross-embodiment
            description for output data types.

        Returns:
            The calculation's initial state.

        Raises:
            DatasetError: If the calculation could not be started.
        """
        # The server derives the job from the request, so retrying is safe.
        session = thread_local_session(retry_transient=True)
        response = session.post(
            f"{API_URL}/org/{self.dataset.org_id}/synchronized-dataset/calculate-dataset-statistics",
            json=CalculateDatasetStatisticsRequest(
                synchronized_dataset_id=self.id,
                input_cross_embodiment_description=input_cross_embodiment_description,
                output_cross_embodiment_description=output_cross_embodiment_description,
            ).model_dump(mode="json"),
            headers=get_auth().get_headers(),
        )
        if not response.ok:
            raise DatasetError(
                extract_error_detail(response)
                or "Failed to start calculating dataset statistics."
            )
        return DatasetStatisticsJob.model_validate(response.json())

    def _poll_statistics_job(self, job_id: str) -> DatasetStatisticsJob | None:
        """Read a statistics calculation's state, tolerating transient failures.

        Args:
            job_id: The statistics job to poll.

        Returns:
            The calculation's state, or None when this read should simply be
            retried on the next poll.

        Raises:
            DatasetError: If the calculation failed, or the read failed in a way
                that further polling cannot resolve.
        """
        session = thread_local_session(retry_transient=True, retry_read_timeout=True)
        try:
            response = session.get(
                f"{API_URL}/org/{self.dataset.org_id}/synchronized-dataset"
                f"/dataset-statistics-progress/{job_id}",
                headers=get_auth().get_headers(),
            )
        except requests.RequestException as exc:
            logger.debug(f"Dataset statistics progress poll failed: {exc}")
            return None

        if not response.ok:
            if response.status_code in _FATAL_PROGRESS_STATUS_CODES:
                raise DatasetError(
                    extract_error_detail(response)
                    or "Calculating dataset statistics failed."
                )
            logger.debug(
                "Dataset statistics progress returned "
                f"{response.status_code}; retrying."
            )
            return None

        job = DatasetStatisticsJob.model_validate(response.json())
        if job.status is DatasetStatisticsJobStatus.FAILED:
            raise DatasetError(job.error or "Calculating dataset statistics failed.")
        return job

    def _await_statistics_job(self, job: DatasetStatisticsJob) -> DatasetStatisticsJob:
        """Block until a statistics calculation completes, reporting progress.

        Args:
            job: The calculation's current state.

        Returns:
            The calculation's completed state.

        Raises:
            DatasetError: If the calculation fails or exceeds the deadline.
        """
        if job.status is DatasetStatisticsJobStatus.COMPLETE:
            logger.debug(f"Dataset statistics already calculated (job {job.job_id}).")
            return job

        deadline = time.monotonic() + DATASET_STATISTICS_TIMEOUT_S
        pbar = tqdm(
            total=job.num_recordings,
            desc="Calculating dataset statistics",
            unit="recording",
        )
        try:
            pbar.n = min(job.num_completed_recordings, job.num_recordings)
            pbar.refresh()
            aggregating = False
            while job.status is not DatasetStatisticsJobStatus.COMPLETE:
                if time.monotonic() >= deadline:
                    raise DatasetError(
                        f"Timed out after {DATASET_STATISTICS_TIMEOUT_S:.0f}s "
                        f"waiting for dataset statistics (job {job.job_id}, "
                        f"status {job.status.value}, "
                        f"{job.num_completed_recordings}/{job.num_recordings} "
                        "recordings). The calculation is still running; "
                        "calculating again resumes it rather than starting over."
                    )
                time.sleep(DATASET_STATISTICS_POLL_INTERVAL_S)
                polled = self._poll_statistics_job(job.job_id)
                if polled is None:
                    continue

                job = polled
                completed = min(job.num_completed_recordings, job.num_recordings)
                if completed > pbar.n:
                    pbar.update(completed - pbar.n)
                if (
                    job.status is DatasetStatisticsJobStatus.AGGREGATING
                    and not aggregating
                ):
                    aggregating = True
                    pbar.set_description("Aggregating dataset statistics")
                # Keep the elapsed clock moving so aggregating does not look
                # like a hang once the recording count stops changing.
                pbar.refresh()
        finally:
            pbar.close()
        return job

    def _fetch_statistics_result(self, job_id: str) -> SynchronizedDatasetStatistics:
        """Fetch a completed statistics calculation's result.

        Args:
            job_id: The completed statistics job.

        Returns:
            SynchronizedDatasetStatistics containing the calculated statistics.

        Raises:
            DatasetError: If the result could not be fetched.
        """
        session = thread_local_session(retry_transient=True, retry_read_timeout=True)
        response = session.get(
            f"{API_URL}/org/{self.dataset.org_id}/synchronized-dataset"
            f"/dataset-statistics/{job_id}",
            headers=get_auth().get_headers(),
            timeout=DATASET_STATISTICS_RESULT_TIMEOUT_S,
        )
        if not response.ok:
            raise DatasetError(
                extract_error_detail(response) or "Failed to fetch dataset statistics."
            )
        return SynchronizedDatasetStatistics.model_validate(response.json())
