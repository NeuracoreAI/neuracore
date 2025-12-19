"""Dataset management with lazy-loading generator."""

import logging
import sys
import time
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Optional, Union

import requests
from neuracore_types import Dataset as DatasetModel
from neuracore_types import DataSpec, DataType
from neuracore_types import Recording as RecordingModel
from neuracore_types import (
    RobotDataSpec,
    SynchronizationDetails,
    SynchronizationProgress,
    SynchronizeDatasetRequest,
)
from neuracore_types import SynchronizedDataset as SynchronizedDatasetModel
from tqdm import tqdm

from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.data.recording import Recording
from neuracore.core.data.synced_dataset import SynchronizedDataset

from ..auth import Auth, get_auth
from ..const import API_URL
from ..exceptions import DatasetError

DEFAULT_CACHE_DIR = Path.home() / ".neuracore" / "training" / "dataset_cache"
PAGE_SIZE = 30

logger = logging.getLogger(__name__)


class Dataset:
    """Class representing a dataset in Neuracore."""

    def __init__(
        self,
        id: str,
        org_id: str,
        name: str,
        size_bytes: int,
        tags: list[str],
        data_types: list[DataType],
        is_shared: bool,
        recordings: list[dict] | list[Recording] | None = None,
    ):
        """Initialize a Dataset instance.

        Args:
            id: Unique identifier for the dataset.
            org_id: Organization ID associated with the dataset.
            name: Human-readable name for the dataset.
            size_bytes: Total size of the dataset in bytes.
            tags: List of tags associated with the dataset.
            data_types: List of data types present in the dataset.
            is_shared: Whether the dataset is shared.
            recordings: List of recording dictionaries.
            If not provided, the dataset will be fetched from the Neuracore API.

        Attributes:
            cache_dir: Directory path for caching dataset recordings.
            _recordings_cache: Internal list of cached recordings.
            _num_recordings: Number of recordings in the dataset,
            or None if not fetched from the Neuracore API.
            _start_after: Internal dictionary for tracking
            the start of the next page of recordings.
        """
        self.id = id
        self.org_id = org_id
        self.name = name
        self.size_bytes = size_bytes
        self.tags = tags
        self.is_shared = is_shared
        self.data_types = data_types or []

        self.cache_dir = DEFAULT_CACHE_DIR
        self._recordings_cache: list[Recording] = (
            [
                self._wrap_raw_recording(r) if isinstance(r, dict) else r
                for r in recordings
            ]
            if recordings
            else []
        )
        self._num_recordings: int | None = len(recordings) if recordings else None
        self._start_after: dict | None = None
        self._robot_ids: list[str] | None = None

    def _wrap_raw_recording(self, raw_recording: dict) -> Recording:
        """Wrap a raw recording dict into a Recording object.

        Args:
            raw_recording: A dict containing the raw recording data

        Returns:
            A Recording object
        """
        recording_model = RecordingModel.model_validate(raw_recording)
        return Recording(
            dataset=self,
            recording_id=recording_model.id,
            total_bytes=recording_model.total_bytes,
            robot_id=recording_model.robot_id,
            instance=recording_model.instance,
            start_time=recording_model.start_time,
            end_time=recording_model.end_time,
        )

    def _initialize_num_recordings(self) -> None:
        """Fetch total number of recordings without loading them."""
        try:
            response = requests.post(
                f"{API_URL}/org/{self.org_id}/recording/by-dataset/{self.id}",
                headers=get_auth().get_headers(),
                params={"limit": 1, "is_shared": self.is_shared},
                json=None,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            self._num_recordings = data.get("total", 0)
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch recording count for Dataset {self.id}: {e}")
            self._num_recordings = 0

    def _fetch_next_page(self) -> list[Recording]:
        """Fetch the next page of recordings and append to cache (lazy)."""
        if (
            self._num_recordings is not None
            and len(self._recordings_cache) >= self._num_recordings
        ):
            return []

        params = {"limit": PAGE_SIZE, "is_shared": self.is_shared}
        payload = self._start_after or None

        response = requests.post(
            f"{API_URL}/org/{self.org_id}/recording/by-dataset/{self.id}",
            headers=get_auth().get_headers(),
            params=params,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        batch = data.get("data", [])
        if not batch:
            return []

        self._start_after = batch[-1]
        self._num_recordings = data.get("total", self._num_recordings)

        wrapped = [self._wrap_raw_recording(r) for r in batch]
        self._recordings_cache.extend(wrapped)
        return wrapped

    def _recordings_generator(self) -> Generator[Recording, None, None]:
        """A generator yielding Recordings for this dataset.

        This generator handles four cases:
        1. All recordings are pre-loaded into the cache.
        2. Not all recordings are in the cache and no pagination state.
        3. Partially fetched with pagination state.
        4. Fetch remaining recordings from API.

        In case 1, the generator yields all recordings from the cache.
        In case 2, the generator resets the cache and fetches recordings from start.
        In case 3, the generator yields the remaining recordings from the
            cache and then fetches the next page of recordings.
        In case 4, the generator fetches the next page of recordings
            from API and yields them.
        The generator stops when all recordings have been yielded or an error occurs.

        Returns:
            A generator yielding Recording objects.
        """
        if self._num_recordings is None:
            self._initialize_num_recordings()

        assert self._num_recordings is not None

        # Case 0: Explicitly known to have zero recordings
        if self._num_recordings == 0:
            return

        if self._recordings_cache:

            # Case 1: All recordings pre-loaded, yield from cache only
            if len(self._recordings_cache) >= self._num_recordings:
                yield from self._recordings_cache
                return

            # Case 2: Not all recordings in cache and no pagination state
            if self._start_after is None:
                # Reset unreliable cache ready for fetching from API
                self._recordings_cache = []

            # Case 3: Partially fetched with pagination state
            else:
                yield from self._recordings_cache

        # Case 4: Fetch remaining recordings from API (from beginning or next page)
        while True:
            recordings = self._fetch_next_page()
            if not recordings:
                return
            yield from recordings

    @staticmethod
    def get_by_id(id: str, non_exist_ok: bool = False) -> Optional["Dataset"]:
        """Retrieve an existing dataset by ID.

        Args:
            id: Unique identifier of the dataset to retrieve.
            non_exist_ok: If True, returns None when dataset is not found
                instead of raising an exception.

        Returns:
            The Dataset instance if found, or None if non_exist_ok is True
            and the dataset doesn't exist.

        Raises:
            DatasetError: If the dataset is not found and non_exist_ok is False.
        """
        auth: Auth = get_auth()
        org_id = get_current_org()
        req = requests.get(
            f"{API_URL}/org/{org_id}/datasets/{id}",
            headers=auth.get_headers(),
        )
        if req.status_code != 200:
            if non_exist_ok:
                return None
            raise DatasetError(f"Dataset with ID '{id}' not found.")
        dataset_model = DatasetModel.model_validate(req.json())
        return Dataset(
            id=dataset_model.id,
            org_id=org_id,
            name=dataset_model.name,
            size_bytes=dataset_model.size_bytes,
            tags=dataset_model.tags,
            is_shared=dataset_model.is_shared,
            data_types=list(dataset_model.all_data_types.keys()),
        )

    @staticmethod
    def get_by_name(name: str, non_exist_ok: bool = False) -> Optional["Dataset"]:
        """Retrieve an existing dataset by name.

        Args:
            name: Name of the dataset to retrieve.
            non_exist_ok: If True, returns None when dataset is not found
                instead of raising an exception.

        Returns:
            The Dataset instance if found, or None if non_exist_ok is True
            and the dataset doesn't exist.

        Raises:
            DatasetError: If the dataset is not found and non_exist_ok is False.
        """
        auth: Auth = get_auth()
        org_id = get_current_org()
        response = requests.get(
            f"{API_URL}/org/{org_id}/datasets/search/by-name",
            params={"name": name},
            headers=auth.get_headers(),
        )
        if response.status_code != 200:
            if non_exist_ok:
                return None
            raise DatasetError(f"Dataset '{name}' not found.")
        dataset_model = DatasetModel.model_validate(response.json())
        return Dataset(
            id=dataset_model.id,
            org_id=org_id,
            name=dataset_model.name,
            size_bytes=dataset_model.size_bytes,
            tags=dataset_model.tags,
            is_shared=dataset_model.is_shared,
            data_types=list(dataset_model.all_data_types.keys()),
        )

    @staticmethod
    def create(
        name: str,
        description: str | None = None,
        tags: list[str] | None = None,
        shared: bool = False,
    ) -> "Dataset":
        """Create a new dataset or return existing one with the same name.

        Creates a new dataset with the specified parameters. If a dataset
        with the same name already exists, returns the existing dataset
        instead of creating a duplicate.

        Args:
            name: Unique name for the dataset.
            description: Optional description of the dataset contents and purpose.
            tags: Optional list of tags for organizing and searching datasets.
            shared: Whether the dataset should be shared/open-source.
                Note that setting shared=True is only available to specific
                members allocated by the Neuracore team.

        Returns:
            The newly created Dataset instance, or existing dataset if
            name already exists.
        """
        ds = Dataset.get_by_name(name, non_exist_ok=True)
        if ds is None:
            ds = Dataset._create_dataset(name, description, tags, shared=shared)
        else:
            logger.info(f"Dataset '{name}' already exist.")
        return ds

    @staticmethod
    def _create_dataset(
        name: str,
        description: str | None = None,
        tags: list[str] | None = None,
        shared: bool = False,
    ) -> "Dataset":
        """Create a new dataset via API call.

        Args:
            name: Unique name for the dataset.
            description: Optional description of the dataset.
            tags: Optional list of tags for the dataset.
            shared: Whether the dataset should be shared.
                Note that setting shared=True is only available to specific
                members allocated by the Neuracore team.

        Returns:
            The newly created Dataset instance.

        Raises:
            requests.HTTPError: If the API request fails.
        """
        auth: Auth = get_auth()
        org_id = get_current_org()
        response = requests.post(
            f"{API_URL}/org/{org_id}/datasets",
            headers=auth.get_headers(),
            json={
                "name": name,
                "description": description,
                "tags": tags,
                "is_shared": shared,
            },
        )
        response.raise_for_status()
        dataset_model = DatasetModel.model_validate(response.json())
        return Dataset(
            id=dataset_model.id,
            org_id=org_id,
            name=dataset_model.name,
            size_bytes=dataset_model.size_bytes,
            tags=dataset_model.tags,
            is_shared=dataset_model.is_shared,
            data_types=list(dataset_model.all_data_types.keys()),
        )

    def _synchronize(
        self,
        frequency: int = 0,
        robot_data_spec: RobotDataSpec | None = None,
    ) -> SynchronizedDatasetModel:
        """Synchronize the dataset with specified frequency and data types.

        Args:
            frequency: Frequency at which to synchronize the dataset.
                If 0, uses the default frequency.
            robot_data_spec: Dict specifying robot id to
                data types and their names to include in synchronization.
                If None, will use all available data types from the dataset.

        Returns:
            SynchronizedDataset instance containing synchronized data.

        Raises:
            requests.HTTPError: If the API request fails.
            DatasetError: If frequency is not greater than 0.
        """
        response = requests.post(
            f"{API_URL}/org/{self.org_id}/synchronize/synchronize-dataset",
            headers=get_auth().get_headers(),
            json=SynchronizeDatasetRequest(
                dataset_id=self.id,
                synchronization_details=SynchronizationDetails(
                    frequency=frequency,
                    robot_data_spec=robot_data_spec,
                    max_delay_s=sys.float_info.max,
                    allow_duplicates=True,
                ),
            ).model_dump(mode="json"),
        )
        response.raise_for_status()
        return SynchronizedDatasetModel.model_validate(response.json())

    def _get_synchronization_progress(
        self, synchronized_dataset_id: str
    ) -> SynchronizationProgress:
        """Get synchronization progress for this dataset.

        Returns:
            Synchronization progress for the dataset.
        """
        response = requests.get(
            f"{API_URL}/org/{self.org_id}/synchronize/synchronization-progress/{synchronized_dataset_id}",
            headers=get_auth().get_headers(),
        )
        response.raise_for_status()
        return SynchronizationProgress.model_validate(response.json())

    def synchronize(
        self,
        frequency: int = 0,
        robot_data_spec: RobotDataSpec | None = None,
        prefetch_videos: bool = False,
        max_prefetch_workers: int = 4,
    ) -> SynchronizedDataset:
        """Synchronize the dataset with specified frequency and data types.

        Args:
            frequency: Frequency at which to synchronize the dataset.
                If 0, uses the default frequency.
            robot_data_spec: Dict specifying robot id to
                data types and their names to include in synchronization.
                If None, will use all available data types from the dataset.
            prefetch_videos: Whether to prefetch video data for the synchronized data.
            max_prefetch_workers: Number of threads to use for prefetching videos.

        Returns:
            SynchronizedDataset instance containing synchronized data.

        Raises:
            requests.HTTPError: If the API request fails.
            DatasetError: If frequency is not greater than 0.
        """
        if robot_data_spec is None:
            robot_data_spec = {}
            for rid in self.robot_ids:
                robot_data_spec[rid] = {dt: [] for dt in self.data_types}

        synced_dataset = self._synchronize(
            frequency=frequency, robot_data_spec=robot_data_spec
        )
        synchronization_progress = self._get_synchronization_progress(synced_dataset.id)
        total = synced_dataset.num_demonstrations
        processed = synchronization_progress.num_synchronized_demonstrations
        if total != processed:
            pbar = tqdm(total=total, desc="Synchronizing dataset", unit="recording")
            pbar.n = processed
            pbar.refresh()
            while processed < total:
                time.sleep(5.0)
                synchronization_progress = self._get_synchronization_progress(
                    synced_dataset.id
                )
                new_processed = synchronization_progress.num_synchronized_demonstrations
                if new_processed > processed:
                    pbar.update(new_processed - processed)
                    processed = new_processed
            pbar.close()
        else:
            logger.info("Dataset is already synchronized.")
        return SynchronizedDataset(
            id=synced_dataset.id,
            dataset=self,
            frequency=frequency,
            robot_data_spec=robot_data_spec,
            prefetch_videos=prefetch_videos,
            max_prefetch_workers=max_prefetch_workers,
        )

    def get_full_data_spec(self, robot_id: str) -> DataSpec:
        """Get full data spec for a given robot ID in the dataset.

        Args:
            robot_id: The robot ID to get the data spec for.

        Returns:
            A dictionary mapping DataType to list of data names.
        """
        response = requests.get(
            f"{API_URL}/org/{self.org_id}/datasets/{self.id}/full-data-spec/{robot_id}",
            headers=get_auth().get_headers(),
        )
        response.raise_for_status()
        return response.json()

    @property
    def robot_ids(self) -> list[str]:
        """Get robot IDs present in the synchronized dataset.

        Returns:
            List of robot IDs in the synchronized dataset.
        """
        if self._robot_ids is None:
            response = requests.get(
                f"{API_URL}/org/{self.org_id}/datasets/{self.id}/robot_ids",
                headers=get_auth().get_headers(),
            )
            response.raise_for_status()
            self._robot_ids = response.json()
        return self._robot_ids

    def __iter__(self) -> Iterator[Recording]:
        """Yield recordings one by one, fetching pages lazily."""
        return self._recordings_generator()

    def __getitem__(self, index: int | slice) -> Union[Recording, "Dataset"]:
        """Support for indexing and slicing dataset episodes.

        Args:
            index: Integer index or slice object for accessing episodes.

        Returns:
            Recording object for a single episode or
            Dataset object for a slice of episodes.

        Raises:
            IndexError: If the index is out of range.
            TypeError: If the index is not an integer or slice.
        """
        if isinstance(index, int):
            if index < 0:
                index += len(self)
            if index < 0 or index >= len(self):
                raise IndexError("Dataset index out of range")

            # Load pages until index is available in cache
            while index >= len(self._recordings_cache):
                if not self._fetch_next_page():
                    raise IndexError("Dataset index out of range")
            return self._recordings_cache[index]

        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            # Load pages until stop index is available
            while stop > len(self._recordings_cache):
                if not self._fetch_next_page():
                    break
            return Dataset(
                org_id=self.org_id,
                id=self.id,
                name=self.name,
                tags=self.tags,
                size_bytes=self.size_bytes,
                is_shared=self.is_shared,
                data_types=self.data_types,
                recordings=self._recordings_cache[start:stop:step],
            )

        else:
            raise TypeError("Dataset indices must be int or slice")

    def __len__(self) -> int:
        """Return the number of recordings in the dataset.

        Returns:
            int: The number of recordings in the dataset.

        Raises:
            DatasetError: If the number of recordings is not available.
        """
        if self._num_recordings is None:
            self._initialize_num_recordings()
        return self._num_recordings or 0
