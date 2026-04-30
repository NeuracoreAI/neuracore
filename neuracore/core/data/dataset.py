"""Dataset management with lazy-loading generator."""

import logging
import sys
import time
from collections.abc import Generator, Iterator
from typing import Optional, Union

import requests
from neuracore_types import CrossEmbodimentUnion
from neuracore_types import Dataset as DatasetModel
from neuracore_types import DatasetUpdateRequest, DataType, EmbodimentDescription
from neuracore_types import Recording as RecordingModel
from neuracore_types import (
    SynchronizationDetails,
    SynchronizationProgress,
    SynchronizeDatasetRequest,
)
from neuracore_types import SynchronizedDataset as SynchronizedDatasetModel
from tqdm import tqdm

from neuracore.api.globals import GlobalSingleton
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.data.recording import Recording
from neuracore.core.data.synced_dataset import SynchronizedDataset
from neuracore.core.exceptions import RobotError
from neuracore.core.robot import get_robot
from neuracore.core.utils import backend_utils

from ..auth import Auth, get_auth
from ..const import API_URL, DEFAULT_RECORDING_CACHE_DIR
from ..exceptions import AuthenticationError, DatasetError
from ..utils.http_errors import extract_error_detail
from ..utils.http_session import get_session

PAGE_SIZE = 30
SYNC_PROGRESS_POLL_INTERVAL_S = 5.0

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
        description: str | None = None,
        recordings: list[dict] | list[Recording] | None = None,
    ):
        """Initialize a Dataset instance.

        Args:
            id: Unique identifier for the dataset.
            org_id: Organization ID of the user not for the org owner of the dataset
            name: Human-readable name for the dataset.
            size_bytes: Total size of the dataset in bytes.
            tags: List of tags associated with the dataset.
            data_types: List of data types present in the dataset.
            is_shared: Whether the dataset is shared.
            description: Description of the dataset.
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
        self.description = description
        self.data_types = data_types or []
        self.cache_dir = DEFAULT_RECORDING_CACHE_DIR
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
            metadata=recording_model.metadata,
        )

    def _initialize_num_recordings(self) -> None:
        """Fetch total number of recordings without loading them."""
        try:
            response = get_session().post(
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

        response = get_session().post(
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
        req = get_session().get(
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
            description=dataset_model.description,
            data_types=list(dataset_model.all_data_types.keys()),
        )

    @staticmethod
    def get_by_name(name: str, non_exist_ok: bool = False) -> Optional["Dataset"]:
        """Retrieve an existing dataset by name.

        Args:
            name: Name of the dataset to retrieve.
            non_exist_ok: If True, returns None when dataset is not found
                or when connection fails, instead of raising an exception.

        Returns:
            The Dataset instance if found, or None if non_exist_ok is True
            and the dataset doesn't exist or connection failed.

        Raises:
            DatasetError: If the dataset is not found and non_exist_ok is False,
                or if connection fails and non_exist_ok is False.
        """
        auth: Auth = get_auth()
        org_id = get_current_org()
        try:
            response = get_session().get(
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
                description=dataset_model.description,
                data_types=list(dataset_model.all_data_types.keys()),
            )
        except requests.exceptions.ConnectionError:
            if non_exist_ok:
                return None
            raise DatasetError("Failed to connect to server to retrieve dataset.")

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
            DatasetError: If the API request fails.
        """
        auth: Auth = get_auth()
        org_id = get_current_org()
        response = get_session().post(
            f"{API_URL}/org/{org_id}/datasets",
            headers=auth.get_headers(),
            json={
                "name": name,
                "description": description,
                "tags": tags,
                "is_shared": shared,
            },
        )
        if not response.ok:
            detail = extract_error_detail(response)
            error_message = detail or f"{response.status_code} {response.reason}"
            raise DatasetError(f"Failed to create dataset: {error_message}")

        dataset_model = DatasetModel.model_validate(response.json())
        return Dataset(
            id=dataset_model.id,
            org_id=org_id,
            name=dataset_model.name,
            size_bytes=dataset_model.size_bytes,
            tags=dataset_model.tags,
            is_shared=dataset_model.is_shared,
            description=dataset_model.description,
            data_types=list(dataset_model.all_data_types.keys()),
        )

    def delete(self) -> None:
        """Delete this dataset from Neuracore."""
        response = get_session().delete(
            f"{API_URL}/org/{self.org_id}/datasets/{self.id}",
            headers=get_auth().get_headers(),
        )
        response.raise_for_status()

    def _format_failure_summary(self, failed_recording_ids: list[str]) -> str:
        id_to_name = {r.id: r.name for r in self._recordings_cache}
        auth_headers = get_auth().get_headers()
        for recording_id in failed_recording_ids:
            if recording_id not in id_to_name:
                try:
                    response = get_session().get(
                        f"{API_URL}/org/{self.org_id}/recording/{recording_id}",
                        headers=auth_headers,
                    )
                    response.raise_for_status()
                    recording_model = RecordingModel.model_validate(response.json())
                    id_to_name[recording_id] = recording_model.metadata.name
                except Exception:
                    logger.debug("Failed to fetch name for recording %s", recording_id)
                    id_to_name[recording_id] = recording_id
        return "".join(
            f"\n{id_to_name[recording_id]}" for recording_id in failed_recording_ids
        )

    def _raise_sync_failure(
        self,
        failed_recording_ids: list[str],
        processed: int | None = None,
        total: int | None = None,
    ) -> None:
        recording_names = self._format_failure_summary(failed_recording_ids)
        progress_line = (
            f"\n({processed}/{total} recordings synchronized)."
            if processed is not None and total is not None
            else ""
        )
        raise DatasetError(
            f"Synchronization failed for dataset '{self.name}'.\n\n"
            f"Problematic recordings:\n{recording_names}\n\n"
            "These recordings might have missing or extra sensor data or "
            f"invalid synchronization parameters were provided.{progress_line}"
        )

    def _synchronize(
        self,
        frequency: int = 0,
        max_delay_s: float = sys.float_info.max,
        allow_duplicates: bool = True,
        trim_start_end: bool = True,
        cross_embodiment_union: CrossEmbodimentUnion | None = None,
    ) -> SynchronizedDatasetModel:
        """Synchronize the dataset with specified frequency and data types.

        Args:
            frequency: Frequency at which to synchronize the dataset.
                If 0, uses the default frequency.
            cross_embodiment_union: Dict specifying robot name to
                data types and their names to include in synchronization.
                If None, will use all available data types from the dataset.
            max_delay_s: Maximum allowed delay for synchronization.
            allow_duplicates: Whether duplicate points are allowed when syncing.
            trim_start_end: Whether to trim start/end during synchronization.

        Returns:
            SynchronizedDataset instance containing synchronized data.

        Raises:
            requests.HTTPError: If the API request fails.
            DatasetError: If frequency is not greater than 0.
        """
        response = get_session().post(
            f"{API_URL}/org/{self.org_id}/synchronize/synchronize-dataset",
            headers=get_auth().get_headers(),
            json=SynchronizeDatasetRequest(
                dataset_id=self.id,
                synchronization_details=SynchronizationDetails(
                    frequency=frequency,
                    max_delay_s=max_delay_s,
                    allow_duplicates=allow_duplicates,
                    trim_start_end=trim_start_end,
                    cross_embodiment_union=cross_embodiment_union,
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
        response = get_session().get(
            f"{API_URL}/org/{self.org_id}/synchronize/synchronization-progress/{synchronized_dataset_id}",
            headers=get_auth().get_headers(),
        )
        if response.status_code == 409:
            detail = extract_error_detail(response) or "Synchronization failed."
            prefix = "Synchronization failed for recording(s): "
            if prefix in detail:
                failed_ids = [
                    rid.strip()
                    for rid in detail.split(prefix, 1)[1].split(",")
                    if rid.strip()
                ]
                self._raise_sync_failure(failed_ids)
            raise DatasetError(detail)
        response.raise_for_status()
        return SynchronizationProgress.model_validate(response.json())

    def synchronize(
        self,
        cross_embodiment_union: CrossEmbodimentUnion | None = None,
        frequency: int = 0,
        prefetch_videos: bool = False,
        max_prefetch_workers: int = 4,
        max_delay_s: float = sys.float_info.max,
        allow_duplicates: bool = True,
        trim_start_end: bool = True,
    ) -> SynchronizedDataset:
        """Synchronize the dataset with specified frequency and data types.

        Args:
            frequency: Frequency at which to synchronize the dataset.
                If 0, uses the default frequency.
            cross_embodiment_union: Dict specifying robot IDs to data types and
                their names to include in synchronization.
            prefetch_videos: Whether to prefetch video data for the synchronized data.
            max_prefetch_workers: Number of threads to use for prefetching videos.
            max_delay_s: Maximum allowed delay for synchronization.
            allow_duplicates: Whether duplicate points are allowed when syncing.
            trim_start_end: Whether to trim start/end during synchronization.

        Returns:
            SynchronizedDataset instance containing synchronized data.

        Raises:
            requests.HTTPError: If the API request fails.
            DatasetError: If frequency is not greater than 0.
        """
        synced_dataset = self._synchronize(
            frequency=frequency,
            max_delay_s=max_delay_s,
            allow_duplicates=allow_duplicates,
            trim_start_end=trim_start_end,
            cross_embodiment_union=cross_embodiment_union,
        )

        total = synced_dataset.num_demonstrations
        synchronization_progress = self._get_synchronization_progress(synced_dataset.id)
        processed = synchronization_progress.num_synchronized_demonstrations
        if synchronization_progress.has_failures:
            self._raise_sync_failure(
                synchronization_progress.failed_recording_ids, processed, total
            )
        if total != processed:
            pbar = tqdm(total=total, desc="Synchronizing dataset", unit="recording")
            pbar.n = processed
            pbar.refresh()
            while processed < total:
                time.sleep(SYNC_PROGRESS_POLL_INTERVAL_S)
                synchronization_progress = self._get_synchronization_progress(
                    synced_dataset.id
                )
                if synchronization_progress.has_failures:
                    pbar.close()
                    self._raise_sync_failure(
                        synchronization_progress.failed_recording_ids, processed, total
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
            cross_embodiment_union=cross_embodiment_union,
            prefetch_videos=prefetch_videos,
            max_prefetch_workers=max_prefetch_workers,
        )

    def get_full_embodiment_description(self, robot_id: str) -> EmbodimentDescription:
        """Get full embodiment description for a given robot ID in the dataset.

        Args:
            robot_id: The robot ID to get the embodiment description for.

        Returns:
            An EmbodimentDescription object containing the data spec for the robot.
        """
        # Best-effort resolution without additional network calls.
        # If we can resolve a robot_name to an ID, do so; otherwise delegate to server.
        response = get_session().get(
            f"{API_URL}/org/{self.org_id}/datasets/{self.id}/full-embodiment-description/{robot_id}",
            headers=get_auth().get_headers(),
        )
        response.raise_for_status()
        raw_description = response.json()
        return {
            DataType(data_type): {
                int(index): name for index, name in indexed_names.items()
            }
            for data_type, indexed_names in raw_description.items()
        }

    @property
    def robot_ids(self) -> list[str]:
        """Get robot IDs present in the synchronized dataset.

        Returns:
            List of robot IDs in the synchronized dataset.
        """
        if self._robot_ids is None:
            response = get_session().get(
                f"{API_URL}/org/{self.org_id}/datasets/{self.id}/robot_ids",
                headers=get_auth().get_headers(),
            )
            response.raise_for_status()
            self._robot_ids = response.json()
            # Populate names to keep order consistent with IDs
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
                description=self.description,
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

    def _refresh_dataset_metadata(self) -> None:
        """Refresh the dataset metadata from the cloud.

        Ideally we should implement If-Match & Etag header.
        """
        auth = get_auth()
        org_id = get_current_org()
        try:
            req = get_session().get(
                f"{API_URL}/org/{org_id}/datasets/{self.id}",
                headers=auth.get_headers(),
            )
            req.raise_for_status()
            dataset_model = DatasetModel.model_validate(req.json())
            assert dataset_model.id == self.id

            self.name = dataset_model.name
            self.size_bytes = dataset_model.size_bytes
            self.tags = dataset_model.tags
            self.description = dataset_model.description
            self.is_shared = dataset_model.is_shared
            self.data_types = list(dataset_model.all_data_types.keys())

        except AuthenticationError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error fetching dataset metadata: {e}")

    def _update_metadata(self, dataset_metadata: DatasetUpdateRequest) -> None:
        """Update the metadata of a dataset.

        Args:
            dataset_metadata: The metadata to update the dataset with

        Raises:
            RuntimeError: Rases if there is an error updating the metadata in the cloud.
            ConfigError: If there is an error trying to get the config
            AuthenticationError: If there is an error with authentication
        """
        auth = get_auth()
        org_id = get_current_org()
        try:
            response = get_session().put(
                f"{API_URL}/org/{org_id}/datasets/{self.id}",
                headers=auth.get_headers(),
                json=dataset_metadata.model_dump(mode="json"),
            )
            response.raise_for_status()
            if dataset_metadata.name is not None:
                self.name = dataset_metadata.name
            if dataset_metadata.description is not None:
                self.description = dataset_metadata.description
            if dataset_metadata.tags is not None:
                self.tags = dataset_metadata.tags

        except AuthenticationError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error updating recording metadata: {e}")

    def set_name(self, name: str) -> None:
        """Update a dataset name programmatically.

        This call is blocking.

        Args:
            name: The name to update the dataset with.

        Raises:
            RuntimeError: Rases if there is an error updating the metadata in the cloud.
            ConfigError: If there is an error trying to get the config
            AuthenticationError: If there is an error with authentication
        """
        self._refresh_dataset_metadata()
        self._update_metadata(
            DatasetUpdateRequest(
                name=name, description=self.description, tags=self.tags
            )
        )

    def set_description(self, description: str) -> None:
        """Update a dataset description programmatically.

        This call is blocking.

        Args:
            description: The description to update the dataset with.

        Raises:
            RuntimeError: Rases if there is an error updating the metadata in the cloud.
            ConfigError: If there is an error trying to get the config
            AuthenticationError: If there is an error with authentication
        """
        self._refresh_dataset_metadata()
        self._update_metadata(
            DatasetUpdateRequest(
                name=self.name, description=description, tags=self.tags
            )
        )

    def set_tags(self, tags: list[str]) -> None:
        """Update a dataset tags programmatically.

        This call is blocking.

        Args:
            tags: The tags to update the dataset with.

        Raises:
            RuntimeError: Rases if there is an error updating the metadata in the cloud.
            ConfigError: If there is an error trying to get the config
            AuthenticationError: If there is an error with authentication
        """
        self._refresh_dataset_metadata()
        self._update_metadata(
            DatasetUpdateRequest(
                name=self.name, description=self.description, tags=tags
            )
        )

    def add_tag(self, tag: str) -> None:
        """Add a tag to a dataset programmatically.

        This call is blocking.

        Args:
            tag: The tag to add to the dataset.

        Raises:
            RuntimeError: Rases if there is an error updating the metadata in the cloud.
            ConfigError: If there is an error trying to get the config
            AuthenticationError: If there is an error with authentication
        """
        self._refresh_dataset_metadata()
        self._update_metadata(
            DatasetUpdateRequest(
                name=self.name, description=self.description, tags=self.tags + [tag]
            )
        )

    def start_recording(self, robot_name: str | None = None, instance: int = 0) -> None:
        """Start recording data to this dataset for a specific robot.

        Args:
            robot_name: Robot identifier. If not provided, uses the currently
                active robot from the global state.
            instance: Instance number of the robot for multi-instance scenarios.

        Raises:
            RobotError: If this dataset or robot cannot be used for recording.
        """
        global_state = GlobalSingleton()

        # Recording is routed through global session state, so this Dataset
        # instance must be the one currently connected via connect_dataset().
        active_dataset = global_state._active_dataset
        if active_dataset is None:
            raise RobotError("No active dataset. Call connect_dataset() first.")
        if active_dataset is not self:
            raise RobotError(
                "Can only start recording on the active dataset. "
                "Call connect_dataset() first."
            )

        # Use the active robot when no name is provided. If a name is provided,
        # resolve that exact robot/instance pair from the robot registry.
        robot = (
            global_state._active_robot
            if robot_name is None
            else get_robot(robot_name, instance)
        )
        if robot is None:
            raise RobotError(
                "No active robot. Call init() first or provide robot_name."
            )

        # Shared datasets may only receive recordings from shared robots, and
        # private datasets may only receive recordings from private robots. This
        # prevents accidentally mixing public/open-source data with private data.
        if robot.shared and not self.is_shared:
            raise RobotError(
                "Shared robot cannot be used with a non-shared dataset. "
                "If you requested a shared dataset, creation may have failed "
                "because you are not authorized to upload shared datasets or "
                "an existing non-shared dataset with the same name was reused. "
                f"Active dataset: '{self.name}' ({self.id})."
            )
        if not robot.shared and self.is_shared:
            raise RobotError(
                "Non-shared robot cannot be used with a shared dataset. "
                "Shared datasets require shared robots. If you requested a "
                "shared robot, ensure connect_robot(shared=True) succeeded. "
                f"Active dataset: '{self.name}' ({self.id})."
            )

        # Dataset-level validation is complete; the Robot owns the lower-level
        # recording lifecycle and will raise if a recording is already active.
        robot.start_recording(self.id)

    def cancel_recording(
        self, robot_name: str | None = None, instance: int = 0
    ) -> None:
        """Cancel the current recording for a specific robot without saving any data.

        Args:
            robot_name: Robot identifier. If not provided, uses the currently
                active robot from the global state.
            instance: Instance number of the robot for multi-instance scenarios.

        Raises:
            RobotError: If no robot is active and no robot_name is provided.
        """
        global_state = GlobalSingleton()

        # Cancellation through a Dataset instance should only affect recordings
        # for the dataset currently connected in this session.
        active_dataset = global_state._active_dataset
        if active_dataset is None:
            raise RobotError("No active dataset. Call connect_dataset() first.")
        if active_dataset is not self:
            raise RobotError(
                "Can only cancel recordings on the active dataset. "
                "Call connect_dataset() first."
            )

        # Match start_recording(): default to the active robot for the current
        # session, or resolve a specific robot/instance when explicitly named.
        robot = (
            global_state._active_robot
            if robot_name is None
            else get_robot(robot_name, instance)
        )
        if robot is None:
            raise RobotError(
                "No active robot. Call init() first or provide robot_name."
            )

        # Cancelling is intentionally idempotent from the caller's perspective:
        # if this robot has no active recording, there is nothing to cancel.
        if not robot.is_recording():
            return

        # The robot tracks the current recording ID internally. If that state is
        # missing despite is_recording(), avoid calling the backend with an empty ID.
        recording_id = robot.get_current_recording_id()
        if not recording_id:
            return

        # Delegate the backend cancellation and local state cleanup to Robot,
        # which owns the recording lifecycle below the Dataset API.
        robot.cancel_recording(recording_id)

    def stop_recording(
        self, robot_name: str | None = None, instance: int = 0, wait: bool = False
    ) -> None:
        """Stop recording data for a specific robot.

        Ends the current recording session for the specified robot. Optionally
        waits for all data streams to finish uploading before returning.

        Args:
            robot_name: Robot identifier. If not provided, uses the currently
                active robot from the global state.
            instance: Instance number of the robot for multi-instance scenarios.
            wait: Whether to block until all data streams have finished uploading
                to the backend storage.

        Raises:
            RobotError: If no robot is active and no robot_name is provided.
        """
        global_state = GlobalSingleton()

        # Stopping through a Dataset instance should only affect recordings for
        # the dataset currently connected in this session.
        active_dataset = global_state._active_dataset
        if active_dataset is None:
            raise RobotError("No active dataset. Call connect_dataset() first.")
        if active_dataset is not self:
            raise RobotError(
                "Can only stop recordings on the active dataset. "
                "Call connect_dataset() first."
            )

        # Match start_recording() and cancel_recording(): default to the active
        # robot, or resolve a specific robot/instance when explicitly named.
        robot = (
            global_state._active_robot
            if robot_name is None
            else get_robot(robot_name, instance)
        )
        if robot is None:
            raise RobotError(
                "No active robot. Call init() first or provide robot_name."
            )

        # Stopping is a no-op if the robot has no active recording, matching the
        # public API behavior for recordings stopped by another node.
        if not robot.is_recording():
            return

        # The robot tracks the current recording ID internally. Stopping needs a
        # concrete ID because the lower-level API finalizes that specific session.
        recording_id = robot.get_current_recording_id()
        if not recording_id:
            raise ValueError("Recording_id is None, no current recording")

        # Delegate finalization to Robot, which handles backend state and local
        # recording lifecycle cleanup.
        robot.stop_recording(recording_id)

        if not wait:
            return

        # When requested, block until backend upload traces have appeared and
        # drained. This preserves the existing wait semantics.
        is_traces_registered = False
        while True:
            data_traces = backend_utils.get_active_data_traces(recording_id)
            if len(data_traces) > 0:
                is_traces_registered = True
            elif len(data_traces) == 0 and is_traces_registered:
                break
            time.sleep(0.2)
