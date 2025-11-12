"""Dataset management."""

import logging
import time
from pathlib import Path
from typing import Optional, Union

import requests
from neuracore_types import DataType, SyncedDataset
from tqdm import tqdm

from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.data.recording import Recording
from neuracore.core.data.synced_dataset import SynchronizedDataset

from ..auth import Auth, get_auth
from ..const import API_URL
from ..exceptions import DatasetError

DEFAULT_CACHE_DIR = Path.home() / ".neuracore" / "training" / "dataset_cache"


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
        is_shared: bool,
        data_types: list[DataType],
        recordings: Optional[list[dict]] = None,
    ):
        """Initialize a dataset from server response data.

        Args:
            id: Unique identifier for the dataset.
            org_id: The id of the organization this dataset belongs to.
            name: Name of the dataset.
            size_bytes: Size of the dataset in bytes.
            tags: List of tags associated with the dataset.
            is_shared: Whether the dataset is shared/open-source.
            recordings: Optional list of recordings in the dataset.
                If not provided, recordings will be fetched from the server.
            data_types: Optional list of DataType available in the dataset.
        """
        self.id = id
        self.org_id = org_id
        self.name = name
        self.size_bytes = size_bytes
        self.tags = tags
        self.is_shared = is_shared
        self.recordings = recordings or self._get_recordings()
        self.data_types = data_types or []
        self.num_recordings = len(self.recordings)
        self._recording_idx = 0
        self.cache_dir = DEFAULT_CACHE_DIR

    def _get_recordings(self) -> list[dict]:
        """Get the list of recordings in the dataset.

        Returns:
            List of recordings in the dataset.

        Raises:
            requests.HTTPError: If the API request fails.
        """
        auth = get_auth()
        response = requests.get(
            f"{API_URL}/org/{self.org_id}/datasets/{self.id}/recordings",
            headers=auth.get_headers(),
        )
        response.raise_for_status()
        data = response.json()
        return data["recordings"]

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
        dataset_json = req.json()
        return Dataset(
            id=dataset_json["id"],
            org_id=org_id,
            name=dataset_json["name"],
            size_bytes=dataset_json["size_bytes"],
            tags=dataset_json["tags"],
            is_shared=dataset_json["is_shared"],
            data_types=list(dataset_json.get("all_data_types", {}).keys()),
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
        req = requests.get(
            f"{API_URL}/org/{org_id}/datasets/search/by-name",
            params={"name": name},
            headers=auth.get_headers(),
        )
        if req.status_code != 200:
            if non_exist_ok:
                return None
            raise DatasetError(f"Dataset '{name}' not found.")
        dataset_json = req.json()
        return Dataset(
            id=dataset_json["id"],
            org_id=org_id,
            name=dataset_json["name"],
            size_bytes=dataset_json["size_bytes"],
            tags=dataset_json["tags"],
            is_shared=dataset_json["is_shared"],
            data_types=list(dataset_json.get("all_data_types", {}).keys()),
        )

    @staticmethod
    def create(
        name: str,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
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
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
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
        dataset_json = response.json()
        return Dataset(
            id=dataset_json["id"],
            org_id=org_id,
            name=dataset_json["name"],
            size_bytes=dataset_json["size_bytes"],
            tags=dataset_json["tags"],
            is_shared=dataset_json["is_shared"],
            data_types=list(dataset_json.get("all_data_types", {}).keys()),
        )

    def _synchronize(
        self, frequency: int = 0, data_types: Optional[list[DataType]] = None
    ) -> SyncedDataset:
        """Synchronize the dataset with specified frequency and data types.

        Args:
            frequency: Frequency at which to synchronize the dataset.
                If 0, uses the default frequency.
            data_types: List of DataType to include in synchronization.
                If None, uses the default data types from the dataset.

        Returns:
            SyncedDataset instance containing synchronized data.

        Raises:
            requests.HTTPError: If the API request fails.
            DatasetError: If frequency is not greater than 0.
        """
        response = requests.post(
            f"{API_URL}/org/{self.org_id}/synchronize/synchronize-dataset",
            headers=get_auth().get_headers(),
            json={
                "dataset_id": self.id,
                "frequency": frequency,
                "data_types": data_types,
            },
        )
        response.raise_for_status()
        dataset_json = response.json()
        return SyncedDataset.model_validate(dataset_json)

    def synchronize(
        self,
        frequency: int = 0,
        data_types: Optional[list[DataType]] = None,
        prefetch_videos: bool = False,
    ) -> SynchronizedDataset:
        """Synchronize the dataset with specified frequency and data types.

        Args:
            frequency: Frequency at which to synchronize the dataset.
                If 0, uses the default frequency.
            data_types: List of DataType to include in synchronization.
                If None, uses the default data types from the dataset.
            prefetch_videos: Whether to prefetch video data for the synchronized data.

        Returns:
            SynchronizedDataset instance containing synchronized data.

        Raises:
            requests.HTTPError: If the API request fails.
            DatasetError: If frequency is not greater than 0.
        """
        synced_dataset = self._synchronize(frequency=frequency, data_types=data_types)
        total = synced_dataset.num_demonstrations
        processed = synced_dataset.num_processed_demonstrations
        if total != processed:
            pbar = tqdm(total=total, desc="Synchronizing dataset", unit="recording")
            pbar.n = processed
            pbar.refresh()
            while processed < total:
                time.sleep(5.0)
                synced_dataset = self._synchronize(
                    frequency=frequency, data_types=data_types
                )
                new_processed = synced_dataset.num_processed_demonstrations
                if new_processed > processed:
                    pbar.update(new_processed - processed)
                    processed = new_processed
            pbar.close()
        else:
            logger.info("Dataset is already synchronized.")
        return SynchronizedDataset(
            dataset=self,
            frequency=frequency,
            data_types=data_types,
            dataset_description=synced_dataset.dataset_description,
            prefetch_videos=prefetch_videos,
        )

    def __iter__(self) -> "Dataset":
        """Initialize iterator over recordings in the dataset.

        Returns:
            Self for iteration over recordings.
        """
        self._recording_idx = 0
        return self

    def __len__(self) -> int:
        """Get the number of recordings in the dataset.

        Returns:
            Number of demonstration recordings in the dataset.
        """
        return self.num_recordings

    def __getitem__(self, idx: Union[int, slice]) -> Union[Recording, "Dataset"]:
        """Support for indexing and slicing dataset recordings.

        Args:
            idx: Integer index or slice object for accessing recordings.

        Returns:
            Recording instance for a single recording if idx is an integer,
            or a new Dataset instance if idx is a slice.

        Raises:
            IndexError: If the index is out of range.
            TypeError: If the index is not an integer or slice.
        """
        if isinstance(idx, slice):
            # Handle slice
            recordings = self.recordings[idx.start : idx.stop : idx.step]
            return Dataset(
                id=self.id,
                org_id=self.org_id,
                name=self.name,
                size_bytes=self.size_bytes,
                tags=self.tags,
                is_shared=self.is_shared,
                recordings=recordings,
                data_types=self.data_types,
            )
        else:
            # Handle single index
            if isinstance(idx, int):
                if idx < 0:  # Handle negative indices
                    idx += len(self.recordings)
                if not 0 <= idx < len(self.recordings):
                    raise IndexError("Dataset index out of range")
                return Recording(
                    dataset=self,
                    recording_id=self.recordings[idx]["id"],
                    size_bytes=self.recordings[idx]["total_bytes"],
                    robot_id=self.recordings[idx]["robot_id"],
                    instance=self.recordings[idx]["instance"],
                )
            raise TypeError(
                f"Dataset indices must be integers or slices, not {type(idx)}"
            )

    def __next__(self) -> Recording:
        """Get the next recording in the dataset iteration.

        Returns:
            Recording instance for the next recording in the dataset.

        Raises:
            StopIteration: If there are no more recordings to iterate over.
        """
        if self._recording_idx >= len(self.recordings):
            raise StopIteration

        recording = self.recordings[self._recording_idx]
        self._recording_idx += 1  # Increment counter
        return Recording(
            dataset=self,
            recording_id=recording["id"],
            size_bytes=recording["total_bytes"],
            robot_id=recording["robot_id"],
            instance=recording["instance"],
        )
