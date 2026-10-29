"""Recording class for managing synchronized data streams in a dataset."""

import sys
from typing import TYPE_CHECKING, Any

import requests
from neuracore_types import Codec, CrossEmbodimentUnion, DataType
from neuracore_types import Recording as RecordingModel
from neuracore_types import (
    RecordingMetadata,
    RecordingTag,
    RecordingTagRequest,
    SynchronizationDetails,
)

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.data.synced_recording import SynchronizedRecording
from neuracore.core.exceptions import AuthenticationError, SynchronizationError
from neuracore.core.utils.embodiment_description_utils import extract_data_types
from neuracore.core.utils.http_session import thread_local_session

if TYPE_CHECKING:
    from neuracore.core.data.dataset import Dataset

_METADATA_ROUTE = "metadata"
_TAGS_ROUTE = "tags"


class Recording:
    """Class representing a recording episode in a dataset.

    This class provides methods to synchronize the recording with a specified
    frequency and data types, and to iterate over the synchronized data.
    """

    def __init__(
        self,
        dataset: "Dataset",
        recording_id: str,
        total_bytes: int,
        robot_id: str,
        instance: int,
        start_time: float,
        end_time: float,
        metadata: RecordingMetadata,
        data_types: set[DataType] | None = None,
        encoding: Codec | None = None,
        sensor_manifest: dict[DataType, list[str]] | None = None,
        deleted: bool = False,
    ):
        """Initialize episode iterator for a specific recording.

        Args:
            dataset: Parent Dataset instance.
            recording_id: Unique identifier for the recording episode.
            total_bytes: Size of the recording episode in bytes.
            robot_id: The robot that created this recording.
            instance: The instance of the robot that created this recording.
            start_time: Unix timestamp when recording started.
            end_time: Unix timestamp when recording ended.
            metadata: Metadata associated with the recording.
            data_types: Set of DataTypes present in this recording.
            encoding: Codec used for this recording's RGB camera video. None
                when there is nothing to report -- no RGB cameras, cameras
                that disagreed on a codec, or a recording predating the field.
            sensor_manifest: Sensor names recorded for each data type,
                captured at finalize. Empty for recordings finalized before
                this field existed.
            deleted: Whether the recording has been deleted.
        """
        self.dataset = dataset
        self.id = recording_id
        self.total_bytes = total_bytes
        self.robot_id = robot_id
        self.instance = instance
        self.start_time = start_time
        self.end_time = end_time
        self.name = metadata.name
        self.metadata = metadata
        self.data_types: set[DataType] = data_types or set()
        self.encoding = encoding
        self.sensor_manifest: dict[DataType, list[str]] = sensor_manifest or {}
        self.deleted = deleted
        self._raw = {
            "id": recording_id,
            "total_bytes": total_bytes,
            "robot_id": robot_id,
            "instance": instance,
        }

    def __getitem__(self, key: str) -> object:
        """Support old dict-style access dynamically."""
        try:
            return self._raw[key]
        except KeyError:
            raise KeyError(f"Recording has no key '{key}'")

    def download(self, filepath: str) -> bytes:
        """Download an original file without sending API credentials to storage."""
        url = f"{API_URL}/org/{self.dataset.org_id}/recording/{self.id}"
        response = thread_local_session().get(
            f"{url}/download_url",
            params={"filepath": filepath},
            headers=get_auth().get_headers(),
            timeout=(10, 60),
        )
        response.raise_for_status()
        # Use a separate unauthenticated request for signed storage URLs.
        with requests.get(response.json()["url"], timeout=(10, 300)) as download:
            download.raise_for_status()
            return download.content

    def synchronize(
        self,
        frequency: int = 0,
        cross_embodiment_union: CrossEmbodimentUnion | None = None,
        max_delay_s: float = sys.float_info.max,
        allow_duplicates: bool = True,
        trim_start_end: bool = True,
        trim_no_movement_at_start_threshold: float | None = None,
    ) -> SynchronizedRecording:
        """Synchronize the episode with specified frequency and data types.

        Args:
            frequency: Frequency at which to synchronize the episode.
                Use 0 for aperiodic data.
            cross_embodiment_union: Dict specifying data types and their
                names to include in synchronization. If None, will use all
                available data types from the dataset.
            max_delay_s: Maximum allowed delay for synchronization.
            allow_duplicates: Whether duplicate points are allowed when syncing.
            trim_start_end: Whether to trim start/end during synchronization.
            trim_no_movement_at_start_threshold: If set, drop frames at the
                start of the episode while every joint position stays within
                this threshold of its value in the first frame. None keeps
                every frame.

        Raises:
            SynchronizationError: If synchronization fails.
        """
        if frequency < 0:
            raise SynchronizationError("Frequency must be >= 0")

        data_types = None
        if cross_embodiment_union is not None:
            data_types = extract_data_types(cross_embodiment_union)

        # check valid data types if provided
        if data_types is not None:
            if not all(isinstance(data_type, DataType) for data_type in data_types):
                raise ValueError(
                    "Invalid data types provided. "
                    "All items must be DataType enum values."
                )
            if not set(data_types).issubset(set(self.dataset.data_types)):
                raise SynchronizationError(
                    "Invalid data type requested for synchronization"
                )

        return SynchronizedRecording(
            dataset=self.dataset,
            recording_id=self.id,
            recording_name=self.name,
            robot_id=self.robot_id,
            instance=self.instance,
            synchronization_details=SynchronizationDetails(
                frequency=frequency,
                cross_embodiment_union=cross_embodiment_union,
                max_delay_s=max_delay_s,
                allow_duplicates=allow_duplicates,
                trim_start_end=trim_start_end,
                trim_no_movement_at_start_threshold=(
                    trim_no_movement_at_start_threshold
                ),
            ),
        )

    def __iter__(self) -> None:
        """Initialize iterator over synchronized recording data.

        Raises:
            RuntimeError: Always raised to indicate that this method is not
            supported for unsynchronized recordings.
        """
        raise RuntimeError(
            "Only synchronized recordings can be iterated over. "
            "Use the synchronize method to create a synchronized recording."
        )

    def _refresh_recording_metadata(self) -> None:
        """Refresh the recording metadata as it is stored in this dataset.

        A recording's name, notes and tags are kept per dataset, so they are
        read from the recording's entry in `self.dataset`.

        Ideally we should implement If-Match & Etag header as well.
        """
        auth = get_auth()
        org_id = get_current_org()
        try:
            session = thread_local_session()
            response = session.get(
                f"{API_URL}/org/{org_id}/datasets/{self.dataset.id}"
                f"/recording/{self.id}",
                headers=auth.get_headers(),
                params={"is_shared": self.dataset.is_shared},
            )
            response.raise_for_status()
            updated_recording = RecordingModel.model_validate(response.json())
            self.metadata = updated_recording.metadata
            self.name = self.metadata.name
        except AuthenticationError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error fetching recording metadata: {e}")

    def _change_dataset_entry(
        self, method: str, route: str, **request_kwargs: Any
    ) -> None:
        """Send a change to this recording's entry in its dataset.

        The API answers with the changed entry, so its metadata replaces the
        local metadata.

        Args:
            method: HTTP method of the request.
            route: Route below the dataset entry, e.g. `metadata`.
            **request_kwargs: Body or query parameters of the request.

        Raises:
            RuntimeError: If there is an error updating the entry in the cloud.
            ConfigError: If there is an error trying to get the config
            AuthenticationError: If there is an error with authentication
        """
        auth = get_auth()
        org_id = get_current_org()
        try:
            session = thread_local_session()
            response = session.request(
                method,
                f"{API_URL}/org/{org_id}/recording/{self.dataset.id}"
                f"/{self.id}/{route}",
                headers=auth.get_headers(),
                **request_kwargs,
            )
            response.raise_for_status()
            self.metadata = RecordingMetadata.model_validate(
                response.json()["metadata"]
            )
            self.name = self.metadata.name
        except AuthenticationError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error updating recording metadata: {e}")

    @property
    def tags(self) -> list[RecordingTag]:
        """Tags of this recording in its dataset, as last read from the cloud."""
        return list(self.metadata.tags)

    def add_tag(self, name: str) -> None:
        """Add a user tag to this recording in its dataset.

        This call is blocking. The name is normalized with
        `neuracore_types.normalize_user_tag_name`. Adding a tag that the
        recording already has changes nothing.

        Args:
            name: Name of the tag.

        Raises:
            ValueError: If the name is not a valid tag name.
            RuntimeError: If there is an error updating the tags in the cloud.
            ConfigError: If there is an error trying to get the config
            AuthenticationError: If there is an error with authentication
        """
        tag = RecordingTagRequest(name=name)
        self._change_dataset_entry("POST", _TAGS_ROUTE, json=tag.model_dump())

    def remove_tag(self, name: str) -> None:
        """Remove a user tag from this recording in its dataset.

        This call is blocking. The name is normalized as in `add_tag`.
        Removing a tag that the recording does not have changes nothing.
        QA tags cannot be removed.

        Args:
            name: Name of the tag.

        Raises:
            ValueError: If the name is not a valid tag name.
            RuntimeError: If there is an error updating the tags in the cloud.
            ConfigError: If there is an error trying to get the config
            AuthenticationError: If there is an error with authentication
        """
        tag = RecordingTagRequest(name=name)
        self._change_dataset_entry("DELETE", _TAGS_ROUTE, params=tag.model_dump())

    def set_notes(self, notes: str) -> None:
        """Update a recordings notes programmatically.

        This call is blocking.

        Args:
            notes: The notes to update the recording with.

        Raises:
            RuntimeError: If there is an error updating the metadata in the cloud.
            ConfigError: If there is an error trying to get the config
            AuthenticationError: If there is an error with authentication
        """
        self._refresh_recording_metadata()
        self._change_dataset_entry(
            "PUT", _METADATA_ROUTE, json={"name": self.name, "notes": notes}
        )
