"""Recording class for managing synchronized data streams in a dataset."""

from typing import TYPE_CHECKING

import requests
from neuracore_types import DataSpec, DataType
from neuracore_types import Recording as RecordingModel
from neuracore_types import RecordingMetadata, RecordingStatus, RobotDataSpec

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.data.synced_recording import SynchronizedRecording
from neuracore.core.exceptions import AuthenticationError, SynchronizationError
from neuracore.core.utils.robot_data_spec_utils import extract_data_types

if TYPE_CHECKING:
    from neuracore.core.data.dataset import Dataset


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
        """
        self.dataset = dataset
        self.id = recording_id
        self.total_bytes = total_bytes
        self.robot_id = robot_id
        self.instance = instance
        self.start_time = start_time
        self.end_time = end_time
        # Store human-friendly recording name when available.
        self.name = getattr(metadata, "name", None) or recording_id
        self.metadata = metadata
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

    def synchronize(
        self,
        frequency: int = 0,
        data_spec: DataSpec | None = None,
        robot_data_spec: RobotDataSpec | None = None,
    ) -> SynchronizedRecording:
        """Synchronize the episode with specified frequency and data types.

        Args:
            frequency: Frequency at which to synchronize the episode.
                Use 0 for aperiodic data.
            data_spec: Dict specifying data types and their names to include
                in synchronization. If None, will use all available data
                types from the dataset.
            robot_data_spec: Dict specifying robot id to
                data types and their names to include in synchronization.
                If None, will use all available data types from the dataset.

        Raises:
            SynchronizationError: If synchronization fails.
        """
        if frequency < 0:
            raise SynchronizationError("Frequency must be >= 0")

        if data_spec is not None:
            # Special case for backend.
            # TODO: Find a better way to handle this.
            robot_data_spec = {
                "": data_spec,
            }

        data_types = extract_data_types(robot_data_spec) if robot_data_spec else None
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
            frequency=frequency,
            robot_data_spec=robot_data_spec,
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
        """Refresh the recording metadata from the cloud.

        Ideally we should implement If-Match & Etag header as well.
        """
        auth = get_auth()
        org_id = get_current_org()
        try:
            response = requests.get(
                f"{API_URL}/org/{org_id}/recording/{self.id}",
                headers=auth.get_headers(),
            )
            response.raise_for_status()
            updated_recording = RecordingModel.model_validate(response.json())
            self.metadata = updated_recording.metadata
        except AuthenticationError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error fetching recording metadata: {e}")

    def _update_metadata(self, recording_metadata: RecordingMetadata) -> None:
        """Update the metadata of a recording.

        Args:
            recording_metadata: The metadata to update the recording with

        Raises:
            RuntimeError: Rases if there is an error updating the metadata in the cloud.
            ConfigError: If there is an error trying to get the config
            AuthenticationError: If there is an error with authentication
        """
        auth = get_auth()
        org_id = get_current_org()
        try:

            response = requests.put(
                f"{API_URL}/org/{org_id}/recording/{self.id}/metadata",
                headers=auth.get_headers(),
                json=recording_metadata.model_dump(mode="json"),
            )
            response.raise_for_status()
            updated_recording = RecordingModel.model_validate(response.json())
            self.metadata = updated_recording.metadata
        except AuthenticationError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error updating recording metadata: {e}")

    def set_status(self, status: RecordingStatus) -> None:
        """Mark a recording with a specific status programmatically.

        This call is blocking.

        Args:
            status: The status to set the recording to e.g. `RecordingStatus.FLAGGED`

        Raises:
            RuntimeError: Rases if there is an error updating the metadata in the cloud.
            ConfigError: If there is an error trying to get the config
            AuthenticationError: If there is an error with authentication
        """
        self._refresh_recording_metadata()
        new_metadata = self.metadata.model_copy()
        new_metadata.status = status
        self._update_metadata(new_metadata)

    def set_notes(self, notes: str) -> None:
        """Update a recordings notes programmatically.

        This call is blocking.

        Args:
            notes: The notes to update the recording with.

        Raises:
            RuntimeError: Rases if there is an error updating the metadata in the cloud.
            ConfigError: If there is an error trying to get the config
            AuthenticationError: If there is an error with authentication
        """
        self._refresh_recording_metadata()
        new_metadata = self.metadata.model_copy()
        new_metadata.notes = notes
        self._update_metadata(new_metadata)
