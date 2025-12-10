"""Abstract base class for uploading recording data to cloud storage buckets.

This module provides the foundation for implementing bucket uploaders that handle
recording data streams and track active stream counts via API calls.
"""

import threading
from abc import ABC, abstractmethod

import requests
from neuracore_types import DataType, RecordingDataStreamStatus

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.streaming.recording_state_manager import get_recording_state_manager

TRACE_FILE = "trace.json"


class BucketUploader(ABC):
    """Abstract base class for uploading recording data to cloud storage buckets.

    This class provides common functionality for managing recording uploads,
    including tracking the number of active streams and communicating with
    the recording API. Concrete implementations must define the finish method
    to handle the actual upload completion logic.
    """

    def __init__(
        self,
        recording_id: str,
    ):
        """Initialize the bucket uploader.

        Args:
            recording_id: Unique identifier for the recording being uploaded.
        """
        self.recording_id = recording_id
        self._recording_manager = get_recording_state_manager()

    def _register_data_stream(self, data_type: DataType) -> str:
        """Register a backend DataStream for this recording.

        Returns:
            The stream id from the backend
        """
        if data_type is None:
            raise ValueError("data_type cannot be None")

        if self._recording_manager.is_recording_expired(self.recording_id):
            raise ValueError(f"Recording {self.recording_id} is expired")

        org_id = get_current_org()
        try:
            response = requests.post(
                f"{API_URL}/org/{org_id}/recording/{self.recording_id}/streams",
                json={"data_type": data_type.value},
                headers=get_auth().get_headers(),
            )
            response.raise_for_status()
            body = response.json()
            return body.get("id")
        except requests.exceptions.RequestException as e:
            raise RuntimeError("Failed to register data stream: ", e)

    def _mark_data_stream_complete(self, stream_id: str) -> None:
        """Mark a DataStream as fully uploaded."""
        if not stream_id:
            return

        if self._recording_manager.is_recording_expired(self.recording_id):
            return

        org_id = get_current_org()
        try:
            requests.put(
                f"{API_URL}/org/{org_id}/recording/{self.recording_id}/streams/{stream_id}",
                json={
                    "status": RecordingDataStreamStatus.UPLOAD_COMPLETE,
                    "upload_progress": 100,
                },
                headers=get_auth().get_headers(),
            )
        except requests.exceptions.RequestException:
            pass

    @abstractmethod
    def finish(self) -> threading.Thread:
        """Complete the upload process and return a thread for async execution.

        This method must be implemented by concrete subclasses to define
        the specific upload completion logic. It should return a thread
        that can be used to perform the upload operation asynchronously.

        Returns:
            A thread object that will execute the upload completion logic.
        """
        pass
