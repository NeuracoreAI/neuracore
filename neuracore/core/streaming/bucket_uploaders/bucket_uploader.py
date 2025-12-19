"""Abstract base class for uploading recording data to cloud storage buckets.

This module provides the foundation for implementing bucket uploaders that handle
recording data traces and track active trace counts via API calls.
"""

import threading
from abc import ABC, abstractmethod
from typing import Any

import requests
from neuracore_types import DataType, RecordingDataTraceStatus

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.streaming.recording_state_manager import get_recording_state_manager

TRACE_FILE = "trace.json"


class BucketUploader(ABC):
    """Abstract base class for uploading recording data to cloud storage buckets.

    This class provides common functionality for managing recording uploads,
    including tracking the number of active traces and communicating with
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

    def _register_data_trace(self, data_type: DataType) -> str:
        """Register a backend DataTrace for this recording.

        Returns:
            The trace id from the backend
        """
        if data_type is None:
            raise ValueError("data_type cannot be None")

        if self._recording_manager.is_recording_expired(self.recording_id):
            raise ValueError(f"Recording {self.recording_id} is expired")

        org_id = get_current_org()
        try:
            response = requests.post(
                f"{API_URL}/org/{org_id}/recording/{self.recording_id}/traces",
                json={"data_type": data_type.value},
                headers=get_auth().get_headers(),
            )
            response.raise_for_status()
            body = response.json()
            return body.get("id")
        except requests.exceptions.RequestException:
            raise RuntimeError("Failed to register data trace")

    def _update_data_trace(
        self,
        trace_id: str,
        status: RecordingDataTraceStatus,
        uploaded_bytes: int | None = None,
        total_bytes: int | None = None,
    ) -> None:
        """Update the status of a backend DataTrace.

        Args:
            trace_id: The id of the DataTrace to update.
            status: The status of the DataTrace.
            uploaded_bytes: The number of bytes uploaded so far.
            total_bytes: The total number of bytes to upload.
        """
        if not trace_id:
            return

        if self._recording_manager.is_recording_expired(self.recording_id):
            return

        data_trace_payload: dict[str, Any] = {"status": status}
        if uploaded_bytes is not None:
            data_trace_payload["uploaded_bytes"] = uploaded_bytes
        if total_bytes is not None:
            data_trace_payload["total_bytes"] = total_bytes

        org_id = get_current_org()
        try:
            requests.put(
                f"{API_URL}/org/{org_id}/recording/{self.recording_id}/traces/{trace_id}",
                json=data_trace_payload,
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
