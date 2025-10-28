"""Abstract base class for uploading recording data to cloud storage buckets.

This module provides the foundation for implementing bucket uploaders that handle
recording data streams and track active stream counts via API calls.
"""

import threading
from abc import ABC, abstractmethod

import requests

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.streaming.recording_state_manager import get_recording_state_manager


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

    def _update_num_active_streams(self, delta: int) -> None:
        """Update the number of active streams for this recording.

        Makes an API call to increment or decrement the active stream count
        for the recording. This is used to track how many streams are currently
        being processed or uploaded.

        Args:
            delta: Change in stream count. Must be 1 (increment) or -1 (decrement).

        Raises:
            AssertionError: If delta is not 1 or -1.
            requests.HTTPError: If the API request fails.
            ValueError: If the response status code is not 200.
        """
        assert delta in (1, -1), "Value must be 1 or -1"
        org_id = get_current_org()
        if self._recording_manager.is_recording_expired(self.recording_id):
            return
        try:
            requests.put(
                f"{API_URL}/org/{org_id}/recording/{self.recording_id}/update_num_active_streams",
                params={
                    "delta": delta,
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
