import threading
from abc import ABC, abstractmethod

import requests

from neuracore.core.auth import get_auth
from neuracore.core.const import API_URL


class BucketUploader(ABC):
    """Bucket uploader."""

    def __init__(
        self,
        recording_id: str,
    ):
        """Init.

        Args:
            recording_id: Recording ID
        """
        self.recording_id = recording_id

    def _update_num_active_streams(self, delta: int) -> None:
        """Increment or decrement the number of streams.

        Args:
            inc: True to increment, False to decrement
        """

        assert delta in (1, -1), "Value must be 1 or -1"
        response = requests.put(
            f"{API_URL}/recording/{self.recording_id}/update_num_active_streams",
            params={
                "delta": delta,
            },
            headers=get_auth().get_headers(),
        )
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError("Failed to update number of active streams")

    @abstractmethod
    def finish(self) -> threading.Thread:
        pass
