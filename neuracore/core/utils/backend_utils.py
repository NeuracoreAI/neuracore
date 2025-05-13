import base64
import hashlib

import requests

from neuracore.core.auth import get_auth
from neuracore.core.const import API_URL
from neuracore.core.nc_types import DataType


# TODO: Receive num active stream updates from the server with the recording
# state rather than polling
def get_num_active_streams(recording_id: str) -> int:
    """Get the number of active streams for a recording.

    Args:
        recording_id: Recording ID
    """
    response = requests.get(
        f"{API_URL}/recording/{recording_id}/get_num_active_streams",
        headers=get_auth().get_headers(),
    )
    response.raise_for_status()
    if response.status_code != 200:
        raise ValueError("Failed to update number of active streams")
    return int(response.json()["num_active_streams"])


def synced_dataset_key(sync_freq: int, data_types: list[DataType]) -> str:
    """Generate a unique key for a synced dataset."""
    names = [data_type.value for data_type in data_types]
    names.sort()
    long_name = "".join([str(sync_freq)] + names).encode()
    return (
        base64.urlsafe_b64encode(hashlib.md5(long_name).digest()).decode().rstrip("=")
    )
