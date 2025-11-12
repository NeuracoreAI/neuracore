"""Backend utility functions for Neuracore recording and dataset management.

This module provides utility functions for interacting with the Neuracore backend,
including monitoring active streams and generating unique identifiers for
synchronized datasets.
"""

import base64
import hashlib

import requests
from neuracore_types import DataType

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL


# TODO: Receive num active stream updates from the server with the recording
# state rather than polling
def get_num_active_streams(recording_id: str) -> int:
    """Get the number of active streams for a recording.

    Queries the backend to determine how many data streams are currently
    active for a specific recording. This is used to monitor upload progress
    and determine when all streams have completed processing.

    Args:
        recording_id: Unique identifier for the recording to check.

    Returns:
        The number of streams currently active for the recording.

    Raises:
        requests.HTTPError: If the API request fails.
        ValueError: If the response indicates an error or has an unexpected format.
        ConfigError: If there is an error trying to get the current org
    """
    org_id = get_current_org()
    response = requests.get(
        f"{API_URL}/org/{org_id}/recording/{recording_id}/get_num_active_streams",
        headers=get_auth().get_headers(),
    )
    response.raise_for_status()
    if response.status_code != 200:
        raise ValueError("Failed to update number of active streams")
    return int(response.json()["num_active_streams"])


def synced_dataset_key(sync_freq: int, data_types: list[DataType]) -> str:
    """Generate a unique key for a synced dataset configuration.

    Creates a deterministic identifier based on synchronization frequency
    and data types. This key is used to identify datasets that share the
    same synchronization parameters, enabling efficient data organization
    and retrieval.

    Args:
        sync_freq: Synchronization frequency in Hz for the dataset.
        data_types: List of data types included in the synchronized dataset.

    Returns:
        A URL-safe base64-encoded hash that uniquely identifies the
        synchronization configuration.
    """
    names = [data_type.value for data_type in data_types]
    names.sort()
    long_name = "".join([str(sync_freq)] + names).encode()
    return (
        base64.urlsafe_b64encode(hashlib.md5(long_name).digest()).decode().rstrip("=")
    )
