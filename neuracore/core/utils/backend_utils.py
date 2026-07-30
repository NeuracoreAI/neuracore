"""Backend utility functions for Neuracore recording and dataset management.

This module provides utility functions for interacting with the Neuracore backend,
including monitoring active data traces and generating unique identifiers for
synchronized datasets.
"""

import base64
import hashlib

from neuracore_types import DataType, RecordingDataTrace

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.utils.http_session import thread_local_session


def get_active_data_traces(recording_id: str) -> list[RecordingDataTrace]:
    """Get all active data traces for a recording.

    Args:
        recording_id: Unique identifier for the recording to check.

    Returns:
        A list of `RecordingDataTrace` instances representing the active
        data traces for the recording. Returns an empty list if no active
        traces are found.

    Raises:
        requests.HTTPError: If the API request fails.
        ValueError: If the response has an unexpected format.
        ConfigError: If there is an error trying to get the current org.
        requests.RequestException: If the API request fails or times out.
    """
    org_id = get_current_org()
    session = thread_local_session()
    response = session.get(
        f"{API_URL}/org/{org_id}/recording/{recording_id}/traces/active",
        headers=get_auth().get_headers(),
        timeout=10,
    )
    if response.status_code == 404:
        return []
    response.raise_for_status()
    data = response.json() or []
    return [RecordingDataTrace.model_validate(item) for item in data]


def is_recording_upload_complete(recording_id: str) -> bool:
    """Check whether all expected traces for a recording are uploaded.

    Args:
        recording_id: Unique identifier of the recording to check.

    Returns:
        True when all expected traces are uploaded, otherwise False.

    Raises:
        requests.HTTPError: If the API returns an unsuccessful response.
        requests.RequestException: If the request fails or times out.
        TypeError: If the response is not a boolean.
    """
    org_id = get_current_org()
    session = thread_local_session(retry_read_timeout=True)
    response = session.get(
        f"{API_URL}/org/{org_id}/recording/{recording_id}/traces/complete",
        headers=get_auth().get_headers(),
        timeout=3,
    )
    response.raise_for_status()

    data = response.json()
    if not isinstance(data, bool):
        raise TypeError("Expected a boolean recording upload-completion response")

    return data


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
