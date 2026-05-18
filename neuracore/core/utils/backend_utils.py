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
from neuracore.core.utils.http_session import Session


class RecordingNotFoundError(LookupError):
    """Raised when the backend has no record of the requested recording.

    Previously a 404 on /traces/active was silently coerced to an empty list,
    which is indistinguishable from a recording that has legitimately drained
    to zero active traces. That ambiguity caused stop_recording(wait=True) to
    either terminate prematurely or hang forever depending on which state the
    poller was in when the 404 landed. Callers should now branch on this
    exception explicitly.
    """


def get_recording_processing_status(recording_id: str) -> dict:
    """Return the backend's view of a recording's processing state.

    There are two distinct success milestones the client previously could
    not distinguish:

    1. **Uploaded** — the daemon successfully transferred all blobs and the
       backend accepted them. The local state DB marks
       ``progress_reported='reported'`` and ``stop_recording(wait=True)``
       returns. This is what the SDK historically reports as "done".

    2. **Viewable** — server-side processing/transcoding has completed and
       the dashboard can play the video back. There is no client-visible
       signal of this today; "video loads forever" is the symptom.

    This helper queries the backend for whichever processing-status fields
    the server publishes (``status``, ``processing_status``, ``viewable``,
    ``urdf_present`` are all returned when present). Unknown fields are
    surfaced verbatim so callers can branch on whatever the API exposes.

    Args:
        recording_id: Unique identifier for the recording.

    Returns:
        A dict mirroring the backend response. At minimum contains a
        ``recording_id`` key; ``status``/``processing_status``/``viewable``
        are populated when the backend reports them.

    Raises:
        RecordingNotFoundError: If the backend returns 404 for the recording.
        requests.HTTPError: For any other non-success status code.
        ConfigError: If the current organization cannot be resolved.
    """
    org_id = get_current_org()
    with Session() as session:
        response = session.get(
            f"{API_URL}/org/{org_id}/recording/{recording_id}",
            headers=get_auth().get_headers(),
        )
    if response.status_code == 404:
        raise RecordingNotFoundError(
            f"Recording {recording_id!r} not found in org {org_id!r}."
        )
    response.raise_for_status()
    payload = response.json() or {}
    if not isinstance(payload, dict):
        return {"recording_id": recording_id, "raw": payload}
    payload.setdefault("recording_id", recording_id)
    return payload


def get_active_data_traces(recording_id: str) -> list[RecordingDataTrace]:
    """Get all active data traces for a recording.

    Args:
        recording_id: Unique identifier for the recording to check.

    Returns:
        A list of `RecordingDataTrace` instances representing the active
        data traces for the recording. Returns an empty list if the recording
        exists but currently has no active traces.

    Raises:
        RecordingNotFoundError: If the backend returns 404 for the recording
            (recording was never created server-side, was deleted, or routed
            to a different org).
        requests.HTTPError: If the API request fails for any other reason.
        ValueError: If the response has an unexpected format.
        ConfigError: If there is an error trying to get the current org.
    """
    org_id = get_current_org()
    with Session() as session:
        response = session.get(
            f"{API_URL}/org/{org_id}/recording/{recording_id}/traces/active",
            headers=get_auth().get_headers(),
        )
    if response.status_code == 404:
        raise RecordingNotFoundError(
            f"Recording {recording_id!r} not found in org {org_id!r}. "
            "It may have been deleted, never registered, or belong to a "
            "different organization than the one the client is using."
        )
    response.raise_for_status()
    data = response.json() or []
    return [RecordingDataTrace.model_validate(item) for item in data]


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
