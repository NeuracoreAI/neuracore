import requests

from neuracore.core.auth import get_auth
from neuracore.core.const import API_URL


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
