from __future__ import annotations
import requests
from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL

_HTTP_TIMEOUT_SECONDS = 60

def get_presigned_upload_url(*, recording_id: str, filepath: str, content_type: str) -> str:
    org_id = get_current_org()
    resp = requests.get(
        f"{API_URL}/org/{org_id}/recording/{recording_id}/resumable_upload_url",
        params={"filepath": filepath, "content_type": content_type},
        headers=get_auth().get_headers(),
        timeout=_HTTP_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    url = resp.json().get("url")
    if not url:
        raise RuntimeError("API did not return an upload URL")
    return url
