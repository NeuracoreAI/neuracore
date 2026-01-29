"""HTTP error helpers for extracting backend error details."""

from __future__ import annotations

from typing import Any

import requests


def extract_error_detail(response: requests.Response) -> str | None:
    """Extract error detail from an HTTP error response."""
    try:
        payload: Any = response.json()
    except ValueError:
        return response.text

    if not isinstance(payload, dict):
        return str(payload)

    detail_payload = payload.get("detail", payload)
    if not isinstance(detail_payload, dict):
        return str(detail_payload)

    return detail_payload.get("exception") or detail_payload.get("error")
