"""HTTP error helpers for extracting backend error details."""

from __future__ import annotations

from typing import Any

import requests


def _error_detail(response: requests.Response) -> Any:
    """Return the detail of a JSON error response, or the whole body if it has none.

    Raises:
        ValueError: If the body is not JSON.
    """
    payload: Any = response.json()
    if isinstance(payload, dict):
        return payload.get("detail", payload)
    return payload


def extract_error_detail(response: requests.Response) -> str | None:
    """Extract error detail from an HTTP error response."""
    try:
        detail = _error_detail(response)
    except ValueError:
        return response.text
    return detail.get("error") if isinstance(detail, dict) else str(detail)


def extract_error_code(response: requests.Response) -> str | None:
    """Extract the machine readable error code from an HTTP error response."""
    try:
        detail = _error_detail(response)
    except ValueError:
        return None
    return detail.get("error_code") if isinstance(detail, dict) else None
