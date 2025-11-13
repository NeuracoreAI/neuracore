"""Helpers for fetching the current user's organization IDs from the API."""

from __future__ import annotations

from typing import Optional, Set

import requests

from neuracore.core.const import API_URL


def fetch_org_ids(access_token: str) -> Optional[Set[str]]:
    """Return the set of org IDs for the authenticated user."""
    try:
        response = requests.get(
            f"{API_URL}/org-management/my-orgs",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        if response.status_code != 200:
            return None

        payload = response.json()
        if not isinstance(payload, list):
            return None

        org_ids: Set[str] = set()
        for membership in payload:
            org_obj = (membership or {}).get("org") or {}
            org_id_value = org_obj.get("id")
            if isinstance(org_id_value, str):
                org_ids.add(org_id_value)
        return org_ids
    except Exception:
        return None
