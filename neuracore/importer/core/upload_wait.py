"""Wait for an import's recordings to finish uploading."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable

import requests

from neuracore.core.utils import backend_utils
from neuracore.importer.core.exceptions import ImporterError

UPLOAD_POLL_INTERVAL_S = 1.0
# Per pending recording, as main's per-episode stop_recording(wait=True).
UPLOAD_TIMEOUT_PER_RECORDING_S = 15 * 60.0


def wait_for_uploads(
    recording_ids: Iterable[str],
    *,
    logger: logging.Logger,
    poll_interval_s: float = UPLOAD_POLL_INTERVAL_S,
    timeout_per_recording_s: float = UPLOAD_TIMEOUT_PER_RECORDING_S,
) -> None:
    """Block until every recording is uploaded.

    Raises:
        ImporterError: If no recording completes within timeout_per_recording_s
            times the number of recordings still pending.
    """
    pending = set(recording_ids)
    if not pending:
        return
    logger.info("Waiting for %s recording(s) to finish uploading.", len(pending))
    budget_s = timeout_per_recording_s * len(pending)
    deadline = time.monotonic() + budget_s
    while True:
        uploaded = {rid for rid in pending if _is_uploaded(rid, logger)}
        if uploaded:
            pending -= uploaded
            budget_s = timeout_per_recording_s * len(pending)
            deadline = time.monotonic() + budget_s
        if not pending:
            return
        if time.monotonic() >= deadline:
            raise ImporterError(
                f"No upload completed for {budget_s:.0f}s; "
                f"{len(pending)} recording(s) pending: {sorted(pending)}"
            )
        time.sleep(poll_interval_s)


def _is_uploaded(recording_id: str, logger: logging.Logger) -> bool:
    try:
        return backend_utils.is_recording_upload_complete(recording_id)
    except requests.RequestException as error:
        logger.debug("Upload check for %s failed: %s", recording_id, error)
        return False
