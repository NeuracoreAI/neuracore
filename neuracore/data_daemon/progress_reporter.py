"""Progress report API integration."""

import asyncio
import logging
from typing import Any

import aiohttp

from neuracore.data_daemon.auth_management.auth_manager import get_auth
from neuracore.data_daemon.const import (
    API_URL,
    BACKEND_API_MAX_BACKOFF_SECONDS,
    BACKEND_API_MAX_RETRIES,
    BACKEND_API_RETRYABLE_STATUS_CODES,
)
from neuracore.data_daemon.event_emitter import Emitter, get_emitter

logger = logging.getLogger(__name__)


class ProgressReporter:
    """Send progress reports to the Neuracore backend."""

    def __init__(self, client_session: aiohttp.ClientSession) -> None:
        """Subscribe to progress report events."""
        self.client_session = client_session
        self._emitter = get_emitter()
        self._emitter.on(Emitter.PROGRESS_REPORT, self.report_progress)

    async def report_progress(
        self,
        start_time: float,
        end_time: float,
        traces: Any,
    ) -> None:
        """Post a progress report for the provided trace records."""
        if not traces:
            return

        if not traces[0].recording_id:
            logger.warning("Progress report missing recording_id; skipping request.")
            return
        trace_map: dict[str, int] = {}

        for trace in traces:
            trace_map[trace.trace_id] = trace.total_bytes

        body = {
            "recording_id": traces[0].recording_id,
            "start_time": float(start_time),
            "end_time": float(end_time),
            "robot_id": traces[0].robot_id,
            "robot_name": traces[0].robot_name,
            "instance": int(traces[0].robot_instance),
            "dataset_id": traces[0].dataset_id,
            "dataset_name": traces[0].dataset_name,
            "traces": trace_map,
        }

        auth = get_auth()
        org_id = await auth.get_org_id()
        recording_id = traces[0].recording_id
        last_error: str | None = None

        url = f"{API_URL}/org/{org_id}/recording/register-traces"

        for attempt in range(BACKEND_API_MAX_RETRIES):
            try:
                async with self.client_session.post(
                    url,
                    json=body,
                    headers=await auth.get_headers(self.client_session),
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status < 400:
                        logger.info(
                            "Progress report success for recording %s", recording_id
                        )
                        self._emitter.emit(Emitter.PROGRESS_REPORTED, recording_id)
                        return

                    error_text = await response.text()
                    last_error = f"HTTP {response.status}: {error_text}"
                    logger.warning(
                        "Progress report failed (attempt %d/%d) to %s: %s %s",
                        attempt + 1,
                        BACKEND_API_MAX_RETRIES,
                        url,
                        response.status,
                        error_text,
                    )

                    if response.status not in BACKEND_API_RETRYABLE_STATUS_CODES:
                        break  # Non-retryable error, stop immediately

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = f"{type(exc).__name__}: {exc!r}"
                logger.warning(
                    "Progress report request failed (attempt %d/%d) to %s: %s: %r",
                    attempt + 1,
                    BACKEND_API_MAX_RETRIES,
                    url,
                    type(exc).__name__,
                    exc,
                )

            if attempt < BACKEND_API_MAX_RETRIES - 1:
                delay = min(2**attempt, BACKEND_API_MAX_BACKOFF_SECONDS)
                await asyncio.sleep(delay)

        self._emitter.emit(
            Emitter.PROGRESS_REPORT_FAILED,
            recording_id,
            last_error or "Unknown error",
        )
