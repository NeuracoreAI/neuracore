"""Progress report API integration."""

import asyncio
import logging
from typing import Any

import aiohttp
from neuracore_types.upload.upload import TracesMetadataRequest

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
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

        recording_id = traces[0].recording_id
        if not recording_id:
            logger.warning("Progress report missing recording_id; skipping request.")
            return
        trace_map: dict[str, int] = {}

        for trace in traces:
            if trace.total_bytes is None:
                logger.warning(
                    "Progress report skipped for %s; trace %s missing total_bytes",
                    recording_id,
                    trace.trace_id,
                )
                self._emitter.emit(
                    Emitter.PROGRESS_REPORT_FAILED,
                    recording_id,
                    f"Trace {trace.trace_id} missing total_bytes",
                )
                return
            trace_map[trace.trace_id] = trace.total_bytes

        body = TracesMetadataRequest(traces=trace_map)
        total_bytes = sum(trace_map.values())

        loop = asyncio.get_running_loop()
        auth = get_auth()
        org_id = await loop.run_in_executor(None, get_current_org)
        headers = await loop.run_in_executor(None, auth.get_headers)
        recording_id = traces[0].recording_id
        last_error: str | None = None

        url = f"{API_URL}/org/{org_id}/recording/{recording_id}/traces-metadata"
        logger.info(
            "Sending progress report for recording %s: trace_count=%d total_bytes=%d "
            "start_time=%.3f end_time=%.3f",
            recording_id,
            len(trace_map),
            total_bytes,
            start_time,
            end_time,
        )

        for attempt in range(BACKEND_API_MAX_RETRIES):
            try:
                async with self.client_session.post(
                    url,
                    json=body.model_dump(),
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status < 400:
                        logger.info(
                            "Progress report sent successfully for recording %s "
                            "(trace_count=%d)",
                            recording_id,
                            len(trace_map),
                        )
                        self._emitter.emit(Emitter.PROGRESS_REPORTED, recording_id)
                        return
                    if response.status == 401:
                        logger.info("Access token expired, refreshing token")
                        await loop.run_in_executor(None, auth.login)
                        headers = await loop.run_in_executor(None, auth.get_headers)
                        continue
                    error_text = await response.text()
                    last_error = f"HTTP {response.status}: {error_text}"
                    logger.warning(
                        "Progress report failed (attempt %d/%d): %s %s",
                        attempt + 1,
                        BACKEND_API_MAX_RETRIES,
                        response.status,
                        error_text,
                    )

                    if response.status not in BACKEND_API_RETRYABLE_STATUS_CODES:
                        break

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = str(exc)
                logger.warning(
                    "Progress report request failed (attempt %d/%d): %s",
                    attempt + 1,
                    BACKEND_API_MAX_RETRIES,
                    exc,
                )

            if attempt < BACKEND_API_MAX_RETRIES - 1:
                delay = min(2**attempt, BACKEND_API_MAX_BACKOFF_SECONDS)
                await asyncio.sleep(delay)

        logger.error(
            "Progress report failed after retries for recording %s: %s",
            recording_id,
            last_error or "Unknown error",
        )
        self._emitter.emit(
            Emitter.PROGRESS_REPORT_FAILED,
            recording_id,
            last_error or "Unknown error",
        )
