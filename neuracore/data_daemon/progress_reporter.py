"""Progress report API integration."""

import asyncio
import logging
from typing import Any

import aiohttp

from neuracore.data_daemon.auth_management.auth_manager import get_auth
from neuracore.data_daemon.const import API_URL
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

        try:
            async with self.client_session.post(
                f"{API_URL}/{org_id}/recording/register-traces",
                json=body,
                headers=await auth.get_headers(self.client_session),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status >= 400:
                    logger.warning(
                        "Progress report failed: %s %s",
                        response.status,
                        response.text,
                    )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning("Progress report request failed: %s", exc)
            self._emitter.emit(
                Emitter.PROGRESS_REPORT_FAILED,
                traces[0].recording_id,
                str(exc),
            )
        self._emitter.emit(Emitter.PROGRESS_REPORTED, traces[0].recording_id)
