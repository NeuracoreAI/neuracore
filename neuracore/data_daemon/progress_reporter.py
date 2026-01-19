"""Progress report API integration."""

import logging
import os
from typing import Any

import requests

from neuracore.data_daemon.auth_management.auth_manager import get_auth
from neuracore.data_daemon.event_emitter import Emitter, emitter

# from neuracore.data_daemon.models import TraceRecord

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("NEURACORE_BASE_API_URL", "https://api.neuracore.app/api")


class ProgressReporter:
    """Send progress reports to the Neuracore backend."""

    def __init__(self) -> None:
        """Subscribe to progress report events."""
        emitter.on(Emitter.PROGRESS_REPORT, self.report_progress)

    def report_progress(
        self,
        start_time: float,
        end_time: float,
        # traces: list[TraceRecord]
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
        org_id = auth.get_org_id()

        try:
            response = requests.post(
                f"{BASE_URL}/{org_id}/recording/register-traces",
                json=body,
                headers=auth.get_headers(),
                timeout=10,
            )
            if response.status_code >= 400:
                logger.warning(
                    "Progress report failed: %s %s",
                    response.status_code,
                    response.text,
                )
        except requests.exceptions.RequestException as exc:
            logger.warning("Progress report request failed: %s", exc)
            emitter.emit(
                Emitter.PROGRESS_REPORT_FAILED,
                traces[0].recording_id,
                exc.response.reason if exc.response else str(exc),
            )
        emitter.emit(Emitter.PROGRESS_REPORTED, traces[0].recording_id)
