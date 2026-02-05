"""Abstract base class for managing trace registration and status updates.

This module provides common functionality for registering traces with the backend
API and updating their status during upload operations.
"""

import asyncio
import logging
from abc import ABC
from typing import Any
from uuid import UUID

import aiohttp
from neuracore_types import DataType, RecordingDataTraceStatus

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.data_daemon.const import API_URL

logger = logging.getLogger(__name__)


class TraceManager(ABC):
    """Abstract base class for managing recording data traces."""

    def __init__(self, client_session: aiohttp.ClientSession):
        """Initialize the trace manager.

        Args:
            client_session: Shared aiohttp session for HTTP requests.
        """
        self.client_session = client_session

    async def _register_data_trace(
        self, recording_id: str, data_type: DataType, trace_id: UUID
    ) -> bool:
        """Register a backend DataTrace for a recording.

        Args:
            recording_id: The recording ID
            data_type: The data type being uploaded
            trace_id: The trace ID to register with the backend

        Returns:
            True if registration succeeded, False otherwise
        """
        if data_type is None:
            logger.error("data_type cannot be None")
            return False

        try:
            loop = asyncio.get_running_loop()
            auth = get_auth()
            auth.login()
            org_id = get_current_org()
            headers = auth.get_headers()

            for attempt in range(2):
                async with self.client_session.post(
                    f"{API_URL}/org/{org_id}/recording/{recording_id}/traces",
                    json={"data_type": data_type.value, "trace_id": str(trace_id)},
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 401 and attempt == 0:
                        logger.info("Access token expired, refreshing token")
                        await loop.run_in_executor(None, auth.login)
                        continue

                    if response.status >= 400:
                        error = await response.text()
                        logger.error(f"Failed to register data trace: {error}")
                        return False

                    logger.info(
                        f"Registered trace {trace_id} for recording {recording_id}"
                    )
                    return True

            return False

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"Failed to register data trace: {e}")
            return False

    async def _update_data_trace(
        self,
        recording_id: str,
        trace_id: str,
        status: RecordingDataTraceStatus,
        uploaded_bytes: int | None = None,
        total_bytes: int | None = None,
    ) -> bool:
        """Update the status of a backend DataTrace.

        Args:
            recording_id: The recording ID
            trace_id: The trace ID
            status: The status of the DataTrace
            uploaded_bytes: The number of bytes uploaded so far
            total_bytes: The total number of bytes to upload

        Returns:
            True if update succeeded, False otherwise
        """
        if not trace_id:
            logger.warning("No trace ID provided for update")
            return False

        data_trace_payload: dict[str, Any] = {"status": status}
        if uploaded_bytes is not None:
            data_trace_payload["uploaded_bytes"] = uploaded_bytes
        if total_bytes is not None:
            data_trace_payload["total_bytes"] = total_bytes

        try:
            loop = asyncio.get_running_loop()
            auth = get_auth()
            org_id = get_current_org()

            for attempt in range(2):
                headers = auth.get_headers()

                async with self.client_session.put(
                    f"{API_URL}/org/{org_id}/recording/{recording_id}/traces/{trace_id}",
                    json=data_trace_payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 401 and attempt == 0:
                        logger.info("Access token expired, refreshing token")
                        await loop.run_in_executor(None, auth.login)
                        continue

                    if response.status >= 400:
                        error = await response.text()
                        logger.warning(
                            f"Failed to update data trace: "
                            f"HTTP {response.status}: {error}"
                        )
                        return False

                    logger.debug(f"Updated trace {trace_id} with status {status}")
                    return True

            return False

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Failed to update data trace: {e}")
            return False
