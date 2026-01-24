"""Abstract base class for managing trace registration and status updates.

This module provides common functionality for registering traces with the backend
API and updating their status during upload operations.
"""

import asyncio
import logging
from abc import ABC
from typing import Any

import aiohttp
from neuracore_types import DataType, RecordingDataTraceStatus

from neuracore.data_daemon.auth_management.auth_manager import get_auth
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
        self, recording_id: str, data_type: DataType
    ) -> str | None:
        """Register a backend DataTrace for a recording.

        Args:
            recording_id: The recording ID
            data_type: The data type being uploaded

        Returns:
            The backend trace ID, or None if registration failed
        """
        if data_type is None:
            logger.error("data_type cannot be None")
            return None

        try:
            auth = get_auth()
            org_id = await auth.get_org_id()

            async with self.client_session.post(
                f"{API_URL}/org/{org_id}/recording/{recording_id}/traces",
                json={"data_type": data_type.value},
                headers=await auth.get_headers(self.client_session),
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status >= 400:
                    error = await response.text()
                    logger.error(f"Failed to register data trace: {error}")
                    return None

                body = await response.json()
                backend_trace_id = body.get("id")

                if backend_trace_id:
                    logger.info(
                        f"Registered backend trace {backend_trace_id} "
                        f"for recording {recording_id}"
                    )
                else:
                    logger.error("No trace ID returned from backend")

                return backend_trace_id

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"Failed to register data trace: {e}")
            return None

    async def _update_data_trace(
        self,
        recording_id: str,
        backend_trace_id: str,
        status: RecordingDataTraceStatus,
        uploaded_bytes: int | None = None,
        total_bytes: int | None = None,
    ) -> bool:
        """Update the status of a backend DataTrace.

        Args:
            recording_id: The recording ID
            backend_trace_id: The backend trace ID
            status: The status of the DataTrace
            uploaded_bytes: The number of bytes uploaded so far
            total_bytes: The total number of bytes to upload

        Returns:
            True if update succeeded, False otherwise
        """
        if not backend_trace_id:
            logger.warning("No backend trace ID provided for update")
            return False

        data_trace_payload: dict[str, Any] = {"status": status}
        if uploaded_bytes is not None:
            data_trace_payload["uploaded_bytes"] = uploaded_bytes
        if total_bytes is not None:
            data_trace_payload["total_bytes"] = total_bytes

        try:
            auth = get_auth()
            org_id = await auth.get_org_id()

            async with self.client_session.put(
                f"{API_URL}/org/{org_id}/recording/{recording_id}/traces/{backend_trace_id}",
                json=data_trace_payload,
                headers=await auth.get_headers(self.client_session),
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status >= 400:
                    error = await response.text()
                    logger.warning(
                        f"Failed to update data trace: HTTP {response.status}: {error}"
                    )
                    return False

                logger.debug(
                    f"Updated backend trace {backend_trace_id} with status {status}"
                )
                return True

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Failed to update data trace: {e}")
            return False
