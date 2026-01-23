"""Connection manager for network monitoring.

This module provides a connection manager that monitors network connectivity
to the Neuracore API and emits events when connection state changes.
"""

import asyncio
import logging

import aiohttp

from neuracore.data_daemon.const import API_URL
from neuracore.data_daemon.event_emitter import Emitter, get_emitter

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages network connectivity checks and emits connection state events.

    Runs a background thread that periodically checks connectivity and emits
    IS_CONNECTED events to the state manager when connection state changes.
    """

    def __init__(
        self,
        client_session: aiohttp.ClientSession,
        timeout: float = 5.0,
        check_interval: float = 10.0,
    ) -> None:
        """Initialize the connection manager.

        Args:
            client_session: aiohttp ClientSession for making requests
            emitter: Event emitter for broadcasting connection state
            timeout: Timeout in seconds for connectivity checks
            check_interval: Seconds between connectivity checks
        """
        self.client_session = client_session
        self._timeout = timeout
        self._check_interval = check_interval
        self._is_connected = False
        self._stopped = False
        self._connection_task: asyncio.Task | None = None

        self._emitter = get_emitter()
        self._emitter.emit(Emitter.IS_CONNECTED, self._is_connected)

    async def start(self) -> None:
        """Start the connectivity check loop."""
        self._stopped = False
        self._connection_task = asyncio.create_task(self._check_loop())
        logger.info("ConnectionManager started")

    async def stop(self) -> None:
        """Stop the connectivity check loop."""
        self._stopped = True
        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
        logger.info("ConnectionManager stopped")

    async def _check_loop(self) -> None:
        """Periodically check connectivity and emit events on state change."""
        while not self._stopped:
            try:
                is_connected = await self._check_connectivity()

                # Emit event if state changed
                if is_connected != self._is_connected:
                    self._is_connected = is_connected
                    self._emitter.emit(Emitter.IS_CONNECTED, is_connected)
                    logger.info(f"{'Connected' if is_connected else 'Disconnected'}")
                await asyncio.sleep(self._check_interval)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in connectivity check loop: {e}", exc_info=True)
                await asyncio.sleep(self._check_interval)

    async def _check_connectivity(self) -> bool:
        """Check if we have network connectivity to the API.

        Makes a HEAD request to the API URL to verify connectivity.

        Returns:
            True if connected, False otherwise
        """
        try:
            async with self.client_session.head(
                API_URL,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as response:
                return response.status < 500

        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    def is_connected(self) -> bool:
        """Get current connection state.

        Returns:
            True if connected, False otherwise
        """
        return self._is_connected

    def get_available_bandwidth(self) -> int | None:
        """Get available upload bandwidth in bytes per second.

        Returns:
            Available bandwidth in bytes/second, or None if unlimited/unknown
        """
        # TODO: bandwidth monitoring
        return None
