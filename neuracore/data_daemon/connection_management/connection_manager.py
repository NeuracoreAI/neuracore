"""Connection manager for network monitoring.

This module provides a connection manager that monitors network connectivity
to the Neuracore API and emits events when connection state changes.
"""

import logging
import threading
import time

import requests

from neuracore.data_daemon.const import API_URL
from neuracore.data_daemon.event_emitter import Emitter, emitter

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages network connectivity checks and emits connection state events.

    Runs a background thread that periodically checks connectivity and emits
    IS_CONNECTED events to the state manager when connection state changes.
    """

    def __init__(
        self,
        *,
        timeout: float = 5.0,
        check_interval: float = 10.0,
        offline_mode: bool = False,
    ) -> None:
        """Initialize the connection manager.

        Args:
            config_manager: Config to resolve for Connection Manager
            timeout: Timeout in seconds for connectivity checks
            check_interval: Seconds between connectivity checks
            offline_mode: Daemon is in offline mode; skip connectivity checks
        """
        self._timeout = timeout
        self._check_interval = check_interval
        self._is_connected = False
        self._running = False
        self._checker_thread: threading.Thread | None = None
        self._offline_mode = offline_mode

        emitter.emit(Emitter.IS_CONNECTED, self._is_connected)

    def start(self) -> None:
        """Start the connection monitoring thread."""
        if self._offline_mode:
            logger.info("ConnectionManager in offline mode")
            return
        if self._running:
            logger.warning("ConnectionManager already running")
            return

        self._running = True
        self._checker_thread = threading.Thread(
            target=self._check_loop,
            name="connection-checker",
            daemon=False,
        )
        self._checker_thread.start()
        logger.info("ConnectionManager started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the connection monitoring thread.

        Args:
            timeout: Maximum time to wait for thread to stop
        """
        if self._offline_mode:
            logger.info("ConnectionManager in offline mode")
            return
        if not self._running:
            logger.warning("ConnectionManager not running")
            return

        logger.info("Stopping ConnectionManager...")
        self._running = False

        if self._checker_thread and self._checker_thread.is_alive():
            self._checker_thread.join(timeout=timeout)

        logger.info("ConnectionManager stopped")

    def _check_loop(self) -> None:
        """Background loop that checks connectivity and emits state changes."""
        logger.info("Connection checking loop started")

        while self._running:
            try:
                new_state = self._check_connectivity()

                # Emit event if state changed
                if new_state != self._is_connected:
                    self._is_connected = new_state
                    logger.info(f"Connection state changed: {new_state}")
                    emitter.emit(Emitter.IS_CONNECTED, new_state)

                time.sleep(self._check_interval)

            except Exception as e:
                logger.error(f"Error in connection check loop: {e}", exc_info=True)
                time.sleep(self._check_interval)

        logger.info("Connection checking loop stopped")

    def _check_connectivity(self) -> bool:
        """Check if we have network connectivity to the API.

        Makes a HEAD request to the API URL to verify connectivity.

        Returns:
            True if connected, False otherwise
        """
        try:
            response = requests.head(API_URL, timeout=self._timeout)
            return response.status_code < 500

        except requests.exceptions.RequestException:
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
