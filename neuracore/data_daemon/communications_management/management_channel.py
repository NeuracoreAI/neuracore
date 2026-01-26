"""High-level wrapper exposing NC / NDD contexts."""

from __future__ import annotations

import logging

import zmq

from neuracore.data_daemon.bootstrap import DaemonBootstrap
from neuracore.data_daemon.communications_management.communications_manager import (
    SOCKET_PATH,
    CommunicationsManager,
)
from neuracore.data_daemon.communications_management.data_bridge import Daemon
from neuracore.data_daemon.communications_management.producer import Producer

logger = logging.getLogger(__name__)


class ManagementChannel:
    """High-level wrapper exposing NC / NDD contexts.

    - `get_nc_context()` → Producer (neuracore client side)
    - `get_ndd_context()` → Daemon instance or None if already running

    Note: For production use, prefer using DaemonBootstrap directly
    which provides proper lifecycle management. This class is maintained
    for backward compatibility and testing.
    """

    def __init__(self) -> None:
        """Initialize the management channel.

        This initializes the ZeroMQ context used by the
        management channel.
        """
        self._ctx = zmq.Context.instance()
        self._bootstrap: DaemonBootstrap | None = None

    def get_nc_context(self) -> Producer:
        """Return a producer-side context used by neuracore."""
        comm = CommunicationsManager()
        return Producer(comm_manager=comm)

    def get_ndd_context(self) -> Daemon | None:
        """Return a daemon context, or None if one is already running.

        Uses DaemonBootstrap for proper initialization of all subsystems
        including StateManager, UploadManager, and other async services.

        Returns:
            Daemon instance if successful, None if already running or failed.
        """
        if SOCKET_PATH.exists():
            logger.warning(
                "NDD context requested but socket already exists at %s; "
                "assuming another daemon is running.",
                SOCKET_PATH,
            )
            return None

        self._bootstrap = DaemonBootstrap()
        context = self._bootstrap.start()

        if context is None:
            logger.error("Failed to initialize daemon context")
            return None

        return Daemon(
            recording_disk_manager=context.recording_disk_manager,
            comm_manager=context.comm_manager,
        )

    def shutdown(self) -> None:
        """Shutdown the daemon and cleanup resources.

        Should be called after daemon.run() exits to properly
        cleanup all subsystems.
        """
        if isinstance(self._bootstrap, DaemonBootstrap):
            self._bootstrap.stop()
            self._bootstrap = None
