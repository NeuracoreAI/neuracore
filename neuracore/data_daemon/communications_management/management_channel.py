"""High-level wrapper exposing NC / NDD contexts."""

from __future__ import annotations

import logging

import zmq

from neuracore.data_daemon.communications_management.communications_manager import (
    SOCKET_PATH,
    CommunicationsManager,
)
from neuracore.data_daemon.communications_management.data_bridge import Daemon
from neuracore.data_daemon.communications_management.producer import Producer
from neuracore.data_daemon.event_loop_manager import EventLoopManager
from neuracore.data_daemon.recording_encoding_disk_manager import (
    recording_disk_manager as rdm_module,
)

RecordingDiskManager = rdm_module.RecordingDiskManager

logger = logging.getLogger(__name__)


class ManagementChannel:
    """High-level wrapper exposing NC / NDD contexts.

    - `get_nc_context()` → Producer (neuracore client side)
    - `get_ndd_context()` → Daemon instance or None if already running
    """

    def __init__(self) -> None:
        """Initialize the management channel.

        This initializes the ZeroMQ context used by the
        management channel.
        """
        self._ctx = zmq.Context.instance()

    def get_nc_context(self) -> Producer:
        """Return a producer-side context used by neuracore."""
        comm = CommunicationsManager()
        return Producer(comm_manager=comm)

    def get_ndd_context(self) -> Daemon | None:
        """Return a daemon context, or None if one is already running.

        We try to bind via CommunicationsManager; if it fails with EADDRINUSE,
        we infer another daemon is already running and return None.

        NOTE: This is intentionally simple. More advanced logic could try to
        probe for a stale socket.
        """
        if SOCKET_PATH.exists():
            logger.warning(
                "NDD context requested but socket already exists at %s; "
                "assuming another daemon is running.",
                SOCKET_PATH,
            )
            return None

        comm = CommunicationsManager()

        loop_manager = EventLoopManager()
        try:
            loop_manager.start()
        except Exception:
            logger.exception("Failed to start EventLoopManager")
            return None

        try:
            recording_disk_manager = RecordingDiskManager(loop_manager=loop_manager)
        except Exception:
            logger.exception("Failed to initialize RecordingDiskManager")
            loop_manager.stop()
            return None

        return Daemon(
            recording_disk_manager=recording_disk_manager,
            comm_manager=comm,
            loop_manager=loop_manager,
        )
