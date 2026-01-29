"""High-level wrapper exposing NC / NDD contexts."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import zmq

from neuracore.data_daemon.bootstrap import DaemonBootstrap
from neuracore.data_daemon.communications_management.communications_manager import (
    SOCKET_PATH,
    CommunicationsManager,
)
from neuracore.data_daemon.communications_management.data_bridge import Daemon
from neuracore.data_daemon.communications_management.producer import Producer
from neuracore.data_daemon.const import (
    DEFAULT_RECORDING_ROOT_PATH,
    RECORDING_EVENTS_SOCKET_PATH,
)
from neuracore.data_daemon.lifecycle.daemon_lifecycle import (
    DaemonLifecycleError,
    startup,
)
from neuracore.data_daemon.recording_encoding_disk_manager import (
    recording_disk_manager as rdm_module,
)
from neuracore.data_daemon.state_management.state_store_sqlite import SqliteStateStore

RecordingDiskManager = rdm_module.RecordingDiskManager

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

    def get_nc_context(
        self, producer_id: str | None = None, recording_id: str | None = None
    ) -> Producer:
        """Return a producer-side context used by neuracore."""
        comm = CommunicationsManager()
        return Producer(comm_manager=comm, id=producer_id, recording_id=recording_id)

    async def get_ndd_context(self) -> Daemon | None:
        """Return a daemon context, or None if one is already running.

        Uses DaemonBootstrap for proper initialization of all subsystems
        including StateManager, UploadManager, and other async services.

        Returns:
            Daemon instance if successful, None if already running or failed.
        """
        pid_path = Path(
            os.environ.get(
                "NEURACORE_DAEMON_PID_PATH",
                str(Path.home() / ".neuracore" / "daemon.pid"),
            )
        )
        db_path = Path(
            os.environ.get(
                "NEURACORE_DAEMON_DB_PATH",
                str(Path.home() / ".neuracore" / "data_daemon" / "state.db"),
            )
        )
        recordings_root = Path(
            os.environ.get(
                "NEURACORE_DAEMON_RECORDINGS_ROOT",
                str(DEFAULT_RECORDING_ROOT_PATH),
            )
        )
        manage_pid = os.environ.get("NEURACORE_DAEMON_MANAGE_PID", "1") != "0"

        if SOCKET_PATH.exists() and pid_path.exists():
            try:
                pid_value = int(pid_path.read_text(encoding="utf-8").strip())
            except Exception:
                pid_value = None
            if pid_value is not None:
                try:
                    os.kill(pid_value, 0)
                    logger.warning(
                        "NDD context requested but socket already exists at %s; "
                        "daemon appears to be running (pid=%s).",
                        SOCKET_PATH,
                        pid_value,
                    )
                    return None
                except ProcessLookupError:
                    pass
                except PermissionError:
                    logger.warning(
                        "NDD context requested but socket already exists at %s; "
                        "permission denied checking pid=%s.",
                        SOCKET_PATH,
                        pid_value,
                    )
                    return None

        store = SqliteStateStore(db_path)
        try:
            await startup(
                pid_path=pid_path,
                socket_paths=(SOCKET_PATH, RECORDING_EVENTS_SOCKET_PATH),
                db_path=db_path,
                recordings_root=recordings_root,
                store=store,
                recover_sqlite=True,
                manage_pid=manage_pid,
            )
        except DaemonLifecycleError:
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
