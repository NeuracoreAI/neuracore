"""Runner entrypoint for the Neuracore data daemon."""

from __future__ import annotations

import logging
import sys

from neuracore.data_daemon.bootstrap import DaemonBootstrap
from neuracore.data_daemon.communications_management.communications_manager import (
    SOCKET_PATH,
)
from neuracore.data_daemon.communications_management.data_bridge import Daemon

logger = logging.getLogger(__name__)


def main() -> None:
    """Start the daemon and block until it exits.

    Initialization sequence:
    1. Check if daemon is already running (socket exists)
    2. DaemonBootstrap.start() initializes all subsystems:
       - EventLoopManager (General + Encoder loops)
       - Async services on General Loop (StateManager, UploadManager, etc.)
       - RecordingDiskManager (workers on respective loops)
       - CommunicationsManager (ZMQ sockets)
    3. Create Daemon with initialized context
    4. Daemon.run() enters blocking ZMQ message loop
    5. On exit, DaemonBootstrap.stop() shuts down all subsystems
    """
    if SOCKET_PATH.exists():
        logger.error(
            "Socket already exists at %s; another daemon may be running",
            SOCKET_PATH,
        )
        sys.exit(1)

    bootstrap = DaemonBootstrap()
    context = bootstrap.start()

    if context is None:
        logger.error("Failed to start daemon")
        sys.exit(1)

    daemon = Daemon(
        recording_disk_manager=context.recording_disk_manager,
        comm_manager=context.comm_manager,
    )

    try:
        logger.info("Daemon starting main loop...")
        daemon.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except SystemExit:
        pass
    finally:
        bootstrap.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
