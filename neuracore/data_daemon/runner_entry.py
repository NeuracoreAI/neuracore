"""Runner entrypoint for the Neuracore data daemon."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from neuracore.data_daemon.bootstrap import DaemonBootstrap
from neuracore.data_daemon.const import RECORDING_EVENTS_SOCKET_PATH, SOCKET_PATH
from neuracore.data_daemon.lifecycle.daemon_lifecycle import (
    install_signal_handlers,
    shutdown,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Runner entrypoint for the Neuracore data daemon.

    This function bootstraps the daemon, starts it, and then waits for
    a signal to stop. The daemon is stopped when the function returns.

    Environment variables affecting this function:

    NEURACORE_DAEMON_PID_PATH
        Path to the pid file for the daemon.

    NEURACORE_DAEMON_DB_PATH
        Path to the SQLite database file for the daemon's state.

    The daemon will exit with a status code of 1 if the socket at
    NEURACORE_DAEMON_SOCKET_PATH already exists.

    The daemon will shut down when it receives a SIGINT or SIGTERM signal.
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
    if SOCKET_PATH.exists():
        logger.error(
            "Socket already exists at %s; another daemon may be running",
            SOCKET_PATH,
        )
        sys.exit(1)

    try:
        bootstrap = DaemonBootstrap()
        context = bootstrap.start()

        if context is None:
            logger.error("Failed to start daemon")

        install_signal_handlers(lambda _signum: None)

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except SystemExit:
        pass
    finally:
        bootstrap.stop()
        shutdown(
            pid_path=pid_path,
            socket_paths=(SOCKET_PATH, RECORDING_EVENTS_SOCKET_PATH),
            db_path=db_path,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
