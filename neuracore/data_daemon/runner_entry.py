"""Runner entrypoint for the Neuracore data daemon."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from neuracore.data_daemon.communications_management.management_channel import (
    ManagementChannel,
)
from neuracore.data_daemon.const import RECORDING_EVENTS_SOCKET_PATH, SOCKET_PATH
from neuracore.data_daemon.lifecycle.daemon_lifecycle import (
    install_signal_handlers,
    shutdown,
)


def main() -> None:
    """Start the daemon and block until it exits."""
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
    management_channel = ManagementChannel()
    daemon_instance = management_channel.get_ndd_context()
    if daemon_instance is None:
        sys.exit(1)

    install_signal_handlers(lambda _signum: None)

    try:
        daemon_instance.run()
    except SystemExit:
        sys.exit(1)
    finally:
        shutdown(
            pid_path=pid_path,
            socket_paths=(SOCKET_PATH, RECORDING_EVENTS_SOCKET_PATH),
            db_path=db_path,
            shutdown_steps=(),
        )


if __name__ == "__main__":
    main()
