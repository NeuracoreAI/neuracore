"""Helper functions for the data daemon."""

import os
from pathlib import Path


def get_daemon_pid_path() -> Path:
    """Return the path to the file where the data daemon's PID is stored.

    This path is determined by the environment variable NEURACORE_DAEMON_PID_PATH.
    If this variable is not set, the path defaults to ~/.neuracore/daemon.pid.

    :return: Path to the PID file
    """
    return Path(
        os.environ.get(
            "NEURACORE_DAEMON_PID_PATH",
            str(Path.home() / ".neuracore" / "daemon.pid"),
        )
    )


def get_daemon_db_path() -> Path:
    """Return the path to the SQLite database file used by the data daemon.

    This path is determined by the environment variable NEURACORE_DAEMON_DB_PATH.
    If this variable is not set, the path defaults to
    ~/.neuracore/data_daemon/state.db.

    :return: Path to the SQLite database file
    """
    return Path(
        os.environ.get(
            "NEURACORE_DAEMON_DB_PATH",
            str(Path.home() / ".neuracore" / "data_daemon" / "state.db"),
        )
    )
