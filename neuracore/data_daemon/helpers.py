"""Helper functions for the data daemon."""

import os
from datetime import datetime, timezone
from pathlib import Path

from neuracore.data_daemon.const import DEFAULT_DAEMON_DB_PATH


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
    """Return the SQLite DB path for the data daemon.

    Uses `NEURACORE_DAEMON_DB_PATH` if set; otherwise defaults to
    `~/.neuracore/data_daemon/state.db`.
    """
    env_path = os.getenv("NEURACORE_DAEMON_DB_PATH")
    if env_path:
        return Path(env_path).expanduser()

    return DEFAULT_DAEMON_DB_PATH


def get_daemon_recordings_root_path() -> Path:
    """Return the root directory used to store recording trace files.

    This path is determined by NEURACORE_DAEMON_RECORDINGS_ROOT. If not set,
    it defaults to a sibling of the DB path: <db_dir>/recordings.
    """
    default_root = get_daemon_db_path().parent / "recordings"
    return Path(
        os.environ.get(
            "NEURACORE_DAEMON_RECORDINGS_ROOT",
            str(default_root),
        )
    )


def utc_now() -> datetime:
    """Return the current time as a Unix timestamp or datetime object."""
    return datetime.now(timezone.utc)
