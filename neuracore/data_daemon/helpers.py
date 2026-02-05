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
    env_override = os.environ.get("NEURACORE_DAEMON_DB_PATH")
    if env_override:
        return Path(env_override)

    # Prefer a host-mounted workspace path if present (outside the container home).
    host_default = Path("/home/steven/Documents/neuracore/state.db")
    if host_default.parent.exists():
        return host_default

    return Path(Path.home() / ".neuracore" / "data_daemon" / "state.db")


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
