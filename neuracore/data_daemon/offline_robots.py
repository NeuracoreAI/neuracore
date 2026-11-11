"""URDF packages of robots connected offline, saved for the data daemon.

An offline ``connect_robot`` cannot reach the server, so it saves the URDF
package it would have uploaded under the recordings root, in a hidden directory
so the daemon's recording sweeps and ``data-daemon reset`` do not take it for a
recording::

    .offline_robots/<local robot id>/package.zip

The robot's name and details travel with each recording started with it. The
daemon registers a robot instance when a recording made with it is about to
upload, uploads the package first if the server needs it, and then deletes it.
A package nothing is recorded with is deleted once it is a day old. The package
is written to a temporary name and renamed into place, so the daemon never
reads a half-written file.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from neuracore.data_daemon.helpers import get_daemon_recordings_root_path

OFFLINE_ROBOTS_DIR = ".offline_robots"
PACKAGE_FILENAME = "package.zip"


def offline_robot_dir(robot_id: str) -> Path:
    """Return the directory a robot connected offline is saved in.

    Args:
        robot_id: The robot's local id.

    Returns:
        The robot's directory under the recordings root.
    """
    return get_daemon_recordings_root_path() / OFFLINE_ROBOTS_DIR / robot_id


def save_offline_robot(robot_id: str, package: bytes) -> Path:
    """Save the URDF package of a robot connected offline for the data daemon.

    Args:
        robot_id: The robot's local id.
        package: The URDF package, exactly as it would be uploaded.

    Returns:
        The package's path.
    """
    path = offline_robot_dir(robot_id) / PACKAGE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomically(path, package)
    return path


def _write_atomically(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` so readers see the old file or the new one.

    The temporary name is this thread's own, so threads saving the same robot
    do not write into one file.
    """
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        _write_then_rename(tmp, path, data)
    except FileNotFoundError:
        # The daemon removes the directory once it has sent the package, which
        # can happen between creating it and writing into it.
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_then_rename(tmp, path, data)


def _write_then_rename(tmp: Path, path: Path, data: bytes) -> None:
    with open(tmp, "wb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path)
