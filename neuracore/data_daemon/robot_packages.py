"""URDF packages ``nc.connect_robot`` saves for the data daemon to upload.

``connect_robot`` goes through the daemon, online or offline, so it builds the
package it would upload and saves it under the recordings root, in a hidden
directory the daemon's recording sweeps never take for a recording::

    .robot_packages/<unique name>.zip

and names it in its connect request. Once the server has registered the robot,
the daemon uploads the package if the server needs it and deletes it. The
package is written to a temporary name and renamed into place, so the daemon
never reads a half-written file. ``data-daemon reset`` deletes the directory
along with the recordings.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from neuracore.data_daemon.helpers import get_daemon_recordings_root_path

ROBOT_PACKAGES_DIR = ".robot_packages"


def save_robot_package(package: bytes) -> Path:
    """Save a robot's URDF package for the data daemon to upload.

    Args:
        package: The URDF package, exactly as it would be uploaded.

    Returns:
        The package's path, to name in the connect request.
    """
    root = get_daemon_recordings_root_path() / ROBOT_PACKAGES_DIR
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{uuid.uuid4().hex}.zip"
    tmp = path.with_name(f"{path.name}.tmp")
    with open(tmp, "wb") as file:
        file.write(package)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path)
    return path
