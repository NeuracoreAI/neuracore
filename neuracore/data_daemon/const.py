"""Constants shared between the SDK and the data daemon.

The daemon owns its own runtime constants in Rust
([rust/data_daemon_shared/src/](../../rust/data_daemon_shared/src/)); what
remains here is only what the SDK needs to agree with it on — default on-disk
locations and the active profile name.
"""

import os
import pathlib
from pathlib import Path

DEFAULT_DAEMON_STARTUP_TIMEOUT_SECONDS = 20

# Uploads Configuration paths and files
CONFIG_DIR = Path.home() / ".neuracore"
CONFIG_FILE = "config.json"
CONFIG_ENCODING = "utf-8"

DEFAULT_RECORDING_ROOT_PATH = (
    pathlib.Path.home() / ".neuracore" / "data_daemon" / "recordings"
)
DEFAULT_DAEMON_DB_PATH = Path.home() / ".neuracore" / "data_daemon" / "state.db"

# default profile name
DEFAULT_PROFILE_NAME = "default_profile"


def active_profile_name() -> str:
    """Return the active daemon profile name, mirroring the launch resolution."""
    return os.environ.get("NEURACORE_DAEMON_PROFILE") or DEFAULT_PROFILE_NAME
