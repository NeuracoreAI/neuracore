"""Resolution of the bundled data-daemon binary.

Kept dependency-free so the SDK can import it without pulling in the daemon's
runtime modules.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

REBUILD_COMMAND_HINT = (
    "`bash rust/scripts/build_wheel_artefacts.sh && pip install -e .`"
)

INSTALL_COMMAND_HINT = "`pip install --force-reinstall neuracore`"

MISSING_BINARY_HINT = (
    "The neuracore data-daemon binary was not found. It ships in the neuracore "
    "wheel (prebuilt for Linux x86_64 and Apple-Silicon macOS); reinstall with "
    f"{INSTALL_COMMAND_HINT} on one of those platforms, or build it into a "
    f"source checkout with {REBUILD_COMMAND_HINT}."
)


class DaemonBinaryNotFoundError(RuntimeError):
    """Raised when the bundled data-daemon binary is missing from the install."""


def data_daemon_binary_path() -> Path | None:
    """Return the path to the data-daemon binary, if available.

    The binary ships in ``neuracore/data_daemon/bin/`` inside the ``neuracore``
    wheel (prebuilt for Linux x86_64 and Apple-Silicon macOS). Returns ``None``
    when it is absent — e.g. a source/editable install without
    ``rust/scripts/build_wheel_artefacts.sh`` having been run.
    """
    candidate = files("neuracore.data_daemon") / "bin" / "data-daemon"
    path = Path(str(candidate))
    return path if path.is_file() else None


def require_data_daemon_binary() -> Path:
    """Return the data-daemon binary path or raise with install instructions."""
    path = data_daemon_binary_path()
    if path is None:
        raise DaemonBinaryNotFoundError(MISSING_BINARY_HINT)
    return path
