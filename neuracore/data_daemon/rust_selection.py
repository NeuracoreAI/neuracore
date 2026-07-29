"""Runtime selection between the Rust data daemon and the legacy Python one.

Centralises the daemon choice used by both the CLI entry point
([__main__.py](neuracore/data_daemon/__main__.py)) and SDK-side producer
routing in
[neuracore/core/streaming/data_stream.py](neuracore/core/streaming/data_stream.py)
so both surfaces agree on which daemon is in play for a given process.

Kept dependency-free so the SDK can import it without pulling in the daemon's
heavyweight runtime modules.
"""

from __future__ import annotations

import os
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

_TRUTHY_VALUES = frozenset({"1", "true", "yes", "y"})
_FALSY_VALUES = frozenset({"0", "false", "no", "n"})


def is_rust_daemon_enabled() -> bool:
    """Return True when the Rust data daemon should handle this process.

    The Rust daemon is the default. ``NCD_RUST_DAEMON`` overrides in both
    directions: a falsy value pins the process to the legacy Python daemon, and
    a truthy value demands Rust even where the bundled binary is missing (the
    launcher reports its own fallback in that case).
    """
    raw = os.environ.get("NCD_RUST_DAEMON", "").strip().lower()
    if raw in _FALSY_VALUES:
        return False
    if raw in _TRUTHY_VALUES:
        return True
    return _bundled_binary_available()


@lru_cache(maxsize=1)
def _bundled_binary_available() -> bool:
    """Return True when the daemon binary ships in this install, cached.

    Whether the binary is on disk cannot change within a process, while
    ``is_rust_daemon_enabled`` sits on per-frame paths such as
    [log_camera_data](neuracore/api/logging.py) — so the resource traversal in
    ``rust_daemon_binary_path`` must not run once per logged sample.
    """
    return rust_daemon_binary_path() is not None


def rust_daemon_binary_path() -> Path | None:
    """Return the path to the Rust data-daemon binary, if available.

    The binary ships in ``neuracore/data_daemon/bin/`` inside the ``neuracore``
    wheel (prebuilt for Linux x86_64 and Apple-Silicon macOS). Returns ``None``
    when it is absent — e.g. a source/editable install without
    ``rust/scripts/build_wheel_artefacts.sh`` having been run.
    """
    candidate = files("neuracore.data_daemon") / "bin" / "data-daemon"
    path = Path(str(candidate))
    return path if path.is_file() else None
