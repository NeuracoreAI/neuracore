"""Runtime selection between the legacy Python daemon and the Rust rewrite.

Centralises the ``NCD_RUST_DAEMON`` environment-variable check used by both
the CLI entry point ([__main__.py](neuracore/data_daemon/__main__.py)) and
SDK-side producer routing in
[neuracore/core/streaming/data_stream.py](neuracore/core/streaming/data_stream.py)
so both surfaces agree on which daemon is in play for a given process.

Kept dependency-free so the SDK can import it without pulling in the daemon's
heavyweight runtime modules.
"""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path

_TRUTHY_VALUES = frozenset({"1", "true", "yes", "y"})


def is_rust_daemon_enabled() -> bool:
    """Return True when ``NCD_RUST_DAEMON`` selects the Rust data daemon."""
    return os.environ.get("NCD_RUST_DAEMON", "").strip().lower() in _TRUTHY_VALUES


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
