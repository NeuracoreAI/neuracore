"""Entry point for ``python -m neuracore.data_daemon``.

By default this runs the Python data daemon CLI. When the
``NCD_RUST_DAEMON`` environment variable is truthy and the bundled Rust
data-daemon binary is present, execution is handed off to it instead — the
Rust binary is the eventual replacement for this package (see
``docs/data-daemon-rewrite.md``). The flag keeps the Python daemon available
throughout the rollout window.
"""

from __future__ import annotations

import os
import sys
from importlib.resources import files

from neuracore.data_daemon.rust_selection import rust_daemon_enabled


def _rust_binary_path() -> str | None:
    """Return the path to the bundled Rust data-daemon binary, if present."""
    candidate = files("neuracore.data_daemon") / "bin" / "data-daemon"
    return str(candidate) if candidate.is_file() else None


def main() -> None:
    """Dispatch to the Rust data daemon when enabled, else the Python CLI."""
    if rust_daemon_enabled():
        binary = _rust_binary_path()
        if binary is not None:
            os.execv(binary, [binary, *sys.argv[1:]])
        print(
            "NCD_RUST_DAEMON is set but the bundled Rust data-daemon binary "
            "was not found; falling back to the Python daemon.",
            file=sys.stderr,
        )

    # Imported lazily so that handing off to the Rust binary above does not
    # pay the cost of importing the full Python daemon stack.
    from neuracore.data_daemon.main import main as run_python_cli

    run_python_cli()


if __name__ == "__main__":
    main()
