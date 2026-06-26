"""Entry point for ``python -m neuracore.data_daemon``.

By default this runs the Python data daemon CLI. When the
``NCD_RUST_DAEMON`` environment variable is truthy and the bundled Rust
data-daemon binary is present, execution is handed off to it instead. The
flag keeps the Python daemon available throughout the rollout window.
"""

from __future__ import annotations

import os
import sys

from neuracore.data_daemon.rust_selection import (
    is_rust_daemon_enabled,
    rust_daemon_binary_path,
)


def main() -> None:
    """Dispatch to the Rust data daemon when enabled, else the Python CLI."""
    if is_rust_daemon_enabled():
        binary = rust_daemon_binary_path()
        if binary is None:
            print(
                "NCD_RUST_DAEMON is set but the bundled Rust data-daemon binary "
                "was not found; falling back to the Python daemon.",
                file=sys.stderr,
            )
        else:
            try:
                # maturin's wheel `include` can drop the bundled binary without
                # the executable bit, which would make execv fail and silently
                # fall back to the Python daemon. Restore it best-effort first.
                if not os.access(binary, os.X_OK):
                    try:
                        os.chmod(binary, 0o755)
                    except OSError:
                        pass
                os.execv(str(binary), [str(binary), *sys.argv[1:]])
            except OSError as error:
                # The binary is present but couldn't be executed (e.g. not
                # executable, ENOEXEC); fall back to the Python daemon rather
                # than crashing the rollout.
                print(
                    f"NCD_RUST_DAEMON is set but the bundled Rust data-daemon "
                    f"binary could not be executed ({error}); falling back to "
                    "the Python daemon.",
                    file=sys.stderr,
                )

    # Imported lazily so that handing off to the Rust binary above does not
    # pay the cost of importing the full Python daemon stack.
    from neuracore.data_daemon.main import main as run_python_cli

    run_python_cli()


if __name__ == "__main__":
    main()
