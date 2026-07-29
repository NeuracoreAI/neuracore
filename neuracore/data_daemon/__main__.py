"""Entry point for ``python -m neuracore.data_daemon``.

By default this hands off to the bundled Rust data-daemon binary, falling back
to the Python data daemon CLI when that binary is absent or unusable. Setting
``NCD_RUST_DAEMON`` to a falsy value pins the process to the Python daemon.
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
                    "The bundled Rust data-daemon binary could not be "
                    f"executed ({error}); falling back to the Python daemon.",
                    file=sys.stderr,
                )

    # Imported lazily so that handing off to the Rust binary above does not
    # pay the cost of importing the full Python daemon stack.
    from neuracore.data_daemon.main import main as run_python_cli

    run_python_cli()


if __name__ == "__main__":
    main()
