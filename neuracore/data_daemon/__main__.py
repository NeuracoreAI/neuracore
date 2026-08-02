"""Entry point for ``python -m neuracore.data_daemon``.

Hands off to the bundled data-daemon binary, replacing this process so signals
and exit codes pass through unchanged.
"""

from __future__ import annotations

import os
import sys

from neuracore.data_daemon.binary import require_data_daemon_binary


def main() -> None:
    """Replace this process with the bundled data-daemon binary."""
    binary = require_data_daemon_binary()
    # maturin's wheel `include` can drop the bundled binary's executable bit,
    # which would make execv fail. Restore it best-effort first.
    if not os.access(binary, os.X_OK):
        try:
            os.chmod(binary, 0o755)
        except OSError:
            pass
    os.execv(str(binary), [str(binary), *sys.argv[1:]])


if __name__ == "__main__":
    main()
