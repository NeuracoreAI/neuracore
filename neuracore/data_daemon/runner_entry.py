"""Runner entrypoint for the Neuracore data daemon."""

from __future__ import annotations

import sys

from neuracore.data_daemon.communications_management.management_channel import (
    ManagementChannel,
)


def main() -> None:
    """Start the daemon and block until it exits."""
    management_channel = ManagementChannel()
    daemon_instance = management_channel.get_ndd_context()
    if daemon_instance is None:
        sys.exit(1)

    try:
        daemon_instance.run()
    except SystemExit:
        sys.exit(1)


if __name__ == "__main__":
    main()
