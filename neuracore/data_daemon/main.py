"""Main entry point for the Neuracore data daemon CLI."""

import argparse

from neuracore.data_daemon.config_manager.args_handler import (
    add_common_config_args,
    handle_install,
    handle_launch,
    handle_list_profile,
    handle_profile_create,
    handle_profile_show,
    handle_profile_update,
    handle_status,
    handle_stop,
    handle_uninstall,
    handle_update,
)


def main() -> None:
    """Handlers for nc-data-daemon CLI commands."""
    parser = argparse.ArgumentParser(
        prog="nc-data-daemon",
        description="Neuracore Data Daemon CLI",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # nc-data-daemon update [options...]
    update_cmd = subparsers.add_parser("update", help="Persist user config overrides.")
    add_common_config_args(update_cmd)
    update_cmd.set_defaults(handler=handle_update)

    profile_parser = subparsers.add_parser("profile", help="Manage daemon profiles.")
    profile_subparsers = profile_parser.add_subparsers(
        dest="profile_command", required=True
    )

    create_parser = profile_subparsers.add_parser("create", help="Create a profile.")
    create_parser.add_argument("--name", required=True, help="Profile name.")
    create_parser.set_defaults(handler=handle_profile_create)

    update_parser = profile_subparsers.add_parser(
        "update", help="Update an existing profile."
    )
    update_parser.add_argument("--name", required=True, help="Profile name to update.")
    add_common_config_args(update_parser)
    update_parser.set_defaults(handler=handle_profile_update)

    show_parser = profile_subparsers.add_parser(
        "show", help="Show a profile's configuration."
    )
    show_parser.add_argument("--name", required=True, help="Profile name to show.")
    show_parser.set_defaults(handler=handle_profile_show)

    launch_parser = subparsers.add_parser("launch", help="Launch the data daemon.")
    launch_parser.set_defaults(handler=handle_launch)

    stop_parser = subparsers.add_parser("stop", help="Stop the data daemon.")
    stop_parser.set_defaults(handler=handle_stop)

    status_parser = subparsers.add_parser("status", help="Show daemon status.")
    status_parser.set_defaults(handler=handle_status)

    install_parser = subparsers.add_parser(
        "install", help="Install the data daemon as a system service."
    )
    install_parser.set_defaults(handler=handle_install)

    uninstall_parser = subparsers.add_parser(
        "uninstall", help="Uninstall the data daemon system service."
    )
    uninstall_parser.set_defaults(handler=handle_uninstall)

    list_profile_parser = subparsers.add_parser(
        "list-profiles", help="List all configured daemon profiles."
    )
    list_profile_parser.set_defaults(handler=handle_list_profile)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
