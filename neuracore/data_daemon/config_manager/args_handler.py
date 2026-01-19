"""Handlers for nc-data-daemon CLI commands."""

import argparse
from typing import Any

from neuracore.data_daemon.config_manager.config import ConfigManager
from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.config_manager.helpers import parse_bytes
from neuracore.data_daemon.config_manager.profiles import (
    ProfileAlreadyExist,
    ProfileManager,
    ProfileNotFound,
)

profile_manager = ProfileManager()
config_manager = ConfigManager(profile_manager)


def add_common_config_args(parser: argparse.ArgumentParser) -> None:
    """Register common daemon configuration flags on an argparse parser.

    Args:
        parser: The argparse parser (or subparser) to attach configuration
            arguments to.

    Returns:
        None
    """
    parser.add_argument(
        "--storage-limit",
        "--storage_limit",
        type=parse_bytes,
        help="Storage limit in bytes.",
    )
    parser.add_argument(
        "--bandwidth-limit",
        "--bandwidth_limit",
        type=parse_bytes,
        help="Bandwidth limit in bytes per second.",
    )
    parser.add_argument(
        "--storage-path",
        "--storage_path",
        "--path_to_store_record",
        dest="path_to_store_record",
        help="Path where records should be stored.",
    )
    parser.add_argument(
        "--num-threads",
        "--num_threads",
        type=int,
        help="Number of worker threads.",
    )
    parser.add_argument(
        "--keep-wakelock-while-upload",
        "--keep_wakelock_while_upload",
        dest="keep_wakelock_while_upload",
        action="store_true",
        default=None,
        help="Keep a wakelock while uploading.",
    )
    parser.add_argument(
        "--offline",
        dest="offline",
        action="store_true",
        default=None,
        help="Run in offline mode.",
    )
    parser.add_argument(
        "--api-key",
        "--api_key",
        dest="api_key",
        help="API key used for authenticating the daemon.",
    )
    parser.add_argument(
        "--current-org-id",
        "--current_org_id",
        dest="current_org_id",
        help="Active organisation ID for scoping daemon operations.",
    )


def _extract_config_updates(args: argparse.Namespace) -> dict[str, Any]:
    """Extract DaemonConfig field values from parsed CLI arguments.

    Args:
        args: Parsed argparse namespace containing CLI flags for various commands.

    Returns:
        A dict of DaemonConfig field names to values, excluding keys that are not part
        of DaemonConfig and excluding values that are None.
    """
    allowed = set(DaemonConfig.model_fields.keys())
    raw = vars(args)
    return {k: v for k, v in raw.items() if k in allowed and v is not None}


def handle_profile_create(args: argparse.Namespace) -> None:
    """Handle the ``profile create`` CLI command.

    Args:
        args: Parsed CLI arguments containing the profile name.

    Returns:
        None
    """
    try:
        profile_manager.create_profile(args.name)
        print(f"Created profile {args.name!r}.")
    except ProfileAlreadyExist as exc:
        print(exc)


def handle_profile_update(args: argparse.Namespace) -> None:
    """Handle the ``profile update`` CLI command.

    Args:
        args: Parsed CLI arguments containing the profile name and any
            configuration fields to update.

    Returns:
        None
    """
    updates = _extract_config_updates(args)
    validated_updates = DaemonConfig.model_validate(updates).model_dump(
        exclude_none=True
    )

    try:
        profile_manager.update_profile(args.name, validated_updates)
        print(f"Updated profile {args.name!r}.")
    except ProfileNotFound as exc:
        print(exc)


def handle_profile_show(args: argparse.Namespace) -> None:
    """Handle the ``profile show`` CLI command.

    Args:
        args: Parsed CLI arguments containing the profile name.

    Returns:
        None
    """
    try:
        config = profile_manager.get_profile(args.name)
    except ProfileNotFound as exc:
        print(exc)
        return

    print(config.model_dump_json(indent=2))


def handle_list_profile(args: argparse.Namespace) -> None:
    """Handle the ``list-profiles`` CLI command.

    Args:
        args: Parsed CLI arguments for the command.

    Returns:
        None
    """
    profiles = profile_manager.list_profiles()
    if not profiles:
        print("No profiles found.")
        return

    for name in profiles:
        print(name)


def handle_launch(args: argparse.Namespace) -> None:
    """Handle the ``launch`` CLI command.

    This is a scaffolding implementation that resolves the effective
    configuration for the current run and prints it. Actual process
    management for the daemon will be added in a later ticket.

    Args:
        args: Parsed CLI arguments for the command.

    Returns:
        None
    """
    print("launch command is not implemented yet.")


def handle_stop(args: argparse.Namespace) -> None:
    """Handle the ``stop`` CLI command.

    Args:
        args: Parsed CLI arguments for the command.

    Returns:
        None
    """
    print("Stop command is not implemented yet.")


def handle_install(args: argparse.Namespace) -> None:
    """Handle the ``install`` CLI command.

    Args:
        args: Parsed CLI arguments for the command.

    Returns:
        None
    """
    print("Install command is not implemented yet.")


def handle_uninstall(args: argparse.Namespace) -> None:
    """Handle the ``uninstall`` CLI command.

    Args:
        args: Parsed CLI arguments for the command.

    Returns:
        None
    """
    print("Uninstall command is not implemented yet.")


def handle_update(args: argparse.Namespace) -> None:
    """Handle the ` update`` CLI command.

    Args:
        args: Parsed CLI arguments containing configuration fields to update.

    Returns:
        None
    """
    updates = _extract_config_updates(args)
    validated_updates = DaemonConfig.model_validate(updates).model_dump(
        exclude_none=True
    )

    print(config_manager.resolve_effective_config(validated_updates))
