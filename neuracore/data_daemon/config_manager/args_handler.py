"""Handlers for nc-data-daemon CLI commands."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from neuracore.data_daemon.config_manager.config import ConfigManager
from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.config_manager.helpers import parse_bytes
from neuracore.data_daemon.config_manager.profiles import (
    ProfileAlreadyExist,
    ProfileManager,
    ProfileNotFound,
)
from neuracore.data_daemon.const import RECORDING_EVENTS_SOCKET_PATH, SOCKET_PATH
from neuracore.data_daemon.helpers import get_daemon_db_path, get_daemon_pid_path
from neuracore.data_daemon.lifecycle.daemon_lifecycle import shutdown

profile_manager = ProfileManager()
config_manager = ConfigManager(profile_manager)

daemon_state_dir_path = Path.home() / ".neuracore"
pid_file_path = get_daemon_pid_path()


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


def _ensure_daemon_state_dir_exists() -> None:
    """Ensure the daemon state directory exists.

    Args:
        args: Parsed CLI arguments for the command.

    Returns:
        None
    """
    pid_file_path.parent.mkdir(parents=True, exist_ok=True)


def _read_pid_from_file(pid_path: Path) -> int | None:
    """Read an integer PID from the pid file.

    Args:
        args: Parsed CLI arguments for the command.

    Returns:
        None
    """
    try:
        pid_text = pid_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None

    if not pid_text:
        return None

    try:
        pid_value = int(pid_text)
    except ValueError:
        return None

    if pid_value <= 0:
        return None

    return pid_value


def _pid_is_running(pid_value: int) -> bool:
    """Check whether a PID is running.

    Args:
        pid_value: Process ID to check.

    Returns:
        True if the process exists, otherwise False.
    """
    try:
        os.kill(pid_value, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_pid(pid_value: int) -> bool:
    """Terminate a process using SIGTERM.

    Args:
        pid_value: Process ID to terminate.

    Returns:
        True if the PID does not exist after the call, otherwise False.
    """
    try:
        os.kill(pid_value, signal.SIGTERM)
        return True
    except ProcessLookupError:
        return True
    except PermissionError:
        return False


def _force_kill(pid_value: int) -> bool:
    """Forcefully terminate a process using SIGKILL.

    Args:
        pid_value: Process ID to terminate.

    Returns:
        True if the PID does not exist after the call, otherwise False.
    """
    try:
        os.kill(pid_value, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return True
    except PermissionError:
        return False


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

    Args:
        args: Parsed CLI arguments for the command.

    Returns:
        None
    """
    pid_path = pid_file_path
    _ensure_daemon_state_dir_exists()
    existing_pid = _read_pid_from_file(pid_path)
    if existing_pid is not None:
        if _pid_is_running(existing_pid):
            print(f"Daemon already running (pid={existing_pid}).")
            sys.exit(1)
        try:
            pid_path.unlink()
        except FileNotFoundError:
            pass

    runner_command = [
        sys.executable,
        "-m",
        "neuracore.data_daemon.runner_entry",
    ]

    env = os.environ.copy()
    env["NEURACORE_DAEMON_PID_PATH"] = str(pid_path)
    env["NEURACORE_DAEMON_MANAGE_PID"] = "0"

    daemon_process = subprocess.Popen(
        runner_command,
        start_new_session=True,
        close_fds=True,
        cwd=str(Path.cwd()),
        env=env,
    )

    spawned_daemon_pid = daemon_process.pid

    time.sleep(0.1)
    if daemon_process.poll() is not None:
        print("Daemon failed to start.")
        sys.exit(1)

    pid_path.write_text(str(spawned_daemon_pid), encoding="utf-8")
    print(f"Daemon launched (pid={spawned_daemon_pid}).")


def handle_stop(args: argparse.Namespace) -> None:
    """Handle the ``stop`` CLI command.

    Stop the daemon if it is running.

    Returns:
        None
    """
    pid_path = pid_file_path
    db_path = get_daemon_db_path()
    pid_value = _read_pid_from_file(pid_path)

    if pid_value is None:
        print("Daemon is not running.")
        return

    # Stale process
    if not _pid_is_running(pid_value):
        shutdown(
            pid_path=pid_path,
            socket_paths=(SOCKET_PATH, RECORDING_EVENTS_SOCKET_PATH),
            db_path=db_path,
        )
        print("Daemon stopped.")
        return

    # Send sigterm
    terminate_ok = _terminate_pid(pid_value)
    if not terminate_ok:
        print(f"Permission denied sending SIGTERM to pid={pid_value}.")
        sys.exit(1)
    if _wait_handler(pid_value, 10):
        # Trigger socket cleanup, pid cleanup and db checkpointing
        shutdown(
            pid_path=pid_path,
            socket_paths=(SOCKET_PATH, RECORDING_EVENTS_SOCKET_PATH),
            db_path=db_path,
        )
        print("Daemon stopped.")
        return

    # Send sigkill instead
    kill_ok = _force_kill(pid_value)
    if not kill_ok:
        print(f"Permission denied sending SIGKILL to pid={pid_value}.")
        sys.exit(1)
    if _wait_handler(pid_value, 5):
        shutdown(
            pid_path=pid_path,
            socket_paths=(SOCKET_PATH, RECORDING_EVENTS_SOCKET_PATH),
            db_path=db_path,
        )
        print("Daemon stopped (forced).")
        return

    print(f"Failed to stop daemon (pid={pid_value}).")
    sys.exit(1)


def handle_status(args: argparse.Namespace) -> None:
    """Handle the ``status`` CLI command.

    Args:
        args: Parsed CLI arguments for the command.

    Returns:
        None
    """
    pid_path = pid_file_path
    pid_value = _read_pid_from_file(pid_path)
    if pid_value is None:
        print("Daemon not running.")
        return

    if not _pid_is_running(pid_value):
        try:
            pid_path.unlink()
        except FileNotFoundError:
            pass
        print("Daemon not running.")
        return

    print(f"Daemon running (pid={pid_value}).")


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


def _wait_handler(pid: int, timeout: int = 2) -> bool:
    """Wait for a process to terminate, given its pid.

    Args:
        pid: Process ID to wait for.
        timeout: Optional timeout in seconds to wait for the process to terminate.

    Returns:
        True if the process terminated within the given timeout, False otherwise.
    """
    start = time.time()
    while time.time() - start < timeout:
        if not _pid_is_running(pid):
            return True
        time.sleep(0.1)
    return False
