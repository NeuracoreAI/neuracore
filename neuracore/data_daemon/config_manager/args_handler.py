"""Handlers for nc-data-daemon CLI commands."""

from __future__ import annotations

import signal
from pathlib import Path

import typer

from neuracore.data_daemon.config_manager.cli_options import (
    ApiKeyOption,
    BackgroundOption,
    BandwidthLimitOption,
    CurrentOrgIdOption,
    DebugOption,
    KeepWakelockOption,
    LaunchProfileOption,
    NumThreadsOption,
    OfflineOption,
    ProfileNameCreateArgument,
    ProfileNameDeleteArgument,
    ProfileNameGetArgument,
    ProfileNameUpdateArgument,
    StorageLimitOption,
    StoragePathOption,
)
from neuracore.data_daemon.config_manager.helpers import collect_config_updates
from neuracore.data_daemon.config_manager.profiles import (
    ProfileAlreadyExist,
    ProfileManager,
    ProfileNotFound,
)
from neuracore.data_daemon.const import DEFAULT_PROFILE_NAME, SOCKET_PATH
from neuracore.data_daemon.helpers import get_daemon_db_path, get_daemon_pid_path
from neuracore.data_daemon.lifecycle.daemon_os_control import (
    DaemonLifecycleError,
    cleanup_stale_client_state,
    force_kill,
    launch_new_daemon_subprocess,
    pid_is_running,
    read_pid_from_file,
    terminate_pid,
    wait_for_exit,
)
from neuracore.data_daemon.lifecycle.runtime_recovery import shutdown

profile_manager = ProfileManager()
profile_app = typer.Typer(help="Manage daemon profiles.")


def _update_profile(
    *,
    profile_name: str,
    create_if_missing: bool,
    storage_limit: int | None,
    bandwidth_limit: int | None,
    path_to_store_record: str | None,
    num_threads: int | None,
    keep_wakelock_while_upload: bool | None,
    offline: bool | None,
    api_key: str | None,
    current_org_id: str | None,
) -> None:
    updates = collect_config_updates(
        storage_limit=storage_limit,
        bandwidth_limit=bandwidth_limit,
        path_to_store_record=path_to_store_record,
        num_threads=num_threads,
        keep_wakelock_while_upload=keep_wakelock_while_upload,
        offline=offline,
        api_key=api_key,
        current_org_id=current_org_id,
    )

    if create_if_missing:
        try:
            profile_manager.create_profile(profile_name)
        except ProfileAlreadyExist:
            pass
        except Exception:
            typer.echo(
                f"Failed to create default profile {profile_name!r}.",
                err=True,
            )
            raise typer.Exit(code=1)

    try:
        profile_manager.update_profile(profile_name, updates)
        typer.echo(f"Updated profile {profile_name!r}.")
    except ProfileNotFound as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)


@profile_app.command("create")
def run_profile_create(name: ProfileNameCreateArgument) -> None:
    """Create a profile."""
    try:
        profile_manager.create_profile(name)
        typer.echo(f"Created profile {name!r}.")
    except ProfileAlreadyExist as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)


@profile_app.command("update")
def run_profile_update(
    name: ProfileNameUpdateArgument = None,
    storage_limit: StorageLimitOption = None,
    bandwidth_limit: BandwidthLimitOption = None,
    path_to_store_record: StoragePathOption = None,
    num_threads: NumThreadsOption = None,
    keep_wakelock_while_upload: KeepWakelockOption = None,
    offline: OfflineOption = None,
    api_key: ApiKeyOption = None,
    current_org_id: CurrentOrgIdOption = None,
) -> None:
    """Update an existing profile."""
    if not name:
        name = DEFAULT_PROFILE_NAME

    _update_profile(
        profile_name=name,
        create_if_missing=False,
        storage_limit=storage_limit,
        bandwidth_limit=bandwidth_limit,
        path_to_store_record=path_to_store_record,
        num_threads=num_threads,
        keep_wakelock_while_upload=keep_wakelock_while_upload,
        offline=offline,
        api_key=api_key,
        current_org_id=current_org_id,
    )


@profile_app.command("get")
def run_profile_get(
    name: ProfileNameGetArgument = None,
) -> None:
    """Get a profile's configuration."""
    if not name:
        name = DEFAULT_PROFILE_NAME
    try:
        config = profile_manager.get_profile(name)
    except ProfileNotFound as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    typer.echo(config.model_dump_json(indent=2))


@profile_app.command("delete")
def run_profile_delete(
    name: ProfileNameDeleteArgument,
) -> None:
    """Delete a profile."""
    if name == DEFAULT_PROFILE_NAME:
        typer.echo(
            f"Cannot delete default profile {DEFAULT_PROFILE_NAME!r}.",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        profile_manager.delete_profile(name)
        typer.echo(f"Deleted profile {name!r}.")
    except ProfileNotFound as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)


@profile_app.command("list")
def run_list_profiles() -> None:
    """List all configured daemon profiles."""
    profiles = profile_manager.list_profiles()
    if not profiles:
        typer.echo("No profiles found.")
        return

    for name in profiles:
        typer.echo(name)


def run_launch(
    profile: LaunchProfileOption = None,
    background: BackgroundOption = False,
    debug: DebugOption = False,
) -> None:
    """Launch the data daemon."""
    pid_path = get_daemon_pid_path()
    db_path = get_daemon_db_path()

    env_overrides: dict[str, str] = {}
    if profile is not None:
        try:
            profile_manager.get_profile(profile)
        except ProfileNotFound as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1)
        env_overrides["NEURACORE_DAEMON_PROFILE"] = profile

    if debug:
        env_overrides["NDD_DEBUG"] = "true"

    try:
        daemon_process = launch_new_daemon_subprocess(
            pid_path=pid_path,
            db_path=db_path,
            background=background,
            env_overrides=env_overrides or None,
        )
    except DaemonLifecycleError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    spawned_pid = daemon_process.pid
    typer.echo(f"Daemon launched (pid={spawned_pid}).")

    if background:
        return

    try:
        daemon_process.wait()
    except KeyboardInterrupt:
        try:
            daemon_process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            return
        daemon_process.wait()


def run_stop() -> None:
    """Stop the data daemon."""
    pid_path = get_daemon_pid_path()
    db_path = get_daemon_db_path()

    pid_value = read_pid_from_file(pid_path)
    if pid_value is None:
        typer.echo("Daemon is not running.")
        return

    if not pid_is_running(pid_value):
        shutdown(
            pid_path=pid_path,
            socket_paths=(Path(str(SOCKET_PATH)),),
            db_path=db_path,
        )
        typer.echo("Daemon stopped.")
        return

    if not terminate_pid(pid_value):
        typer.echo(f"Permission denied sending SIGTERM to pid={pid_value}.", err=True)
        raise typer.Exit(code=1)

    if wait_for_exit(pid_value, timeout_s=10.0):
        shutdown(
            pid_path=pid_path,
            socket_paths=(Path(str(SOCKET_PATH)),),
            db_path=db_path,
        )
        typer.echo("Daemon stopped.")
        return

    if not force_kill(pid_value):
        typer.echo(f"Permission denied sending SIGKILL to pid={pid_value}.", err=True)
        raise typer.Exit(code=1)

    if wait_for_exit(pid_value, timeout_s=5.0):
        shutdown(
            pid_path=pid_path,
            socket_paths=(Path(str(SOCKET_PATH)),),
            db_path=db_path,
        )
        typer.echo("Daemon stopped (forced).")
        return

    typer.echo(f"Failed to stop daemon (pid={pid_value}).", err=True)
    raise typer.Exit(code=1)


def run_status() -> None:
    """Show daemon status."""
    pid_path = get_daemon_pid_path()
    db_path = get_daemon_db_path()

    pid_value = read_pid_from_file(pid_path)
    if pid_value is None:
        typer.echo("Daemon not running.")
        return

    if not pid_is_running(pid_value):
        cleanup_stale_client_state(
            pid_path=pid_path,
            db_path=db_path,
            socket_paths=(str(SOCKET_PATH),),
        )
        typer.echo("Daemon not running.")
        return

    typer.echo(f"Daemon running (pid={pid_value}).")


def run_install() -> None:
    """Install the data daemon as a system service."""
    typer.echo("Install command is not implemented yet.")


def run_uninstall() -> None:
    """Uninstall the data daemon system service."""
    typer.echo("Uninstall command is not implemented yet.")
