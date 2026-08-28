"""OS-facing control helpers for daemon process management."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import IO, Any, cast

import filelock

from neuracore.data_daemon.binary import (
    INSTALL_COMMAND_HINT,
    REBUILD_COMMAND_HINT,
    data_daemon_binary_path,
    require_data_daemon_binary,
)
from neuracore.data_daemon.const import DEFAULT_DAEMON_STARTUP_TIMEOUT_SECONDS
from neuracore.data_daemon.helpers import get_daemon_db_path, get_daemon_pid_path

# Reported for a daemon that answers the health probe but has no version
# service, which is every daemon built before that service existed.
UNKNOWN_DAEMON_VERSION = "unknown (daemon too old to report a version)"

STOP_COMMAND_HINT = (
    "`neuracore data-daemon stop` (or `python -m neuracore.data_daemon stop`)"
)

STALE_EXTENSION_MESSAGE = (
    "The neuracore native extension of this install has no daemon version "
    "query, so it is older than this neuracore code. Build the artefacts "
    f"again with {REBUILD_COMMAND_HINT} (source checkouts), or install the "
    f"package again with {INSTALL_COMMAND_HINT}."
)

# The daemon's own stop path budgets 10 seconds for a graceful SIGTERM exit
# (see rust/data_daemon/src/cli/stop.rs). A just-launched idle daemon exits
# well inside that; the wait mostly reaps the child promptly.
STALE_DAEMON_STOP_WAIT_SECONDS = 10.0

logger = logging.getLogger(__name__)


class DaemonLifecycleError(RuntimeError):
    """Raised when daemon lifecycle checks fail."""


class DaemonVersionMismatchError(DaemonLifecycleError):
    """Raised when the running daemon comes from a different neuracore version."""


# Cargo needs semver, so an alpha release carries `15.0.1-alpha.7` in the rust
# workspace while PyPI normalises the same release to `15.0.1a7`. Pre-releases
# are the only versions where the two spellings differ.
_CARGO_PRE_RELEASE = re.compile(r"-alpha\.(\d+)$")


def _pep440(version: str) -> str:
    """Return a cargo version string in the PEP 440 spelling."""
    return _CARGO_PRE_RELEASE.sub(r"a\1", version)


def _sdk_version() -> str | None:
    """Return the installed neuracore package version, or None without metadata.

    A bare source tree that was never pip-installed has no metadata and so no
    version to compare against the daemon.
    """
    try:
        return importlib.metadata.version("neuracore")
    except importlib.metadata.PackageNotFoundError:
        return None


def read_pid_from_file(pid_path: Path) -> int | None:
    """Read an integer PID from `pid_path`, returning None if missing/invalid."""
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

    return pid_value if pid_value > 0 else None


def _import_data_bridge() -> Any | None:
    """Return the native bridge module, or None when it cannot be imported."""
    try:
        return cast(Any, importlib.import_module("neuracore.data_daemon._data_bridge"))
    except (ImportError, OSError):
        return None


def _daemon_ready(pid_path: Path, *, timeout_s: float = 0.0) -> bool:
    """Return True once the daemon answers an IPC health probe."""
    pid_value = read_pid_from_file(pid_path)
    if pid_value is None or not pid_is_running(pid_value):
        return False
    data_bridge = _import_data_bridge()
    if data_bridge is None:
        return False

    try:
        ready_pid = data_bridge.wait_until_ready(timeout_s)
    except (AttributeError, RuntimeError):
        return False
    return ready_pid == pid_value


def _format_version_mismatch(
    daemon_version: str | None, sdk_version: str, just_launched: bool
) -> str:
    """Build the user-facing message for a daemon from a different version."""
    reported_version = (
        UNKNOWN_DAEMON_VERSION if daemon_version is None else daemon_version
    )
    opening = (
        f"The running data daemon was built from neuracore {reported_version}, "
        f"but neuracore {sdk_version} is installed. "
    )
    if just_launched:
        return opening + (
            "This daemon was just launched from the daemon binary of this "
            "install, so that binary is stale; it was stopped again. Build "
            f"the binary again with {REBUILD_COMMAND_HINT} (source "
            f"checkouts), or install the package again with {INSTALL_COMMAND_HINT}."
        )
    if data_daemon_binary_path() is None:
        return opening + (
            "This install has no daemon binary to replace it: install the "
            f"package again with {INSTALL_COMMAND_HINT}, then stop the old daemon "
            f"with {STOP_COMMAND_HINT}. The next use of neuracore starts the "
            "daemon of this install automatically."
        )
    return opening + (
        f"Stop the old daemon with {STOP_COMMAND_HINT}. The next use of "
        "neuracore starts the daemon of this install automatically. In a "
        "source checkout, the bundled daemon can itself be older than the "
        f"installed package: build it again with {REBUILD_COMMAND_HINT}."
    )


def _check_daemon_version(timeout_s: float, *, just_launched: bool = False) -> None:
    """Raise unless the running daemon matches the installed neuracore version.

    Callers run this right after a passing health probe and pass the probe's
    own time budget, so a daemon that stays silent for the whole budget has no
    version service and is too old.

    The check is skipped with a warning when the package has no installed
    metadata (there is no version to compare), when the native extension is
    not importable (defensive only: a passing health probe has already
    imported it), and when the query fails in transport, where the daemon's
    version is not the problem. A native extension that imports but has no
    ``daemon_version`` raises a plain :class:`DaemonLifecycleError` naming the
    extension, not the daemon, as stale.
    """
    sdk_version = _sdk_version()
    if sdk_version is None:
        logger.warning(
            "Skipped the daemon version check: the neuracore package has no "
            "installed metadata."
        )
        return
    data_bridge = _import_data_bridge()
    if data_bridge is None:
        logger.warning(
            "Skipped the daemon version check: the native extension is not "
            "importable."
        )
        return
    try:
        daemon_version = cast("str | None", data_bridge.daemon_version(timeout_s))
    except AttributeError:
        raise DaemonLifecycleError(STALE_EXTENSION_MESSAGE) from None
    except RuntimeError as error:
        logger.warning("Skipped the daemon version check: %s", error)
        return
    if daemon_version is None or _pep440(daemon_version) != sdk_version:
        raise DaemonVersionMismatchError(
            _format_version_mismatch(daemon_version, sdk_version, just_launched)
        )


def _is_zombie(pid_value: int) -> bool:
    """Return True if pid_value is a zombie process (Linux /proc only)."""
    try:
        stat = Path(f"/proc/{pid_value}/stat").read_text(encoding="utf-8")
        # State is field 3, after the comm field enclosed in parens.
        state = stat.split(")")[1].split()[0]
        return state == "Z"
    except OSError:
        return False


def pid_is_running(pid_value: int) -> bool:
    """Return True if pid_value exists and is not a zombie."""
    try:
        os.kill(pid_value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return not _is_zombie(pid_value)


def _build_daemon_launch_env(
    *,
    pid_path: Path,
    db_path: Path,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the environment for launching the daemon subprocess.

    The daemon manages its own PID file (see
    [rust/data_daemon/src/cli/launch.rs](../../rust/data_daemon/src/cli/launch.rs)),
    so the parent only tells it where that file lives.
    """
    environment = os.environ.copy()
    environment["NEURACORE_DAEMON_PID_PATH"] = str(pid_path)
    environment["NEURACORE_DAEMON_DB_PATH"] = str(db_path)
    if env_overrides:
        environment.update(env_overrides)
    return cast(dict[str, str], environment)


def _start_daemon_subprocess(
    pid_path: Path,
    db_path: Path,
    background: bool,
    env_overrides: dict[str, str] | None = None,
    stdout: int | None = None,
    stderr: int | None = None,
) -> tuple[subprocess.Popen, Path | None]:
    """Start the daemon subprocess with the requested terminal mode.

    Returns the process together with the log path its stderr was routed to
    in background mode (``None`` in the foreground). A long-lived background
    daemon must not inherit an undrained ``subprocess.PIPE`` — once the pipe
    buffer fills, the daemon blocks on its next stderr write and hangs. Sending
    stderr to ``DEVNULL`` avoids that, but throws away the reason for a startup
    failure. Routing to a file gets both: writes never block, and the caller
    can read the daemon's own error output back if it exits prematurely.

    The binary's ``launch`` subcommand stays in the foreground of the spawned
    process so the parent keeps ordinary ``Popen`` semantics over it.
    """
    environment = _build_daemon_launch_env(
        pid_path=pid_path,
        db_path=db_path,
        env_overrides=env_overrides,
    )
    command = [str(require_data_daemon_binary()), "launch"]
    current_working_directory = str(Path.cwd())

    daemon_log_path: Path | None = None
    daemon_log_handle: IO[bytes] | None = None
    if background:
        candidate_log_path = db_path.parent / "daemon.log"
        try:
            candidate_log_path.parent.mkdir(parents=True, exist_ok=True)
            # Truncate so the log reflects this run only; the daemon's own
            # stderr (tracing output / early eprintln failures) lands here.
            daemon_log_handle = open(
                candidate_log_path, "wb", buffering=0
            )  # noqa: SIM115
        except OSError:
            # Fall back to discarding stderr rather than failing the launch.
            daemon_log_handle = None
        else:
            daemon_log_path = candidate_log_path

    try:
        if background:
            stderr_target: int | IO[bytes] = (
                daemon_log_handle
                if daemon_log_handle is not None
                else subprocess.DEVNULL
            )
            process = subprocess.Popen(
                command,
                close_fds=True,
                cwd=current_working_directory,
                env=environment,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr_target,
            )
        else:
            process = subprocess.Popen(
                command,
                close_fds=True,
                cwd=current_working_directory,
                env=environment,
                start_new_session=False,
                stdout=stdout,
                stderr=stderr,
            )
    except OSError as error:
        if daemon_log_handle is not None:
            daemon_log_handle.close()
        raise RuntimeError(f"Failed to start daemon: {error}") from error

    if daemon_log_handle is not None:
        daemon_log_handle.close()
    return process, daemon_log_path


# Cap on how much of the daemon log we fold into a premature-exit error, so a
# verbose-but-then-crashing daemon can't produce a multi-megabyte exception.
_DAEMON_FAILURE_DETAIL_TAIL_BYTES = 8192


def _read_daemon_failure_detail(
    process: subprocess.Popen, daemon_log_path: Path | None
) -> str:
    """Return the trailing daemon output to append to a premature-exit error.

    Background launches route the daemon's stderr to ``daemon_log_path``;
    foreground launches may instead expose a readable ``process.stderr`` pipe.
    Returns a newline-prefixed snippet, or an empty string when no output is
    available.
    """
    output = ""
    if daemon_log_path is not None:
        try:
            log_bytes = daemon_log_path.read_bytes()
        except OSError:
            log_bytes = b""
        tail = log_bytes[-_DAEMON_FAILURE_DETAIL_TAIL_BYTES:]
        output = tail.decode(errors="replace").strip()
    elif process.stderr is not None:
        output = process.stderr.read().decode(errors="replace").strip()
    return f"\n{output}" if output else ""


def launch_daemon_subprocess(
    pid_path: Path,
    db_path: Path,
    background: bool = True,
    timeout_s: float = DEFAULT_DAEMON_STARTUP_TIMEOUT_SECONDS,
    env_overrides: dict[str, str] | None = None,
    stdout: int | None = None,
    stderr: int | None = None,
) -> subprocess.Popen:
    """Launch the daemon subprocess and poll until it is ready.

    The daemon answers a side-effect-free iceoryx2 health query once its IPC
    listener and dispatcher are live; it owns the PID file itself, so the
    parent must not overwrite it.
    """
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    process, daemon_log_path = _start_daemon_subprocess(
        pid_path=pid_path,
        db_path=db_path,
        background=background,
        env_overrides=env_overrides,
        stdout=stdout,
        stderr=stderr,
    )
    poll_interval_s = 0.05
    daemon_startup_timeout_s = time.monotonic() + timeout_s

    while time.monotonic() < daemon_startup_timeout_s:
        if process.poll() is not None:
            detail = _read_daemon_failure_detail(process, daemon_log_path)
            raise RuntimeError(
                f"Daemon process exited unexpectedly during startup "
                f"(exit code {process.returncode}).{detail}"
            )
        if _daemon_ready(pid_path, timeout_s=0.0):
            break
        time.sleep(poll_interval_s)
    else:
        process.terminate()
        raise RuntimeError(
            f"Daemon did not become ready within {timeout_s}s: "
            "IPC health probe never answered."
        )

    return process


def ensure_daemon_running(
    timeout_s: float = DEFAULT_DAEMON_STARTUP_TIMEOUT_SECONDS,
    env_overrides: dict[str, str] | None = None,
) -> int:
    """Ensure the data daemon is running and ready to accept connections.

    A stale PID file from an unclean exit needs no cleanup here: the daemon's
    ``launch`` reclaims it before acquiring its own, alongside any leftover
    iceoryx2 artefacts.

    Both paths check the daemon's build version against the installed
    neuracore version: an adopted daemon can be left over from an earlier
    install, and a daemon just launched from a source checkout can come from a
    binary that was built before the last change.

    The readiness probe and the version check each get the full ``timeout_s``
    budget, so against a daemon too old to serve versions the PID file lock
    can be held for up to twice that budget before this raises; on the launch
    path a version mismatch adds up to ``STALE_DAEMON_STOP_WAIT_SECONDS`` on
    top, because the just-launched daemon is stopped and reaped before the
    raise. Only a version mismatch stops that daemon: a stale-extension error
    leaves it running, since the daemon itself may be the correct version.
    """
    pid_path = get_daemon_pid_path()
    db_path = get_daemon_db_path()
    pid_file_lock = str(pid_path) + ".lock"

    os.environ.setdefault("NEURACORE_DAEMON_PID_PATH", str(pid_path))
    os.environ.setdefault("NEURACORE_DAEMON_DB_PATH", str(db_path))

    with filelock.FileLock(pid_file_lock):
        existing_pid = read_pid_from_file(pid_path)
        if existing_pid is not None and pid_is_running(existing_pid):
            if _daemon_ready(pid_path, timeout_s=timeout_s):
                _check_daemon_version(timeout_s)
                return existing_pid
            raise DaemonLifecycleError(
                f"Daemon process is running (pid={existing_pid}) but did not "
                "become ready to accept IPC commands."
            )

        process = launch_daemon_subprocess(
            pid_path=pid_path,
            db_path=db_path,
            background=True,
            timeout_s=timeout_s,
            env_overrides=env_overrides,
        )
        try:
            _check_daemon_version(timeout_s, just_launched=True)
        except DaemonVersionMismatchError:
            process.terminate()
            try:
                process.wait(timeout=STALE_DAEMON_STOP_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "The stale daemon (pid=%s) did not exit after SIGTERM.",
                    process.pid,
                )
            raise
        return process.pid


__all__ = [
    "DaemonLifecycleError",
    "DaemonVersionMismatchError",
    "ensure_daemon_running",
    "launch_daemon_subprocess",
    "pid_is_running",
    "read_pid_from_file",
]
