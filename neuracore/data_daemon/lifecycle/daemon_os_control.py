"""OS-facing control helpers for daemon process management."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import FrameType
from typing import cast

import filelock

from neuracore.data_daemon.const import SOCKET_PATH
from neuracore.data_daemon.helpers import get_daemon_db_path, get_daemon_pid_path

# cspell:ignore WNOHANG waitpid

logger = logging.getLogger(__name__)

# /proc and other Linux-only primitives are skipped on non-Linux platforms.
# Callers should not rely on Linux-only semantics; helpers degrade gracefully.
_IS_LINUX = sys.platform.startswith("linux")


class DaemonLifecycleError(RuntimeError):
    """Raised when daemon lifecycle checks fail."""


def get_daemon_log_path() -> Path:
    """Return the path for the background daemon's combined stdout/stderr log."""
    return Path(
        os.environ.get(
            "NEURACORE_DAEMON_LOG_PATH",
            str(Path.home() / ".neuracore" / "logs" / "daemon.log"),
        )
    )


def open_daemon_log_stream() -> tuple[int, Path] | tuple[None, None]:
    """Open (or rotate) the daemon log file and return (fd, path).

    The returned file descriptor is owned by the caller and must be closed
    after passing it to subprocess.Popen. Returns (None, None) when the
    log destination cannot be created (caller falls back to DEVNULL).
    """
    log_path = get_daemon_log_path()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Rotate manually on each daemon launch so each spawn starts a fresh
        # file but the previous N runs are preserved for post-mortem.
        if log_path.exists() and log_path.stat().st_size > 0:
            handler = RotatingFileHandler(
                str(log_path), maxBytes=1, backupCount=5
            )
            try:
                handler.doRollover()
            finally:
                handler.close()
        fd = os.open(
            log_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        return fd, log_path
    except OSError as error:
        logger.warning(
            "Failed to open daemon log file %s: %s; falling back to DEVNULL",
            log_path,
            error,
        )
        return None, None


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


def _is_zombie(pid_value: int) -> bool:
    """Return True if pid_value is a zombie process.

    Linux uses /proc/<pid>/stat. macOS/BSD have no /proc filesystem, so we
    fall back to a non-blocking waitpid for our own children; for foreign
    PIDs we simply report False (best-effort, matches pre-existing behavior).
    """
    if _IS_LINUX:
        try:
            stat = Path(f"/proc/{pid_value}/stat").read_text(encoding="utf-8")
            # State is field 3, after the comm field enclosed in parens.
            state = stat.split(")")[1].split()[0]
            return state == "Z"
        except OSError:
            return False

    # Non-Linux fallback: a zombie child of *this* process will be reaped
    # by waitpid(WNOHANG); for unrelated PIDs we cannot detect zombie state.
    try:
        reaped_pid, _ = os.waitpid(pid_value, os.WNOHANG)
        return reaped_pid != 0
    except (ChildProcessError, OSError):
        return False


def _try_reap_zombie_child(pid_value: int) -> bool:
    """Attempt a non-blocking waitpid to reap a zombie child; return True if reaped."""
    try:
        reaped_pid, _ = os.waitpid(pid_value, os.WNOHANG)
        return reaped_pid != 0
    except ChildProcessError:
        return False
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


def management_socket_is_alive(socket_path: Path = SOCKET_PATH) -> bool:
    """Return True when *something* is listening on the management socket.

    A bare file at the path means nothing — leftover sockets from crashed
    daemons routinely accumulate. We actively try to connect: that is the
    only reliable signal that a second daemon would race with an existing
    one over /tmp/ndd/management.sock.
    """
    import socket as _socket

    if not socket_path.exists():
        return False

    probe = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    probe.settimeout(0.25)
    try:
        probe.connect(str(socket_path))
    except (ConnectionRefusedError, FileNotFoundError):
        return False
    except OSError:
        # E.g. PermissionError, timeout — assume something is there and
        # let the caller surface a clear error rather than racing.
        return True
    else:
        return True
    finally:
        try:
            probe.close()
        except OSError:
            pass


def cleanup_stale_client_state(
    *,
    pid_path: Path,
    db_path: Path,
    socket_paths: Sequence[str],
) -> None:
    """Clean up stale pid/sockets/db state when no running daemon corresponds to it."""
    existing_pid = read_pid_from_file(pid_path)
    if existing_pid is not None and pid_is_running(existing_pid):
        return

    sockets_present = any(Path(path).exists() for path in socket_paths)
    pid_file_present = pid_path.exists()

    if existing_pid is None and not pid_file_present and not sockets_present:
        return

    from neuracore.data_daemon.lifecycle.runtime_recovery import (
        cleanup_stale_shared_slot_segments,
        shutdown,
    )

    shutdown(
        pid_path=pid_path,
        socket_paths=tuple(Path(path) for path in socket_paths),
        db_path=db_path,
    )
    cleanup_stale_shared_slot_segments()


def _build_daemon_runner_command() -> list[str]:
    """Build the command used to launch the daemon runner entrypoint."""
    return [
        sys.executable,
        "-m",
        "neuracore.data_daemon.runner_entry",
    ]


def _build_daemon_launch_env(
    *,
    pid_path: Path,
    db_path: Path,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the environment for launching the daemon subprocess."""
    environment = os.environ.copy()
    environment["NEURACORE_DAEMON_PID_PATH"] = str(pid_path)
    environment["NEURACORE_DAEMON_DB_PATH"] = str(db_path)
    environment["NEURACORE_DAEMON_MANAGE_PID"] = "0"
    if env_overrides:
        environment.update(env_overrides)
    return cast(dict[str, str], environment)


def _start_daemon_subprocess(
    *,
    pid_path: Path,
    db_path: Path,
    background: bool,
    env_overrides: dict[str, str] | None = None,
    stdout: int | None = None,
    stderr: int | None = None,
) -> subprocess.Popen:
    """Start the daemon runner subprocess with the requested terminal mode."""
    environment = _build_daemon_launch_env(
        pid_path=pid_path,
        db_path=db_path,
        env_overrides=env_overrides,
    )
    current_working_directory = str(Path.cwd())

    try:
        if background:
            # Background daemons must not silently discard stderr — auth/SSL/
            # upload failures need to land somewhere a user can find them.
            log_fd, log_path = open_daemon_log_stream()
            try:
                stdout_target = log_fd if log_fd is not None else subprocess.DEVNULL
                # stderr is still piped during startup so launch_daemon_subprocess
                # can capture an immediate-exit error; once startup completes the
                # daemon's own logging handlers (configured to the same log file
                # via NEURACORE_DAEMON_LOG_PATH) keep writing.
                if log_path is not None:
                    environment["NEURACORE_DAEMON_LOG_PATH"] = str(log_path)
                process = subprocess.Popen(
                    _build_daemon_runner_command(),
                    close_fds=True,
                    cwd=current_working_directory,
                    env=environment,
                    start_new_session=True,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_target,
                    stderr=subprocess.PIPE,
                )
            finally:
                if log_fd is not None:
                    os.close(log_fd)
            return process

        return subprocess.Popen(
            _build_daemon_runner_command(),
            close_fds=True,
            cwd=current_working_directory,
            env=environment,
            start_new_session=False,
            stdout=stdout,
            stderr=stderr,
        )
    except OSError as error:
        raise RuntimeError(f"Failed to start daemon: {error}") from error


def launch_daemon_subprocess(
    *,
    pid_path: Path,
    db_path: Path,
    background: bool = True,
    timeout_s: float = 10.0,
    env_overrides: dict[str, str] | None = None,
    stdout: int | None = None,
    stderr: int | None = None,
) -> subprocess.Popen:
    """Launch the daemon runner subprocess and poll until it is ready."""
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    process = _start_daemon_subprocess(
        pid_path=pid_path,
        db_path=db_path,
        background=background,
        env_overrides=env_overrides,
        stdout=stdout,
        stderr=stderr,
    )
    socket_poll_interval_s = 0.05
    daemon_startup_timeout_s = time.monotonic() + timeout_s

    while time.monotonic() < daemon_startup_timeout_s:
        if process.poll() is not None:
            stderr_output = ""
            if process.stderr is not None:
                stderr_output = process.stderr.read().decode(errors="replace").strip()
            detail = f"\n{stderr_output}" if stderr_output else ""
            raise RuntimeError(
                f"Daemon process exited unexpectedly during startup "
                f"(exit code {process.returncode}).{detail}"
            )
        if SOCKET_PATH.exists():
            break
        time.sleep(socket_poll_interval_s)
    else:
        process.terminate()
        raise RuntimeError(
            f"Daemon did not become ready within {timeout_s}s: "
            f"socket {SOCKET_PATH} never appeared."
        )

    pid_path.write_text(str(process.pid), encoding="utf-8")
    return process


def launch_new_daemon_subprocess(
    *,
    pid_path: Path,
    db_path: Path,
    background: bool,
    timeout_s: float = 10.0,
    env_overrides: dict[str, str] | None = None,
    stdout: int | None = None,
    stderr: int | None = None,
) -> subprocess.Popen:
    """Launch a new daemon subprocess, rejecting an already-running daemon."""
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_file_lock = str(pid_path) + ".lock"

    with filelock.FileLock(pid_file_lock):
        existing_pid = read_pid_from_file(pid_path)
        if existing_pid is not None and pid_is_running(existing_pid):
            raise DaemonLifecycleError(f"Daemon already running (pid={existing_pid})")

        if management_socket_is_alive():
            raise DaemonLifecycleError(
                f"A daemon already owns {SOCKET_PATH} (no pid file present). "
                "Stop that daemon before launching a new one."
            )

        cleanup_stale_client_state(
            pid_path=pid_path,
            db_path=db_path,
            socket_paths=(str(SOCKET_PATH),),
        )

        return launch_daemon_subprocess(
            pid_path=pid_path,
            db_path=db_path,
            background=background,
            timeout_s=timeout_s,
            env_overrides=env_overrides,
            stdout=stdout,
            stderr=stderr,
        )


def ensure_daemon_running(
    *,
    timeout_s: float = 10.0,
    env_overrides: dict[str, str] | None = None,
) -> int:
    """Ensure the data daemon is running and ready to accept connections."""
    pid_path = get_daemon_pid_path()
    db_path = get_daemon_db_path()
    pid_file_lock = str(pid_path) + ".lock"

    os.environ.setdefault("NEURACORE_DAEMON_PID_PATH", str(pid_path))
    os.environ.setdefault("NEURACORE_DAEMON_DB_PATH", str(db_path))

    with filelock.FileLock(pid_file_lock):
        existing_pid = read_pid_from_file(pid_path)
        if existing_pid is not None and pid_is_running(existing_pid):
            return existing_pid

        # Defend against the MANAGE_PID=0 race: a manually-launched daemon
        # may be holding the management socket without owning daemon.pid.
        # Spawning a second daemon in that state causes both to fight over
        # /tmp/ndd/management.sock and the client to use whichever bound
        # second. If something is already serving the socket, adopt it
        # instead of starting another one.
        if management_socket_is_alive():
            raise DaemonLifecycleError(
                f"A daemon already owns {SOCKET_PATH} but no pid file is "
                "present (likely launched with NEURACORE_DAEMON_MANAGE_PID=0). "
                "Stop that daemon first or set NEURACORE_DAEMON_MANAGE_PID=1 "
                "so its lifecycle is tracked."
            )

        cleanup_stale_client_state(
            pid_path=pid_path,
            db_path=db_path,
            socket_paths=(str(SOCKET_PATH),),
        )
        process = launch_daemon_subprocess(
            pid_path=pid_path,
            db_path=db_path,
            background=True,
            timeout_s=timeout_s,
            env_overrides=env_overrides,
        )
        return process.pid


def acquire_pid_file(pid_path: Path) -> bool:
    """Create a pid file atomically; return True if created or stale cleared."""
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        file_descriptor = os.open(pid_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        existing_pid = read_pid_from_file(pid_path)
        if existing_pid and pid_is_running(existing_pid):
            raise DaemonLifecycleError(f"Daemon already running (pid={existing_pid})")
        try:
            pid_path.unlink()
        except FileNotFoundError:
            pass
        return acquire_pid_file(pid_path)
    with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    return True


def remove_pid_file(pid_path: Path) -> None:
    """Remove the pid file if present."""
    try:
        pid_path.unlink(missing_ok=True)
    except FileNotFoundError:
        return


def install_signal_handlers(
    *,
    on_shutdown: Callable[[int], None] | None = None,
    on_reload: Callable[[], None] | None = None,
) -> None:
    """Install signal handlers for graceful shutdown and optional reload."""

    def _handle_shutdown(signal_number: int, _stack_frame: FrameType | None) -> None:
        if on_shutdown:
            on_shutdown(signal_number)
        raise KeyboardInterrupt

    def _handle_reload(_signal_number: int, _stack_frame: FrameType | None) -> None:
        if on_reload:
            on_reload()

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _handle_reload)


def terminate_pid(pid_value: int) -> bool:
    """Send SIGTERM to the given PID."""
    try:
        os.kill(pid_value, signal.SIGTERM)
        return True
    except ProcessLookupError:
        return True
    except PermissionError:
        return False


def force_kill(pid_value: int) -> bool:
    """Send SIGKILL to the given PID."""
    try:
        os.kill(pid_value, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return True
    except PermissionError:
        return False


def wait_for_exit(pid_value: int, *, timeout_s: float) -> bool:
    """Wait for a PID to stop running until a timeout elapses."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _try_reap_zombie_child(pid_value) or not pid_is_running(pid_value):
            return True
        time.sleep(0.1)
    return False


__all__ = [
    "DaemonLifecycleError",
    "acquire_pid_file",
    "cleanup_stale_client_state",
    "ensure_daemon_running",
    "force_kill",
    "install_signal_handlers",
    "launch_daemon_subprocess",
    "launch_new_daemon_subprocess",
    "pid_is_running",
    "read_pid_from_file",
    "remove_pid_file",
    "terminate_pid",
    "wait_for_exit",
]
