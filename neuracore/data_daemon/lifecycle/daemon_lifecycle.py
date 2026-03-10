"""Lifecycle helpers for daemon startup, recovery, and shutdown."""

from __future__ import annotations

import logging
import os
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from types import FrameType

from neuracore.data_daemon.const import RECORDING_EVENTS_SOCKET_PATH, SOCKET_PATH
from neuracore.data_daemon.helpers import get_daemon_db_path, get_daemon_pid_path
from neuracore.data_daemon.models import TraceErrorCode, TraceStatus
from neuracore.data_daemon.state_management.state_store import StateStore

logger = logging.getLogger(__name__)


class DaemonLifecycleError(RuntimeError):
    """Raised when daemon lifecycle checks fail."""


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


def pid_is_running(pid_value: int) -> bool:
    """Return True if `pid_value` exists (or cannot be checked due to permissions)."""
    try:
        os.kill(pid_value, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def wait_for_socket_paths(
    socket_paths: Sequence[str],
    *,
    timeout_s: float = 5.0,
    poll_s: float = 0.05,
    process: subprocess.Popen | None = None,
) -> None:
    """Wait for daemon socket files to be created.

    Polls the filesystem until all `socket_paths` exist or the timeout elapses.
    If `process` is provided, the wait fails fast if the process exits before
    the sockets are created.

    Args:
        socket_paths: Paths that must exist before the daemon is considered ready.
        timeout_s: Maximum seconds to wait before raising.
        poll_s: Poll interval in seconds.
        process: Optional subprocess handle for the daemon.

    Raises:
        RuntimeError: If the daemon process exits early (when `process` is given),
            or if the sockets are not created within `timeout_s`.
    """
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                "Data daemon exited before becoming ready. To get full error trace, "
                "run daemon directly with `nc-data-daemon launch`."
            )

        if all(Path(p).exists() for p in socket_paths):
            return

        time.sleep(poll_s)

    raise RuntimeError(
        "Data daemon did not become ready: expected sockets not created."
    )


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

    sockets_present = any(Path(p).exists() for p in socket_paths)
    pid_file_present = pid_path.exists()

    if existing_pid is None and not pid_file_present and not sockets_present:
        return

    shutdown(
        pid_path=pid_path,
        socket_paths=tuple(Path(p) for p in socket_paths),
        db_path=db_path,
    )


def launch_daemon_subprocess(
    *,
    pid_path: Path,
    db_path: Path,
    timeout_s: float = 5.0,
) -> int:
    """Launch the daemon runner as a detached subprocess and wait for readiness.

    Args:
        pid_path: Path to write the daemon pid file.
        db_path: Path to the daemon SQLite state database.
        timeout_s: Max seconds to wait for sockets.

    Returns:
        Spawned daemon pid.

    Raises:
        RuntimeError: If the daemon fails to start or become ready.
    """
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    runner_command = [
        sys.executable,
        "-m",
        "neuracore.data_daemon.runner_entry",
    ]

    env = os.environ.copy()
    env["NEURACORE_DAEMON_PID_PATH"] = str(pid_path)
    env["NEURACORE_DAEMON_DB_PATH"] = str(db_path)
    env["NEURACORE_DAEMON_MANAGE_PID"] = "0"

    try:
        proc = subprocess.Popen(
            runner_command,
            start_new_session=True,
            close_fds=True,
            cwd=str(Path.cwd()),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to start daemon: {exc}") from exc

    wait_for_socket_paths(
        (str(SOCKET_PATH), str(RECORDING_EVENTS_SOCKET_PATH)),
        timeout_s=timeout_s,
        process=proc,
    )

    pid_path.write_text(str(proc.pid), encoding="utf-8")
    return proc.pid


def ensure_daemon_running(*, timeout_s: float = 5.0) -> int:
    """Ensure the data daemon is running and ready to accept connections.

    Returns:
        Daemon pid (existing or newly spawned).

    Raises:
        RuntimeError: If the daemon cannot be started or becomes unhealthy.
    """
    pid_path = get_daemon_pid_path()
    db_path = get_daemon_db_path()

    os.environ.setdefault("NEURACORE_DAEMON_PID_PATH", str(pid_path))
    os.environ.setdefault("NEURACORE_DAEMON_DB_PATH", str(db_path))

    existing_pid = read_pid_from_file(pid_path)
    if existing_pid is not None and pid_is_running(existing_pid):
        wait_for_socket_paths(
            (str(SOCKET_PATH), str(RECORDING_EVENTS_SOCKET_PATH)),
            timeout_s=timeout_s,
        )
        return existing_pid

    cleanup_stale_client_state(
        pid_path=pid_path,
        db_path=db_path,
        socket_paths=(str(SOCKET_PATH), str(RECORDING_EVENTS_SOCKET_PATH)),
    )
    return launch_daemon_subprocess(
        pid_path=pid_path, db_path=db_path, timeout_s=timeout_s
    )


def acquire_pid_file(pid_path: Path) -> bool:
    """Create a pid file atomically; return True if created or stale cleared."""
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(pid_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        existing = read_pid_from_file(pid_path)
        if existing and pid_is_running(existing):
            raise DaemonLifecycleError(f"Daemon already running (pid={existing})")
        try:
            pid_path.unlink()
        except FileNotFoundError:
            pass
        return acquire_pid_file(pid_path)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    return True


def remove_pid_file(pid_path: Path) -> None:
    """Remove the pid file if present."""
    try:
        pid_path.unlink(missing_ok=True)
    except FileNotFoundError:
        logger.error("Failed to remove pid file %s", pid_path)


def cleanup_socket_files(paths: Iterable[Path]) -> None:
    """Remove socket files that exist on disk."""
    for socket_path in paths:
        if socket_path.exists():
            try:
                socket_path.unlink()
            except OSError as exc:
                logger.warning("Failed to remove socket file %s: %s", socket_path, exc)


def validate_or_recover_sqlite(db_path: Path, *, recover: bool = True) -> bool:
    """Validate SQLite integrity, optionally recover by rotating corrupt DB."""
    if not db_path.exists():
        return True

    try:
        conn = sqlite3.connect(str(db_path))
        try:
            result = conn.execute("PRAGMA integrity_check").fetchone()
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        logger.error("Failed to open SQLite database: %s", exc)
        result = None

    ok = result is not None and result[0] == "ok"
    if ok:
        return True
    if not recover:
        raise DaemonLifecycleError("SQLite integrity check failed")

    ts = int(time.time())
    corrupt_path = db_path.with_suffix(db_path.suffix + f".corrupt-{ts}")
    db_path.rename(corrupt_path)
    logger.warning("SQLite corruption detected; rotated to %s", corrupt_path)
    return False


def checkpoint_sqlite(db_path: Path) -> None:
    """Checkpoint SQLite WAL to disk."""
    if not db_path.exists():
        return
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        logger.warning("SQLite checkpoint failed: %s", exc)


def _iter_trace_dirs(recordings_root: Path) -> Iterator[Path]:
    if not recordings_root.exists():
        return
    for recording_dir in recordings_root.iterdir():
        if not recording_dir.is_dir():
            continue
        for data_type_dir in recording_dir.iterdir():
            if not data_type_dir.is_dir():
                continue
            for trace_dir in data_type_dir.iterdir():
                if trace_dir.is_dir():
                    yield trace_dir


def _trace_dir_has_files(trace_dir: Path) -> bool:
    try:
        return any(trace_dir.iterdir())
    except FileNotFoundError:
        return False


async def reconcile_state_with_filesystem(
    store: StateStore, recordings_root: Path
) -> None:
    """Sync stored traces with disk contents, cleaning orphans and flagging gaps."""
    traces = await store.list_traces()
    trace_paths = {Path(str(trace.path)) for trace in traces}

    for trace in traces:
        trace_path = Path(str(trace.path))
        if not trace_path.exists() or not _trace_dir_has_files(trace_path):
            await store.record_error(
                trace.trace_id,
                "Trace data missing or incomplete on disk",
                error_code=TraceErrorCode.WRITE_FAILED,
                status=TraceStatus.FAILED,
            )
            continue
        elif trace.status == TraceStatus.UPLOADING:
            await store.update_status(trace.trace_id, TraceStatus.PAUSED)

    for trace_dir in _iter_trace_dirs(recordings_root):
        if trace_dir not in trace_paths:
            for child in trace_dir.rglob("*"):
                if child.is_file():
                    child.unlink()
            trace_dir.rmdir()


def install_signal_handlers(
    on_shutdown: Callable[[int], None],
    *,
    on_reload: Callable[[], None] | None = None,
) -> None:
    """Install signal handlers for graceful shutdown and optional reload."""

    def _handle_shutdown(signum: int, _frame: FrameType | None) -> None:
        on_shutdown(signum)
        raise KeyboardInterrupt

    def _handle_reload(_signum: int, _frame: FrameType | None) -> None:
        if on_reload:
            on_reload()

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _handle_reload)


async def startup(
    *,
    pid_path: Path,
    socket_paths: Iterable[Path],
    db_path: Path,
    recordings_root: Path,
    store: StateStore,
    recover_sqlite: bool = True,
    manage_pid: bool = True,
) -> None:
    """Run startup checks and cleanup."""
    if manage_pid:
        existing = read_pid_from_file(pid_path)
        if existing and not pid_is_running(existing):
            try:
                pid_path.unlink()
            except FileNotFoundError:
                pass
        acquire_pid_file(pid_path)

    cleanup_socket_files(socket_paths)
    sqlite_recovered = not validate_or_recover_sqlite(db_path, recover=recover_sqlite)
    if sqlite_recovered:
        logger.warning("SQLite recovered by rotation; new DB will be created.")

    recordings_root.mkdir(parents=True, exist_ok=True)
    await reconcile_state_with_filesystem(store, recordings_root)


def shutdown(
    *,
    pid_path: Path,
    socket_paths: Iterable[Path],
    db_path: Path,
) -> None:
    """Run shutdown steps and cleanup."""
    checkpoint_sqlite(db_path)
    cleanup_socket_files(socket_paths)
    remove_pid_file(pid_path)


def terminate_pid(pid_value: int) -> bool:
    """Send SIGTERM to the given PID.

    Args:
        pid_value: Process ID to terminate.

    Returns:
        True if the process does not exist or the signal was sent successfully.
        False if the signal could not be sent due to permissions.
    """
    try:
        os.kill(pid_value, signal.SIGTERM)
        return True
    except ProcessLookupError:
        return True
    except PermissionError:
        return False


def force_kill(pid_value: int) -> bool:
    """Send SIGKILL to the given PID.

    Args:
        pid_value: Process ID to kill.

    Returns:
        True if the process does not exist or the signal was sent successfully.
        False if the signal could not be sent due to permissions.
    """
    try:
        os.kill(pid_value, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return True
    except PermissionError:
        return False


def wait_for_exit(pid_value: int, *, timeout_s: float) -> bool:
    """Wait for a PID to stop running until a timeout elapses.

    Args:
        pid_value: Process ID to wait on.
        timeout_s: Maximum seconds to wait.

    Returns:
        True if the process exited within the timeout, otherwise False.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not pid_is_running(pid_value):
            return True
        time.sleep(0.1)
    return False
