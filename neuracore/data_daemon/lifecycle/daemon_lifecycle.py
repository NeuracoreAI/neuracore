"""Lifecycle helpers for daemon startup, recovery, and shutdown."""

from __future__ import annotations

import logging
import os
import signal
import sqlite3
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from types import FrameType

from neuracore.data_daemon.models import TraceErrorCode, TraceStatus
from neuracore.data_daemon.state_management.state_store import StateStore

logger = logging.getLogger(__name__)


class DaemonLifecycleError(RuntimeError):
    """Raised when daemon lifecycle checks fail."""


def _pid_is_running(pid_value: int) -> bool:
    try:
        os.kill(pid_value, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read_pid(pid_path: Path) -> int | None:
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


def acquire_pid_file(pid_path: Path) -> bool:
    """Create a pid file atomically; return True if created or stale cleared."""
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(pid_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        existing = _read_pid(pid_path)
        if existing and _pid_is_running(existing):
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
        pid_path.unlink()
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
    traces = store.list_traces()
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
        existing = _read_pid(pid_path)
        if existing and not _pid_is_running(existing):
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
    """Run shutdown steps and cleanup, returning a report of actions taken."""
    checkpoint_sqlite(db_path)
    cleanup_socket_files(socket_paths)
    remove_pid_file(pid_path)
