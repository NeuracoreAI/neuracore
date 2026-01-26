"""Lifecycle helpers for daemon startup, recovery, and shutdown."""

from __future__ import annotations

import logging
import os
import signal
import sqlite3
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

from neuracore.data_daemon.models import TraceErrorCode, TraceStatus
from neuracore.data_daemon.state_management.state_store import StateStore

logger = logging.getLogger(__name__)


class DaemonLifecycleError(RuntimeError):
    """Raised when daemon lifecycle checks fail."""


@dataclass(frozen=True)
class StartupReport:
    """Summary of cleanup and reconciliation performed during startup."""

    stale_pid_removed: bool = False
    sockets_cleaned: tuple[Path, ...] = ()
    sqlite_recovered: bool = False
    orphaned_dirs_removed: tuple[Path, ...] = ()
    missing_traces: tuple[str, ...] = ()
    interrupted_traces: tuple[str, ...] = ()
    paused_traces: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShutdownReport:
    """Summary of cleanup performed during shutdown."""

    pid_removed: bool = False
    sqlite_checkpointed: bool = False
    sockets_cleaned: tuple[Path, ...] = ()


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


def release_pid_file(pid_path: Path) -> bool:
    """Remove the pid file if present."""
    try:
        pid_path.unlink()
        return True
    except FileNotFoundError:
        return False


def cleanup_socket_files(paths: Iterable[Path]) -> tuple[Path, ...]:
    """Remove socket files that exist on disk; return removed paths."""
    cleaned: list[Path] = []
    for socket_path in paths:
        if socket_path.exists():
            try:
                socket_path.unlink()
                cleaned.append(socket_path)
            except OSError as exc:
                logger.warning("Failed to remove socket file %s: %s", socket_path, exc)
    return tuple(cleaned)


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


def checkpoint_sqlite(db_path: Path) -> bool:
    """Checkpoint SQLite WAL to disk."""
    if not db_path.exists():
        return False
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()
        return True
    except sqlite3.DatabaseError as exc:
        logger.warning("SQLite checkpoint failed: %s", exc)
        return False


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


def reconcile_state_with_filesystem(
    store: StateStore, recordings_root: Path
) -> StartupReport:
    """Sync stored traces with disk contents, cleaning orphans and flagging gaps."""
    missing: list[str] = []
    interrupted: list[str] = []
    paused: list[str] = []

    traces = store.list_traces()
    trace_paths = {Path(trace.path) for trace in traces}

    for trace in traces:
        trace_path = Path(trace.path)
        if not trace_path.exists() or not _trace_dir_has_files(trace_path):
            missing.append(trace.trace_id)
            store.record_error(
                trace.trace_id,
                "Trace data missing or incomplete on disk",
                error_code=TraceErrorCode.WRITE_FAILED,
                status=TraceStatus.FAILED,
            )
            continue

        if trace.status == TraceStatus.WRITING:
            interrupted.append(trace.trace_id)
            store.record_error(
                trace.trace_id,
                "Trace interrupted on shutdown",
                error_code=TraceErrorCode.WRITE_FAILED,
                status=TraceStatus.FAILED,
            )
        elif trace.status == TraceStatus.UPLOADING:
            store.update_status(trace.trace_id, TraceStatus.PAUSED)
            paused.append(trace.trace_id)

    orphaned: list[Path] = []
    for trace_dir in _iter_trace_dirs(recordings_root):
        if trace_dir not in trace_paths:
            orphaned.append(trace_dir)
            for child in trace_dir.rglob("*"):
                if child.is_file():
                    child.unlink()
            trace_dir.rmdir()

    return StartupReport(
        orphaned_dirs_removed=tuple(orphaned),
        missing_traces=tuple(missing),
        interrupted_traces=tuple(interrupted),
        paused_traces=tuple(paused),
    )


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


def startup(
    *,
    pid_path: Path,
    socket_paths: Iterable[Path],
    db_path: Path,
    recordings_root: Path,
    store: StateStore,
    recover_sqlite: bool = True,
    manage_pid: bool = True,
) -> StartupReport:
    """Run startup checks and cleanup, returning a report of actions taken."""
    stale_pid_removed = False
    if manage_pid:
        existing = _read_pid(pid_path)
        if existing and not _pid_is_running(existing):
            try:
                pid_path.unlink()
                stale_pid_removed = True
            except FileNotFoundError:
                pass
        acquire_pid_file(pid_path)

    sockets_cleaned = cleanup_socket_files(socket_paths)
    sqlite_recovered = not validate_or_recover_sqlite(db_path, recover=recover_sqlite)
    if sqlite_recovered:
        logger.warning("SQLite recovered by rotation; new DB will be created.")

    recordings_root.mkdir(parents=True, exist_ok=True)
    reconcile_report = reconcile_state_with_filesystem(store, recordings_root)

    return StartupReport(
        stale_pid_removed=stale_pid_removed,
        sockets_cleaned=sockets_cleaned,
        sqlite_recovered=sqlite_recovered,
        orphaned_dirs_removed=reconcile_report.orphaned_dirs_removed,
        missing_traces=reconcile_report.missing_traces,
        interrupted_traces=reconcile_report.interrupted_traces,
        paused_traces=reconcile_report.paused_traces,
    )


def shutdown(
    *,
    pid_path: Path,
    socket_paths: Iterable[Path],
    db_path: Path,
    shutdown_steps: Iterable[Callable[[], None]] = (),
) -> ShutdownReport:
    """Run shutdown steps and cleanup, returning a report of actions taken."""
    for step in shutdown_steps:
        step()

    sqlite_checkpointed = checkpoint_sqlite(db_path)
    sockets_cleaned = cleanup_socket_files(socket_paths)
    pid_removed = release_pid_file(pid_path)
    return ShutdownReport(
        pid_removed=pid_removed,
        sqlite_checkpointed=sqlite_checkpointed,
        sockets_cleaned=sockets_cleaned,
    )
