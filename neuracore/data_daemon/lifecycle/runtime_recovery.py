"""Recovery and on-disk state helpers for daemon runtime state."""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

from neuracore.data_daemon.lifecycle.daemon_os_control import (
    DaemonLifecycleError,
    remove_pid_file,
)
from neuracore.data_daemon.models import TraceErrorCode, TraceUploadStatus
from neuracore.data_daemon.state_management.state_store import StateStore

logger = logging.getLogger(__name__)


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
            )
            continue
        if trace.upload_status == TraceUploadStatus.UPLOADING:
            await store.update_upload_status(trace.trace_id, TraceUploadStatus.PAUSED)

    for trace_dir in _iter_trace_dirs(recordings_root):
        if trace_dir not in trace_paths:
            for child in trace_dir.rglob("*"):
                if child.is_file():
                    child.unlink()
            trace_dir.rmdir()


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


__all__ = [
    "checkpoint_sqlite",
    "cleanup_socket_files",
    "reconcile_state_with_filesystem",
    "shutdown",
    "validate_or_recover_sqlite",
]
