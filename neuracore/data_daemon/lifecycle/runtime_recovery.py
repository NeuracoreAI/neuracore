"""Recovery and on-disk state helpers for daemon runtime state."""

from __future__ import annotations

import logging
import shutil
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

_SHARED_MEMORY_DIR = Path("/dev/shm")
_ZEROBUFFER_LOCK_DIR = Path("/tmp/zerobuffer")
_NEURACORE_SHARED_RING_PREFIX = "neuracore-ring-buffer-"
_NEURACORE_SHARED_SLOT_PREFIX = "neuracore-slots-"

try:
    import zerobuffer.platform as _shared_platform
    from zerobuffer.oieb_view import OIEBView
    from zerobuffer.shared_memory import SharedMemoryFactory
    from zerobuffer.types import align_to_boundary

    _SHARED_MEMORY_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised in envs without deps
    _shared_platform = None
    OIEBView = None
    SharedMemoryFactory = None
    align_to_boundary = None
    _SHARED_MEMORY_IMPORT_ERROR = exc


class SharedMemoryCapacityError(RuntimeError):
    """Raised when /dev/shm lacks space for a new shared-memory allocation."""


def _format_bytes(value: int) -> str:
    """Render a byte count in a compact human-readable form."""
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(max(0, value))
    unit = units[0]
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            break
        amount /= 1024.0
    if unit == "B":
        return f"{int(amount)} {unit}"
    return f"{amount:.1f} {unit}"


def shared_memory_required_bytes(payload_size: int, *, metadata_size: int) -> int:
    """Return the actual POSIX shared-memory bytes needed for a payload."""
    if payload_size <= 0:
        raise ValueError("payload_size must be > 0")
    if metadata_size <= 0:
        raise ValueError("metadata_size must be > 0")
    if align_to_boundary is None or OIEBView is None:
        raise RuntimeError(
            "Shared-memory sizing unavailable"
            if _SHARED_MEMORY_IMPORT_ERROR is None
            else f"Shared-memory sizing unavailable: {_SHARED_MEMORY_IMPORT_ERROR}"
        )
    return (
        int(align_to_boundary(OIEBView.SIZE))
        + int(align_to_boundary(metadata_size))
        + int(align_to_boundary(payload_size))
    )


def shared_memory_free_bytes(shm_dir: Path = _SHARED_MEMORY_DIR) -> int:
    """Return currently available bytes on the POSIX shared-memory mount."""
    usage = shutil.disk_usage(shm_dir)
    return int(usage.free)


def ensure_shared_memory_capacity(
    required_bytes: int,
    *,
    shm_dir: Path = _SHARED_MEMORY_DIR,
) -> int:
    """Raise when /dev/shm cannot hold a new shared-memory allocation."""
    if required_bytes <= 0:
        raise ValueError("required_bytes must be > 0")
    free_bytes = shared_memory_free_bytes(shm_dir)
    if free_bytes < required_bytes:
        raise SharedMemoryCapacityError(
            "Insufficient shared memory in "
            f"{shm_dir}: requires {_format_bytes(required_bytes)} but only "
            f"{_format_bytes(free_bytes)} is available. "
            "Clean stale neuracore shared-memory buffers or increase /dev/shm."
        )
    return free_bytes


def _unlink_shared_memory_artifacts(
    buffer_name: str,
    *,
    shm_dir: Path,
    temp_dir: Path,
) -> None:
    if SharedMemoryFactory is not None:
        try:
            SharedMemoryFactory.remove(buffer_name)
        except Exception:
            pass
    try:
        (shm_dir / buffer_name).unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning(
            "Failed to remove shared-memory object %s from %s: %s",
            buffer_name,
            shm_dir,
            exc,
        )

    for semaphore_name in (
        f"sem.sem-r-{buffer_name}",
        f"sem.sem-w-{buffer_name}",
    ):
        try:
            (shm_dir / semaphore_name).unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Failed to remove semaphore %s: %s", semaphore_name, exc)

    try:
        (temp_dir / f"{buffer_name}.lock").unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Failed to remove lock file for %s: %s", buffer_name, exc)


def cleanup_stale_shared_memory_buffers(
    *,
    shm_dir: Path = _SHARED_MEMORY_DIR,
    temp_dir: Path = _ZEROBUFFER_LOCK_DIR,
) -> int:
    """Remove stale neuracore shared-memory buffers and matching semaphores."""
    if SharedMemoryFactory is None or OIEBView is None or _shared_platform is None:
        logger.debug(
            "Skipping shared-memory cleanup: %s",
            _SHARED_MEMORY_IMPORT_ERROR or "zerobuffer unavailable",
        )
        return 0
    if not shm_dir.exists():
        return 0

    cleaned = 0
    for shm_path in shm_dir.iterdir():
        buffer_name = shm_path.name
        if not buffer_name.startswith(_NEURACORE_SHARED_RING_PREFIX):
            continue

        shm = None
        oieb = None
        try:
            shm = SharedMemoryFactory.open(buffer_name)
            oieb = OIEBView(shm.get_memoryview(0, OIEBView.SIZE))
            reader_pid = int(getattr(oieb, "reader_pid", 0))
            writer_pid = int(getattr(oieb, "writer_pid", 0))
            reader_dead = reader_pid == 0 or not _shared_platform.process_exists(
                reader_pid
            )
            writer_dead = writer_pid == 0 or not _shared_platform.process_exists(
                writer_pid
            )
            if not (reader_dead and writer_dead):
                continue

            logger.info(
                "Cleaning up orphaned shared-memory buffer: %s "
                "(reader_pid=%d, writer_pid=%d)",
                buffer_name,
                reader_pid,
                writer_pid,
            )
            cleaned += 1
            _unlink_shared_memory_artifacts(
                buffer_name,
                shm_dir=shm_dir,
                temp_dir=temp_dir,
            )
        except Exception:
            logger.warning(
                "Failed to inspect shared-memory buffer %s; removing stale artifacts",
                buffer_name,
                exc_info=True,
            )
            cleaned += 1
            _unlink_shared_memory_artifacts(
                buffer_name,
                shm_dir=shm_dir,
                temp_dir=temp_dir,
            )
        finally:
            if oieb is not None:
                try:
                    oieb.dispose()
                except Exception:
                    pass
            if shm is not None:
                try:
                    shm.close()
                except Exception:
                    pass

    return cleaned


def cleanup_stale_shared_slot_segments(
    *,
    shm_dir: Path = _SHARED_MEMORY_DIR,
) -> int:
    """Remove stale daemon-owned shared-slot segments from /dev/shm."""
    if not shm_dir.exists():
        return 0

    cleaned = 0
    for shm_path in shm_dir.iterdir():
        if not shm_path.name.startswith(_NEURACORE_SHARED_SLOT_PREFIX):
            continue
        try:
            shm_path.unlink()
            cleaned += 1
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning(
                "Failed to remove shared-slot segment %s: %s",
                shm_path,
                exc,
            )

    return cleaned


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
    "cleanup_stale_shared_slot_segments",
    "cleanup_stale_shared_memory_buffers",
    "cleanup_socket_files",
    "ensure_shared_memory_capacity",
    "reconcile_state_with_filesystem",
    "SharedMemoryCapacityError",
    "shared_memory_free_bytes",
    "shared_memory_required_bytes",
    "shutdown",
    "validate_or_recover_sqlite",
]
