from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from neuracore.data_daemon.const import DEFAULT_SHARED_MEMORY_SIZE
from neuracore.data_daemon.lifecycle.daemon_os_control import (
    acquire_pid_file,
    pid_is_running,
    read_pid_from_file,
)
from neuracore.data_daemon.lifecycle.runtime_recovery import (
    SharedMemoryCapacityError,
    cleanup_socket_files,
    cleanup_stale_shared_memory_buffers,
    ensure_shared_memory_capacity,
    reconcile_state_with_filesystem,
    shared_memory_required_bytes,
    shutdown,
    validate_or_recover_sqlite,
)
from neuracore.data_daemon.models import (
    DataType,
    ProgressReportStatus,
    TraceErrorCode,
    TraceRecord,
    TraceRegistrationStatus,
    TraceUploadStatus,
    TraceWriteStatus,
)
from neuracore.data_daemon.state_management.state_store_sqlite import SqliteStateStore


class _InMemoryStore:
    def __init__(self, traces: list[TraceRecord]) -> None:
        self._traces = {trace.trace_id: trace for trace in traces}

    async def list_traces(self) -> list[TraceRecord]:
        return list(self._traces.values())

    async def record_error(
        self,
        trace_id: str,
        error_message: str,
        error_code: TraceErrorCode | None = None,
    ) -> None:
        self._traces[trace_id] = replace(
            self._traces[trace_id],
            upload_status=TraceUploadStatus.FAILED,
            error_message=error_message,
            error_code=error_code,
            last_updated=datetime.now(),
        )

    async def update_upload_status(
        self, trace_id: str, upload_status: TraceUploadStatus
    ) -> None:
        self._traces[trace_id] = replace(
            self._traces[trace_id],
            upload_status=upload_status,
            last_updated=datetime.now(),
        )

    async def get_trace(self, trace_id: str) -> TraceRecord | None:
        return self._traces.get(trace_id)


def test_cleanup_socket_files_removes_paths(tmp_path: Path) -> None:
    socket_path = tmp_path / "daemon.sock"
    socket_path.write_text("stale", encoding="utf-8")

    cleanup_socket_files([socket_path])
    assert not socket_path.exists()


def test_validate_or_recover_sqlite_rotates_corrupt_db(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"not-a-db")

    ok = validate_or_recover_sqlite(db_path, recover=True)
    assert ok is False
    assert not db_path.exists()
    assert any(path.name.startswith("state.db.corrupt-") for path in tmp_path.iterdir())


def test_cleanup_stale_shared_memory_buffers_removes_stale_shared_slot_segments(
    tmp_path: Path,
) -> None:
    shm_dir = tmp_path / "dev-shm"
    shm_dir.mkdir()

    stale_names = (
        "neuracore-slots-stale-1",
        "neuracore-slots-stale-2",
    )
    for buffer_name in stale_names:
        (shm_dir / buffer_name).write_bytes(b"shm")

    live_name = "neuracore-keep-live"
    (shm_dir / live_name).write_bytes(b"shm")

    cleaned = cleanup_stale_shared_memory_buffers(shm_dir=shm_dir)

    assert cleaned == len(stale_names)
    for stale_name in stale_names:
        assert not (shm_dir / stale_name).exists()
    assert (shm_dir / live_name).exists()


def test_ensure_shared_memory_capacity_raises_when_tmpfs_is_full(
    tmp_path: Path,
    monkeypatch,
) -> None:
    shm_dir = tmp_path / "dev-shm"
    shm_dir.mkdir()

    monkeypatch.setattr(
        "neuracore.data_daemon.lifecycle.runtime_recovery.shared_memory_free_bytes",
        lambda _shm_dir=shm_dir: 1024,
    )

    with pytest.raises(SharedMemoryCapacityError, match="Insufficient shared memory"):
        ensure_shared_memory_capacity(2048, shm_dir=shm_dir)


def test_shared_memory_required_bytes_matches_default_allocation() -> None:
    assert (
        shared_memory_required_bytes(DEFAULT_SHARED_MEMORY_SIZE, metadata_size=4096)
        == DEFAULT_SHARED_MEMORY_SIZE + 4096
    )


def test_runtime_recovery_primitives_reconcile_missing_and_orphaned_traces(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "state.db"
    pid_path = tmp_path / "daemon.pid"
    socket_path = tmp_path / "daemon.sock"
    recordings_root = tmp_path / "recordings"
    missing_trace_path = recordings_root / "rec-1" / "CUSTOM_1D" / "trace-missing"
    now = datetime.now()
    trace = TraceRecord(
        trace_id="trace-missing",
        write_status=TraceWriteStatus.INITIALIZING,
        registration_status=TraceRegistrationStatus.PENDING,
        upload_status=TraceUploadStatus.PENDING,
        recording_id="rec-1",
        data_type=DataType.CUSTOM_1D,
        data_type_name="custom",
        dataset_id=None,
        dataset_name=None,
        robot_name=None,
        robot_id=None,
        robot_instance=1,
        path=str(missing_trace_path),
        bytes_written=0,
        total_bytes=None,
        bytes_uploaded=0,
        progress_reported=ProgressReportStatus.PENDING,
        expected_trace_count_reported=0,
        error_code=None,
        error_message=None,
        created_at=now,
        last_updated=now,
        num_upload_attempts=0,
        next_retry_at=None,
        stopped_at=None,
    )

    store = _InMemoryStore([trace])

    orphan_dir = recordings_root / "rec-1" / "CUSTOM_1D" / "trace-orphan"
    orphan_dir.mkdir(parents=True, exist_ok=True)
    (orphan_dir / "batch_000001.raw").write_text("orphan", encoding="utf-8")
    socket_path.write_text("stale", encoding="utf-8")

    assert acquire_pid_file(pid_path) is True
    cleanup_socket_files((socket_path,))
    assert validate_or_recover_sqlite(db_path, recover=True) is True
    recordings_root.mkdir(parents=True, exist_ok=True)
    asyncio.run(reconcile_state_with_filesystem(store, recordings_root))

    updated = asyncio.run(store.get_trace("trace-missing"))
    assert updated is not None
    assert updated.upload_status == TraceUploadStatus.FAILED
    assert updated.error_code == TraceErrorCode.WRITE_FAILED
    assert not orphan_dir.exists()
    assert not socket_path.exists()
    assert pid_is_running(read_pid_from_file(pid_path) or -1)


def test_runtime_recovery_primitives_initialize_store_before_reconcile(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy-state.db"
    recordings_root = tmp_path / "recordings"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE traces (
                trace_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                recording_id TEXT NOT NULL,
                data_type TEXT,
                data_type_name TEXT,
                dataset_id TEXT,
                dataset_name TEXT,
                robot_name TEXT,
                robot_id TEXT,
                robot_instance INTEGER,
                path TEXT,
                bytes_written INTEGER,
                total_bytes INTEGER,
                bytes_uploaded INTEGER DEFAULT 0,
                progress_reported TEXT NOT NULL DEFAULT 'pending',
                expected_trace_count_reported INTEGER NOT NULL DEFAULT 0,
                error_code TEXT,
                error_message TEXT,
                stopped_at DATETIME,
                num_upload_attempts INTEGER NOT NULL DEFAULT 0,
                next_retry_at DATETIME,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_updated DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            INSERT INTO traces (
                trace_id,
                status,
                recording_id,
                data_type,
                data_type_name,
                path,
                bytes_written,
                total_bytes,
                progress_reported,
                expected_trace_count_reported
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "trace-legacy",
                "written",
                "rec-legacy",
                "CUSTOM_1D",
                "custom",
                "/tmp/trace-legacy.bin",
                10,
                10,
                "pending",
                0,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    store = SqliteStateStore(db_path)
    assert validate_or_recover_sqlite(db_path, recover=True) is True
    asyncio.run(store.init_async_store())
    recordings_root.mkdir(parents=True, exist_ok=True)
    asyncio.run(reconcile_state_with_filesystem(store, recordings_root))

    conn = sqlite3.connect(str(db_path))
    try:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(traces)").fetchall()
        }
        assert "write_status" in columns
        assert "registration_status" in columns
        assert "upload_status" in columns
    finally:
        conn.close()
    asyncio.run(store.close())


def test_reconcile_pauses_uploading_traces(tmp_path: Path) -> None:
    recordings_root = tmp_path / "recordings"
    trace_path = recordings_root / "rec-2" / "CUSTOM_1D" / "trace-upload"
    trace_path.mkdir(parents=True, exist_ok=True)
    (trace_path / "batch_000001.raw").write_text("data", encoding="utf-8")

    now = datetime.now()
    trace = TraceRecord(
        trace_id="trace-upload",
        write_status=TraceWriteStatus.WRITTEN,
        registration_status=TraceRegistrationStatus.REGISTERED,
        upload_status=TraceUploadStatus.UPLOADING,
        recording_id="rec-2",
        data_type=DataType.CUSTOM_1D,
        data_type_name="custom",
        dataset_id=None,
        dataset_name=None,
        robot_name=None,
        robot_id=None,
        robot_instance=1,
        path=str(trace_path),
        bytes_written=10,
        total_bytes=10,
        bytes_uploaded=0,
        progress_reported=ProgressReportStatus.PENDING,
        expected_trace_count_reported=0,
        error_code=None,
        error_message=None,
        created_at=now,
        last_updated=now,
        num_upload_attempts=0,
        next_retry_at=None,
        stopped_at=None,
    )
    store = _InMemoryStore([trace])

    asyncio.run(reconcile_state_with_filesystem(store, recordings_root))

    updated = asyncio.run(store.get_trace("trace-upload"))
    assert updated is not None
    assert updated.upload_status == TraceUploadStatus.PAUSED


def test_reconcile_marks_empty_trace_dir_as_incomplete(tmp_path: Path) -> None:
    recordings_root = tmp_path / "recordings"

    trace_path = recordings_root / "rec-3" / "CUSTOM_1D" / "trace-empty"
    trace_path.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    trace = TraceRecord(
        trace_id="trace-empty",
        write_status=TraceWriteStatus.WRITTEN,
        registration_status=TraceRegistrationStatus.REGISTERED,
        upload_status=TraceUploadStatus.PENDING,
        recording_id="rec-3",
        data_type=DataType.CUSTOM_1D,
        data_type_name="custom",
        dataset_id=None,
        dataset_name=None,
        robot_name=None,
        robot_id=None,
        robot_instance=1,
        path=str(trace_path),
        bytes_written=10,
        total_bytes=10,
        bytes_uploaded=0,
        progress_reported=ProgressReportStatus.PENDING,
        expected_trace_count_reported=0,
        error_code=None,
        error_message=None,
        created_at=now,
        last_updated=now,
        num_upload_attempts=0,
        next_retry_at=None,
        stopped_at=None,
    )
    store = _InMemoryStore([trace])

    asyncio.run(reconcile_state_with_filesystem(store, recordings_root))

    updated = asyncio.run(store.get_trace("trace-empty"))
    assert updated is not None
    assert updated.upload_status == TraceUploadStatus.FAILED
    assert updated.error_code == TraceErrorCode.WRITE_FAILED


def test_shutdown_removes_pid_and_sockets(tmp_path: Path) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text("123", encoding="utf-8")
    db_path = tmp_path / "state.db"
    sqlite3.connect(str(db_path)).close()
    socket_path = tmp_path / "daemon.sock"
    events_path = tmp_path / "events.sock"
    socket_path.write_text("stale", encoding="utf-8")
    events_path.write_text("stale", encoding="utf-8")

    shutdown(
        pid_path=pid_path,
        socket_paths=(socket_path, events_path),
        db_path=db_path,
    )

    assert not socket_path.exists()
    assert not events_path.exists()
    assert not pid_path.exists()
