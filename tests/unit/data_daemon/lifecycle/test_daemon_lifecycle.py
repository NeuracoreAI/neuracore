from __future__ import annotations

import asyncio
import os
import signal
import sqlite3
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from neuracore.data_daemon.lifecycle.daemon_lifecycle import (
    DaemonLifecycleError,
    acquire_pid_file,
    cleanup_socket_files,
    install_signal_handlers,
    reconcile_state_with_filesystem,
    remove_pid_file,
    shutdown,
    startup,
    validate_or_recover_sqlite,
)
from neuracore.data_daemon.models import (
    DataType,
    TraceErrorCode,
    TraceRecord,
    TraceStatus,
)


class _InMemoryStore:
    def __init__(self, traces: list[TraceRecord]) -> None:
        self._traces = {trace.trace_id: trace for trace in traces}

    def list_traces(self) -> list[TraceRecord]:
        return list(self._traces.values())

    async def record_error(
        self,
        trace_id: str,
        error_message: str,
        error_code: TraceErrorCode | None = None,
        status: TraceStatus = TraceStatus.FAILED,
    ) -> None:
        self._traces[trace_id] = replace(
            self._traces[trace_id],
            status=status,
            error_message=error_message,
            error_code=error_code,
            last_updated=datetime.now(),
        )

    async def update_status(self, trace_id: str, status: TraceStatus) -> None:
        self._traces[trace_id] = replace(
            self._traces[trace_id],
            status=status,
            last_updated=datetime.now(),
        )

    async def get_trace(self, trace_id: str) -> TraceRecord | None:
        return self._traces.get(trace_id)


def test_acquire_pid_file_rejects_running_pid(tmp_path: Path) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    with pytest.raises(DaemonLifecycleError):
        acquire_pid_file(pid_path)


def test_acquire_pid_file_clears_stale_pid(tmp_path: Path) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text("999999", encoding="utf-8")

    assert acquire_pid_file(pid_path) is True
    assert pid_path.read_text(encoding="utf-8").strip() == str(os.getpid())


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


def test_startup_reconciles_missing_and_orphaned_traces(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    recordings_root = tmp_path / "recordings"
    missing_trace_path = recordings_root / "rec-1" / "CUSTOM_1D" / "trace-missing"
    now = datetime.now()
    trace = TraceRecord(
        trace_id="trace-missing",
        status=TraceStatus.INITIALIZING,
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
        progress_reported=0,
        error_code=None,
        error_message=None,
        created_at=now,
        last_updated=now,
    )

    store = _InMemoryStore([trace])

    orphan_dir = recordings_root / "rec-1" / "CUSTOM_1D" / "trace-orphan"
    orphan_dir.mkdir(parents=True, exist_ok=True)
    (orphan_dir / "batch_000001.raw").write_text("orphan", encoding="utf-8")

    asyncio.run(
        startup(
            pid_path=tmp_path / "daemon.pid",
            socket_paths=(),
            db_path=db_path,
            recordings_root=recordings_root,
            store=store,
            recover_sqlite=True,
            manage_pid=True,
        )
    )
    updated = asyncio.run(store.get_trace("trace-missing"))
    assert updated is not None
    assert updated.status == TraceStatus.FAILED
    assert updated.error_code == TraceErrorCode.WRITE_FAILED
    assert not orphan_dir.exists()


def test_reconcile_pauses_uploading_traces(tmp_path: Path) -> None:
    recordings_root = tmp_path / "recordings"
    trace_path = recordings_root / "rec-2" / "CUSTOM_1D" / "trace-upload"
    trace_path.mkdir(parents=True, exist_ok=True)
    (trace_path / "batch_000001.raw").write_text("data", encoding="utf-8")

    now = datetime.now()
    trace = TraceRecord(
        trace_id="trace-upload",
        status=TraceStatus.UPLOADING,
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
        progress_reported=0,
        error_code=None,
        error_message=None,
        created_at=now,
        last_updated=now,
    )
    store = _InMemoryStore([trace])

    asyncio.run(reconcile_state_with_filesystem(store, recordings_root))

    updated = asyncio.run(store.get_trace("trace-upload"))
    assert updated is not None
    assert updated.status == TraceStatus.PAUSED


def test_reconcile_marks_empty_trace_dir_as_incomplete(tmp_path: Path) -> None:
    recordings_root = tmp_path / "recordings"

    trace_path = recordings_root / "rec-3" / "CUSTOM_1D" / "trace-empty"
    trace_path.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    trace = TraceRecord(
        trace_id="trace-empty",
        status=TraceStatus.WRITTEN,
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
        progress_reported=0,
        error_code=None,
        error_message=None,
        created_at=now,
        last_updated=now,
    )
    store = _InMemoryStore([trace])

    asyncio.run(reconcile_state_with_filesystem(store, recordings_root))

    updated = asyncio.run(store.get_trace("trace-empty"))
    assert updated is not None
    assert updated.status == TraceStatus.FAILED
    assert updated.error_code == TraceErrorCode.WRITE_FAILED


def test_remove_pid_file_removes(tmp_path: Path) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text("123", encoding="utf-8")
    remove_pid_file(pid_path)
    assert not pid_path.exists()


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


def test_install_signal_handlers_invokes_shutdown() -> None:
    called: list[int] = []

    def on_shutdown(signum: int) -> None:
        called.append(signum)

    orig_term = signal.getsignal(signal.SIGTERM)
    orig_int = signal.getsignal(signal.SIGINT)
    try:
        install_signal_handlers(on_shutdown)
        handler = signal.getsignal(signal.SIGTERM)
        assert handler is not None
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGTERM, None)
        assert called == [signal.SIGTERM]
    finally:
        signal.signal(signal.SIGTERM, orig_term)
        signal.signal(signal.SIGINT, orig_int)
