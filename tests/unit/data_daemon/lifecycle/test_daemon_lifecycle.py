from __future__ import annotations

import os
import signal
import sqlite3
from pathlib import Path

import pytest

from neuracore.data_daemon.lifecycle.daemon_lifecycle import (
    DaemonLifecycleError,
    StartupReport,
    acquire_pid_file,
    cleanup_socket_files,
    install_signal_handlers,
    reconcile_state_with_filesystem,
    release_pid_file,
    shutdown,
    startup,
    validate_or_recover_sqlite,
)
from neuracore.data_daemon.models import DataType, TraceStatus
from neuracore.data_daemon.state_management.state_store_sqlite import SqliteStateStore


def _create_trace(
    store: SqliteStateStore,
    trace_id: str,
    recording_id: str,
    path: Path,
) -> None:
    store.create_trace(
        trace_id=trace_id,
        recording_id=recording_id,
        data_type=DataType.CUSTOM_1D,
        data_type_name="custom",
        robot_instance=1,
        path=str(path),
    )


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

    cleaned = cleanup_socket_files([socket_path])
    assert cleaned == (socket_path,)
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
    store = SqliteStateStore(db_path)

    missing_trace_path = recordings_root / "rec-1" / "CUSTOM_1D" / "trace-missing"
    _create_trace(store, "trace-missing", "rec-1", missing_trace_path)
    store.update_status("trace-missing", TraceStatus.WRITING)

    orphan_dir = recordings_root / "rec-1" / "CUSTOM_1D" / "trace-orphan"
    orphan_dir.mkdir(parents=True, exist_ok=True)
    (orphan_dir / "batch_000001.raw").write_text("orphan", encoding="utf-8")

    report = startup(
        pid_path=tmp_path / "daemon.pid",
        socket_paths=(),
        db_path=db_path,
        recordings_root=recordings_root,
        store=store,
        recover_sqlite=True,
        manage_pid=True,
    )

    assert isinstance(report, StartupReport)
    assert "trace-missing" in report.missing_traces
    assert orphan_dir in report.orphaned_dirs_removed


def test_reconcile_pauses_uploading_traces(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    recordings_root = tmp_path / "recordings"
    store = SqliteStateStore(db_path)

    trace_path = recordings_root / "rec-2" / "CUSTOM_1D" / "trace-upload"
    trace_path.mkdir(parents=True, exist_ok=True)
    (trace_path / "batch_000001.raw").write_text("data", encoding="utf-8")

    _create_trace(store, "trace-upload", "rec-2", trace_path)
    store.mark_trace_as_written("trace-upload", 10)
    store.update_status("trace-upload", TraceStatus.UPLOADING)

    report = reconcile_state_with_filesystem(store, recordings_root)

    trace = store.get_trace("trace-upload")
    assert trace is not None
    assert trace.status == TraceStatus.PAUSED
    assert "trace-upload" in report.paused_traces


def test_reconcile_marks_empty_trace_dir_as_incomplete(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    recordings_root = tmp_path / "recordings"
    store = SqliteStateStore(db_path)

    trace_path = recordings_root / "rec-3" / "CUSTOM_1D" / "trace-empty"
    trace_path.mkdir(parents=True, exist_ok=True)

    _create_trace(store, "trace-empty", "rec-3", trace_path)
    store.mark_trace_as_written("trace-empty", 10)

    report = reconcile_state_with_filesystem(store, recordings_root)

    trace = store.get_trace("trace-empty")
    assert trace is not None
    assert trace.status == TraceStatus.FAILED
    assert "trace-empty" in report.missing_traces


def test_release_pid_file_removes(tmp_path: Path) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text("123", encoding="utf-8")
    assert release_pid_file(pid_path) is True
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

    report = shutdown(
        pid_path=pid_path,
        socket_paths=(socket_path, events_path),
        db_path=db_path,
        shutdown_steps=(),
    )

    assert report.pid_removed is True
    assert report.sqlite_checkpointed is True
    assert not socket_path.exists()
    assert not events_path.exists()


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
