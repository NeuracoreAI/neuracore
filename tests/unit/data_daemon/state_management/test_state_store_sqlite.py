from __future__ import annotations

import threading
from pathlib import Path

import pytest
from sqlalchemy import select, text

from neuracore.data_daemon.event_emitter import Emitter, emitter
from neuracore.data_daemon.models import (
    DATA_TYPE_CONTENT_MAPPING,
    DataType,
    TraceErrorCode,
    TraceStatus,
)
from neuracore.data_daemon.state_management.state_manager import StateManager
from neuracore.data_daemon.state_management.state_store_sqlite import SqliteStateStore
from neuracore.data_daemon.state_management.tables import traces

DATA_TYPES = list(DATA_TYPE_CONTENT_MAPPING.keys())
PRIMARY_DATA_TYPE = DATA_TYPES[0]
SECONDARY_DATA_TYPE = DATA_TYPES[1] if len(DATA_TYPES) > 1 else DATA_TYPES[0]
ROBOT_INSTANCE = 1


@pytest.fixture
def store(tmp_path: Path) -> SqliteStateStore:
    return SqliteStateStore(tmp_path / "state.db")


def _get_trace_row(store: SqliteStateStore, trace_id: str) -> dict | None:
    with store._engine.begin() as conn:
        row = (
            conn.execute(select(traces).where(traces.c.trace_id == trace_id))
            .mappings()
            .first()
        )
        return dict(row) if row else None


def _cleanup_state_manager(manager: StateManager) -> None:
    emitter.remove_listener(Emitter.TRACE_WRITTEN, manager._handle_trace_written)
    emitter.remove_listener(Emitter.START_TRACE, manager.create_trace)
    emitter.remove_listener(Emitter.UPLOAD_COMPLETE, manager.handle_upload_complete)
    emitter.remove_listener(Emitter.UPLOADED_BYTES, manager.update_bytes_uploaded)
    emitter.remove_listener(Emitter.UPLOAD_FAILED, manager.handle_upload_failed)
    emitter.remove_listener(Emitter.STOP_RECORDING, manager.handle_stop_recording)
    emitter.remove_listener(Emitter.IS_CONNECTED, manager.handle_is_connected)


def test_create_trace_inserts_row(store: SqliteStateStore) -> None:
    store.create_trace(
        "trace-1",
        "rec-1",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-1.bin",
        total_bytes=128,
        robot_instance=ROBOT_INSTANCE,
    )

    row = _get_trace_row(store, "trace-1")
    assert row is not None
    assert row["trace_id"] == "trace-1"
    assert row["recording_id"] == "rec-1"
    assert row["data_type"] == PRIMARY_DATA_TYPE
    assert row["path"] == "/tmp/trace-1.bin"
    assert row["total_bytes"] == 128
    assert row["status"] == TraceStatus.PENDING
    assert row["bytes_written"] == 0
    assert row["bytes_uploaded"] == 0
    assert row["progress_reported"] == 0
    assert row["error_message"] is None
    assert row["robot_instance"] == ROBOT_INSTANCE


def test_create_trace_updates_existing(store: SqliteStateStore) -> None:
    store.create_trace(
        "trace-2",
        "rec-1",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-2.bin",
        total_bytes=10,
        robot_instance=ROBOT_INSTANCE,
    )
    store.create_trace(
        "trace-2",
        "rec-2",
        SECONDARY_DATA_TYPE,
        data_type_name="secondary",
        path="/tmp/trace-2.mp4",
        total_bytes=20,
        robot_instance=ROBOT_INSTANCE,
    )

    row = _get_trace_row(store, "trace-2")
    assert row is not None
    assert row["recording_id"] == "rec-2"
    assert row["data_type"] == SECONDARY_DATA_TYPE
    assert row["path"] == "/tmp/trace-2.mp4"
    assert row["total_bytes"] == 20
    assert row["status"] == TraceStatus.PENDING


def test_update_bytes_uploaded_accumulates(store: SqliteStateStore) -> None:
    store.create_trace(
        "trace-3",
        "rec-3",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-3.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    store.update_bytes_uploaded("trace-3", 5)
    row = _get_trace_row(store, "trace-3")
    assert row["bytes_uploaded"] == 5
    store.update_bytes_uploaded("trace-3", 7)
    row = _get_trace_row(store, "trace-3")
    assert row["bytes_uploaded"] == 7
    assert row is not None


def test_bytes_uploaded_persisted_across_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    store = SqliteStateStore(db_path)
    store.create_trace(
        "trace-restart",
        "rec-restart",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-restart.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    store.update_bytes_uploaded("trace-restart", 1024)
    row = _get_trace_row(store, "trace-restart")
    assert row is not None
    assert row["bytes_uploaded"] == 1024

    restarted_store = SqliteStateStore(db_path)
    row_after = _get_trace_row(restarted_store, "trace-restart")
    assert row_after is not None
    assert row_after["bytes_uploaded"] == 1024


def test_update_status_sets_error(store: SqliteStateStore) -> None:
    store.create_trace(
        "trace-4",
        "rec-4",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-4.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    store.update_status("trace-4", TraceStatus.WRITTEN)
    store.update_status("trace-4", TraceStatus.FAILED, error_message="boom")

    row = _get_trace_row(store, "trace-4")
    assert row is not None
    assert row["status"] == TraceStatus.FAILED
    assert row["error_message"] == "boom"


def test_record_error_sets_code_and_status(store: SqliteStateStore) -> None:
    store.create_trace(
        "trace-4b",
        "rec-4b",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-4b.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    store.record_error(
        "trace-4b",
        error_message="disk full",
        error_code=TraceErrorCode.DISK_FULL,
    )

    row = _get_trace_row(store, "trace-4b")
    assert row is not None
    assert row["status"] == TraceStatus.FAILED
    assert row["error_message"] == "disk full"
    assert row["error_code"] == TraceErrorCode.DISK_FULL.value


def test_claim_ready_traces_filters_ready_written(store: SqliteStateStore) -> None:
    store.create_trace(
        "trace-5",
        "rec-5",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-5.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    store.create_trace(
        "trace-6",
        "rec-6",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-6.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    store.create_trace(
        "trace-7",
        "rec-7",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-7.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    with store._engine.begin() as conn:
        conn.execute(
            traces.update()
            .where(traces.c.trace_id == "trace-5")
            .values(status=TraceStatus.WRITTEN, ready_for_upload=1)
        )
        conn.execute(
            traces.update()
            .where(traces.c.trace_id == "trace-6")
            .values(status=TraceStatus.WRITTEN, ready_for_upload=0)
        )
        conn.execute(
            traces.update()
            .where(traces.c.trace_id == "trace-7")
            .values(status=TraceStatus.PENDING, ready_for_upload=1)
        )

    claimed = store.claim_ready_traces(limit=10)
    assert [row["trace_id"] for row in claimed] == ["trace-5"]


def test_mark_trace_as_written_sets_total_bytes_and_ready_flag(
    store: SqliteStateStore,
) -> None:
    store.create_trace(
        "trace-6b",
        "rec-6b",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-6b.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    with store._engine.begin() as conn:
        conn.execute(
            traces.update()
            .where(traces.c.trace_id == "trace-6b")
            .values(bytes_written=64)
        )

    store.mark_trace_as_written("trace-6b", 64)

    row = _get_trace_row(store, "trace-6b")
    assert row is not None
    assert row["status"] == TraceStatus.WRITTEN
    assert row["total_bytes"] == 64
    assert row["ready_for_upload"] == 1
    assert row["progress_reported"] == 0


def test_claim_ready_traces_marks_uploading(store: SqliteStateStore) -> None:
    store.create_trace(
        "trace-7b",
        "rec-7b",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-7b.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    with store._engine.begin() as conn:
        conn.execute(
            traces.update()
            .where(traces.c.trace_id == "trace-7b")
            .values(status=TraceStatus.WRITTEN, ready_for_upload=1)
        )

    claimed = store.claim_ready_traces(limit=10)
    assert [row["trace_id"] for row in claimed] == ["trace-7b"]
    assert claimed[0]["status"] == TraceStatus.UPLOADING
    assert claimed[0]["ready_for_upload"] == 0


def test_delete_trace_removes_row(store: SqliteStateStore) -> None:
    store.create_trace(
        "trace-7",
        "rec-7",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-7.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    store.delete_trace("trace-7")

    assert _get_trace_row(store, "trace-7") is None


def test_find_ready_traces_returns_only_ready(store: SqliteStateStore) -> None:
    store.create_trace(
        "trace-8",
        "rec-8",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-8.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    store.create_trace(
        "trace-9",
        "rec-9",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-9.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    with store._engine.begin() as conn:
        conn.execute(
            traces.update()
            .where(traces.c.trace_id == "trace-8")
            .values(status=TraceStatus.WRITTEN, ready_for_upload=1)
        )
        conn.execute(
            traces.update()
            .where(traces.c.trace_id == "trace-9")
            .values(status=TraceStatus.WRITTEN, ready_for_upload=0)
        )

    ready = store.find_ready_traces()
    assert sorted(trace.trace_id for trace in ready) == ["trace-8"]


def test_mark_recording_reported_updates_all_traces(
    store: SqliteStateStore,
) -> None:
    store.create_trace(
        "trace-10",
        "rec-10",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-10.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    store.create_trace(
        "trace-11",
        "rec-10",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-11.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    store.mark_recording_reported("rec-10")

    row_10 = _get_trace_row(store, "trace-10")
    row_11 = _get_trace_row(store, "trace-11")
    assert row_10 is not None
    assert row_11 is not None
    assert row_10["progress_reported"] == 1
    assert row_11["progress_reported"] == 1


def test_find_unreported_traces_filters_reported(store: SqliteStateStore) -> None:
    store.create_trace(
        "trace-12",
        "rec-12",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-12.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    store.create_trace(
        "trace-13",
        "rec-13",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-13.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    store.mark_recording_reported("rec-13")

    unreported = store.find_unreported_traces()
    assert sorted(trace.trace_id for trace in unreported) == ["trace-12"]


def test_wal_mode_enabled(store: SqliteStateStore) -> None:
    with store._engine.begin() as conn:
        mode = conn.execute(text("PRAGMA journal_mode;")).scalar_one()
    assert str(mode).lower() == "wal"


def test_state_transition_sequence(store: SqliteStateStore) -> None:
    store.create_trace(
        "trace-transition",
        "rec-transition",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-transition.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    store.update_status("trace-transition", TraceStatus.WRITING)
    store.mark_trace_as_written("trace-transition", 256)

    claimed = store.claim_ready_traces(limit=1)
    assert [row["trace_id"] for row in claimed] == ["trace-transition"]
    assert claimed[0]["status"] == TraceStatus.UPLOADING

    store.update_status("trace-transition", TraceStatus.UPLOADED)
    row = _get_trace_row(store, "trace-transition")
    assert row is not None
    assert row["status"] == TraceStatus.UPLOADED


def test_invalid_state_transition_rejected(store: SqliteStateStore) -> None:
    store.create_trace(
        "trace-invalid",
        "rec-invalid",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-invalid.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    with pytest.raises(ValueError, match="Invalid status transition"):
        store.update_status("trace-invalid", TraceStatus.UPLOADED)

    row = _get_trace_row(store, "trace-invalid")
    assert row is not None
    assert row["status"] == TraceStatus.PENDING


def test_multiple_state_managers_share_sqlite_db(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    store_one = SqliteStateStore(db_path)
    store_two = SqliteStateStore(db_path)
    manager_one = StateManager(store_one)
    manager_two = StateManager(store_two)
    barrier = threading.Barrier(2)

    def worker(manager: StateManager, trace_id: str) -> None:
        barrier.wait()
        manager.create_trace(
            trace_id,
            "rec-shared",
            DataType.CUSTOM_1D,
            "custom",
            1,
            path=f"/tmp/{trace_id}.bin",
            total_bytes=64,
        )

    try:
        thread_one = threading.Thread(
            target=worker, args=(manager_one, "trace-multi-1")
        )
        thread_two = threading.Thread(
            target=worker, args=(manager_two, "trace-multi-2")
        )
        thread_one.start()
        thread_two.start()
        thread_one.join()
        thread_two.join()
    finally:
        _cleanup_state_manager(manager_one)
        _cleanup_state_manager(manager_two)

    assert store_one.get_trace("trace-multi-1") is not None
    assert store_one.get_trace("trace-multi-2") is not None


def test_concurrent_writes_do_not_lock(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    store_one = SqliteStateStore(db_path)
    store_two = SqliteStateStore(db_path)
    store_one.create_trace(
        "trace-concurrent",
        "rec-concurrent",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-concurrent.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def worker(store: SqliteStateStore, bytes_uploaded: int) -> None:
        try:
            barrier.wait()
            for _ in range(5):
                store.update_bytes_uploaded("trace-concurrent", bytes_uploaded)
        except Exception as exc:
            errors.append(exc)

    thread_one = threading.Thread(target=worker, args=(store_one, 10))
    thread_two = threading.Thread(target=worker, args=(store_two, 20))
    thread_one.start()
    thread_two.start()
    thread_one.join()
    thread_two.join()

    assert errors == []
    row = _get_trace_row(store_one, "trace-concurrent")
    assert row is not None
    assert row["bytes_uploaded"] in {10, 20}


def test_race_conditions_on_rapid_state_changes(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    store_one = SqliteStateStore(db_path)
    store_two = SqliteStateStore(db_path)
    store_one.create_trace(
        "trace-race",
        "rec-race",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-race.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    store_one.update_status("trace-race", TraceStatus.WRITING)

    barrier = threading.Barrier(2)
    errors: list[str] = []

    def worker(store: SqliteStateStore) -> None:
        try:
            barrier.wait()
            store.update_status("trace-race", TraceStatus.WRITTEN)
            store.update_status("trace-race", TraceStatus.UPLOADING)
            store.update_status("trace-race", TraceStatus.UPLOADED)
        except ValueError as exc:
            errors.append(str(exc))

    thread_one = threading.Thread(target=worker, args=(store_one,))
    thread_two = threading.Thread(target=worker, args=(store_two,))
    thread_one.start()
    thread_two.start()
    thread_one.join()
    thread_two.join()

    assert errors
    row = _get_trace_row(store_one, "trace-race")
    assert row is not None
    assert row["status"] == TraceStatus.UPLOADED


def test_state_recovery_after_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    store = SqliteStateStore(db_path)
    store.create_trace(
        "trace-recover",
        "rec-recover",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-recover.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    store.update_status("trace-recover", TraceStatus.WRITING)
    store.mark_trace_as_written("trace-recover", 512)

    recovered_store = SqliteStateStore(db_path)
    recovered_trace = recovered_store.get_trace("trace-recover")
    assert recovered_trace is not None
    assert recovered_trace.status == TraceStatus.WRITTEN
    assert recovered_trace.ready_for_upload == 1

    claimed = recovered_store.claim_ready_traces(limit=1)
    assert [row["trace_id"] for row in claimed] == ["trace-recover"]
    assert claimed[0]["status"] == TraceStatus.UPLOADING
