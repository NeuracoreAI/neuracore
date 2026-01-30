from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select, text

from neuracore.data_daemon.models import (
    DATA_TYPE_CONTENT_MAPPING,
    ProgressReportStatus,
    TraceErrorCode,
    TraceStatus,
)
from neuracore.data_daemon.state_management.state_store_sqlite import SqliteStateStore
from neuracore.data_daemon.state_management.tables import traces

DATA_TYPES = list(DATA_TYPE_CONTENT_MAPPING.keys())
PRIMARY_DATA_TYPE = DATA_TYPES[0]
SECONDARY_DATA_TYPE = DATA_TYPES[1] if len(DATA_TYPES) > 1 else DATA_TYPES[0]
ROBOT_INSTANCE = 1


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> SqliteStateStore:
    store = SqliteStateStore(tmp_path / "state.db")
    await store.init_async_store()
    yield store
    await store._engine.dispose()


async def _get_trace_row(store: SqliteStateStore, trace_id: str) -> dict | None:
    async with store._engine.begin() as conn:
        result = await conn.execute(select(traces).where(traces.c.trace_id == trace_id))
        row = result.mappings().first()
        return dict(row) if row else None


@pytest.mark.asyncio
async def test_upsert_trace_metadata_inserts_row(store: SqliteStateStore) -> None:
    trace = await store.upsert_trace_metadata(
        trace_id="trace-1",
        recording_id="rec-1",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-1.bin",
        total_bytes=128,
        robot_instance=ROBOT_INSTANCE,
    )

    assert trace.status == TraceStatus.INITIALIZING
    row = await _get_trace_row(store, "trace-1")
    assert row is not None
    assert row["trace_id"] == "trace-1"
    assert row["recording_id"] == "rec-1"
    assert row["data_type"] == PRIMARY_DATA_TYPE
    assert row["path"] == "/tmp/trace-1.bin"
    assert row["total_bytes"] == 128
    assert row["status"] == TraceStatus.INITIALIZING
    assert row["bytes_written"] is None
    assert row["bytes_uploaded"] == 0
    assert row["progress_reported"] == ProgressReportStatus.PENDING
    assert row["error_message"] is None
    assert row["robot_instance"] == ROBOT_INSTANCE


@pytest.mark.asyncio
async def test_upsert_trace_metadata_updates_existing(store: SqliteStateStore) -> None:
    await store.upsert_trace_metadata(
        trace_id="trace-2",
        recording_id="rec-1",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-2.bin",
        total_bytes=10,
        robot_instance=ROBOT_INSTANCE,
    )
    trace = await store.upsert_trace_metadata(
        trace_id="trace-2",
        recording_id="rec-1",
        data_type=SECONDARY_DATA_TYPE,
        data_type_name="secondary",
        path="/tmp/trace-2.mp4",
        total_bytes=20,
        robot_instance=ROBOT_INSTANCE,
    )

    assert trace.status == TraceStatus.INITIALIZING
    row = await _get_trace_row(store, "trace-2")
    assert row is not None
    assert row["recording_id"] == "rec-1"
    assert row["data_type"] == SECONDARY_DATA_TYPE
    assert row["path"] == "/tmp/trace-2.mp4"
    assert row["total_bytes"] == 20
    assert row["status"] == TraceStatus.INITIALIZING


@pytest.mark.asyncio
async def test_upsert_trace_bytes_inserts_row(store: SqliteStateStore) -> None:
    trace = await store.upsert_trace_bytes(
        trace_id="trace-bytes-1",
        recording_id="rec-bytes-1",
        bytes_written=64,
    )

    assert trace.status == TraceStatus.PENDING_BYTES
    row = await _get_trace_row(store, "trace-bytes-1")
    assert row is not None
    assert row["trace_id"] == "trace-bytes-1"
    assert row["recording_id"] == "rec-bytes-1"
    assert row["bytes_written"] == 64
    assert row["total_bytes"] == 64
    assert row["status"] == TraceStatus.PENDING_BYTES
    assert row["bytes_uploaded"] == 0


@pytest.mark.asyncio
async def test_update_bytes_uploaded_sets_value(store: SqliteStateStore) -> None:
    await store.upsert_trace_metadata(
        trace_id="trace-3",
        recording_id="rec-3",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-3.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    await store.update_bytes_uploaded("trace-3", 5)
    row = await _get_trace_row(store, "trace-3")
    assert row is not None
    assert row["bytes_uploaded"] == 5

    await store.update_bytes_uploaded("trace-3", 7)
    row = await _get_trace_row(store, "trace-3")
    assert row is not None
    assert row["bytes_uploaded"] == 7


@pytest.mark.asyncio
async def test_update_status_sets_error(store: SqliteStateStore) -> None:
    await store.upsert_trace_metadata(
        trace_id="trace-4",
        recording_id="rec-4",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-4.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    await store.upsert_trace_bytes(
        trace_id="trace-4",
        recording_id="rec-4",
        bytes_written=64,
    )

    await store.update_status("trace-4", TraceStatus.FAILED, error_message="boom")

    row = await _get_trace_row(store, "trace-4")
    assert row is not None
    assert row["status"] == TraceStatus.FAILED
    assert row["error_message"] == "boom"


@pytest.mark.asyncio
async def test_record_error_sets_code_and_status(store: SqliteStateStore) -> None:
    await store.upsert_trace_metadata(
        trace_id="trace-4b",
        recording_id="rec-4b",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-4b.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    await store.record_error(
        "trace-4b",
        error_message="disk full",
        error_code=TraceErrorCode.DISK_FULL,
    )

    row = await _get_trace_row(store, "trace-4b")
    assert row is not None
    assert row["status"] == TraceStatus.FAILED
    assert row["error_message"] == "disk full"
    assert row["error_code"] == TraceErrorCode.DISK_FULL.value


@pytest.mark.asyncio
async def test_join_pattern_metadata_then_bytes_transitions_to_written(
    store: SqliteStateStore,
) -> None:
    """Test INITIALIZING + bytes -> WRITTEN transition."""
    trace = await store.upsert_trace_metadata(
        trace_id="trace-6b",
        recording_id="rec-6b",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-6b.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    assert trace.status == TraceStatus.INITIALIZING

    trace = await store.upsert_trace_bytes(
        trace_id="trace-6b",
        recording_id="rec-6b",
        bytes_written=64,
    )

    assert trace.status == TraceStatus.WRITTEN
    row = await _get_trace_row(store, "trace-6b")
    assert row is not None
    assert row["status"] == TraceStatus.WRITTEN
    assert row["bytes_written"] == 64
    assert row["total_bytes"] == 64
    assert row["progress_reported"] == ProgressReportStatus.PENDING


@pytest.mark.asyncio
async def test_join_pattern_bytes_then_metadata_transitions_to_written(
    store: SqliteStateStore,
) -> None:
    """Test PENDING_BYTES + metadata -> WRITTEN transition."""
    trace = await store.upsert_trace_bytes(
        trace_id="trace-6c",
        recording_id="rec-6c",
        bytes_written=128,
    )
    assert trace.status == TraceStatus.PENDING_BYTES

    trace = await store.upsert_trace_metadata(
        trace_id="trace-6c",
        recording_id="rec-6c",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-6c.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    assert trace.status == TraceStatus.WRITTEN
    row = await _get_trace_row(store, "trace-6c")
    assert row is not None
    assert row["status"] == TraceStatus.WRITTEN
    assert row["bytes_written"] == 128
    assert row["total_bytes"] == 128
    assert row["progress_reported"] == ProgressReportStatus.PENDING


@pytest.mark.asyncio
async def test_delete_trace_removes_row(store: SqliteStateStore) -> None:
    await store.upsert_trace_metadata(
        trace_id="trace-7",
        recording_id="rec-7",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-7.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    await store.delete_trace("trace-7")

    assert await _get_trace_row(store, "trace-7") is None


@pytest.mark.asyncio
async def test_find_ready_traces_returns_only_ready(store: SqliteStateStore) -> None:
    await store.upsert_trace_metadata(
        trace_id="trace-8",
        recording_id="rec-8",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-8.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    await store.upsert_trace_bytes(
        trace_id="trace-8",
        recording_id="rec-8",
        bytes_written=64,
    )

    await store.upsert_trace_metadata(
        trace_id="trace-9",
        recording_id="rec-9",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-9.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    await store.upsert_trace_bytes(
        trace_id="trace-9",
        recording_id="rec-9",
        bytes_written=64,
    )
    await store.update_status("trace-9", TraceStatus.UPLOADING)

    ready = await store.find_ready_traces()
    assert [trace.trace_id for trace in ready] == ["trace-8"]


@pytest.mark.asyncio
async def test_mark_recording_reported_updates_all_traces(
    store: SqliteStateStore,
) -> None:
    await store.upsert_trace_metadata(
        trace_id="trace-10",
        recording_id="rec-10",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-10.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    await store.upsert_trace_metadata(
        trace_id="trace-11",
        recording_id="rec-10",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-11.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    await store.mark_recording_reported("rec-10")

    row_10 = await _get_trace_row(store, "trace-10")
    row_11 = await _get_trace_row(store, "trace-11")
    assert row_10 is not None
    assert row_11 is not None
    assert row_10["progress_reported"] == ProgressReportStatus.REPORTED
    assert row_11["progress_reported"] == ProgressReportStatus.REPORTED


@pytest.mark.asyncio
async def test_find_unreported_traces_filters_reported(store: SqliteStateStore) -> None:
    await store.upsert_trace_metadata(
        trace_id="trace-12",
        recording_id="rec-12",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-12.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    await store.upsert_trace_metadata(
        trace_id="trace-13",
        recording_id="rec-13",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-13.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    await store.mark_recording_reported("rec-13")

    unreported = await store.find_unreported_traces()
    assert sorted(trace.trace_id for trace in unreported) == ["trace-12"]


@pytest.mark.asyncio
async def test_bytes_uploaded_persisted_across_restart(tmp_path: Path) -> None:
    """Test that bytes_uploaded is persisted and readable after restart."""
    db_path = tmp_path / "state.db"
    store = SqliteStateStore(db_path)
    await store.init_async_store()

    try:
        await store.upsert_trace_metadata(
            trace_id="trace-restart",
            recording_id="rec-restart",
            data_type=PRIMARY_DATA_TYPE,
            data_type_name="primary",
            path="/tmp/trace-restart.bin",
            robot_instance=ROBOT_INSTANCE,
        )
        await store.update_bytes_uploaded("trace-restart", 1024)

        row = await _get_trace_row(store, "trace-restart")
        assert row is not None
        assert row["bytes_uploaded"] == 1024
    finally:
        await store._engine.dispose()

    # Simulate restart by creating new store instance
    restarted_store = SqliteStateStore(db_path)
    await restarted_store.init_async_store()

    try:
        row_after = await _get_trace_row(restarted_store, "trace-restart")
        assert row_after is not None
        assert row_after["bytes_uploaded"] == 1024
    finally:
        await restarted_store._engine.dispose()


@pytest.mark.asyncio
async def test_wal_mode_enabled(store: SqliteStateStore) -> None:
    """Test that WAL journal mode is enabled for the database."""
    async with store._engine.begin() as conn:
        result = await conn.execute(text("PRAGMA journal_mode;"))
        mode = result.scalar_one()
    assert str(mode).lower() == "wal"


@pytest.mark.asyncio
async def test_state_transition_sequence(store: SqliteStateStore) -> None:
    """Test a valid sequence of state transitions.

    Flow: INITIALIZING -> WRITTEN (via bytes) -> UPLOADING -> UPLOADED
    """
    trace = await store.upsert_trace_metadata(
        trace_id="trace-transition",
        recording_id="rec-transition",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-transition.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    assert trace.status == TraceStatus.INITIALIZING

    trace = await store.upsert_trace_bytes(
        trace_id="trace-transition",
        recording_id="rec-transition",
        bytes_written=256,
    )
    assert trace.status == TraceStatus.WRITTEN

    row = await _get_trace_row(store, "trace-transition")
    assert row is not None
    assert row["status"] == TraceStatus.WRITTEN

    await store.update_status("trace-transition", TraceStatus.UPLOADING)
    row = await _get_trace_row(store, "trace-transition")
    assert row is not None
    assert row["status"] == TraceStatus.UPLOADING

    await store.update_status("trace-transition", TraceStatus.UPLOADED)
    row = await _get_trace_row(store, "trace-transition")
    assert row is not None
    assert row["status"] == TraceStatus.UPLOADED


@pytest.mark.asyncio
async def test_invalid_state_transition_rejected(store: SqliteStateStore) -> None:
    """Test that invalid state transitions are rejected."""
    await store.upsert_trace_metadata(
        trace_id="trace-invalid",
        recording_id="rec-invalid",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-invalid.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    with pytest.raises(ValueError, match="Invalid status transition"):
        await store.update_status("trace-invalid", TraceStatus.UPLOADED)

    row = await _get_trace_row(store, "trace-invalid")
    assert row is not None
    assert row["status"] == TraceStatus.INITIALIZING


@pytest.mark.asyncio
async def test_state_recovery_after_restart(tmp_path: Path) -> None:
    """Test that state is properly recovered after a restart."""
    db_path = tmp_path / "state.db"
    store = SqliteStateStore(db_path)
    await store.init_async_store()

    try:
        await store.upsert_trace_metadata(
            trace_id="trace-recover",
            recording_id="rec-recover",
            data_type=PRIMARY_DATA_TYPE,
            data_type_name="primary",
            path="/tmp/trace-recover.bin",
            robot_instance=ROBOT_INSTANCE,
        )
        await store.upsert_trace_bytes(
            trace_id="trace-recover",
            recording_id="rec-recover",
            bytes_written=512,
        )
    finally:
        await store._engine.dispose()

    recovered_store = SqliteStateStore(db_path)
    await recovered_store.init_async_store()

    try:
        recovered_trace = await recovered_store.get_trace("trace-recover")
        assert recovered_trace is not None
        assert recovered_trace.status == TraceStatus.WRITTEN

        ready = await recovered_store.find_ready_traces()
        assert [trace.trace_id for trace in ready] == ["trace-recover"]

        await recovered_store.update_status("trace-recover", TraceStatus.UPLOADING)
        updated = await recovered_store.get_trace("trace-recover")
        assert updated is not None
        assert updated.status == TraceStatus.UPLOADING
    finally:
        await recovered_store._engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_writes_do_not_lock(tmp_path: Path) -> None:
    """Test that concurrent writes to the same trace don't cause deadlocks."""
    db_path = tmp_path / "state.db"
    store = SqliteStateStore(db_path)
    await store.init_async_store()

    try:
        await store.upsert_trace_metadata(
            trace_id="trace-concurrent",
            recording_id="rec-concurrent",
            data_type=PRIMARY_DATA_TYPE,
            data_type_name="primary",
            path="/tmp/trace-concurrent.bin",
            robot_instance=ROBOT_INSTANCE,
        )

        errors: list[Exception] = []

        async def worker(bytes_uploaded: int) -> None:
            try:
                for _ in range(5):
                    await store.update_bytes_uploaded(
                        "trace-concurrent", bytes_uploaded
                    )
            except Exception as exc:
                errors.append(exc)

        await asyncio.gather(worker(10), worker(20))

        assert errors == []
        row = await _get_trace_row(store, "trace-concurrent")
        assert row is not None
        assert row["bytes_uploaded"] in {10, 20}
    finally:
        await store._engine.dispose()


@pytest.mark.asyncio
async def test_race_conditions_on_rapid_state_changes(tmp_path: Path, caplog) -> None:
    """Test that rapid state changes don't corrupt data."""
    db_path = tmp_path / "state.db"
    store = SqliteStateStore(db_path)
    await store.init_async_store()

    try:
        await store.upsert_trace_metadata(
            trace_id="trace-race",
            recording_id="rec-race",
            data_type=PRIMARY_DATA_TYPE,
            data_type_name="primary",
            path="/tmp/trace-race.bin",
            robot_instance=ROBOT_INSTANCE,
        )
        await store.upsert_trace_bytes(
            trace_id="trace-race",
            recording_id="rec-race",
            bytes_written=64,
        )

        errors: list[str] = []

        async def worker() -> None:
            try:
                await store.update_status("trace-race", TraceStatus.UPLOADING)
                await store.update_status("trace-race", TraceStatus.UPLOADED)
            except ValueError as exc:
                errors.append(str(exc))

        await asyncio.gather(worker(), worker())

        row = await _get_trace_row(store, "trace-race")
        assert row is not None
        assert row["status"] == TraceStatus.UPLOADED
    finally:
        await store._engine.dispose()
