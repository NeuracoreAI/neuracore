from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select, text

from neuracore.data_daemon.models import (
    DATA_TYPE_CONTENT_MAPPING,
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
    return store


async def _get_trace_row(store: SqliteStateStore, trace_id: str) -> dict | None:
    async with store._engine.begin() as conn:
        result = await conn.execute(select(traces).where(traces.c.trace_id == trace_id))
        row = result.mappings().first()
        return dict(row) if row else None


@pytest.mark.asyncio
async def test_create_trace_inserts_row(store: SqliteStateStore) -> None:
    await store.create_trace(
        "trace-1",
        "rec-1",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-1.bin",
        total_bytes=128,
        robot_instance=ROBOT_INSTANCE,
    )

    row = await _get_trace_row(store, "trace-1")
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


@pytest.mark.asyncio
async def test_create_trace_updates_existing(store: SqliteStateStore) -> None:
    await store.create_trace(
        "trace-2",
        "rec-1",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-2.bin",
        total_bytes=10,
        robot_instance=ROBOT_INSTANCE,
    )
    await store.create_trace(
        "trace-2",
        "rec-2",
        SECONDARY_DATA_TYPE,
        data_type_name="secondary",
        path="/tmp/trace-2.mp4",
        total_bytes=20,
        robot_instance=ROBOT_INSTANCE,
    )

    row = await _get_trace_row(store, "trace-2")
    assert row is not None
    assert row["recording_id"] == "rec-2"
    assert row["data_type"] == SECONDARY_DATA_TYPE
    assert row["path"] == "/tmp/trace-2.mp4"
    assert row["total_bytes"] == 20
    assert row["status"] == TraceStatus.PENDING


@pytest.mark.asyncio
async def test_update_bytes_uploaded_accumulates(store: SqliteStateStore) -> None:
    await store.create_trace(
        "trace-3",
        "rec-3",
        PRIMARY_DATA_TYPE,
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
    await store.create_trace(
        "trace-4",
        "rec-4",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-4.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    await store.update_status("trace-4", TraceStatus.WRITTEN)
    await store.update_status("trace-4", TraceStatus.FAILED, error_message="boom")

    row = await _get_trace_row(store, "trace-4")
    assert row is not None
    assert row["status"] == TraceStatus.FAILED
    assert row["error_message"] == "boom"


@pytest.mark.asyncio
async def test_record_error_sets_code_and_status(store: SqliteStateStore) -> None:
    await store.create_trace(
        "trace-4b",
        "rec-4b",
        PRIMARY_DATA_TYPE,
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
async def test_claim_ready_traces_filters_ready_written(
    store: SqliteStateStore,
) -> None:
    await store.create_trace(
        "trace-5",
        "rec-5",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-5.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    await store.create_trace(
        "trace-6",
        "rec-6",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-6.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    await store.create_trace(
        "trace-7",
        "rec-7",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-7.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    async with store._engine.begin() as conn:
        await conn.execute(
            traces.update()
            .where(traces.c.trace_id == "trace-5")
            .values(status=TraceStatus.WRITTEN, ready_for_upload=1)
        )
        await conn.execute(
            traces.update()
            .where(traces.c.trace_id == "trace-6")
            .values(status=TraceStatus.WRITTEN, ready_for_upload=0)
        )
        await conn.execute(
            traces.update()
            .where(traces.c.trace_id == "trace-7")
            .values(status=TraceStatus.PENDING, ready_for_upload=1)
        )

    claimed = await store.claim_ready_traces(limit=10)
    assert [row["trace_id"] for row in claimed] == ["trace-5"]


@pytest.mark.asyncio
async def test_mark_trace_as_written_sets_total_bytes_and_ready_flag(
    store: SqliteStateStore,
) -> None:
    await store.create_trace(
        "trace-6b",
        "rec-6b",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-6b.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    async with store._engine.begin() as conn:
        await conn.execute(
            traces.update()
            .where(traces.c.trace_id == "trace-6b")
            .values(bytes_written=64)
        )

    await store.mark_trace_as_written("trace-6b", 64)

    row = await _get_trace_row(store, "trace-6b")
    assert row is not None
    assert row["status"] == TraceStatus.WRITTEN
    assert row["total_bytes"] == 64
    assert row["ready_for_upload"] == 1
    assert row["progress_reported"] == 0


@pytest.mark.asyncio
async def test_claim_ready_traces_marks_uploading(store: SqliteStateStore) -> None:
    await store.create_trace(
        "trace-7b",
        "rec-7b",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-7b.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    async with store._engine.begin() as conn:
        await conn.execute(
            traces.update()
            .where(traces.c.trace_id == "trace-7b")
            .values(status=TraceStatus.WRITTEN, ready_for_upload=1)
        )

    claimed = await store.claim_ready_traces(limit=10)
    assert [row["trace_id"] for row in claimed] == ["trace-7b"]
    assert claimed[0]["status"] == TraceStatus.UPLOADING
    assert claimed[0]["ready_for_upload"] == 0


@pytest.mark.asyncio
async def test_delete_trace_removes_row(store: SqliteStateStore) -> None:
    await store.create_trace(
        "trace-7",
        "rec-7",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-7.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    await store.delete_trace("trace-7")

    assert await _get_trace_row(store, "trace-7") is None


@pytest.mark.asyncio
async def test_find_ready_traces_returns_only_ready(store: SqliteStateStore) -> None:
    await store.create_trace(
        "trace-8",
        "rec-8",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-8.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    await store.create_trace(
        "trace-9",
        "rec-9",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-9.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    async with store._engine.begin() as conn:
        await conn.execute(
            traces.update()
            .where(traces.c.trace_id == "trace-8")
            .values(status=TraceStatus.WRITTEN, ready_for_upload=1)
        )
        await conn.execute(
            traces.update()
            .where(traces.c.trace_id == "trace-9")
            .values(status=TraceStatus.WRITTEN, ready_for_upload=0)
        )

    ready = await store.find_ready_traces()
    assert sorted(trace.trace_id for trace in ready) == ["trace-8"]


@pytest.mark.asyncio
async def test_mark_recording_reported_updates_all_traces(
    store: SqliteStateStore,
) -> None:
    await store.create_trace(
        "trace-10",
        "rec-10",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-10.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    await store.create_trace(
        "trace-11",
        "rec-10",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-11.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    await store.mark_recording_reported("rec-10")

    row_10 = await _get_trace_row(store, "trace-10")
    row_11 = await _get_trace_row(store, "trace-11")
    assert row_10 is not None
    assert row_11 is not None
    assert row_10["progress_reported"] == 1
    assert row_11["progress_reported"] == 1


@pytest.mark.asyncio
async def test_find_unreported_traces_filters_reported(store: SqliteStateStore) -> None:
    await store.create_trace(
        "trace-12",
        "rec-12",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-12.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    await store.create_trace(
        "trace-13",
        "rec-13",
        PRIMARY_DATA_TYPE,
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

    await store.create_trace(
        "trace-restart",
        "rec-restart",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-restart.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    await store.update_bytes_uploaded("trace-restart", 1024)

    row = await _get_trace_row(store, "trace-restart")
    assert row is not None
    assert row["bytes_uploaded"] == 1024

    # Simulate restart by creating new store instance
    restarted_store = SqliteStateStore(db_path)
    await restarted_store.init_async_store()

    row_after = await _get_trace_row(restarted_store, "trace-restart")
    assert row_after is not None
    assert row_after["bytes_uploaded"] == 1024


@pytest.mark.asyncio
async def test_wal_mode_enabled(store: SqliteStateStore) -> None:
    """Test that WAL journal mode is enabled for the database."""
    async with store._engine.begin() as conn:
        result = await conn.execute(text("PRAGMA journal_mode;"))
        mode = result.scalar_one()
    assert str(mode).lower() == "wal"


@pytest.mark.asyncio
async def test_state_transition_sequence(store: SqliteStateStore) -> None:
    """Test a valid sequence of state transitions."""
    await store.create_trace(
        "trace-transition",
        "rec-transition",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-transition.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    await store.update_status("trace-transition", TraceStatus.WRITING)
    await store.mark_trace_as_written("trace-transition", 256)

    claimed = await store.claim_ready_traces(limit=1)
    assert [row["trace_id"] for row in claimed] == ["trace-transition"]
    assert claimed[0]["status"] == TraceStatus.UPLOADING

    await store.update_status("trace-transition", TraceStatus.UPLOADED)
    row = await _get_trace_row(store, "trace-transition")
    assert row is not None
    assert row["status"] == TraceStatus.UPLOADED


@pytest.mark.asyncio
async def test_invalid_state_transition_rejected(store: SqliteStateStore) -> None:
    """Test that invalid state transitions are rejected."""
    await store.create_trace(
        "trace-invalid",
        "rec-invalid",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-invalid.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    with pytest.raises(ValueError, match="Invalid status transition"):
        await store.update_status("trace-invalid", TraceStatus.UPLOADED)

    row = await _get_trace_row(store, "trace-invalid")
    assert row is not None
    assert row["status"] == TraceStatus.PENDING


@pytest.mark.asyncio
async def test_state_recovery_after_restart(tmp_path: Path) -> None:
    """Test that state is properly recovered after a restart."""
    db_path = tmp_path / "state.db"
    store = SqliteStateStore(db_path)
    await store.init_async_store()

    await store.create_trace(
        "trace-recover",
        "rec-recover",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-recover.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    await store.update_status("trace-recover", TraceStatus.WRITING)
    await store.mark_trace_as_written("trace-recover", 512)

    # Simulate restart
    recovered_store = SqliteStateStore(db_path)
    await recovered_store.init_async_store()

    recovered_trace = await recovered_store.get_trace("trace-recover")
    assert recovered_trace is not None
    assert recovered_trace.status == TraceStatus.WRITTEN
    assert recovered_trace.ready_for_upload == 1

    claimed = await recovered_store.claim_ready_traces(limit=1)
    assert [row["trace_id"] for row in claimed] == ["trace-recover"]
    assert claimed[0]["status"] == TraceStatus.UPLOADING


@pytest.mark.asyncio
async def test_concurrent_writes_do_not_lock(tmp_path: Path) -> None:
    """Test that concurrent writes to the same trace don't cause deadlocks."""
    db_path = tmp_path / "state.db"
    store = SqliteStateStore(db_path)
    await store.init_async_store()

    await store.create_trace(
        "trace-concurrent",
        "rec-concurrent",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-concurrent.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    errors: list[Exception] = []

    async def worker(bytes_uploaded: int) -> None:
        try:
            for _ in range(5):
                await store.update_bytes_uploaded("trace-concurrent", bytes_uploaded)
        except Exception as exc:
            errors.append(exc)

    await asyncio.gather(worker(10), worker(20))

    assert errors == []
    row = await _get_trace_row(store, "trace-concurrent")
    assert row is not None
    assert row["bytes_uploaded"] in {10, 20}


@pytest.mark.asyncio
async def test_race_conditions_on_rapid_state_changes(tmp_path: Path, caplog) -> None:
    """Test that rapid state changes don't corrupt data."""
    db_path = tmp_path / "state.db"
    store = SqliteStateStore(db_path)
    await store.init_async_store()

    await store.create_trace(
        "trace-race",
        "rec-race",
        PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-race.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    await store.update_status("trace-race", TraceStatus.WRITING)

    errors: list[str] = []

    async def worker() -> None:
        try:
            await store.update_status("trace-race", TraceStatus.WRITTEN)
            await store.update_status("trace-race", TraceStatus.UPLOADING)
            await store.update_status("trace-race", TraceStatus.UPLOADED)
        except ValueError as exc:
            errors.append(str(exc))

    await asyncio.gather(worker(), worker())

    # At least one worker should hit a race (error raised or warning logged)
    assert errors or "Failed to update trace status: Trace trace-race" in caplog.text
    row = await _get_trace_row(store, "trace-race")
    assert row is not None
    assert row["status"] == TraceStatus.UPLOADED
