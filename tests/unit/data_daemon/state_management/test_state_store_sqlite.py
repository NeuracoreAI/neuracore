from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select, text, update

from neuracore.data_daemon.models import (
    DATA_TYPE_CONTENT_MAPPING,
    ProgressReportStatus,
    TraceErrorCode,
    TraceRegistrationStatus,
    TraceUploadStatus,
    TraceWriteStatus,
)
from neuracore.data_daemon.state_management.state_store_sqlite import SqliteStateStore
from neuracore.data_daemon.state_management.tables import recordings, traces

DATA_TYPES = list(DATA_TYPE_CONTENT_MAPPING.keys())
PRIMARY_DATA_TYPE = DATA_TYPES[0]
SECONDARY_DATA_TYPE = DATA_TYPES[1] if len(DATA_TYPES) > 1 else DATA_TYPES[0]
ROBOT_INSTANCE = 1


def _create_legacy_schema_db(db_path: Path) -> None:
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
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_updated DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                num_upload_attempts INTEGER NOT NULL DEFAULT 0,
                next_retry_at DATETIME
            )
            """
        )
        conn.execute("CREATE INDEX idx_traces_status ON traces(status)")
        conn.execute("CREATE INDEX idx_traces_next_retry_at ON traces(next_retry_at)")
        conn.commit()
    finally:
        conn.close()


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


async def _get_recording_row(store: SqliteStateStore, recording_id: str) -> dict | None:
    async with store._engine.begin() as conn:
        result = await conn.execute(
            select(recordings).where(recordings.c.recording_id == recording_id)
        )
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
        robot_instance=ROBOT_INSTANCE,
    )

    assert trace.write_status == TraceWriteStatus.INITIALIZING
    row = await _get_trace_row(store, "trace-1")
    assert row is not None
    assert row["trace_id"] == "trace-1"
    assert row["recording_id"] == "rec-1"
    assert row["data_type"] == PRIMARY_DATA_TYPE
    assert row["path"] == "/tmp/trace-1.bin"
    assert row["total_bytes"] is None
    assert row["write_status"] == TraceWriteStatus.INITIALIZING
    assert row["bytes_written"] is None
    assert row["bytes_uploaded"] == 0
    assert row["error_message"] is None
    assert row["robot_instance"] == ROBOT_INSTANCE
    recording = await _get_recording_row(store, "rec-1")
    assert recording is None


@pytest.mark.asyncio
async def test_init_async_store_migrates_legacy_trace_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy-state.db"
    _create_legacy_schema_db(db_path)

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    conn = sqlite3.connect(str(db_path))
    try:
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
                bytes_uploaded,
                progress_reported,
                expected_trace_count_reported,
                created_at,
                last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "trace-written",
                "written",
                "rec-legacy",
                PRIMARY_DATA_TYPE.value,
                "primary",
                "/tmp/trace-written.bin",
                100,
                100,
                0,
                "pending",
                0,
                now,
                now,
            ),
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
                bytes_uploaded,
                progress_reported,
                expected_trace_count_reported,
                created_at,
                last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "trace-uploaded",
                "uploaded",
                "rec-legacy",
                PRIMARY_DATA_TYPE.value,
                "primary",
                "/tmp/trace-uploaded.bin",
                200,
                200,
                200,
                "reported",
                1,
                now,
                now,
            ),
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
                progress_reported,
                expected_trace_count_reported,
                created_at,
                last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "trace-failed-no-bytes",
                "failed",
                "rec-failed",
                PRIMARY_DATA_TYPE.value,
                "primary",
                "/tmp/trace-failed-no-bytes.bin",
                "pending",
                0,
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    store = SqliteStateStore(db_path)
    await store.init_async_store()
    try:
        row_written = await _get_trace_row(store, "trace-written")
        row_uploaded = await _get_trace_row(store, "trace-uploaded")
        row_failed = await _get_trace_row(store, "trace-failed-no-bytes")
        recording = await _get_recording_row(store, "rec-legacy")
        failed_recording = await _get_recording_row(store, "rec-failed")

        assert row_written is not None
        assert row_uploaded is not None
        assert row_failed is not None
        assert recording is not None
        assert failed_recording is not None

        assert row_written["write_status"] == TraceWriteStatus.WRITTEN
        assert row_written["registration_status"] == TraceRegistrationStatus.PENDING
        assert row_written["upload_status"] == TraceUploadStatus.PENDING

        assert row_uploaded["write_status"] == TraceWriteStatus.WRITTEN
        assert row_uploaded["registration_status"] == TraceRegistrationStatus.REGISTERED
        assert row_uploaded["upload_status"] == TraceUploadStatus.UPLOADED

        assert row_failed["write_status"] == TraceWriteStatus.FAILED
        assert row_failed["registration_status"] == TraceRegistrationStatus.PENDING
        assert row_failed["upload_status"] == TraceUploadStatus.FAILED

        assert int(recording["expected_trace_count"]) == 2
        assert int(recording["trace_count"]) == 2
        assert int(recording["uploaded_trace_count"]) == 1
        assert int(recording["expected_trace_count_reported"]) == 1
        assert recording["progress_reported"] == ProgressReportStatus.PENDING

        async with store._engine.begin() as async_conn:
            legacy_exists = (
                await async_conn.execute(
                    text(
                        "SELECT 1 FROM sqlite_master "
                        "WHERE type='table' AND name='traces_legacy'"
                    )
                )
            ).scalar_one_or_none()
        assert legacy_exists is None
    finally:
        await store._engine.dispose()


@pytest.mark.asyncio
async def test_upsert_trace_metadata_updates_existing(store: SqliteStateStore) -> None:
    await store.upsert_trace_metadata(
        trace_id="trace-2",
        recording_id="rec-1",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-2.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    trace = await store.upsert_trace_metadata(
        trace_id="trace-2",
        recording_id="rec-1",
        data_type=SECONDARY_DATA_TYPE,
        data_type_name="secondary",
        path="/tmp/trace-2.mp4",
        robot_instance=ROBOT_INSTANCE,
    )

    assert trace.write_status == TraceWriteStatus.INITIALIZING
    row = await _get_trace_row(store, "trace-2")
    assert row is not None
    assert row["recording_id"] == "rec-1"
    assert row["data_type"] == SECONDARY_DATA_TYPE
    assert row["path"] == "/tmp/trace-2.mp4"
    assert row["total_bytes"] is None
    assert row["write_status"] == TraceWriteStatus.INITIALIZING


@pytest.mark.asyncio
async def test_upsert_trace_bytes_inserts_row(store: SqliteStateStore) -> None:
    trace = await store.upsert_trace_bytes(
        trace_id="trace-bytes-1",
        recording_id="rec-bytes-1",
        bytes_written=64,
    )

    assert trace.write_status == TraceWriteStatus.PENDING_METADATA
    row = await _get_trace_row(store, "trace-bytes-1")
    assert row is not None
    assert row["trace_id"] == "trace-bytes-1"
    assert row["recording_id"] == "rec-bytes-1"
    assert row["bytes_written"] == 64
    assert row["total_bytes"] == 64
    assert row["write_status"] == TraceWriteStatus.PENDING_METADATA
    assert row["bytes_uploaded"] == 0


@pytest.mark.asyncio
async def test_upsert_trace_bytes_preserves_existing_total_bytes(
    store: SqliteStateStore,
) -> None:
    await store.upsert_trace_metadata(
        trace_id="trace-bytes-preserve-total",
        recording_id="rec-bytes-preserve-total",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-bytes-preserve-total.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    trace = await store.upsert_trace_bytes(
        trace_id="trace-bytes-preserve-total",
        recording_id="rec-bytes-preserve-total",
        bytes_written=64,
    )

    assert trace.write_status == TraceWriteStatus.WRITTEN
    row = await _get_trace_row(store, "trace-bytes-preserve-total")
    assert row is not None
    assert row["bytes_written"] == 64


@pytest.mark.asyncio
async def test_upsert_trace_bytes_backfills_missing_total_bytes(
    store: SqliteStateStore,
) -> None:
    await store.upsert_trace_metadata(
        trace_id="trace-bytes-backfill-total",
        recording_id="rec-bytes-backfill-total",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-bytes-backfill-total.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    trace = await store.upsert_trace_bytes(
        trace_id="trace-bytes-backfill-total",
        recording_id="rec-bytes-backfill-total",
        bytes_written=96,
    )

    assert trace.write_status == TraceWriteStatus.WRITTEN
    row = await _get_trace_row(store, "trace-bytes-backfill-total")
    assert row is not None
    assert row["bytes_written"] == 96


@pytest.mark.asyncio
async def test_upsert_trace_write_progress_transitions_to_writing_then_written(
    store: SqliteStateStore,
) -> None:
    await store.upsert_trace_metadata(
        trace_id="trace-writing-1",
        recording_id="rec-writing-1",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-writing-1.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    assert await _get_recording_row(store, "rec-writing-1") is None

    trace = await store.upsert_trace_write_progress(
        trace_id="trace-writing-1",
        recording_id="rec-writing-1",
        bytes_written=128,
    )

    assert trace.write_status == TraceWriteStatus.WRITING
    row = await _get_trace_row(store, "trace-writing-1")
    assert row is not None
    assert row["bytes_written"] == 128
    assert row["total_bytes"] is None
    assert row["write_status"] == TraceWriteStatus.WRITING
    recording_row = await _get_recording_row(store, "rec-writing-1")
    assert recording_row is not None
    assert int(recording_row["trace_count"]) == 0

    trace = await store.upsert_trace_bytes(
        trace_id="trace-writing-1",
        recording_id="rec-writing-1",
        bytes_written=64,
    )

    assert trace.write_status == TraceWriteStatus.WRITTEN
    row = await _get_trace_row(store, "trace-writing-1")
    assert row is not None
    assert row["bytes_written"] == 64
    assert row["total_bytes"] == 64
    assert row["write_status"] == TraceWriteStatus.WRITTEN
    recording_row = await _get_recording_row(store, "rec-writing-1")
    assert recording_row is not None
    assert int(recording_row["trace_count"]) == 1


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
async def test_update_upload_status_and_record_error(store: SqliteStateStore) -> None:
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

    await store.update_upload_status("trace-4", TraceUploadStatus.FAILED)
    await store.record_error("trace-4", error_message="boom")

    row = await _get_trace_row(store, "trace-4")
    assert row is not None
    assert row["upload_status"] == TraceUploadStatus.FAILED
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
    assert row["upload_status"] == TraceUploadStatus.PENDING
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
    assert trace.write_status == TraceWriteStatus.INITIALIZING

    trace = await store.upsert_trace_bytes(
        trace_id="trace-6b",
        recording_id="rec-6b",
        bytes_written=64,
    )

    assert trace.write_status == TraceWriteStatus.WRITTEN
    row = await _get_trace_row(store, "trace-6b")
    assert row is not None
    assert row["write_status"] == TraceWriteStatus.WRITTEN
    assert row["bytes_written"] == 64
    assert row["total_bytes"] == 64
    recording_row = await _get_recording_row(store, "rec-6b")
    assert recording_row is not None
    assert int(recording_row["trace_count"]) == 1
    assert recording_row["progress_reported"] == ProgressReportStatus.PENDING

    # Duplicate TRACE_WRITTEN for the same trace must not increment expected count.
    await store.upsert_trace_bytes(
        trace_id="trace-6b",
        recording_id="rec-6b",
        bytes_written=64,
    )
    recording_row = await _get_recording_row(store, "rec-6b")
    assert recording_row is not None
    assert int(recording_row["trace_count"]) == 1


@pytest.mark.asyncio
async def test_join_pattern_bytes_then_metadata_transitions_to_written(
    store: SqliteStateStore,
) -> None:
    """Test PENDING_METADATA + metadata -> WRITTEN transition."""
    trace = await store.upsert_trace_bytes(
        trace_id="trace-6c",
        recording_id="rec-6c",
        bytes_written=128,
    )
    assert trace.write_status == TraceWriteStatus.PENDING_METADATA
    recording_row = await _get_recording_row(store, "rec-6c")
    assert recording_row is None

    trace = await store.upsert_trace_metadata(
        trace_id="trace-6c",
        recording_id="rec-6c",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-6c.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    assert trace.write_status == TraceWriteStatus.WRITTEN
    row = await _get_trace_row(store, "trace-6c")
    assert row is not None
    assert row["write_status"] == TraceWriteStatus.WRITTEN
    assert row["bytes_written"] == 128
    assert row["total_bytes"] == 128
    recording_row = await _get_recording_row(store, "rec-6c")
    assert recording_row is not None
    assert int(recording_row["trace_count"]) == 1
    assert recording_row["progress_reported"] == ProgressReportStatus.PENDING


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
    await store.mark_traces_as_registered(["trace-8"])

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
    await store.mark_traces_as_registered(["trace-9"])
    await store.update_upload_status("trace-9", TraceUploadStatus.UPLOADING)

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

    recording = await _get_recording_row(store, "rec-10")
    assert recording is not None
    assert recording["progress_reported"] == ProgressReportStatus.REPORTED


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
    await store.set_expected_trace_count("rec-12", 1)
    await store.upsert_trace_metadata(
        trace_id="trace-13",
        recording_id="rec-13",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-13.bin",
        robot_instance=ROBOT_INSTANCE,
    )
    await store.set_expected_trace_count("rec-13", 1)

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
    assert trace.write_status == TraceWriteStatus.INITIALIZING

    trace = await store.upsert_trace_bytes(
        trace_id="trace-transition",
        recording_id="rec-transition",
        bytes_written=256,
    )
    assert trace.write_status == TraceWriteStatus.WRITTEN

    row = await _get_trace_row(store, "trace-transition")
    assert row is not None
    assert row["write_status"] == TraceWriteStatus.WRITTEN

    await store.update_upload_status("trace-transition", TraceUploadStatus.UPLOADING)
    row = await _get_trace_row(store, "trace-transition")
    assert row is not None
    assert row["upload_status"] == TraceUploadStatus.UPLOADING

    await store.update_upload_status("trace-transition", TraceUploadStatus.UPLOADED)
    row = await _get_trace_row(store, "trace-transition")
    assert row is not None
    assert row["upload_status"] == TraceUploadStatus.UPLOADED


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

    with pytest.raises(ValueError, match="Trace not found"):
        await store.update_upload_status(
            "trace-invalid-missing", TraceUploadStatus.UPLOADED
        )

    row = await _get_trace_row(store, "trace-invalid")
    assert row is not None
    assert row["write_status"] == TraceWriteStatus.INITIALIZING


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
        await store.mark_traces_as_registered(["trace-recover"])
    finally:
        await store._engine.dispose()

    recovered_store = SqliteStateStore(db_path)
    await recovered_store.init_async_store()

    try:
        recovered_trace = await recovered_store.get_trace("trace-recover")
        assert recovered_trace is not None
        assert recovered_trace.write_status == TraceWriteStatus.WRITTEN

        ready = await recovered_store.find_ready_traces()
        assert [trace.trace_id for trace in ready] == ["trace-recover"]

        await recovered_store.update_upload_status(
            "trace-recover", TraceUploadStatus.UPLOADING
        )
        updated = await recovered_store.get_trace("trace-recover")
        assert updated is not None
        assert updated.upload_status == TraceUploadStatus.UPLOADING
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
                await store.update_upload_status(
                    "trace-race", TraceUploadStatus.UPLOADING
                )
                await store.update_upload_status(
                    "trace-race", TraceUploadStatus.UPLOADED
                )
            except ValueError as exc:
                errors.append(str(exc))

        await asyncio.gather(worker(), worker())

        row = await _get_trace_row(store, "trace-race")
        assert row is not None
        assert row["upload_status"] == TraceUploadStatus.UPLOADED
    finally:
        await store._engine.dispose()


async def _set_attempts_and_retry_at(
    store: SqliteStateStore, trace_id: str, attempts: int, next_retry_at
) -> None:
    async with store._engine.begin() as conn:
        await conn.execute(
            traces.update()
            .where(traces.c.trace_id == trace_id)
            .values(num_upload_attempts=int(attempts), next_retry_at=next_retry_at)
        )


@pytest.mark.asyncio
async def test_create_trace_sets_retry_defaults(store: SqliteStateStore) -> None:
    """New traces should default to 0 attempts and no next_retry_at."""
    await store.upsert_trace_metadata(
        trace_id="trace-retry-defaults",
        recording_id="rec-retry-defaults",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/trace-retry-defaults.bin",
        robot_instance=ROBOT_INSTANCE,
    )

    row = await _get_trace_row(store, "trace-retry-defaults")
    assert row is not None
    assert int(row["num_upload_attempts"]) == 0
    assert row["next_retry_at"] is None


@pytest.mark.asyncio
async def test_retry_fields_persisted_across_restart(tmp_path: Path) -> None:
    """num_upload_attempts and next_retry_at should persist across restart."""
    db_path = tmp_path / "state.db"
    store = SqliteStateStore(db_path)
    await store.init_async_store()
    try:
        await store.upsert_trace_metadata(
            trace_id="trace-retry-persist",
            recording_id="rec-retry-persist",
            data_type=PRIMARY_DATA_TYPE,
            data_type_name="primary",
            path="/tmp/trace-retry-persist.bin",
            robot_instance=ROBOT_INSTANCE,
        )

        next_retry_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(
            seconds=30
        )
        await _set_attempts_and_retry_at(store, "trace-retry-persist", 2, next_retry_at)

        row = await _get_trace_row(store, "trace-retry-persist")
        assert row is not None
        assert int(row["num_upload_attempts"]) == 2
        assert row["next_retry_at"] is not None
    finally:
        await store._engine.dispose()

    restarted = SqliteStateStore(db_path)
    await restarted.init_async_store()
    try:
        row_after = await _get_trace_row(restarted, "trace-retry-persist")
        assert row_after is not None
        assert int(row_after["num_upload_attempts"]) == 2
        assert row_after["next_retry_at"] is not None
    finally:
        await restarted._engine.dispose()


@pytest.mark.asyncio
async def test_find_ready_traces_excludes_future_retries(
    store: SqliteStateStore,
) -> None:
    """A WRITTEN trace with next_retry_at in the future not be returned as ready."""
    await store.upsert_trace_metadata(
        trace_id="trace-future-retry",
        recording_id="rec-id",
        data_type=PRIMARY_DATA_TYPE,
        data_type_name="primary",
        path="/tmp/....bin",
        robot_instance=ROBOT_INSTANCE,
    )
    async with store._engine.begin() as conn:
        await conn.execute(
            traces.update()
            .where(traces.c.trace_id == "trace-future-retry")
            .values(
                write_status=TraceWriteStatus.WRITTEN,
                registration_status=TraceRegistrationStatus.REGISTERED,
                upload_status=TraceUploadStatus.RETRYING,
            )
        )

    future = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=60)
    await _set_attempts_and_retry_at(store, "trace-future-retry", 1, future)

    ready = await store.find_ready_traces()
    assert "trace-future-retry" not in {t.trace_id for t in ready}


@pytest.mark.asyncio
async def test_delete_expired_completed_recordings_deletes_old_completed_rows(
    store: SqliteStateStore,
) -> None:
    await store.set_expected_trace_count("rec-old", 2)
    await store.mark_expected_trace_count_reported("rec-old")
    await store.mark_recording_reported("rec-old")

    old_stopped_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
        hours=(24 * 30) + 1
    )

    async with store._engine.begin() as conn:
        await conn.execute(
            update(recordings)
            .where(recordings.c.recording_id == "rec-old")
            .values(
                expected_trace_count=2,
                uploaded_trace_count=2,
                stopped_at=old_stopped_at,
            )
        )

    deleted = await store.delete_expired_completed_recordings(24 * 30)

    assert deleted == 1
    assert await _get_recording_row(store, "rec-old") is None


@pytest.mark.asyncio
async def test_delete_expired_completed_recordings_keeps_recent_completed_rows(
    store: SqliteStateStore,
) -> None:
    await store.set_expected_trace_count("rec-recent", 2)
    await store.mark_expected_trace_count_reported("rec-recent")
    await store.mark_recording_reported("rec-recent")

    recent_stopped_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
        hours=(24 * 30) - 1
    )

    async with store._engine.begin() as conn:
        await conn.execute(
            update(recordings)
            .where(recordings.c.recording_id == "rec-recent")
            .values(
                expected_trace_count=2,
                uploaded_trace_count=2,
                stopped_at=recent_stopped_at,
            )
        )

    deleted = await store.delete_expired_completed_recordings(24 * 30)

    assert deleted == 0
    assert await _get_recording_row(store, "rec-recent") is not None


@pytest.mark.asyncio
async def test_delete_expired_completed_recordings_keeps_unreported_rows(
    store: SqliteStateStore,
) -> None:
    await store.set_expected_trace_count("rec-unreported", 2)

    old_stopped_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
        hours=(24 * 30) + 1
    )

    async with store._engine.begin() as conn:
        await conn.execute(
            update(recordings)
            .where(recordings.c.recording_id == "rec-unreported")
            .values(
                expected_trace_count=2,
                uploaded_trace_count=2,
                progress_reported=ProgressReportStatus.PENDING,
                stopped_at=old_stopped_at,
            )
        )

    deleted = await store.delete_expired_completed_recordings(24 * 30)

    assert deleted == 0
    assert await _get_recording_row(store, "rec-unreported") is not None


@pytest.mark.asyncio
async def test_delete_expired_completed_recordings_keeps_incomplete_rows(
    store: SqliteStateStore,
) -> None:
    await store.set_expected_trace_count("rec-incomplete", 3)
    await store.mark_expected_trace_count_reported("rec-incomplete")
    await store.mark_recording_reported("rec-incomplete")

    old_stopped_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
        hours=(24 * 30) + 1
    )

    async with store._engine.begin() as conn:
        await conn.execute(
            update(recordings)
            .where(recordings.c.recording_id == "rec-incomplete")
            .values(
                expected_trace_count=3,
                uploaded_trace_count=2,
                stopped_at=old_stopped_at,
            )
        )

    deleted = await store.delete_expired_completed_recordings(24 * 30)

    assert deleted == 0
    assert await _get_recording_row(store, "rec-incomplete") is not None


@pytest.mark.asyncio
async def test_init_async_store_adds_missing_recordings_org_id_column(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    store = SqliteStateStore(db_path)

    async with store._engine.begin() as conn:
        await conn.execute(
            text(
                """
            CREATE TABLE recordings (
                recording_id TEXT PRIMARY KEY,
                expected_trace_count INTEGER NOT NULL DEFAULT 0,
                trace_count INTEGER NOT NULL DEFAULT 0,
                expected_trace_count_reported INTEGER NOT NULL DEFAULT 0,
                uploaded_trace_count INTEGER NOT NULL DEFAULT 0,
                progress_reported TEXT NOT NULL DEFAULT 'pending',
                stopped_at DATETIME,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_updated DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """
            )
        )
        await conn.execute(
            text(
                """
            INSERT INTO recordings (
                recording_id,
                expected_trace_count,
                trace_count,
                expected_trace_count_reported,
                uploaded_trace_count,
                progress_reported
            ) VALUES (
                'rec-old', 1, 1, 0, 0, 'pending'
            )
        """
            )
        )

    await store.init_async_store()

    async with store._engine.begin() as conn:
        columns = (
            (await conn.execute(text("PRAGMA table_info(recordings)"))).mappings().all()
        )
        column_names = {str(row["name"]) for row in columns}
        assert "org_id" in column_names

        row = (
            (
                await conn.execute(
                    text("SELECT org_id FROM recordings WHERE recording_id = 'rec-old'")
                )
            )
            .mappings()
            .one()
        )

    assert row["org_id"] is None
    await store.close()


@pytest.mark.asyncio
async def test_set_recording_org_id_sets_only_when_missing(tmp_path) -> None:
    store = SqliteStateStore(tmp_path / "state.db")
    await store.init_async_store()

    await store.set_expected_trace_count("rec-1", 2)
    await store.set_recording_org_id("rec-1", "org-123")
    await store.set_recording_org_id("rec-1", "org-456")

    async with store._engine.begin() as conn:
        row = (
            (
                await conn.execute(
                    text("SELECT org_id FROM recordings WHERE recording_id = 'rec-1'")
                )
            )
            .mappings()
            .one()
        )

    assert row["org_id"] == "org-123"
    await store.close()
