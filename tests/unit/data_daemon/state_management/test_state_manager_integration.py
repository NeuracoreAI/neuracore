from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import select, text, update

from neuracore.data_daemon.event_emitter import Emitter, get_emitter
from neuracore.data_daemon.models import DataType, TraceErrorCode, TraceStatus
from neuracore.data_daemon.state_management.state_manager import StateManager
from neuracore.data_daemon.state_management.state_store_sqlite import SqliteStateStore
from neuracore.data_daemon.state_management.tables import traces

DEFAULT_TIMEOUT = 2.0


class EventCollector:
    """Collects events and signals when expected count is reached."""

    def __init__(self, expected_count: int = 1) -> None:
        self.events: list[tuple] = []
        self.expected_count = expected_count
        self._done = asyncio.Event()

    def handler(self, *args: Any) -> None:
        self.events.append(args)
        if len(self.events) >= self.expected_count:
            self._done.set()

    async def wait(self, timeout: float = DEFAULT_TIMEOUT) -> list[tuple]:
        await asyncio.wait_for(self._done.wait(), timeout=timeout)
        return self.events


@asynccontextmanager
async def listen_for(event: str, expected_count: int = 1):
    """Context manager that listens for events and cleans up."""
    collector = EventCollector(expected_count)
    emitter = get_emitter()
    emitter.on(event, collector.handler)
    try:
        yield collector
    finally:
        emitter.remove_listener(event, collector.handler)


@asynccontextmanager
async def listen_for_multiple(events: dict[str, int]):
    """Context manager that listens for multiple event types."""
    collectors: dict[str, EventCollector] = {}
    emitter = get_emitter()
    for event, expected_count in events.items():
        collector = EventCollector(expected_count)
        collectors[event] = collector
        emitter.on(event, collector.handler)
    try:
        yield collectors
    finally:
        for event, collector in collectors.items():
            emitter.remove_listener(event, collector.handler)


def _cleanup_state_manager(manager: StateManager) -> None:
    emitter = get_emitter()
    emitter.remove_listener(Emitter.TRACE_WRITTEN, manager._handle_trace_written)
    emitter.remove_listener(Emitter.START_TRACE, manager._handle_start_trace)
    emitter.remove_listener(Emitter.UPLOAD_COMPLETE, manager._handle_upload_complete)
    emitter.remove_listener(Emitter.UPLOADED_BYTES, manager._handle_uploaded_bytes)
    emitter.remove_listener(Emitter.UPLOAD_FAILED, manager._handle_upload_failed)
    emitter.remove_listener(Emitter.STOP_RECORDING, manager._handle_stop_recording)
    emitter.remove_listener(Emitter.IS_CONNECTED, manager._handle_is_connected)
    emitter.remove_listener(
        Emitter.PROGRESS_REPORTED, manager._handle_progress_reported
    )
    emitter.remove_listener(
        Emitter.PROGRESS_REPORT_FAILED, manager._handle_progress_report_failed
    )


@pytest_asyncio.fixture
async def manager_store(tmp_path) -> tuple[StateManager, SqliteStateStore]:
    store = SqliteStateStore(tmp_path / "state.db")
    await store.init_async_store()
    manager = StateManager(store)
    try:
        yield manager, store
    finally:
        _cleanup_state_manager(manager)
        await store.close()


async def _set_created_at(
    store: SqliteStateStore, trace_id: str, created_at: datetime
) -> None:
    async with store._engine.begin() as conn:
        await conn.execute(
            update(traces)
            .where(traces.c.trace_id == trace_id)
            .values(created_at=created_at)
        )


async def _get_trace_status(store: SqliteStateStore, trace_id: str) -> TraceStatus:
    async with store._engine.begin() as conn:
        return (
            await conn.execute(
                select(traces.c.status).where(traces.c.trace_id == trace_id)
            )
        ).scalar_one()


@pytest.mark.asyncio
async def test_trace_written_emits_ready_and_progress_report(manager_store) -> None:
    manager, store = manager_store
    created_early = datetime(2024, 1, 1)
    created_late = datetime(2024, 1, 2)

    await manager._handle_start_trace(
        "trace-1",
        "rec-1",
        DataType.CUSTOM_1D,
        "custom",
        1,
        path="/tmp/trace-1.bin",
        total_bytes=10,
    )
    await manager._handle_start_trace(
        "trace-2",
        "rec-1",
        DataType.CUSTOM_1D,
        "custom",
        1,
        path="/tmp/trace-2.bin",
        total_bytes=10,
    )
    await _set_created_at(store, "trace-1", created_early)
    await _set_created_at(store, "trace-2", created_late)

    await manager._handle_uploaded_bytes("trace-2", 4)
    await manager._handle_is_connected(True)

    async with listen_for_multiple({
        Emitter.READY_FOR_UPLOAD: 2,
        Emitter.PROGRESS_REPORT: 1,
    }) as collectors:
        before_second = datetime.now()
        await manager._handle_trace_written("trace-1", "rec-1", 10)
        await manager._handle_trace_written("trace-2", "rec-1", 10)
        after_second = datetime.now()

        ready_events = await collectors[Emitter.READY_FOR_UPLOAD].wait()
        assert len(ready_events) == 2
        ready_by_id = {event[0]: event for event in ready_events}
        assert ready_by_id["trace-1"] == (
            "trace-1",
            "rec-1",
            "/tmp/trace-1.bin",
            DataType.CUSTOM_1D,
            "custom",
            0,
        )
        assert ready_by_id["trace-2"] == (
            "trace-2",
            "rec-1",
            "/tmp/trace-2.bin",
            DataType.CUSTOM_1D,
            "custom",
            4,
        )

        progress_events = await collectors[Emitter.PROGRESS_REPORT].wait()
        assert len(progress_events) == 1
        start_time, end_time, traces_list = progress_events[0]
        assert start_time == created_early.timestamp()
        assert before_second.timestamp() <= end_time <= after_second.timestamp()
        assert [trace.trace_id for trace in traces_list] == ["trace-1", "trace-2"]
        assert {trace.trace_id: trace.bytes_uploaded for trace in traces_list} == {
            "trace-1": 0,
            "trace-2": 4,
        }


@pytest.mark.asyncio
async def test_uploaded_bytes_updates_store(manager_store) -> None:
    manager, store = manager_store
    await manager._handle_start_trace(
        "trace-uploaded",
        "rec-uploaded",
        DataType.CUSTOM_1D,
        "custom",
        1,
        path="/tmp/trace-uploaded.bin",
    )

    await manager._handle_uploaded_bytes("trace-uploaded", 5)

    trace = await store.get_trace("trace-uploaded")
    assert trace is not None
    assert trace.bytes_uploaded == 5


@pytest.mark.asyncio
async def test_invalid_transition_raises_via_manager(manager_store) -> None:
    manager, _ = manager_store
    await manager._handle_start_trace(
        "trace-invalid",
        "rec-invalid",
        DataType.CUSTOM_1D,
        "custom",
        1,
        path="/tmp/trace-invalid.bin",
    )

    with pytest.raises(ValueError, match="Invalid status transition"):
        await manager.update_status("trace-invalid", TraceStatus.UPLOADED)


@pytest.mark.asyncio
async def test_multiple_state_managers_share_sqlite_db(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    store_one = SqliteStateStore(db_path)
    store_two = SqliteStateStore(db_path)
    await store_one.init_async_store()
    await store_two.init_async_store()
    manager_one = StateManager(store_one)
    manager_two = StateManager(store_two)

    try:
        await manager_one._handle_start_trace(
            "trace-multi-1",
            "rec-shared",
            DataType.CUSTOM_1D,
            "custom",
            1,
            path="/tmp/trace-multi-1.bin",
        )
        await manager_two._handle_start_trace(
            "trace-multi-2",
            "rec-shared",
            DataType.CUSTOM_1D,
            "custom",
            1,
            path="/tmp/trace-multi-2.bin",
        )

        assert await store_one.get_trace("trace-multi-1") is not None
        assert await store_one.get_trace("trace-multi-2") is not None
    finally:
        _cleanup_state_manager(manager_one)
        _cleanup_state_manager(manager_two)
        await store_one.close()
        await store_two.close()


@pytest.mark.asyncio
async def test_concurrent_writes_with_wal(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    store_one = SqliteStateStore(db_path)
    store_two = SqliteStateStore(db_path)
    await store_one.init_async_store()
    await store_two.init_async_store()
    try:
        await store_one.create_trace(
            trace_id="trace-concurrent",
            recording_id="rec-concurrent",
            data_type=DataType.CUSTOM_1D,
            path="/tmp/trace-concurrent.bin",
            data_type_name="custom",
            robot_instance=1,
        )
        async with store_one._engine.begin() as conn:
            mode = (await conn.execute(text("PRAGMA journal_mode;"))).scalar_one()
        assert str(mode).lower() == "wal"

        errors: list[Exception] = []

        async def worker(store: SqliteStateStore, bytes_uploaded: int) -> None:
            try:
                for _ in range(5):
                    await store.update_bytes_uploaded(
                        "trace-concurrent", bytes_uploaded
                    )
            except Exception as exc:
                errors.append(exc)

        await asyncio.gather(
            worker(store_one, 10),
            worker(store_two, 20),
        )

        assert errors == []
        trace = await store_one.get_trace("trace-concurrent")
        assert trace is not None
        assert trace.bytes_uploaded in {10, 20}
    finally:
        await store_one.close()
        await store_two.close()


@pytest.mark.asyncio
async def test_race_conditions_on_rapid_state_changes(tmp_path, caplog) -> None:
    """Test race conditions on rapid state changes.

    Two tasks concurrently update the state of the same trace.
    The first task will update the state to WRITTEN, then UPLOADING, then UPLOADED.
    The second task will update the state to WRITTEN, then UPLOADING, then UPLOADED.
    The test asserts that the final state of the trace is UPLOADED
    and that a ValueError is raised when updating the state.
    """

    db_path = tmp_path / "state.db"
    store_one = SqliteStateStore(db_path)
    store_two = SqliteStateStore(db_path)
    await store_one.init_async_store()
    await store_two.init_async_store()
    manager_one = StateManager(store_one)
    manager_two = StateManager(store_two)
    try:
        await manager_one._handle_start_trace(
            "trace-race",
            "rec-race",
            DataType.CUSTOM_1D,
            "custom",
            1,
            path="/tmp/trace-race.bin",
        )

        errors: list[str] = []

        async def worker(manager: StateManager) -> None:
            try:
                await manager.update_status("trace-race", TraceStatus.WRITTEN)
                await manager.update_status("trace-race", TraceStatus.UPLOADING)
                await manager.update_status("trace-race", TraceStatus.UPLOADED)
            except ValueError as exc:
                errors.append(str(exc))

        with caplog.at_level(logging.INFO):
            await asyncio.gather(
                worker(manager_one),
                worker(manager_two),
            )

        assert (
            errors or "Failed to update trace status: Trace trace-race" in caplog.text
        )
        assert await _get_trace_status(store_one, "trace-race") == TraceStatus.UPLOADED
    finally:
        _cleanup_state_manager(manager_one)
        _cleanup_state_manager(manager_two)
        await store_one.close()
        await store_two.close()


@pytest.mark.asyncio
async def test_state_recovery_after_restart(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    store = SqliteStateStore(db_path)
    await store.init_async_store()
    manager = StateManager(store)
    try:
        await manager._handle_start_trace(
            "trace-recover",
            "rec-recover",
            DataType.CUSTOM_1D,
            "custom",
            1,
            path="/tmp/trace-recover.bin",
        )
        await manager._handle_trace_written("trace-recover", "rec-recover", 8)
    finally:
        _cleanup_state_manager(manager)
        await store.close()

    recovered_store = SqliteStateStore(db_path)
    await recovered_store.init_async_store()
    try:
        recovered_trace = await recovered_store.get_trace("trace-recover")
        assert recovered_trace is not None
        assert recovered_trace.status == TraceStatus.UPLOADING

        ready = await recovered_store.find_ready_traces()
        assert ready == []

        await recovered_store.update_status("trace-recover", TraceStatus.UPLOADED)
        updated = await recovered_store.get_trace("trace-recover")
        assert updated is not None
        assert updated.status == TraceStatus.UPLOADED
    finally:
        await recovered_store.close()


@pytest.mark.asyncio
async def test_simultaneous_recordings_emit_progress_reports(manager_store) -> None:
    manager, _ = manager_store
    for trace_id, recording_id in [
        ("trace-a1", "rec-a"),
        ("trace-a2", "rec-a"),
        ("trace-b1", "rec-b"),
        ("trace-b2", "rec-b"),
    ]:
        await manager._handle_start_trace(
            trace_id,
            recording_id,
            DataType.CUSTOM_1D,
            "custom",
            1,
            path=f"/tmp/{trace_id}.bin",
            total_bytes=10,
        )

    async with listen_for(Emitter.PROGRESS_REPORT, expected_count=2) as collector:
        await manager._handle_is_connected(True)
        await manager._handle_trace_written("trace-a1", "rec-a", 10)
        await manager._handle_trace_written("trace-a2", "rec-a", 10)
        await manager._handle_trace_written("trace-b1", "rec-b", 10)
        await manager._handle_trace_written("trace-b2", "rec-b", 10)

        events = await collector.wait()

        seen_recordings = set()
        for _, _, traces_list in events:
            recording_ids = frozenset(trace.recording_id for trace in traces_list)
            seen_recordings.add(recording_ids)

        assert seen_recordings == {
            frozenset({"rec-a"}),
            frozenset({"rec-b"}),
        }


@pytest.mark.asyncio
async def test_encoder_crash_does_not_block_other_recordings(manager_store) -> None:
    manager, _ = manager_store
    await manager._handle_start_trace(
        "trace-a",
        "rec-a",
        DataType.CUSTOM_1D,
        "custom",
        1,
        path="/tmp/trace-a.bin",
        total_bytes=10,
    )
    await manager._handle_start_trace(
        "trace-b",
        "rec-b",
        DataType.CUSTOM_1D,
        "custom",
        1,
        path="/tmp/trace-b.bin",
        total_bytes=10,
    )

    async with listen_for(Emitter.READY_FOR_UPLOAD, expected_count=2) as collector:
        await manager._handle_is_connected(True)
        await manager._handle_trace_written("trace-a", "rec-a", 0)
        await manager._handle_upload_failed(
            "trace-a",
            0,
            TraceStatus.WRITTEN,
            TraceErrorCode.ENCODE_FAILED,
            "encoder crashed",
        )
        await manager._handle_trace_written("trace-b", "rec-b", 10)

        events = await collector.wait()

        trace_b_events = [e for e in events if e[:2] == ("trace-b", "rec-b")]
        assert trace_b_events, "trace-b should have received READY_FOR_UPLOAD"


@pytest.mark.asyncio
async def test_status_is_uploading_during_active_upload(manager_store) -> None:
    """Verify trace status is UPLOADING between TRACE_WRITTEN and UPLOAD_COMPLETE.

    The Story:
    A single trace completes writing. The state machine should transition
    the trace to UPLOADING status before emitting READY_FOR_UPLOAD, and
    the trace should remain in UPLOADING status until upload completes.

    The Flow:
    1. Create a trace in PENDING state
    2. Emit IS_CONNECTED to enable online mode
    3. Emit TRACE_WRITTEN to signal writing is complete
    4. Capture the trace status from DB before UPLOAD_COMPLETE
    5. Verify status is UPLOADING (not WRITTEN)

    Why This Matters:
    Without proper UPLOADING transition, the same trace could be picked up
    by multiple uploaders causing duplicate uploads and wasted bandwidth.

    Key Assertions:
    - Status is UPLOADING after TRACE_WRITTEN event is processed
    - READY_FOR_UPLOAD event is emitted with correct trace data
    """
    manager, store = manager_store

    await manager._handle_start_trace(
        "trace-upload-status",
        "rec-upload-status",
        DataType.CUSTOM_1D,
        "custom",
        1,
        path="/tmp/trace-upload-status.bin",
        total_bytes=64,
    )

    async with listen_for(Emitter.READY_FOR_UPLOAD) as collector:
        await manager._handle_is_connected(True)
        await manager._handle_trace_written(
            "trace-upload-status", "rec-upload-status", 64
        )

        events = await collector.wait()

        status = await _get_trace_status(store, "trace-upload-status")
        assert status == TraceStatus.UPLOADING, f"Expected UPLOADING, got {status}"

        assert len(events) == 1
        assert events[0] == (
            "trace-upload-status",
            "rec-upload-status",
            "/tmp/trace-upload-status.bin",
            DataType.CUSTOM_1D,
            "custom",
            0,
        )


@pytest.mark.asyncio
async def test_two_traces_same_recording_sequential_completion(manager_store) -> None:
    """Verify two traces in the same recording transition independently.

    The Story:
    A recording produces two traces (e.g., video and sensor data). Both
    complete writing at nearly the same time. Each trace should transition
    to UPLOADING independently without blocking the other.

    The Flow:
    1. Create two traces for the same recording
    2. Emit IS_CONNECTED to enable online mode
    3. Emit TRACE_WRITTEN for both traces in sequence
    4. Verify both traces are in UPLOADING status
    5. Verify READY_FOR_UPLOAD emitted for both traces

    Why This Matters:
    Multi-stream recordings are common. Each trace must be uploadable
    independently to maximize upload throughput and minimize latency.

    Key Assertions:
    - Both traces reach UPLOADING status
    - READY_FOR_UPLOAD emitted for each trace
    - Neither trace blocks the other's state transition
    """
    manager, store = manager_store

    await manager._handle_start_trace(
        "trace-seq-1",
        "rec-seq",
        DataType.CUSTOM_1D,
        "custom",
        1,
        path="/tmp/trace-seq-1.bin",
        total_bytes=32,
    )
    await manager._handle_start_trace(
        "trace-seq-2",
        "rec-seq",
        DataType.CUSTOM_1D,
        "custom",
        1,
        path="/tmp/trace-seq-2.bin",
        total_bytes=48,
    )

    async with listen_for(Emitter.READY_FOR_UPLOAD, expected_count=2) as collector:
        await manager._handle_is_connected(True)
        await manager._handle_trace_written("trace-seq-1", "rec-seq", 32)
        await manager._handle_trace_written("trace-seq-2", "rec-seq", 48)

        events = await collector.wait()

        status_1 = await _get_trace_status(store, "trace-seq-1")
        status_2 = await _get_trace_status(store, "trace-seq-2")
        assert (
            status_1 == TraceStatus.UPLOADING
        ), f"trace-seq-1: expected UPLOADING, got {status_1}"
        assert (
            status_2 == TraceStatus.UPLOADING
        ), f"trace-seq-2: expected UPLOADING, got {status_2}"

        emitted_trace_ids = {event[0] for event in events}
        assert "trace-seq-1" in emitted_trace_ids
        assert "trace-seq-2" in emitted_trace_ids


@pytest.mark.asyncio
async def test_two_traces_staggered_completion(manager_store) -> None:
    """Verify trace-A can upload while trace-B is still writing.

    The Story:
    A recording has two traces. Trace-A finishes writing and starts uploading.
    While trace-A uploads, trace-B is still being written. Trace-B's ongoing
    write should not interfere with trace-A's upload, and vice versa.

    The Flow:
    1. Create two traces for the same recording
    2. Emit IS_CONNECTED to enable online mode
    3. Emit TRACE_WRITTEN for trace-A only
    4. Verify trace-A transitions to UPLOADING and READY_FOR_UPLOAD emits
    5. Verify trace-B remains in PENDING status (still writing)
    6. Emit TRACE_WRITTEN for trace-B
    7. Verify trace-B transitions to UPLOADING independently

    Why This Matters:
    Different data types have different write durations. A 10-second video
    trace should not wait for a 60-second sensor trace to finish writing.

    Key Assertions:
    - Trace-A reaches UPLOADING while trace-B is PENDING
    - Trace-B's eventual completion triggers its own UPLOADING transition
    - No cross-contamination between trace states
    """
    manager, store = manager_store

    await manager._handle_start_trace(
        "trace-stag-a",
        "rec-stag",
        DataType.CUSTOM_1D,
        "custom",
        1,
        path="/tmp/trace-stag-a.bin",
        total_bytes=32,
    )
    await manager._handle_start_trace(
        "trace-stag-b",
        "rec-stag",
        DataType.CUSTOM_1D,
        "custom",
        1,
        path="/tmp/trace-stag-b.bin",
        total_bytes=64,
    )

    async with listen_for(Emitter.READY_FOR_UPLOAD, expected_count=2) as collector:
        await manager._handle_is_connected(True)
        await manager._handle_trace_written("trace-stag-a", "rec-stag", 32)

        status_a = await _get_trace_status(store, "trace-stag-a")
        status_b = await _get_trace_status(store, "trace-stag-b")
        assert (
            status_a == TraceStatus.UPLOADING
        ), f"trace-stag-a: expected UPLOADING, got {status_a}"
        assert (
            status_b == TraceStatus.PENDING
        ), f"trace-stag-b: expected PENDING, got {status_b}"

        await manager._handle_trace_written("trace-stag-b", "rec-stag", 64)

        events = await collector.wait()

        status_b_after = await _get_trace_status(store, "trace-stag-b")
        assert (
            status_b_after == TraceStatus.UPLOADING
        ), f"trace-stag-b: expected UPLOADING, got {status_b_after}"

        status_a_after = await _get_trace_status(store, "trace-stag-a")
        assert (
            status_a_after == TraceStatus.UPLOADING
        ), f"trace-stag-a: expected UPLOADING, got {status_a_after}"

        emitted_trace_ids = [event[0] for event in events]
        assert emitted_trace_ids.count("trace-stag-a") == 1
        assert emitted_trace_ids.count("trace-stag-b") == 1
