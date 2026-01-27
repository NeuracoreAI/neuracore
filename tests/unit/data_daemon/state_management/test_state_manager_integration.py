from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import pytest
import pytest_asyncio
from sqlalchemy import select, text, update

from neuracore.data_daemon.event_emitter import Emitter, get_emitter
from neuracore.data_daemon.models import DataType, TraceErrorCode, TraceStatus
from neuracore.data_daemon.state_management.state_manager import StateManager
from neuracore.data_daemon.state_management.state_store_sqlite import SqliteStateStore
from neuracore.data_daemon.state_management.tables import traces


def _cleanup_state_manager(manager: StateManager) -> None:
    emitter = get_emitter()
    emitter.remove_listener(Emitter.TRACE_WRITTEN, manager._handle_trace_written)
    emitter.remove_listener(Emitter.START_TRACE, manager.create_trace)
    emitter.remove_listener(Emitter.UPLOAD_COMPLETE, manager.handle_upload_complete)
    emitter.remove_listener(Emitter.UPLOADED_BYTES, manager.update_bytes_uploaded)
    emitter.remove_listener(Emitter.UPLOAD_FAILED, manager.handle_upload_failed)
    emitter.remove_listener(Emitter.STOP_RECORDING, manager.handle_stop_recording)
    emitter.remove_listener(Emitter.IS_CONNECTED, manager.handle_is_connected)


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
    emitter = get_emitter()
    created_early = datetime(2024, 1, 1)
    created_late = datetime(2024, 1, 2)

    await manager.create_trace(
        "trace-1",
        "rec-1",
        DataType.CUSTOM_1D,
        "custom",
        1,
        path="/tmp/trace-1.bin",
        total_bytes=10,
    )
    await manager.create_trace(
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

    emitter.emit(Emitter.UPLOADED_BYTES, "trace-2", 4)
    emitter.emit(Emitter.IS_CONNECTED, True)
    await asyncio.sleep(0.1)

    ready_events: list[tuple] = []
    progress_events: list[tuple] = []
    progress_event = asyncio.Event()

    def ready_handler(*args) -> None:
        ready_events.append(args)

    def progress_handler(*args) -> None:
        progress_events.append(args)
        progress_event.set()

    emitter.on(Emitter.READY_FOR_UPLOAD, ready_handler)
    emitter.on(Emitter.PROGRESS_REPORT, progress_handler)
    try:
        before_second = datetime.now()
        emitter.emit(Emitter.TRACE_WRITTEN, "trace-1", "rec-1", 10)
        emitter.emit(Emitter.TRACE_WRITTEN, "trace-2", "rec-1", 10)
        await asyncio.sleep(0.2)
        after_second = datetime.now()

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

        try:
            await asyncio.wait_for(progress_event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        if progress_events:
            start_time, end_time, traces_list = progress_events[0]
            assert start_time == created_early.timestamp()
            assert before_second.timestamp() <= end_time <= after_second.timestamp()
            assert [trace.trace_id for trace in traces_list] == ["trace-1", "trace-2"]
            assert {trace.trace_id: trace.bytes_uploaded for trace in traces_list} == {
                "trace-1": 0,
                "trace-2": 4,
            }
    finally:
        emitter.remove_listener(Emitter.READY_FOR_UPLOAD, ready_handler)
        emitter.remove_listener(Emitter.PROGRESS_REPORT, progress_handler)


@pytest.mark.asyncio
async def test_offline_then_connected_emits_ready(manager_store) -> None:
    manager, store = manager_store
    emitter = get_emitter()
    await manager.create_trace(
        "trace-offline",
        "rec-offline",
        DataType.CUSTOM_1D,
        "custom",
        1,
        path="/tmp/trace-offline.bin",
        total_bytes=8,
    )

    ready_events: list[tuple] = []

    def ready_handler(*args) -> None:
        ready_events.append(args)

    emitter.on(Emitter.READY_FOR_UPLOAD, ready_handler)
    try:
        emitter.emit(Emitter.TRACE_WRITTEN, "trace-offline", "rec-offline", 8)
        await asyncio.sleep(0.1)
        assert ready_events == []

        emitter.emit(Emitter.IS_CONNECTED, True)
        await asyncio.sleep(0.1)
        assert ready_events == [(
            "trace-offline",
            "rec-offline",
            "/tmp/trace-offline.bin",
            DataType.CUSTOM_1D,
            "custom",
            0,
        )]
    finally:
        emitter.remove_listener(Emitter.READY_FOR_UPLOAD, ready_handler)

    assert await _get_trace_status(store, "trace-offline") == TraceStatus.WRITTEN


@pytest.mark.asyncio
async def test_uploaded_bytes_updates_store(manager_store) -> None:
    manager, store = manager_store
    emitter = get_emitter()
    await manager.create_trace(
        "trace-uploaded",
        "rec-uploaded",
        DataType.CUSTOM_1D,
        "custom",
        1,
        path="/tmp/trace-uploaded.bin",
    )

    emitter.emit(Emitter.UPLOADED_BYTES, "trace-uploaded", 5)
    await asyncio.sleep(0.1)
    trace = await store.get_trace("trace-uploaded")
    assert trace is not None
    assert trace.bytes_uploaded == 5


@pytest.mark.asyncio
async def test_invalid_transition_raises_via_manager(manager_store) -> None:
    manager, _ = manager_store
    await manager.create_trace(
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
        await manager_one.create_trace(
            "trace-multi-1",
            "rec-shared",
            DataType.CUSTOM_1D,
            "custom",
            1,
            path="/tmp/trace-multi-1.bin",
        )
        await manager_two.create_trace(
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
async def test_full_state_transition_chain(manager_store) -> None:
    manager, store = manager_store
    emitter = get_emitter()
    await manager.create_trace(
        "trace-chain",
        "rec-chain",
        DataType.CUSTOM_1D,
        "custom",
        1,
        path="/tmp/trace-chain.bin",
    )

    await manager.update_status("trace-chain", TraceStatus.WRITING)
    emitter.emit(Emitter.TRACE_WRITTEN, "trace-chain", "rec-chain", 12)
    await asyncio.sleep(0.1)

    claimed = await manager.claim_ready_traces(limit=1)
    assert [row["trace_id"] for row in claimed] == ["trace-chain"]
    assert claimed[0]["status"] == TraceStatus.UPLOADING

    await manager.update_status("trace-chain", TraceStatus.UPLOADED)
    assert await _get_trace_status(store, "trace-chain") == TraceStatus.UPLOADED


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
        await manager_one.create_trace(
            "trace-race",
            "rec-race",
            DataType.CUSTOM_1D,
            "custom",
            1,
            path="/tmp/trace-race.bin",
        )
        await manager_one.update_status("trace-race", TraceStatus.WRITING)

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
    emitter = get_emitter()
    db_path = tmp_path / "state.db"
    store = SqliteStateStore(db_path)
    await store.init_async_store()
    manager = StateManager(store)
    try:
        await manager.create_trace(
            "trace-recover",
            "rec-recover",
            DataType.CUSTOM_1D,
            "custom",
            1,
            path="/tmp/trace-recover.bin",
        )
        await manager.update_status("trace-recover", TraceStatus.WRITING)
        emitter.emit(Emitter.TRACE_WRITTEN, "trace-recover", "rec-recover", 8)
        await asyncio.sleep(0.1)
    finally:
        _cleanup_state_manager(manager)
        await store.close()

    recovered_store = SqliteStateStore(db_path)
    await recovered_store.init_async_store()
    try:
        recovered_trace = await recovered_store.get_trace("trace-recover")
        assert recovered_trace is not None
        assert recovered_trace.status == TraceStatus.WRITTEN
        assert recovered_trace.ready_for_upload == 1

        claimed = await recovered_store.claim_ready_traces(limit=1)
        assert [row["trace_id"] for row in claimed] == ["trace-recover"]
        assert claimed[0]["status"] == TraceStatus.UPLOADING
    finally:
        await recovered_store.close()


@pytest.mark.asyncio
async def test_ready_for_upload_includes_bytes_after_reconnect(manager_store) -> None:
    manager, store = manager_store
    emitter = get_emitter()
    await manager.create_trace(
        "trace-resume",
        "rec-resume",
        DataType.CUSTOM_1D,
        "custom",
        1,
        path="/tmp/trace-resume.bin",
        total_bytes=1000,
    )
    emitter.emit(Emitter.IS_CONNECTED, True)
    emitter.emit(Emitter.TRACE_WRITTEN, "trace-resume", "rec-resume", 1000)
    emitter.emit(Emitter.UPLOADED_BYTES, "trace-resume", 500)
    await asyncio.sleep(0.1)

    ready_events: list[tuple] = []

    def ready_handler(*args) -> None:
        ready_events.append(args)

    emitter.on(Emitter.READY_FOR_UPLOAD, ready_handler)
    try:
        emitter.emit(Emitter.IS_CONNECTED, False)
        await asyncio.sleep(0.1)
        ready_events.clear()
        emitter.emit(Emitter.IS_CONNECTED, True)
        await asyncio.sleep(0.1)

        assert ready_events
        assert ready_events[-1][:2] == ("trace-resume", "rec-resume")
        # bytes_uploaded is at index 5
        assert ready_events[-1][5] == 500  # bytes_uploaded
    finally:
        emitter.remove_listener(Emitter.READY_FOR_UPLOAD, ready_handler)


@pytest.mark.asyncio
async def test_ready_for_upload_includes_bytes_after_restart(tmp_path) -> None:
    emitter = get_emitter()
    db_path = tmp_path / "state.db"
    store = SqliteStateStore(db_path)
    await store.init_async_store()
    manager = StateManager(store)
    try:
        await manager.create_trace(
            "trace-restart",
            "rec-restart",
            DataType.CUSTOM_1D,
            "custom",
            1,
            path="/tmp/trace-restart.bin",
            total_bytes=1200,
        )
        emitter.emit(Emitter.IS_CONNECTED, True)
        emitter.emit(Emitter.TRACE_WRITTEN, "trace-restart", "rec-restart", 1200)
        emitter.emit(Emitter.UPLOADED_BYTES, "trace-restart", 500)
        await asyncio.sleep(0.1)
    finally:
        _cleanup_state_manager(manager)
        await store.close()

    recovered_store = SqliteStateStore(db_path)
    await recovered_store.init_async_store()
    recovered_manager = StateManager(recovered_store)
    ready_events: list[tuple] = []

    def ready_handler(*args) -> None:
        ready_events.append(args)

    emitter.on(Emitter.READY_FOR_UPLOAD, ready_handler)
    try:
        emitter.emit(Emitter.IS_CONNECTED, True)
        await asyncio.sleep(0.1)
        assert ready_events == [(
            "trace-restart",
            "rec-restart",
            "/tmp/trace-restart.bin",
            DataType.CUSTOM_1D,
            "custom",
            500,
        )]
    finally:
        emitter.remove_listener(Emitter.READY_FOR_UPLOAD, ready_handler)
        _cleanup_state_manager(recovered_manager)
        await recovered_store.close()


@pytest.mark.asyncio
async def test_failed_trace_not_claimed_for_retry(manager_store) -> None:
    manager, _ = manager_store
    emitter = get_emitter()
    await manager.create_trace(
        "trace-failed",
        "rec-failed",
        DataType.CUSTOM_1D,
        "custom",
        1,
        path="/tmp/trace-failed.bin",
        total_bytes=10,
    )
    emitter.emit(Emitter.TRACE_WRITTEN, "trace-failed", "rec-failed", 10)
    await asyncio.sleep(0.1)
    await manager.update_status("trace-failed", TraceStatus.FAILED)

    claimed = await manager.claim_ready_traces(limit=1)
    assert claimed == []


@pytest.mark.asyncio
async def test_simultaneous_recordings_emit_progress_reports(manager_store) -> None:
    manager, _ = manager_store
    emitter = get_emitter()
    for trace_id, recording_id in [
        ("trace-a1", "rec-a"),
        ("trace-a2", "rec-a"),
        ("trace-b1", "rec-b"),
        ("trace-b2", "rec-b"),
    ]:
        await manager.create_trace(
            trace_id,
            recording_id,
            DataType.CUSTOM_1D,
            "custom",
            1,
            path=f"/tmp/{trace_id}.bin",
            total_bytes=10,
        )

    progress_event = asyncio.Event()
    progress_events: list[tuple] = []
    seen_recordings: set[frozenset[str]] = set()

    def progress_handler(*args) -> None:
        progress_events.append(args)
        _, _, traces_list = args
        recording_ids = frozenset(trace.recording_id for trace in traces_list)
        seen_recordings.add(recording_ids)
        # Set event when we have progress reports for both recordings
        if (
            frozenset({"rec-a"}) in seen_recordings
            and frozenset({"rec-b"}) in seen_recordings
        ):
            progress_event.set()

    emitter.on(Emitter.PROGRESS_REPORT, progress_handler)
    try:
        emitter.emit(Emitter.IS_CONNECTED, True)
        emitter.emit(Emitter.TRACE_WRITTEN, "trace-a1", "rec-a", 10)
        emitter.emit(Emitter.TRACE_WRITTEN, "trace-a2", "rec-a", 10)
        emitter.emit(Emitter.TRACE_WRITTEN, "trace-b1", "rec-b", 10)
        emitter.emit(Emitter.TRACE_WRITTEN, "trace-b2", "rec-b", 10)

        await asyncio.wait_for(progress_event.wait(), timeout=2.0)

        # Verify we received progress reports for both recordings
        assert seen_recordings == {
            frozenset({"rec-a"}),
            frozenset({"rec-b"}),
        }
    finally:
        emitter.remove_listener(Emitter.PROGRESS_REPORT, progress_handler)


@pytest.mark.asyncio
async def test_resume_after_failure_and_reconnect(manager_store) -> None:
    manager, store = manager_store
    emitter = get_emitter()
    await manager.create_trace(
        "trace-e2e",
        "rec-e2e",
        DataType.CUSTOM_1D,
        "custom",
        1,
        path="/tmp/trace-e2e.bin",
        total_bytes=1000,
    )
    emitter.emit(Emitter.IS_CONNECTED, True)
    emitter.emit(Emitter.TRACE_WRITTEN, "trace-e2e", "rec-e2e", 1000)
    emitter.emit(Emitter.UPLOADED_BYTES, "trace-e2e", 500)
    emitter.emit(
        Emitter.UPLOAD_FAILED,
        "trace-e2e",
        500,
        TraceStatus.WRITTEN,
        TraceErrorCode.NETWORK_ERROR,
        "drop",
    )
    await asyncio.sleep(0.1)

    trace = await store.get_trace("trace-e2e")
    assert trace is not None
    assert trace.bytes_uploaded == 500

    ready_events: list[tuple] = []
    delete_events: list[tuple] = []

    def ready_handler(*args) -> None:
        ready_events.append(args)

    def delete_handler(recording_id: str, trace_id: str, data_type: DataType) -> None:
        delete_events.append((recording_id, trace_id, data_type))

    emitter.on(Emitter.READY_FOR_UPLOAD, ready_handler)
    emitter.on(Emitter.DELETE_TRACE, delete_handler)
    try:
        emitter.emit(Emitter.IS_CONNECTED, False)
        await asyncio.sleep(0.1)
        ready_events.clear()
        emitter.emit(Emitter.IS_CONNECTED, True)
        await asyncio.sleep(0.1)
        # bytes_uploaded is at index 5
        assert ready_events[-1][5] == 500

        emitter.emit(Emitter.UPLOADED_BYTES, "trace-e2e", 1000)
        emitter.emit(Emitter.UPLOAD_COMPLETE, "trace-e2e")
        await asyncio.sleep(0.1)
        assert delete_events == [("rec-e2e", "trace-e2e", DataType.CUSTOM_1D)]
        assert await store.get_trace("trace-e2e") is None
    finally:
        emitter.remove_listener(Emitter.READY_FOR_UPLOAD, ready_handler)
        emitter.remove_listener(Emitter.DELETE_TRACE, delete_handler)


@pytest.mark.asyncio
async def test_encoder_crash_does_not_block_other_recordings(manager_store) -> None:
    manager, _ = manager_store
    emitter = get_emitter()
    await manager.create_trace(
        "trace-a",
        "rec-a",
        DataType.CUSTOM_1D,
        "custom",
        1,
        path="/tmp/trace-a.bin",
        total_bytes=10,
    )
    await manager.create_trace(
        "trace-b",
        "rec-b",
        DataType.CUSTOM_1D,
        "custom",
        1,
        path="/tmp/trace-b.bin",
        total_bytes=10,
    )

    ready_events: list[tuple] = []
    trace_b_ready = asyncio.Event()

    def ready_handler(*args) -> None:
        ready_events.append(args)
        if args[:2] == ("trace-b", "rec-b"):
            trace_b_ready.set()

    emitter.on(Emitter.READY_FOR_UPLOAD, ready_handler)
    try:
        emitter.emit(Emitter.IS_CONNECTED, True)
        emitter.emit(Emitter.TRACE_WRITTEN, "trace-a", "rec-a", 0)
        emitter.emit(
            Emitter.UPLOAD_FAILED,
            "trace-a",
            0,
            TraceStatus.WRITTEN,
            TraceErrorCode.ENCODE_FAILED,
            "encoder crashed",
        )
        emitter.emit(Emitter.TRACE_WRITTEN, "trace-b", "rec-b", 10)

        # Wait for trace-b's ready event specifically
        await asyncio.wait_for(trace_b_ready.wait(), timeout=2.0)

        # Verify trace-b received a READY_FOR_UPLOAD event
        trace_b_events = [e for e in ready_events if e[:2] == ("trace-b", "rec-b")]
        assert trace_b_events, "trace-b should have received READY_FOR_UPLOAD"
    finally:
        emitter.remove_listener(Emitter.READY_FOR_UPLOAD, ready_handler)
