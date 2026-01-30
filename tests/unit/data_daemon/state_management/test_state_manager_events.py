from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from neuracore.data_daemon.event_emitter import Emitter, get_emitter
from neuracore.data_daemon.models import (
    DataType,
    TraceErrorCode,
    TraceRecord,
    TraceStatus,
)
from neuracore.data_daemon.state_management.state_manager import StateManager


class FakeStateStore:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.stopped: list[str] = []
        self.updated_bytes: list[tuple[str, int]] = []
        self.deleted: list[str] = []
        self.errors: list[tuple[str, str, TraceErrorCode | None, TraceStatus]] = []
        self.marked_written: list[tuple[str, int]] = []
        self.ready_traces: list[TraceRecord] = []
        self.unreported_traces: list[TraceRecord] = []
        self._traces_by_id: dict[str, TraceRecord] = {}
        self._traces_by_recording: dict[str, list[TraceRecord]] = {}

    async def set_stopped_ats(self, recording_id: str) -> None:
        self.stopped.append(recording_id)

    async def create_trace(self, *args, **kwargs) -> None:
        self.created.append(dict(kwargs))

    async def get_trace(self, trace_id: str) -> TraceRecord | None:
        return self._traces_by_id.get(trace_id)

    async def find_traces_by_recording_id(self, recording_id: str) -> list[TraceRecord]:
        return list(self._traces_by_recording.get(recording_id, []))

    def list_traces(self) -> list[TraceRecord]:
        return list(self._traces_by_id.values())

    async def update_bytes_uploaded(self, trace_id: str, bytes_uploaded: int) -> None:
        self.updated_bytes.append((trace_id, bytes_uploaded))

    async def mark_trace_as_written(self, trace_id: str, bytes_written: int) -> None:
        self.marked_written.append((trace_id, bytes_written))
        trace = self._traces_by_id.get(trace_id)
        if trace:
            updated = replace(
                trace,
                status=TraceStatus.WRITTEN,
                bytes_written=bytes_written,
                total_bytes=bytes_written,
            )
            self._traces_by_id[trace_id] = updated
            traces = self._traces_by_recording.get(trace.recording_id, [])
            self._traces_by_recording[trace.recording_id] = [
                updated if t.trace_id == trace_id else t for t in traces
            ]

    async def find_ready_traces(self) -> list[TraceRecord]:
        return list(self.ready_traces)

    async def find_unreported_traces(self) -> list[TraceRecord]:
        return list(self.unreported_traces)

    async def mark_recording_reported(self, recording_id: str) -> None:
        return None

    async def update_status(
        self, trace_id: str, status: TraceStatus, *, error_message=None
    ) -> None:
        return None

    async def record_error(
        self,
        trace_id: str,
        error_message: str,
        error_code: TraceErrorCode | None = None,
        status: TraceStatus = TraceStatus.FAILED,
    ) -> None:
        self.errors.append((trace_id, error_message, error_code, status))

    async def delete_trace(self, trace_id: str) -> None:
        self.deleted.append(trace_id)


def _cleanup_state_manager(manager: StateManager) -> None:
    get_emitter().remove_listener(Emitter.TRACE_WRITTEN, manager._handle_trace_written)
    get_emitter().remove_listener(Emitter.START_TRACE, manager.create_trace)
    get_emitter().remove_listener(
        Emitter.UPLOAD_COMPLETE, manager.handle_upload_complete
    )
    get_emitter().remove_listener(Emitter.UPLOADED_BYTES, manager.update_bytes_uploaded)
    get_emitter().remove_listener(Emitter.UPLOAD_FAILED, manager.handle_upload_failed)
    get_emitter().remove_listener(Emitter.STOP_RECORDING, manager.handle_stop_recording)
    get_emitter().remove_listener(Emitter.IS_CONNECTED, manager.handle_is_connected)


@pytest_asyncio.fixture
async def state_manager() -> tuple[StateManager, FakeStateStore]:
    store = FakeStateStore()
    manager = StateManager(store)
    try:
        yield manager, store
    finally:
        _cleanup_state_manager(manager)


def _make_trace(
    trace_id: str,
    recording_id: str,
    *,
    status: TraceStatus = TraceStatus.PENDING,
    progress_reported: int = 0,
    bytes_written: int = 0,
    total_bytes: int | None = None,
    bytes_uploaded: int = 0,
    created_at: datetime,
    last_updated: datetime,
) -> TraceRecord:
    return TraceRecord(
        trace_id=trace_id,
        status=status,
        recording_id=recording_id,
        data_type=DataType.CUSTOM_1D,
        data_type_name="custom",
        dataset_id=None,
        dataset_name=None,
        robot_name=None,
        robot_id=None,
        robot_instance=0,
        path=f"/tmp/{trace_id}.bin",
        bytes_written=bytes_written,
        total_bytes=total_bytes,
        bytes_uploaded=bytes_uploaded,
        progress_reported=progress_reported,
        error_code=None,
        error_message=None,
        created_at=created_at,
        last_updated=last_updated,
    )


@pytest.mark.asyncio
async def test_stop_recording_emits_stop_all_and_sets_stopped(state_manager) -> None:
    manager, store = state_manager
    received: list[str] = []

    def handler(recording_id: str) -> None:
        received.append(recording_id)

    get_emitter().on(Emitter.STOP_ALL_TRACES_FOR_RECORDING, handler)
    try:
        get_emitter().emit(Emitter.STOP_RECORDING, "rec-1")
        await asyncio.sleep(0.2)
        assert store.stopped == ["rec-1"]
        assert received == ["rec-1"]
    finally:
        get_emitter().remove_listener(Emitter.STOP_ALL_TRACES_FOR_RECORDING, handler)


@pytest.mark.asyncio
async def test_start_trace_creates_trace(state_manager) -> None:
    _, store = state_manager
    get_emitter().emit(
        Emitter.START_TRACE,
        "trace-1",
        "rec-1",
        DataType.CUSTOM_1D,
        "custom",
        None,
        None,
        None,
        None,
        path="/tmp/trace-1.bin",
        total_bytes=128,
    )
    await asyncio.sleep(0.2)

    assert len(store.created) == 1
    payload = store.created[0]
    assert payload["trace_id"] == "trace-1"
    assert payload["recording_id"] == "rec-1"
    assert payload["data_type"] == DataType.CUSTOM_1D
    assert payload["data_type_name"] == "custom"
    assert payload["path"] == "/tmp/trace-1.bin"
    assert payload["total_bytes"] == 128


@pytest.mark.asyncio
async def test_uploaded_bytes_updates_store(state_manager) -> None:
    _, store = state_manager
    get_emitter().emit(Emitter.UPLOADED_BYTES, "trace-2", 42)
    await asyncio.sleep(0.2)

    assert store.updated_bytes == [("trace-2", 42)]


@pytest.mark.asyncio
async def test_upload_complete_emits_delete_and_deletes(state_manager) -> None:
    _, store = state_manager
    received: list[tuple[str, str, DataType]] = []

    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trace = _make_trace(
        "trace-3",
        "rec-3",
        created_at=created_at,
        last_updated=created_at,
    )
    store._traces_by_id["trace-3"] = trace

    def handler(recording_id: str, trace_id: str, data_type: DataType) -> None:
        received.append((recording_id, trace_id, data_type))

    get_emitter().on(Emitter.DELETE_TRACE, handler)
    try:
        get_emitter().emit(Emitter.UPLOAD_COMPLETE, "trace-3")
        await asyncio.sleep(0.2)

        assert store.deleted == ["trace-3"]
        assert received == [("rec-3", "trace-3", DataType.CUSTOM_1D)]
    finally:
        get_emitter().remove_listener(Emitter.DELETE_TRACE, handler)


@pytest.mark.asyncio
async def test_upload_failed_records_error(state_manager) -> None:
    _, store = state_manager
    get_emitter().emit(
        Emitter.UPLOAD_FAILED,
        "trace-4",
        12,
        TraceStatus.FAILED,
        TraceErrorCode.NETWORK_ERROR,
        "lost connection",
    )
    await asyncio.sleep(0.2)

    assert store.errors == [(
        "trace-4",
        "lost connection",
        TraceErrorCode.NETWORK_ERROR,
        TraceStatus.FAILED,
    )]


@pytest.mark.asyncio
async def test_trace_written_emits_ready_for_upload_when_connected(
    state_manager,
) -> None:
    manager, store = state_manager
    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    last_updated = datetime(2024, 1, 2, tzinfo=timezone.utc)
    trace = _make_trace(
        "trace-5",
        "rec-5",
        created_at=created_at,
        last_updated=last_updated,
    )
    store._traces_by_id["trace-5"] = trace
    store._traces_by_recording["rec-5"] = [trace]
    received: list[tuple] = []

    def handler(*args) -> None:
        received.append(args)

    get_emitter().on(Emitter.READY_FOR_UPLOAD, handler)
    try:
        get_emitter().emit(Emitter.IS_CONNECTED, True)
        await asyncio.sleep(0.2)

        get_emitter().emit(Emitter.TRACE_WRITTEN, "trace-5", "rec-5", 64)
        await asyncio.sleep(0.2)

        assert store.marked_written == [("trace-5", 64)]
        assert received == [(
            "trace-5",
            "rec-5",
            "/tmp/trace-5.bin",
            DataType.CUSTOM_1D,
            "custom",
            0,
        )]
    finally:
        get_emitter().remove_listener(Emitter.READY_FOR_UPLOAD, handler)


@pytest.mark.asyncio
async def test_trace_written_emits_progress_report_with_bounds(state_manager) -> None:
    """Test that TRACE_WRITTEN emits progress report with correct time bounds."""
    manager, store = state_manager
    created_early = datetime(2024, 1, 1, tzinfo=timezone.utc)
    created_late = datetime(2024, 1, 2, tzinfo=timezone.utc)
    updated_early = datetime(2024, 1, 3, tzinfo=timezone.utc)
    updated_late = datetime(2024, 1, 4, tzinfo=timezone.utc)

    trace_pending = _make_trace(
        "trace-1",
        "rec-1",
        status=TraceStatus.WRITING,
        created_at=created_early,
        last_updated=updated_early,
    )
    trace_written = _make_trace(
        "trace-2",
        "rec-1",
        status=TraceStatus.WRITTEN,
        bytes_written=10,
        total_bytes=10,
        created_at=created_late,
        last_updated=updated_late,
    )
    store._traces_by_id["trace-1"] = trace_pending
    store._traces_by_id["trace-2"] = trace_written
    store._traces_by_recording["rec-1"] = [trace_pending, trace_written]

    get_emitter().emit(Emitter.IS_CONNECTED, True)
    await asyncio.sleep(0.1)

    ready_events: list[tuple] = []
    progress_events: list[tuple] = []

    def ready_handler(*args) -> None:
        ready_events.append(args)

    def progress_handler(*args) -> None:
        progress_events.append(args)

    get_emitter().on(Emitter.READY_FOR_UPLOAD, ready_handler)
    get_emitter().on(Emitter.PROGRESS_REPORT, progress_handler)
    try:
        get_emitter().emit(Emitter.TRACE_WRITTEN, "trace-1", "rec-1", 10)
        await asyncio.sleep(0.3)

        assert ready_events == [(
            "trace-1",
            "rec-1",
            "/tmp/trace-1.bin",
            DataType.CUSTOM_1D,
            "custom",
            0,
        )]
        assert len(progress_events) == 1
        start_time, end_time, traces = progress_events[0]
        assert start_time == created_early.timestamp()
        assert end_time == updated_late.timestamp()
        assert {trace.trace_id for trace in traces} == {"trace-1", "trace-2"}

        get_emitter().emit(Emitter.UPLOADED_BYTES, "trace-1", 5)
        await asyncio.sleep(0.1)
        assert store.updated_bytes == [("trace-1", 5)]
    finally:
        get_emitter().remove_listener(Emitter.READY_FOR_UPLOAD, ready_handler)
        get_emitter().remove_listener(Emitter.PROGRESS_REPORT, progress_handler)


@pytest.mark.asyncio
async def test_trace_written_waits_for_all_traces_before_progress_report(
    state_manager,
) -> None:
    """Test that progress report waits for all traces before emitting."""
    _, store = state_manager
    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    updated_at = datetime(2024, 1, 2, tzinfo=timezone.utc)
    trace_written = _make_trace(
        "trace-written",
        "rec-1",
        status=TraceStatus.WRITTEN,
        bytes_written=8,
        total_bytes=8,
        created_at=created_at,
        last_updated=updated_at,
    )
    trace_writing = _make_trace(
        "trace-writing",
        "rec-1",
        status=TraceStatus.WRITING,
        bytes_written=4,
        total_bytes=8,
        created_at=created_at,
        last_updated=updated_at,
    )
    store._traces_by_id["trace-written"] = trace_written
    store._traces_by_id["trace-writing"] = trace_writing
    store._traces_by_recording["rec-1"] = [trace_written, trace_writing]

    progress_events: list[tuple] = []

    def progress_handler(*args) -> None:
        progress_events.append(args)

    get_emitter().on(Emitter.PROGRESS_REPORT, progress_handler)
    try:
        get_emitter().emit(Emitter.IS_CONNECTED, True)
        await asyncio.sleep(0.1)

        get_emitter().emit(Emitter.TRACE_WRITTEN, "trace-written", "rec-1", 8)
        await asyncio.sleep(0.3)

        assert progress_events == []
    finally:
        get_emitter().remove_listener(Emitter.PROGRESS_REPORT, progress_handler)


@pytest.mark.asyncio
async def test_recording_completion_isolated_across_recordings(state_manager) -> None:
    """Test that recordings complete independently of each other."""
    _, store = state_manager
    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    updated_at = datetime(2024, 1, 2, tzinfo=timezone.utc)
    trace_a = _make_trace(
        "trace-a",
        "rec-a",
        status=TraceStatus.WRITING,
        created_at=created_at,
        last_updated=updated_at,
    )
    trace_b1 = _make_trace(
        "trace-b1",
        "rec-b",
        status=TraceStatus.WRITTEN,
        bytes_written=10,
        total_bytes=10,
        created_at=created_at,
        last_updated=updated_at,
    )
    trace_b2 = _make_trace(
        "trace-b2",
        "rec-b",
        status=TraceStatus.WRITING,
        bytes_written=0,
        total_bytes=10,
        created_at=created_at,
        last_updated=updated_at,
    )
    store._traces_by_id["trace-a"] = trace_a
    store._traces_by_id["trace-b1"] = trace_b1
    store._traces_by_id["trace-b2"] = trace_b2
    store._traces_by_recording["rec-a"] = [trace_a]
    store._traces_by_recording["rec-b"] = [trace_b1, trace_b2]

    progress_events: list[tuple] = []

    def progress_handler(*args) -> None:
        progress_events.append(args)

    get_emitter().on(Emitter.PROGRESS_REPORT, progress_handler)
    try:
        get_emitter().emit(Emitter.IS_CONNECTED, True)
        await asyncio.sleep(0.1)

        get_emitter().emit(Emitter.TRACE_WRITTEN, "trace-b2", "rec-b", 10)
        await asyncio.sleep(0.3)

        assert len(progress_events) == 1
        _, _, traces = progress_events[0]
        assert {trace.recording_id for trace in traces} == {"rec-b"}
    finally:
        get_emitter().remove_listener(Emitter.PROGRESS_REPORT, progress_handler)


@pytest.mark.asyncio
async def test_upload_failed_does_not_block_other_recordings(state_manager) -> None:
    """Test that upload failure in one recording doesn't block others."""
    _, store = state_manager
    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    updated_at = datetime(2024, 1, 2, tzinfo=timezone.utc)
    trace_a = _make_trace(
        "trace-a",
        "rec-a",
        status=TraceStatus.WRITING,
        created_at=created_at,
        last_updated=updated_at,
    )
    trace_b = _make_trace(
        "trace-b",
        "rec-b",
        status=TraceStatus.PENDING,
        created_at=created_at,
        last_updated=updated_at,
    )
    store._traces_by_id["trace-a"] = trace_a
    store._traces_by_id["trace-b"] = trace_b
    store._traces_by_recording["rec-a"] = [trace_a]
    store._traces_by_recording["rec-b"] = [trace_b]

    ready_events: list[tuple] = []

    def ready_handler(*args) -> None:
        ready_events.append(args)

    get_emitter().on(Emitter.READY_FOR_UPLOAD, ready_handler)
    try:
        get_emitter().emit(Emitter.IS_CONNECTED, True)
        await asyncio.sleep(0.1)

        get_emitter().emit(
            Emitter.UPLOAD_FAILED,
            "trace-a",
            0,
            TraceStatus.WRITTEN,
            TraceErrorCode.DISK_FULL,
            "disk full",
        )
        await asyncio.sleep(0.1)

        get_emitter().emit(Emitter.TRACE_WRITTEN, "trace-b", "rec-b", 10)
        await asyncio.sleep(0.3)

        assert ready_events == [(
            "trace-b",
            "rec-b",
            "/tmp/trace-b.bin",
            DataType.CUSTOM_1D,
            "custom",
            0,
        )]
    finally:
        get_emitter().remove_listener(Emitter.READY_FOR_UPLOAD, ready_handler)
