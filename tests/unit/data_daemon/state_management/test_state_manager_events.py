from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

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

    async def wait_for_count(
        self, count: int, timeout: float = DEFAULT_TIMEOUT
    ) -> list[tuple]:
        self.expected_count = count
        if len(self.events) >= count:
            return self.events
        self._done.clear()
        await asyncio.wait_for(self._done.wait(), timeout=timeout)
        return self.events


@asynccontextmanager
async def listen_for(event: str, expected_count: int = 1) -> EventCollector:
    """Context manager that listens for events and cleans up."""
    collector = EventCollector(expected_count)
    emitter = get_emitter()
    emitter.on(event, collector.handler)
    try:
        yield collector
    finally:
        emitter.remove_listener(event, collector.handler)


@asynccontextmanager
async def listen_for_multiple(events: dict[str, int]) -> dict[str, EventCollector]:
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
        trace = self._traces_by_id.get(trace_id)
        if trace:
            updated = replace(trace, status=status)
            self._traces_by_id[trace_id] = updated
            recording_id = trace.recording_id
            traces = self._traces_by_recording.get(recording_id, [])
            self._traces_by_recording[recording_id] = [
                updated if t.trace_id == trace_id else t for t in traces
            ]

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

    async def upsert_trace_metadata(
        self,
        trace_id: str,
        recording_id: str,
        data_type: DataType,
        path: str,
        data_type_name: str,
        robot_instance: int,
        dataset_id: str | None = None,
        dataset_name: str | None = None,
        robot_name: str | None = None,
        robot_id: str | None = None,
        total_bytes: int | None = None,
    ) -> TraceRecord:
        """Upsert trace with metadata from START_TRACE."""
        now = datetime.now(timezone.utc)
        existing = self._traces_by_id.get(trace_id)
        if existing:
            trace = replace(
                existing,
                data_type=data_type,
                data_type_name=data_type_name,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                robot_name=robot_name,
                robot_id=robot_id,
                robot_instance=robot_instance,
                path=path,
                total_bytes=total_bytes,
                last_updated=now,
            )
        else:
            trace = TraceRecord(
                trace_id=trace_id,
                recording_id=recording_id,
                status=TraceStatus.PENDING,
                data_type=data_type,
                data_type_name=data_type_name,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                robot_name=robot_name,
                robot_id=robot_id,
                robot_instance=robot_instance,
                path=path,
                total_bytes=total_bytes,
                bytes_written=None,
                bytes_uploaded=0,
                progress_reported=0,
                error_code=None,
                error_message=None,
                created_at=now,
                last_updated=now,
            )
        self._traces_by_id[trace_id] = trace
        if recording_id not in self._traces_by_recording:
            self._traces_by_recording[recording_id] = []
        traces = self._traces_by_recording[recording_id]
        self._traces_by_recording[recording_id] = (
            [trace if t.trace_id == trace_id else t for t in traces]
            if any(t.trace_id == trace_id for t in traces)
            else traces + [trace]
        )
        return trace

    async def upsert_trace_bytes(
        self,
        trace_id: str,
        recording_id: str,
        bytes_written: int,
    ) -> TraceRecord:
        """Upsert trace with bytes from TRACE_WRITTEN."""
        now = datetime.now(timezone.utc)
        existing = self._traces_by_id.get(trace_id)
        if existing:
            trace = replace(
                existing,
                bytes_written=bytes_written,
                total_bytes=bytes_written,
                last_updated=now,
            )
        else:
            trace = TraceRecord(
                trace_id=trace_id,
                recording_id=recording_id,
                status=TraceStatus.PENDING,
                data_type=None,
                data_type_name=None,
                dataset_id=None,
                dataset_name=None,
                robot_name=None,
                robot_id=None,
                robot_instance=None,
                path=None,
                total_bytes=bytes_written,
                bytes_written=bytes_written,
                bytes_uploaded=0,
                progress_reported=0,
                error_code=None,
                error_message=None,
                created_at=now,
                last_updated=now,
            )
        self._traces_by_id[trace_id] = trace
        if recording_id not in self._traces_by_recording:
            self._traces_by_recording[recording_id] = []
        traces = self._traces_by_recording[recording_id]
        self._traces_by_recording[recording_id] = (
            [trace if t.trace_id == trace_id else t for t in traces]
            if any(t.trace_id == trace_id for t in traces)
            else traces + [trace]
        )
        return trace


def _cleanup_state_manager(manager: StateManager) -> None:
    get_emitter().remove_listener(Emitter.TRACE_WRITTEN, manager._handle_trace_written)
    get_emitter().remove_listener(Emitter.START_TRACE, manager._handle_start_trace)
    get_emitter().remove_listener(
        Emitter.UPLOAD_COMPLETE, manager._handle_upload_complete
    )
    get_emitter().remove_listener(
        Emitter.UPLOADED_BYTES, manager._handle_uploaded_bytes
    )
    get_emitter().remove_listener(Emitter.UPLOAD_FAILED, manager._handle_upload_failed)
    get_emitter().remove_listener(
        Emitter.STOP_RECORDING, manager._handle_stop_recording
    )
    get_emitter().remove_listener(Emitter.IS_CONNECTED, manager._handle_is_connected)
    get_emitter().remove_listener(
        Emitter.PROGRESS_REPORTED, manager._handle_progress_reported
    )
    get_emitter().remove_listener(
        Emitter.PROGRESS_REPORT_FAILED, manager._handle_progress_report_failed
    )


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
    bytes_written: int | None = 0,
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

    async with listen_for(Emitter.STOP_ALL_TRACES_FOR_RECORDING) as collector:
        await manager._handle_stop_recording("rec-1")

        events = await collector.wait()
        assert events == [("rec-1",)]
        assert store.stopped == ["rec-1"]


@pytest.mark.asyncio
async def test_start_trace_creates_trace(state_manager) -> None:
    """START_TRACE creates trace in PENDING with metadata (partial data)."""
    manager, store = state_manager

    await manager._handle_start_trace(
        "trace-1",
        "rec-1",
        DataType.CUSTOM_1D,
        "custom",
        0,
        None,
        None,
        None,
        None,
        path="/tmp/trace-1.bin",
        total_bytes=128,
    )

    trace = store._traces_by_id.get("trace-1")
    assert trace is not None
    assert trace.trace_id == "trace-1"
    assert trace.recording_id == "rec-1"
    assert trace.data_type == DataType.CUSTOM_1D
    assert trace.data_type_name == "custom"
    assert trace.path == "/tmp/trace-1.bin"
    assert trace.total_bytes == 128
    assert trace.status == TraceStatus.PENDING
    assert trace.bytes_written is None


@pytest.mark.asyncio
async def test_uploaded_bytes_updates_store(state_manager) -> None:
    manager, store = state_manager

    await manager._handle_uploaded_bytes("trace-2", 42)

    assert store.updated_bytes == [("trace-2", 42)]


@pytest.mark.asyncio
async def test_upload_complete_emits_delete_and_deletes(state_manager) -> None:
    manager, store = state_manager

    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trace = _make_trace(
        "trace-3",
        "rec-3",
        created_at=created_at,
        last_updated=created_at,
    )
    store._traces_by_id["trace-3"] = trace

    async with listen_for(Emitter.DELETE_TRACE) as collector:
        await manager._handle_upload_complete("trace-3")

        events = await collector.wait()
        assert events == [("rec-3", "trace-3", DataType.CUSTOM_1D)]
        assert store.deleted == ["trace-3"]


@pytest.mark.asyncio
async def test_upload_failed_records_error(state_manager) -> None:
    manager, store = state_manager

    await manager._handle_upload_failed(
        "trace-4",
        12,
        TraceStatus.FAILED,
        TraceErrorCode.NETWORK_ERROR,
        "lost connection",
    )

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
    """When both START_TRACE and TRACE_WRITTEN arrive, emit READY_FOR_UPLOAD."""
    manager, store = state_manager

    await manager._handle_start_trace(
        "trace-5",
        "rec-5",
        DataType.CUSTOM_1D,
        "custom",
        0,
        None,
        None,
        None,
        None,
        path="/tmp/trace-5.bin",
        total_bytes=64,
    )

    async with listen_for(Emitter.READY_FOR_UPLOAD) as collector:
        await manager._handle_trace_written("trace-5", "rec-5", 64)

        events = await collector.wait()
        assert events == [(
            "trace-5",
            "rec-5",
            "/tmp/trace-5.bin",
            DataType.CUSTOM_1D,
            "custom",
            0,
        )]


@pytest.mark.asyncio
async def test_trace_written_emits_progress_report_with_bounds(state_manager) -> None:
    """Test that progress report is emitted when all traces in recording complete."""
    manager, store = state_manager

    await manager._handle_start_trace(
        "trace-1",
        "rec-1",
        DataType.CUSTOM_1D,
        "custom",
        0,
        None,
        None,
        None,
        None,
        path="/tmp/trace-1.bin",
        total_bytes=10,
    )
    await manager._handle_start_trace(
        "trace-2",
        "rec-1",
        DataType.CUSTOM_1D,
        "custom",
        0,
        None,
        None,
        None,
        None,
        path="/tmp/trace-2.bin",
        total_bytes=10,
    )

    async with listen_for_multiple({
        Emitter.READY_FOR_UPLOAD: 2,
        Emitter.PROGRESS_REPORT: 1,
    }) as collectors:
        await manager._handle_trace_written("trace-1", "rec-1", 10)
        await manager._handle_trace_written("trace-2", "rec-1", 10)

        ready_events = await collectors[Emitter.READY_FOR_UPLOAD].wait()
        assert len(ready_events) == 2

        progress_events = await collectors[Emitter.PROGRESS_REPORT].wait()
        assert len(progress_events) == 1
        start_time, end_time, traces = progress_events[0]
        assert {trace.trace_id for trace in traces} == {"trace-1", "trace-2"}
        assert start_time <= end_time


@pytest.mark.asyncio
async def test_trace_written_waits_for_all_traces_before_progress_report(
    state_manager,
) -> None:
    """Test that progress report waits for all traces before emitting."""
    manager, store = state_manager
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
    trace_pending = _make_trace(
        "trace-pending",
        "rec-1",
        status=TraceStatus.PENDING,
        bytes_written=None,
        total_bytes=8,
        created_at=created_at,
        last_updated=updated_at,
    )
    store._traces_by_id["trace-written"] = trace_written
    store._traces_by_id["trace-pending"] = trace_pending
    store._traces_by_recording["rec-1"] = [trace_written, trace_pending]

    async with listen_for(Emitter.PROGRESS_REPORT) as collector:
        await manager._handle_is_connected(True)
        await manager._handle_trace_written("trace-written", "rec-1", 8)

        with pytest.raises(asyncio.TimeoutError):
            await collector.wait(timeout=0.1)

        assert collector.events == []


@pytest.mark.asyncio
async def test_recording_completion_isolated_across_recordings(state_manager) -> None:
    """Test that recordings complete independently of each other."""
    manager, store = state_manager
    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    updated_at = datetime(2024, 1, 2, tzinfo=timezone.utc)
    trace_a = _make_trace(
        "trace-a",
        "rec-a",
        status=TraceStatus.PENDING,
        bytes_written=None,
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
        status=TraceStatus.PENDING,
        bytes_written=None,
        total_bytes=10,
        created_at=created_at,
        last_updated=updated_at,
    )
    store._traces_by_id["trace-a"] = trace_a
    store._traces_by_id["trace-b1"] = trace_b1
    store._traces_by_id["trace-b2"] = trace_b2
    store._traces_by_recording["rec-a"] = [trace_a]
    store._traces_by_recording["rec-b"] = [trace_b1, trace_b2]

    async with listen_for(Emitter.PROGRESS_REPORT) as collector:
        await manager._handle_is_connected(True)
        await manager._handle_trace_written("trace-b2", "rec-b", 10)

        events = await collector.wait()
        assert len(events) == 1
        _, _, traces = events[0]
        assert {trace.recording_id for trace in traces} == {"rec-b"}


@pytest.mark.asyncio
async def test_upload_failed_does_not_block_other_recordings(state_manager) -> None:
    """Test that upload failure in one recording doesn't block others."""
    manager, store = state_manager
    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    updated_at = datetime(2024, 1, 2, tzinfo=timezone.utc)
    trace_a = _make_trace(
        "trace-a",
        "rec-a",
        status=TraceStatus.PENDING,
        bytes_written=None,
        created_at=created_at,
        last_updated=updated_at,
    )
    trace_b = _make_trace(
        "trace-b",
        "rec-b",
        status=TraceStatus.PENDING,
        bytes_written=None,
        created_at=created_at,
        last_updated=updated_at,
    )
    store._traces_by_id["trace-a"] = trace_a
    store._traces_by_id["trace-b"] = trace_b
    store._traces_by_recording["rec-a"] = [trace_a]
    store._traces_by_recording["rec-b"] = [trace_b]

    async with listen_for(Emitter.READY_FOR_UPLOAD) as collector:
        await manager._handle_is_connected(True)
        await manager._handle_upload_failed(
            "trace-a",
            0,
            TraceStatus.WRITTEN,
            TraceErrorCode.DISK_FULL,
            "disk full",
        )
        await manager._handle_trace_written("trace-b", "rec-b", 10)

        events = await collector.wait()
        assert events == [(
            "trace-b",
            "rec-b",
            "/tmp/trace-b.bin",
            DataType.CUSTOM_1D,
            "custom",
            0,
        )]
