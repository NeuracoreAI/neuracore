from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from neuracore.data_daemon.event_emitter import Emitter, get_emitter
from neuracore.data_daemon.models import (
    DataType,
    ProgressReportStatus,
    TraceErrorCode,
    TraceRecord,
    TraceStatus,
)
from neuracore.data_daemon.state_management.state_manager import StateManager


class FakeStateStore:
    def __init__(self) -> None:
        self.stopped: list[str] = []
        self.updated_bytes: list[tuple[str, int]] = []
        self.deleted: list[str] = []
        self.errors: list[tuple[str, str, TraceErrorCode | None, TraceStatus]] = []
        self.ready_traces: list[TraceRecord] = []
        self.unreported_traces: list[TraceRecord] = []
        self._traces_by_id: dict[str, TraceRecord] = {}
        self._traces_by_recording: dict[str, list[TraceRecord]] = {}

    async def set_stopped_ats(self, recording_id: str) -> None:
        self.stopped.append(recording_id)

    async def get_trace(self, trace_id: str) -> TraceRecord | None:
        return self._traces_by_id.get(trace_id)

    async def find_traces_by_recording_id(self, recording_id: str) -> list[TraceRecord]:
        return list(self._traces_by_recording.get(recording_id, []))

    async def update_bytes_uploaded(self, trace_id: str, bytes_uploaded: int) -> None:
        self.updated_bytes.append((trace_id, bytes_uploaded))

    async def find_ready_traces(self) -> list[TraceRecord]:
        return list(self.ready_traces)

    async def find_unreported_traces(self) -> list[TraceRecord]:
        return list(self.unreported_traces)

    async def mark_recording_reported(self, recording_id: str) -> None:
        return None

    async def update_status(
        self, trace_id: str, status: TraceStatus, *, error_message=None
    ) -> bool:
        trace = self._traces_by_id.get(trace_id)
        if trace is None:
            raise ValueError(f"Trace not found: {trace_id}")
        if trace.status == status:
            return False  # Already at target status
        updated = replace(trace, status=status)
        self._traces_by_id[trace_id] = updated
        recording_id = trace.recording_id
        traces = self._traces_by_recording.get(recording_id, [])
        self._traces_by_recording[recording_id] = [
            updated if t.trace_id == trace_id else t for t in traces
        ]
        return True

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

    def _update_trace_in_recording(self, trace: TraceRecord, recording_id: str) -> None:
        """Helper to update trace in recording mapping."""
        if recording_id not in self._traces_by_recording:
            self._traces_by_recording[recording_id] = []
        traces = self._traces_by_recording[recording_id]
        self._traces_by_recording[recording_id] = (
            [trace if t.trace_id == trace.trace_id else t for t in traces]
            if any(t.trace_id == trace.trace_id for t in traces)
            else traces + [trace]
        )

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
        """Upsert trace with metadata from START_TRACE.

        State transitions:
        - New trace: INITIALIZING
        - PENDING_BYTES -> WRITTEN
        """
        now = datetime.now(timezone.utc)
        existing = self._traces_by_id.get(trace_id)
        if existing:
            new_status = (
                TraceStatus.WRITTEN
                if existing.status == TraceStatus.PENDING_BYTES
                else existing.status
            )
            trace = replace(
                existing,
                status=new_status,
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
                status=TraceStatus.INITIALIZING,
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
                progress_reported=ProgressReportStatus.PENDING,
                error_code=None,
                error_message=None,
                created_at=now,
                last_updated=now,
            )
        self._traces_by_id[trace_id] = trace
        self._update_trace_in_recording(trace, recording_id)
        return trace

    async def upsert_trace_bytes(
        self,
        trace_id: str,
        recording_id: str,
        bytes_written: int,
    ) -> TraceRecord:
        """Upsert trace with bytes from TRACE_WRITTEN.

        State transitions:
        - New trace: PENDING_BYTES
        - INITIALIZING -> WRITTEN
        """
        now = datetime.now(timezone.utc)
        existing = self._traces_by_id.get(trace_id)
        if existing:
            new_status = (
                TraceStatus.WRITTEN
                if existing.status == TraceStatus.INITIALIZING
                else existing.status
            )
            trace = replace(
                existing,
                status=new_status,
                bytes_written=bytes_written,
                total_bytes=bytes_written,
                last_updated=now,
            )
        else:
            trace = TraceRecord(
                trace_id=trace_id,
                recording_id=recording_id,
                status=TraceStatus.PENDING_BYTES,
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
                progress_reported=ProgressReportStatus.PENDING,
                error_code=None,
                error_message=None,
                created_at=now,
                last_updated=now,
            )
        self._traces_by_id[trace_id] = trace
        self._update_trace_in_recording(trace, recording_id)
        return trace


def _cleanup_state_manager(manager: StateManager) -> None:
    get_emitter().remove_listener(Emitter.TRACE_WRITTEN, manager._handle_trace_written)
    get_emitter().remove_listener(Emitter.START_TRACE, manager._handle_start_trace)
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
    status: TraceStatus = TraceStatus.INITIALIZING,
    progress_reported: ProgressReportStatus = ProgressReportStatus.PENDING,
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
    """START_TRACE creates trace in INITIALIZING with metadata (waiting for bytes)."""
    _, store = state_manager
    get_emitter().emit(
        Emitter.START_TRACE,
        "trace-1",
        "rec-1",
        DataType.CUSTOM_1D,
        "custom",
        0,  # robot_instance
        None,
        None,
        None,
        None,
        path="/tmp/trace-1.bin",
        total_bytes=128,
    )
    await asyncio.sleep(0.2)

    # Trace should be in store with metadata but no bytes_written yet
    trace = store._traces_by_id.get("trace-1")
    assert trace is not None
    assert trace.trace_id == "trace-1"
    assert trace.recording_id == "rec-1"
    assert trace.data_type == DataType.CUSTOM_1D
    assert trace.data_type_name == "custom"
    assert trace.path == "/tmp/trace-1.bin"
    assert trace.total_bytes == 128
    assert trace.status == TraceStatus.INITIALIZING
    assert trace.bytes_written is None  # Not complete yet


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
    """When both START_TRACE and TRACE_WRITTEN arrive, emit READY_FOR_UPLOAD."""
    manager, store = state_manager
    received: list[tuple] = []

    def handler(*args) -> None:
        received.append(args)

    get_emitter().on(Emitter.READY_FOR_UPLOAD, handler)
    try:
        # First: START_TRACE with metadata (creates PENDING trace)
        get_emitter().emit(
            Emitter.START_TRACE,
            "trace-5",
            "rec-5",
            DataType.CUSTOM_1D,
            "custom",
            0,  # robot_instance
            None,
            None,
            None,
            None,
            path="/tmp/trace-5.bin",
            total_bytes=64,
        )
        await asyncio.sleep(0.2)

        # No READY_FOR_UPLOAD yet (missing bytes_written)
        assert received == []

        # Second: TRACE_WRITTEN with bytes (completes the trace)
        get_emitter().emit(Emitter.TRACE_WRITTEN, "trace-5", "rec-5", 64)
        await asyncio.sleep(0.2)

        # Now READY_FOR_UPLOAD should be emitted
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
    """Test that progress report is emitted when all traces in recording complete."""
    manager, store = state_manager

    ready_events: list[tuple] = []
    progress_events: list[tuple] = []

    def ready_handler(*args) -> None:
        ready_events.append(args)

    def progress_handler(*args) -> None:
        progress_events.append(args)

    get_emitter().on(Emitter.READY_FOR_UPLOAD, ready_handler)
    get_emitter().on(Emitter.PROGRESS_REPORT, progress_handler)
    try:
        # Create two traces in the same recording via START_TRACE
        get_emitter().emit(
            Emitter.START_TRACE,
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
        get_emitter().emit(
            Emitter.START_TRACE,
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
        await asyncio.sleep(0.2)

        # No progress report yet - neither trace has bytes_written
        assert len(progress_events) == 0

        # Complete first trace
        get_emitter().emit(Emitter.TRACE_WRITTEN, "trace-1", "rec-1", 10)
        await asyncio.sleep(0.2)

        # Still no progress report - trace-2 not complete
        assert len(progress_events) == 0
        assert len(ready_events) == 1  # trace-1 ready for upload

        # Complete second trace
        get_emitter().emit(Emitter.TRACE_WRITTEN, "trace-2", "rec-1", 10)
        await asyncio.sleep(0.3)

        # Now progress report should be emitted
        assert len(ready_events) == 2  # both traces ready
        assert len(progress_events) == 1
        start_time, end_time, traces = progress_events[0]
        assert {trace.trace_id for trace in traces} == {"trace-1", "trace-2"}
        # Verify time bounds are reasonable (both created close to now)
        assert start_time <= end_time
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
    trace_pending = _make_trace(
        "trace-pending",
        "rec-1",
        status=TraceStatus.INITIALIZING,
        bytes_written=None,
        total_bytes=8,
        created_at=created_at,
        last_updated=updated_at,
    )
    store._traces_by_id["trace-written"] = trace_written
    store._traces_by_id["trace-pending"] = trace_pending
    store._traces_by_recording["rec-1"] = [trace_written, trace_pending]

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
        status=TraceStatus.INITIALIZING,
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
        status=TraceStatus.INITIALIZING,
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
        status=TraceStatus.INITIALIZING,
        bytes_written=None,
        created_at=created_at,
        last_updated=updated_at,
    )
    trace_b = _make_trace(
        "trace-b",
        "rec-b",
        status=TraceStatus.INITIALIZING,
        bytes_written=None,
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
