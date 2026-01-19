from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from neuracore.data_daemon.event_emitter import Emitter, emitter
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
        self.errors: list[tuple[str, int, str, TraceErrorCode | None, TraceStatus]] = []
        self.marked_written: list[tuple[str, int]] = []
        self.ready_traces: list[TraceRecord] = []
        self.unreported_traces: list[TraceRecord] = []
        self._traces_by_id: dict[str, TraceRecord] = {}
        self._traces_by_recording: dict[str, list[TraceRecord]] = {}

    def set_stopped_ats(self, recording_id: str) -> None:
        self.stopped.append(recording_id)

    def create_trace(self, *args, **kwargs) -> None:
        self.created.append(dict(kwargs))

    def get_trace(self, trace_id: str) -> TraceRecord | None:
        return self._traces_by_id.get(trace_id)

    def find_traces_by_recording_id(self, recording_id: str) -> list[TraceRecord]:
        return list(self._traces_by_recording.get(recording_id, []))

    def update_bytes_uploaded(self, trace_id: str, bytes_uploaded: int) -> None:
        self.updated_bytes.append((trace_id, bytes_uploaded))

    def mark_trace_as_written(self, trace_id: str, bytes_written: int) -> None:
        self.marked_written.append((trace_id, bytes_written))
        trace = self._traces_by_id.get(trace_id)
        if trace:
            updated = replace(
                trace,
                status=TraceStatus.WRITTEN,
                bytes_written=bytes_written,
                total_bytes=bytes_written,
                ready_for_upload=1,
            )
            self._traces_by_id[trace_id] = updated
            traces = self._traces_by_recording.get(trace.recording_id, [])
            self._traces_by_recording[trace.recording_id] = [
                updated if t.trace_id == trace_id else t for t in traces
            ]

    def find_ready_traces(self) -> list[TraceRecord]:
        return list(self.ready_traces)

    def find_unreported_traces(self) -> list[TraceRecord]:
        return list(self.unreported_traces)

    def claim_ready_traces(self, limit: int = 50) -> list[dict]:
        return []

    def mark_recording_reported(self, recording_id: str) -> None:
        return None

    def update_status(
        self, trace_id: str, status: TraceStatus, *, error_message=None
    ) -> None:
        return None

    def record_error(
        self,
        trace_id: str,
        error_message: str,
        error_code: TraceErrorCode | None = None,
        status: TraceStatus = TraceStatus.FAILED,
    ) -> None:
        self.errors.append((trace_id, error_message, error_code, status))

    def delete_trace(self, trace_id: str) -> None:
        self.deleted.append(trace_id)


def _cleanup_state_manager(manager: StateManager) -> None:
    emitter.remove_listener(Emitter.TRACE_WRITTEN, manager._handle_trace_written)
    emitter.remove_listener(Emitter.START_TRACE, manager.create_trace)
    emitter.remove_listener(Emitter.UPLOAD_COMPLETE, manager.handle_upload_complete)
    emitter.remove_listener(Emitter.UPLOADED_BYTES, manager.update_bytes_uploaded)
    emitter.remove_listener(Emitter.UPLOAD_FAILED, manager.handle_upload_failed)
    emitter.remove_listener(Emitter.STOP_RECORDING, manager.handle_stop_recording)
    emitter.remove_listener(Emitter.IS_CONNECTED, manager.handle_is_connected)


@pytest.fixture
def state_manager() -> tuple[StateManager, FakeStateStore]:
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
    ready_for_upload: int = 0,
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
        ready_for_upload=ready_for_upload,
        progress_reported=progress_reported,
        error_code=None,
        error_message=None,
        created_at=created_at,
        last_updated=last_updated,
    )


def test_stop_recording_emits_stop_all_and_sets_stopped(state_manager) -> None:
    manager, store = state_manager
    received: list[str] = []

    def handler(recording_id: str) -> None:
        received.append(recording_id)

    emitter.on(Emitter.STOP_ALL_TRACES_FOR_RECORDING, handler)
    try:
        emitter.emit(Emitter.STOP_RECORDING, "rec-1")
        assert store.stopped == ["rec-1"]
        assert received == ["rec-1"]
    finally:
        emitter.remove_listener(Emitter.STOP_ALL_TRACES_FOR_RECORDING, handler)


def test_start_trace_creates_trace(state_manager) -> None:
    _, store = state_manager
    emitter.emit(
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
    assert len(store.created) == 1
    payload = store.created[0]
    assert payload["trace_id"] == "trace-1"
    assert payload["recording_id"] == "rec-1"
    assert payload["data_type"] == DataType.CUSTOM_1D
    assert payload["data_type_name"] == "custom"
    assert payload["path"] == "/tmp/trace-1.bin"
    assert payload["total_bytes"] == 128


def test_uploaded_bytes_updates_store(state_manager) -> None:
    _, store = state_manager
    emitter.emit(Emitter.UPLOADED_BYTES, "trace-2", 42)
    assert store.updated_bytes == [("trace-2", 42)]


def test_upload_complete_emits_delete_and_deletes(state_manager) -> None:
    _, store = state_manager
    received: list[tuple[str, str]] = []

    def handler(trace_id: str, path: str) -> None:
        received.append((trace_id, path))

    emitter.on(Emitter.DELETE_TRACE, handler)
    try:
        emitter.emit(Emitter.UPLOAD_COMPLETE, "trace-3", "/tmp/trace-3.bin")
        assert store.deleted == ["trace-3"]
        assert received == [("trace-3", "/tmp/trace-3.bin")]
    finally:
        emitter.remove_listener(Emitter.DELETE_TRACE, handler)


def test_upload_failed_records_error(state_manager) -> None:
    _, store = state_manager
    emitter.emit(
        Emitter.UPLOAD_FAILED,
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


def test_trace_written_emits_ready_for_upload_when_connected(state_manager) -> None:
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

    emitter.on(Emitter.READY_FOR_UPLOAD, handler)
    try:
        emitter.emit(Emitter.IS_CONNECTED, True)
        emitter.emit(Emitter.TRACE_WRITTEN, "trace-5", "rec-5", 64)
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
        emitter.remove_listener(Emitter.READY_FOR_UPLOAD, handler)


def test_is_connected_emits_ready_and_progress_report(state_manager) -> None:
    _, store = state_manager
    t1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2024, 1, 2, tzinfo=timezone.utc)
    t3 = datetime(2024, 1, 3, tzinfo=timezone.utc)

    uploading = _make_trace(
        "trace-6",
        "rec-6",
        status=TraceStatus.UPLOADING,
        created_at=t1,
        last_updated=t2,
    )
    written = _make_trace(
        "trace-7",
        "rec-7",
        status=TraceStatus.WRITTEN,
        ready_for_upload=1,
        bytes_written=10,
        total_bytes=10,
        created_at=t2,
        last_updated=t3,
    )
    store.ready_traces = [written, uploading]
    store.unreported_traces = [written]

    ready_events: list[tuple] = []
    progress_events: list[tuple] = []

    def ready_handler(*args) -> None:
        ready_events.append(args)

    def progress_handler(*args) -> None:
        progress_events.append(args)

    emitter.on(Emitter.READY_FOR_UPLOAD, ready_handler)
    emitter.on(Emitter.PROGRESS_REPORT, progress_handler)
    try:
        emitter.emit(Emitter.IS_CONNECTED, True)
        assert ready_events == [
            (
                "trace-6",
                "rec-6",
                "/tmp/trace-6.bin",
                DataType.CUSTOM_1D,
                "custom",
                0,
            ),
            (
                "trace-7",
                "rec-7",
                "/tmp/trace-7.bin",
                DataType.CUSTOM_1D,
                "custom",
                0,
            ),
        ]
        assert len(progress_events) == 1
        start_time, end_time, traces = progress_events[0]
        assert start_time == t2.timestamp()
        assert end_time == t3.timestamp()
        assert [trace.trace_id for trace in traces] == ["trace-7"]
    finally:
        emitter.remove_listener(Emitter.READY_FOR_UPLOAD, ready_handler)
        emitter.remove_listener(Emitter.PROGRESS_REPORT, progress_handler)
