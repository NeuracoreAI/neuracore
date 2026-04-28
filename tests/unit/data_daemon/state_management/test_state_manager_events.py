from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.event_emitter import Emitter
from neuracore.data_daemon.models import (
    DataType,
    ProgressReportStatus,
    RecordingStopReportStatus,
    TraceErrorCode,
    TraceRecord,
    TraceRegistrationStatus,
    TraceUploadStatus,
    TraceWriteStatus,
)
from neuracore.data_daemon.state_management.state_manager import StateManager


class FakeStateStore:
    def __init__(self) -> None:
        self.stopped: list[str] = []
        self.updated_bytes: list[tuple[str, int]] = []
        self.deleted: list[str] = []
        self.errors: list[tuple[str, str, TraceErrorCode | None]] = []
        self.ready_traces: list[TraceRecord] = []
        self.unreported_traces: list[TraceRecord] = []
        self._traces_by_id: dict[str, TraceRecord] = {}
        self._traces_by_recording: dict[str, list[TraceRecord]] = {}
        self._recording_stopped: dict[str, bool] = {}
        self._recording_progress_reported: dict[str, bool] = {}
        self._recording_stop_report_status: dict[str, RecordingStopReportStatus] = {}
        self._recording_org_ids: dict[str, str] = {}
        self._expected_trace_count_reported: dict[str, bool] = {}
        self._expected_trace_count: dict[str, int] = {}
        self.uploaded_count_increments: list[str] = []

    async def set_stopped_at(self, recording_id: str) -> None:
        self.stopped.append(recording_id)
        self._recording_stopped[recording_id] = True
        now = datetime.now(timezone.utc)
        traces = self._traces_by_recording.get(recording_id, [])
        updated = []
        for t in traces:
            if t.stopped_at is None:
                t = replace(t, stopped_at=now)
                self._traces_by_id[t.trace_id] = t
            updated.append(t)
        self._traces_by_recording[recording_id] = updated

    async def get_trace(self, trace_id: str) -> TraceRecord | None:
        return self._traces_by_id.get(trace_id)

    async def find_traces_by_recording_id(self, recording_id: str) -> list[TraceRecord]:
        return list(self._traces_by_recording.get(recording_id, []))

    async def list_traces(self) -> list[TraceRecord]:
        return list(self._traces_by_id.values())

    async def update_bytes_uploaded(self, trace_id: str, bytes_uploaded: int) -> None:
        self.updated_bytes.append((trace_id, bytes_uploaded))

    async def find_ready_traces(self) -> list[TraceRecord]:
        return list(self.ready_traces)

    async def find_unreported_traces(self) -> list[TraceRecord]:
        return list(self.unreported_traces)

    async def mark_recording_reported(self, recording_id: str) -> None:
        if hasattr(self, "_recording_reporting"):
            self._recording_reporting.discard(recording_id)
        self._recording_progress_reported[recording_id] = True
        traces = self._traces_by_recording.get(recording_id, [])
        updated = []
        for trace in traces:
            reported = replace(trace, progress_reported=ProgressReportStatus.REPORTED)
            self._traces_by_id[reported.trace_id] = reported
            updated.append(reported)
        self._traces_by_recording[recording_id] = updated

    async def recording_has_reported_progress(self, recording_id: str) -> bool:
        return self._recording_progress_reported.get(recording_id, False)

    async def mark_recording_reporting(self, recording_id: str) -> bool:
        if not hasattr(self, "_recording_reporting"):
            self._recording_reporting: set[str] = set()
        if self._recording_progress_reported.get(recording_id, False):
            return False
        if recording_id in self._recording_reporting:
            return False
        self._recording_reporting.add(recording_id)
        return True

    async def mark_recording_pending(self, recording_id: str) -> None:
        if hasattr(self, "_recording_reporting"):
            self._recording_reporting.discard(recording_id)
        self._recording_progress_reported[recording_id] = False
        return None

    async def reset_reporting_recordings_to_pending(self) -> int:
        if not hasattr(self, "_recording_reporting"):
            return 0
        count = len(self._recording_reporting)
        self._recording_reporting.clear()
        return count

    async def mark_recording_stop_report_pending(self, recording_id: str) -> None:
        if (
            self._recording_stop_report_status.get(recording_id)
            == RecordingStopReportStatus.REPORTED
        ):
            return
        self._recording_stop_report_status[recording_id] = (
            RecordingStopReportStatus.PENDING
        )

    async def claim_recording_stop_report(self, recording_id: str) -> bool:
        status = self._recording_stop_report_status.get(
            recording_id, RecordingStopReportStatus.PENDING
        )
        if status != RecordingStopReportStatus.PENDING:
            return False
        self._recording_stop_report_status[recording_id] = (
            RecordingStopReportStatus.REPORTING
        )
        return True

    async def mark_recording_stop_reported(self, recording_id: str) -> None:
        self._recording_stop_report_status[recording_id] = (
            RecordingStopReportStatus.REPORTED
        )

    async def reset_recording_stop_reporting_to_pending(self) -> int:
        count = 0
        for recording_id, status in list(self._recording_stop_report_status.items()):
            if status == RecordingStopReportStatus.REPORTING:
                self._recording_stop_report_status[recording_id] = (
                    RecordingStopReportStatus.PENDING
                )
                count += 1
        return count

    async def recording_stop_has_been_reported(self, recording_id: str) -> bool:
        return (
            self._recording_stop_report_status.get(recording_id)
            == RecordingStopReportStatus.REPORTED
        )

    async def list_pending_recording_stop_reports(self) -> list[str]:
        return [
            recording_id
            for recording_id, status in self._recording_stop_report_status.items()
            if status == RecordingStopReportStatus.PENDING
        ]

    async def delete_uploaded_traces_for_recording(self, recording_id: str) -> int:
        traces = self._traces_by_recording.get(recording_id, [])
        keep: list[TraceRecord] = []
        deleted_count = 0
        for trace in traces:
            if trace.upload_status == TraceUploadStatus.UPLOADED:
                self.deleted.append(trace.trace_id)
                self._traces_by_id.pop(trace.trace_id, None)
                deleted_count += 1
                continue
            keep.append(trace)
        self._traces_by_recording[recording_id] = keep
        return deleted_count

    async def delete_traces_for_recording(self, recording_id: str) -> int:
        traces = self._traces_by_recording.get(recording_id, [])
        for trace in traces:
            self.deleted.append(trace.trace_id)
            self._traces_by_id.pop(trace.trace_id, None)
        self._traces_by_recording[recording_id] = []
        return len(traces)

    async def delete_expired_completed_recordings(self, max_age_hours: int) -> int:
        return 0

    async def list_recording_ids_with_stopped_traces(self) -> list[str]:
        return [
            recording_id
            for recording_id, traces in self._traces_by_recording.items()
            if any(trace.stopped_at is not None for trace in traces)
        ]

    async def reconcile_recordings_from_traces(self) -> None:
        return None

    async def prune_old_empty_recordings(self, max_age_hours: int) -> int:
        return 0

    async def is_recording_stopped(self, recording_id: str) -> bool:
        return self._recording_stopped.get(recording_id, False)

    async def is_expected_trace_count_reported(self, recording_id: str) -> bool:
        return self._expected_trace_count_reported.get(recording_id, False)

    async def get_expected_trace_count(self, recording_id: str) -> int | None:
        return self._expected_trace_count.get(recording_id)

    async def set_expected_trace_count(
        self, recording_id: str, expected_trace_count: int
    ) -> None:
        self._expected_trace_count[recording_id] = int(expected_trace_count)

    async def count_traces_for_recording(self, recording_id: str) -> int:
        traces = self._traces_by_recording.get(recording_id, [])
        return sum(
            1
            for trace in traces
            if trace.write_status == TraceWriteStatus.WRITTEN
            and trace.total_bytes is not None
            and trace.total_bytes > 0
        )

    async def get_progress_report_snapshot(
        self, recording_id: str
    ) -> tuple[float, float, dict[str, int], int] | None:
        traces = self._traces_by_recording.get(recording_id, [])
        eligible = [
            trace
            for trace in traces
            if trace.data_type_name
            and trace.total_bytes is not None
            and trace.total_bytes > 0
        ]
        if not eligible:
            return None
        start_time = min(trace.created_at for trace in eligible).timestamp()
        end_time = datetime.now(timezone.utc).timestamp()
        trace_map = {
            str(trace.trace_id): int(trace.total_bytes or 0) for trace in eligible
        }
        total_bytes = sum(trace_map.values())
        return start_time, end_time, trace_map, total_bytes

    async def find_failed_traces(self) -> list[TraceRecord]:
        return [
            trace
            for trace in self._traces_by_id.values()
            if trace.upload_status == TraceUploadStatus.FAILED
        ]

    async def reset_failed_trace_for_retry(self, trace_id: str) -> None:
        trace = self._traces_by_id.get(trace_id)
        if trace is None:
            raise ValueError(f"Trace not found: {trace_id}")
        now = datetime.now(timezone.utc)
        updated = replace(
            trace,
            write_status=TraceWriteStatus.WRITTEN,
            upload_status=TraceUploadStatus.PENDING,
            error_code=None,
            error_message=None,
            next_retry_at=None,
            num_upload_attempts=0,
            bytes_uploaded=0,
            last_updated=now,
        )
        self._traces_by_id[trace_id] = updated
        self._update_trace_in_recording(updated, updated.recording_id)

    async def reset_retrying_to_written(self) -> int:
        return 0

    async def record_error(
        self,
        trace_id: str,
        error_message: str,
        error_code: TraceErrorCode | None = None,
    ) -> None:
        self.errors.append((trace_id, error_message, error_code))

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
    ) -> TraceRecord:
        """Upsert trace with metadata from START_TRACE.

        State transitions:
        - New trace: INITIALIZING
        - PENDING_METADATA -> WRITTEN
        """
        now = datetime.now(timezone.utc)
        existing = self._traces_by_id.get(trace_id)
        if existing:
            new_status = (
                TraceWriteStatus.WRITTEN
                if existing.write_status == TraceWriteStatus.PENDING_METADATA
                else existing.write_status
            )
            trace = replace(
                existing,
                write_status=new_status,
                data_type=data_type,
                data_type_name=data_type_name,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                robot_name=robot_name,
                robot_id=robot_id,
                robot_instance=robot_instance,
                path=path,
                total_bytes=existing.total_bytes,
                last_updated=now,
                num_upload_attempts=0,
                next_retry_at=None,
            )
        else:
            trace = TraceRecord(
                trace_id=trace_id,
                recording_id=recording_id,
                write_status=TraceWriteStatus.INITIALIZING,
                registration_status=TraceRegistrationStatus.PENDING,
                upload_status=TraceUploadStatus.PENDING,
                data_type=data_type,
                data_type_name=data_type_name,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                robot_name=robot_name,
                robot_id=robot_id,
                robot_instance=robot_instance,
                path=path,
                total_bytes=None,
                bytes_written=None,
                bytes_uploaded=0,
                progress_reported=ProgressReportStatus.PENDING,
                expected_trace_count_reported=0,
                error_code=None,
                error_message=None,
                created_at=now,
                last_updated=now,
                num_upload_attempts=0,
                next_retry_at=None,
                stopped_at=None,
            )
        self._traces_by_id[trace_id] = trace
        self._update_trace_in_recording(trace, recording_id)
        return trace

    async def upsert_trace_write_progress(
        self,
        trace_id: str,
        recording_id: str,
        bytes_written: int,
    ) -> TraceRecord:
        now = datetime.now(timezone.utc)
        existing = self._traces_by_id.get(trace_id)
        if existing:
            new_status = (
                TraceWriteStatus.WRITING
                if existing.write_status
                in (
                    TraceWriteStatus.PENDING,
                    TraceWriteStatus.INITIALIZING,
                    TraceWriteStatus.PENDING_METADATA,
                    TraceWriteStatus.WRITING,
                )
                else existing.write_status
            )
            trace = replace(
                existing,
                write_status=new_status,
                bytes_written=bytes_written,
                last_updated=now,
            )
        else:
            trace = TraceRecord(
                trace_id=trace_id,
                recording_id=recording_id,
                write_status=TraceWriteStatus.WRITING,
                registration_status=TraceRegistrationStatus.PENDING,
                upload_status=TraceUploadStatus.PENDING,
                data_type=None,
                data_type_name=None,
                dataset_id=None,
                dataset_name=None,
                robot_name=None,
                robot_id=None,
                robot_instance=None,
                path=None,
                total_bytes=None,
                bytes_written=bytes_written,
                bytes_uploaded=0,
                progress_reported=ProgressReportStatus.PENDING,
                expected_trace_count_reported=0,
                error_code=None,
                error_message=None,
                created_at=now,
                last_updated=now,
                num_upload_attempts=0,
                next_retry_at=None,
                stopped_at=None,
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
        - New trace: PENDING_METADATA
        - INITIALIZING/WRITING -> WRITTEN
        """
        now = datetime.now(timezone.utc)
        existing = self._traces_by_id.get(trace_id)
        if existing:
            new_status = (
                TraceWriteStatus.WRITTEN
                if existing.write_status
                in (TraceWriteStatus.INITIALIZING, TraceWriteStatus.WRITING)
                else existing.write_status
            )
            trace = replace(
                existing,
                write_status=new_status,
                bytes_written=bytes_written,
                total_bytes=bytes_written,
                last_updated=now,
            )
        else:
            trace = TraceRecord(
                trace_id=trace_id,
                recording_id=recording_id,
                write_status=TraceWriteStatus.PENDING_METADATA,
                registration_status=TraceRegistrationStatus.PENDING,
                upload_status=TraceUploadStatus.PENDING,
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
                expected_trace_count_reported=0,
                error_code=None,
                error_message=None,
                created_at=now,
                last_updated=now,
                num_upload_attempts=0,
                next_retry_at=None,
                stopped_at=None,
            )
        self._traces_by_id[trace_id] = trace
        self._update_trace_in_recording(trace, recording_id)
        return trace

    async def claim_traces_for_registration(
        self,
        limit: int,
        max_wait_s: float,
    ):
        candidates = [
            trace
            for trace in self._traces_by_id.values()
            if trace.write_status == TraceWriteStatus.WRITTEN
            and trace.registration_status == TraceRegistrationStatus.PENDING
        ][:limit]

        claimed: list[TraceRecord] = []
        for trace in candidates:
            updated = replace(
                trace,
                registration_status=TraceRegistrationStatus.REGISTERING,
            )
            self._traces_by_id[trace.trace_id] = updated
            self._update_trace_in_recording(updated, updated.recording_id)
            claimed.append(updated)

        return claimed

    async def mark_traces_as_registering(self, trace_ids: list[str]) -> list[str]:
        return list(trace_ids)

    async def mark_traces_as_registered(self, trace_ids: list[str]) -> list[str]:
        return list(trace_ids)

    async def update_write_status(
        self, trace_id: str, write_status: TraceWriteStatus
    ) -> None:
        trace = self._traces_by_id.get(trace_id)
        if trace is None:
            return
        updated = replace(trace, write_status=write_status)
        self._traces_by_id[trace_id] = updated
        self._update_trace_in_recording(updated, updated.recording_id)

    async def update_registration_status(
        self, trace_id: str, registration_status: TraceRegistrationStatus
    ) -> None:
        trace = self._traces_by_id.get(trace_id)
        if trace is None:
            return
        updated = replace(trace, registration_status=registration_status)
        self._traces_by_id[trace_id] = updated
        self._update_trace_in_recording(updated, updated.recording_id)

    async def update_upload_status(
        self, trace_id: str, upload_status: TraceUploadStatus
    ) -> None:
        trace = self._traces_by_id.get(trace_id)
        if trace is None:
            return
        updated = replace(trace, upload_status=upload_status)
        self._traces_by_id[trace_id] = updated
        self._update_trace_in_recording(updated, updated.recording_id)

    async def increment_uploaded_trace_count(self, recording_id: str) -> None:
        self.uploaded_count_increments.append(recording_id)

    async def set_recording_org_id(self, recording_id: str, org_id: str) -> None:
        self._recording_org_ids[recording_id] = org_id

    async def get_recording_org_id(self, recording_id: str) -> str | None:
        return self._recording_org_ids.get(recording_id)

    async def delete_recording_and_traces_if_fully_uploaded(
        self, recording_id: str
    ) -> bool:
        return False

    async def mark_expected_trace_count_reported(self, recording_id: str) -> None:
        self._expected_trace_count_reported[recording_id] = True

    async def schedule_retry(
        self,
        trace_id: str,
        *,
        next_retry_at: datetime,
        error_code: TraceErrorCode,
        error_message: str,
    ) -> int:
        trace = self._traces_by_id.get(trace_id)
        if trace is None:
            return 0
        updated = replace(
            trace,
            upload_status=TraceUploadStatus.RETRYING,
            next_retry_at=next_retry_at,
            num_upload_attempts=int(trace.num_upload_attempts) + 1,
            error_code=error_code,
            error_message=error_message,
        )
        self._traces_by_id[trace_id] = updated
        self._update_trace_in_recording(updated, updated.recording_id)
        return 1

    async def mark_retry_exhausted(
        self,
        trace_id: str,
        *,
        error_code: TraceErrorCode,
        error_message: str,
    ) -> int:
        trace = self._traces_by_id.get(trace_id)
        if trace is None:
            return 0
        updated = replace(
            trace,
            upload_status=TraceUploadStatus.FAILED,
            error_code=error_code,
            error_message=error_message,
            num_upload_attempts=int(trace.num_upload_attempts) + 1,
        )
        self._traces_by_id[trace_id] = updated
        self._update_trace_in_recording(updated, updated.recording_id)
        return 1


@pytest_asyncio.fixture
async def state_manager(emitter: Emitter) -> tuple[StateManager, FakeStateStore]:
    store = FakeStateStore()
    manager = StateManager(store, emitter=emitter)

    async def _fake_set_expected(recording_id: str, expected_trace_count: int) -> bool:
        await store.set_expected_trace_count(recording_id, expected_trace_count)
        await store.mark_expected_trace_count_reported(recording_id)
        return True

    setattr(manager, "_set_expected_trace_count", _fake_set_expected)
    yield manager, store


def _make_trace(
    trace_id: str,
    recording_id: str,
    *,
    write_status: TraceWriteStatus = TraceWriteStatus.INITIALIZING,
    registration_status: TraceRegistrationStatus = TraceRegistrationStatus.REGISTERED,
    upload_status: TraceUploadStatus = TraceUploadStatus.PENDING,
    progress_reported: ProgressReportStatus = ProgressReportStatus.PENDING,
    bytes_written: int | None = 0,
    total_bytes: int | None = None,
    bytes_uploaded: int = 0,
    num_upload_attempts: int = 0,
    next_retry_at: datetime | None = None,
    created_at: datetime,
    last_updated: datetime,
) -> TraceRecord:
    return TraceRecord(
        trace_id=trace_id,
        write_status=write_status,
        registration_status=registration_status,
        upload_status=upload_status,
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
        expected_trace_count_reported=0,
        error_code=None,
        error_message=None,
        num_upload_attempts=num_upload_attempts,
        next_retry_at=next_retry_at,
        created_at=created_at,
        last_updated=last_updated,
        stopped_at=None,
    )


@pytest.mark.asyncio
async def test_stop_recording_emits_stop_all_and_sets_stopped(
    state_manager, emitter: Emitter
) -> None:
    manager, store = state_manager
    received: list[str] = []

    def handler(recording_id: str) -> None:
        received.append(recording_id)

    emitter.on(Emitter.STOP_ALL_TRACES_FOR_RECORDING, handler)
    try:
        emitter.emit(Emitter.STOP_RECORDING_REQUESTED, "rec-1")
        await asyncio.sleep(0.2)
        assert store.stopped == ["rec-1"]
        assert received == []

        emitter.emit(Emitter.STOP_RECORDING, "rec-1")
        await asyncio.sleep(0.2)
        assert received == ["rec-1"]
    finally:
        emitter.remove_listener(Emitter.STOP_ALL_TRACES_FOR_RECORDING, handler)


@pytest.mark.asyncio
async def test_stop_recording_requested_marks_pending_stop_report_and_starts_reporting(
    state_manager, emitter: Emitter
) -> None:
    manager, store = state_manager
    reported: list[str] = []

    async def fake_report(recording_id: str) -> bool:
        reported.append(recording_id)
        return True

    manager._report_recording_stop = fake_report  # type: ignore[method-assign]

    emitter.emit(Emitter.STOP_RECORDING_REQUESTED, "rec-stop")
    await asyncio.sleep(0.2)

    assert store.stopped == ["rec-stop"]
    assert (
        store._recording_stop_report_status["rec-stop"]
        == RecordingStopReportStatus.PENDING
    )
    assert reported == ["rec-stop"]


@pytest.mark.asyncio
async def test_stop_recording_requested_offline_leaves_pending_stop_report(
    emitter: Emitter,
) -> None:
    store = FakeStateStore()
    manager = StateManager(store, DaemonConfig(offline=True), emitter=emitter)

    called: list[str] = []

    async def fake_report(recording_id: str) -> bool:
        called.append(recording_id)
        return True

    manager._report_recording_stop = fake_report  # type: ignore[method-assign]

    emitter.emit(Emitter.STOP_RECORDING_REQUESTED, "rec-offline")
    await asyncio.sleep(0.2)

    assert store.stopped == ["rec-offline"]
    assert (
        store._recording_stop_report_status["rec-offline"]
        == RecordingStopReportStatus.PENDING
    )
    assert called == []


@pytest.mark.asyncio
async def test_is_connected_replays_pending_stop_reports(
    state_manager, emitter: Emitter
) -> None:
    manager, store = state_manager
    store._recording_stop_report_status["rec-replay"] = (
        RecordingStopReportStatus.REPORTING
    )
    replayed: list[str] = []

    async def fake_report(recording_id: str) -> bool:
        replayed.append(recording_id)
        return True

    manager._report_recording_stop = fake_report  # type: ignore[method-assign]

    emitter.emit(Emitter.IS_CONNECTED, True)
    await asyncio.sleep(0.2)

    assert (
        store._recording_stop_report_status["rec-replay"]
        == RecordingStopReportStatus.PENDING
    )
    assert replayed == ["rec-replay"]


@pytest.mark.asyncio
async def test_stop_recording_sets_expected_trace_count_once(
    state_manager, emitter: Emitter
) -> None:
    manager, store = state_manager
    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trace_a = _make_trace(
        "trace-a",
        "rec-exp",
        created_at=created_at,
        last_updated=created_at,
    )
    trace_b = _make_trace(
        "trace-b",
        "rec-exp",
        created_at=created_at,
        last_updated=created_at,
    )
    store._traces_by_id["trace-a"] = trace_a
    store._traces_by_id["trace-b"] = trace_b
    store._traces_by_recording["rec-exp"] = [trace_a, trace_b]
    store._expected_trace_count["rec-exp"] = 2

    posted: list[tuple[str, int]] = []

    async def fake_set_expected(recording_id: str, expected_trace_count: int) -> None:
        await store.set_expected_trace_count(recording_id, expected_trace_count)
        await store.mark_expected_trace_count_reported(recording_id)
        posted.append((recording_id, expected_trace_count))

    manager._set_expected_trace_count = fake_set_expected  # type: ignore[method-assign]

    emitter.emit(Emitter.STOP_RECORDING, "rec-exp")
    await asyncio.sleep(0.2)

    assert store._expected_trace_count["rec-exp"] == 2
    assert posted == []

    # A repeated stop should not schedule an expected-count post.
    store._expected_trace_count_reported["rec-exp"] = True
    emitter.emit(Emitter.STOP_RECORDING, "rec-exp")
    await asyncio.sleep(0.2)

    assert posted == []


@pytest.mark.asyncio
async def test_is_connected_backfills_expected_trace_count_for_stopped_recordings(
    state_manager, emitter: Emitter
) -> None:
    manager, store = state_manager
    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)

    missing_a = replace(
        _make_trace(
            "trace-missing-a",
            "rec-missing",
            write_status=TraceWriteStatus.WRITTEN,
            bytes_written=10,
            total_bytes=10,
            created_at=created_at,
            last_updated=created_at,
        ),
        stopped_at=created_at,
    )
    missing_b = replace(
        _make_trace(
            "trace-missing-b",
            "rec-missing",
            write_status=TraceWriteStatus.WRITTEN,
            bytes_written=10,
            total_bytes=10,
            created_at=created_at,
            last_updated=created_at,
        ),
        stopped_at=created_at,
    )
    already_reported = replace(
        _make_trace(
            "trace-reported",
            "rec-reported",
            write_status=TraceWriteStatus.WRITTEN,
            bytes_written=10,
            total_bytes=10,
            created_at=created_at,
            last_updated=created_at,
        ),
        stopped_at=created_at,
    )

    store._traces_by_id[missing_a.trace_id] = missing_a
    store._traces_by_id[missing_b.trace_id] = missing_b
    store._traces_by_id[already_reported.trace_id] = already_reported
    store._traces_by_recording["rec-missing"] = [missing_a, missing_b]
    store._traces_by_recording["rec-reported"] = [already_reported]
    store._expected_trace_count["rec-missing"] = 2
    store._expected_trace_count_reported["rec-reported"] = True
    store._expected_trace_count["rec-reported"] = 1

    posted: list[tuple[str, int]] = []

    async def fake_set_expected(recording_id: str, expected_trace_count: int) -> None:
        await store.set_expected_trace_count(recording_id, expected_trace_count)
        await store.mark_expected_trace_count_reported(recording_id)
        posted.append((recording_id, expected_trace_count))

    manager._set_expected_trace_count = fake_set_expected  # type: ignore[method-assign]

    emitter.emit(Emitter.IS_CONNECTED, True)
    await asyncio.sleep(0.2)

    assert store._expected_trace_count["rec-missing"] == 2
    assert store._expected_trace_count["rec-reported"] == 1
    assert posted == [("rec-missing", 2)]


@pytest.mark.asyncio
async def test_stop_recording_does_not_post_expected_when_local_missing(
    state_manager, emitter: Emitter
) -> None:
    manager, store = state_manager
    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trace_a = _make_trace(
        "trace-missing-local-a",
        "rec-missing-local",
        created_at=created_at,
        last_updated=created_at,
    )
    store._traces_by_id[trace_a.trace_id] = trace_a
    store._traces_by_recording["rec-missing-local"] = [trace_a]

    posted: list[tuple[str, int]] = []

    async def fake_set_expected(recording_id: str, expected_trace_count: int) -> None:
        posted.append((recording_id, expected_trace_count))

    manager._set_expected_trace_count = fake_set_expected  # type: ignore[method-assign]

    emitter.emit(Emitter.STOP_RECORDING, "rec-missing-local")
    await asyncio.sleep(0.2)

    assert posted == []


@pytest.mark.asyncio
async def test_stop_recording_uses_existing_local_expected_trace_count(
    state_manager, emitter: Emitter
) -> None:
    manager, store = state_manager
    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trace_a = _make_trace(
        "trace-local-a",
        "rec-local",
        created_at=created_at,
        last_updated=created_at,
    )
    trace_b = _make_trace(
        "trace-local-b",
        "rec-local",
        created_at=created_at,
        last_updated=created_at,
    )
    store._traces_by_id[trace_a.trace_id] = trace_a
    store._traces_by_id[trace_b.trace_id] = trace_b
    store._traces_by_recording["rec-local"] = [trace_a, trace_b]
    # Simulate local expected count already frozen previously.
    store._expected_trace_count["rec-local"] = 7

    posted: list[tuple[str, int]] = []

    async def fake_set_expected(recording_id: str, expected_trace_count: int) -> None:
        posted.append((recording_id, expected_trace_count))

    manager._set_expected_trace_count = fake_set_expected  # type: ignore[method-assign]

    emitter.emit(Emitter.STOP_RECORDING, "rec-local")
    await asyncio.sleep(0.2)

    assert store._expected_trace_count["rec-local"] == 7
    assert posted == []


@pytest.mark.asyncio
async def test_start_trace_creates_trace(state_manager, emitter: Emitter) -> None:
    """START_TRACE creates trace in INITIALIZING with metadata (waiting for bytes)."""
    _, store = state_manager
    emitter.emit(
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
        "/tmp/trace-1.bin",
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
    assert trace.total_bytes is None
    assert trace.write_status == TraceWriteStatus.INITIALIZING
    assert trace.bytes_written is None  # Not complete yet


@pytest.mark.asyncio
async def test_trace_write_progress_creates_writing_trace(
    state_manager, emitter: Emitter
) -> None:
    """TRACE_WRITE_PROGRESS creates a WRITING trace without finalizing it."""
    _, store = state_manager

    emitter.emit(Emitter.TRACE_WRITE_PROGRESS, "trace-progress-1", "rec-1", 32)
    await asyncio.sleep(0.2)

    trace = store._traces_by_id.get("trace-progress-1")
    assert trace is not None
    assert trace.recording_id == "rec-1"
    assert trace.write_status == TraceWriteStatus.WRITING
    assert trace.bytes_written == 32
    assert trace.total_bytes is None


@pytest.mark.asyncio
async def test_uploaded_bytes_updates_store(state_manager, emitter: Emitter) -> None:
    _, store = state_manager
    emitter.emit(Emitter.UPLOADED_BYTES, "trace-2", 42)
    await asyncio.sleep(0.2)

    assert store.updated_bytes == [("trace-2", 42)]


@pytest.mark.asyncio
async def test_upload_complete_emits_delete_and_deletes(
    state_manager, emitter: Emitter
) -> None:
    _, store = state_manager
    received: list[tuple[str, str, DataType]] = []

    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trace = _make_trace(
        "trace-3",
        "rec-3",
        upload_status=TraceUploadStatus.UPLOADED,
        progress_reported=ProgressReportStatus.REPORTED,
        created_at=created_at,
        last_updated=created_at,
    )
    store._traces_by_id["trace-3"] = trace
    store._traces_by_recording["rec-3"] = [trace]
    store._recording_progress_reported["rec-3"] = True

    def handler(recording_id: str, trace_id: str, data_type: DataType) -> None:
        received.append((recording_id, trace_id, data_type))

    emitter.on(Emitter.DELETE_TRACE, handler)
    try:
        emitter.emit(Emitter.UPLOAD_COMPLETE, "trace-3")
        await asyncio.sleep(0.2)

        assert store.deleted == ["trace-3"]
        assert store.uploaded_count_increments == ["rec-3"]
        assert received == [("rec-3", "trace-3", DataType.CUSTOM_1D)]
    finally:
        emitter.remove_listener(Emitter.DELETE_TRACE, handler)


@pytest.mark.asyncio
async def test_upload_failed_does_not_record_error_in_event_suite(
    state_manager, emitter: Emitter
) -> None:
    """UPLOAD_FAILED should not call record_error."""
    _, store = state_manager

    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trace = _make_trace(
        "trace-4",
        "rec-4",
        created_at=created_at,
        last_updated=created_at,
    )
    store._traces_by_id["trace-4"] = trace
    store._traces_by_recording["rec-4"] = [trace]

    ready_events: list[tuple] = []

    def ready_handler(*args) -> None:
        ready_events.append(args)

    emitter.on(Emitter.READY_FOR_UPLOAD, ready_handler)
    try:
        emitter.emit(
            Emitter.UPLOAD_FAILED,
            "trace-4",
            12,
            TraceErrorCode.NETWORK_ERROR,
            "lost connection",
        )
        await asyncio.sleep(0.2)
    finally:
        emitter.remove_listener(Emitter.READY_FOR_UPLOAD, ready_handler)

    assert store.errors == []
    assert ready_events == []


@pytest.mark.asyncio
async def test_trace_written_emits_ready_for_upload_when_connected(
    state_manager, emitter: Emitter
) -> None:
    """When both START_TRACE and TRACE_WRITTEN arrive, emit READY_FOR_UPLOAD."""
    manager, store = state_manager
    received: list[tuple] = []

    def handler(*args) -> None:
        received.append(args)

    emitter.on(Emitter.READY_FOR_UPLOAD, handler)
    try:
        # First: START_TRACE with metadata (creates PENDING trace)
        emitter.emit(
            Emitter.START_TRACE,
            "trace-5",
            "rec-5",
            DataType.CUSTOM_1D,
            "custom",
            0,
            None,
            None,
            None,
            None,
            "/tmp/trace-5.bin",
        )
        await asyncio.sleep(0.2)

        # No READY_FOR_UPLOAD yet (missing bytes_written)
        assert received == []

        emitter.emit(Emitter.IS_CONNECTED, True)
        await asyncio.sleep(0)
        store._expected_trace_count_reported["rec-5"] = True

        # Second: TRACE_WRITTEN with bytes (completes the trace)
        emitter.emit(Emitter.TRACE_WRITTEN, "trace-5", "rec-5", 64)
        await asyncio.sleep(0.2)

        # Registration is now signaled; upload-ready happens after registration stage.
        assert received == []
    finally:
        emitter.remove_listener(Emitter.READY_FOR_UPLOAD, handler)


@pytest.mark.asyncio
async def test_trace_written_emits_progress_report_with_bounds(
    state_manager, emitter: Emitter
) -> None:
    """Test that progress report is emitted when all traces in recording complete."""
    manager, store = state_manager

    ready_events: list[tuple] = []
    progress_events: list[tuple] = []

    def ready_handler(*args) -> None:
        ready_events.append(args)

    def progress_handler(*args) -> None:
        progress_events.append(args)

    emitter.on(Emitter.READY_FOR_UPLOAD, ready_handler)
    emitter.on(Emitter.PROGRESS_REPORT, progress_handler)
    try:
        # Create two traces in the same recording via START_TRACE
        emitter.emit(
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
            "/tmp/trace-1.bin",
        )
        emitter.emit(
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
            "/tmp/trace-2.bin",
        )
        await asyncio.sleep(0.2)

        store._expected_trace_count_reported["rec-1"] = True
        store._expected_trace_count["rec-1"] = 2
        emitter.emit(Emitter.IS_CONNECTED, True)
        await asyncio.sleep(0)

        # Stop the recording so stopped_at gets set on traces
        emitter.emit(Emitter.STOP_RECORDING_REQUESTED, "rec-1")
        await asyncio.sleep(0.1)

        # No progress report yet - neither trace has bytes_written
        assert len(progress_events) == 0

        # Complete first trace
        emitter.emit(Emitter.TRACE_WRITTEN, "trace-1", "rec-1", 10)
        await asyncio.sleep(0.2)

        # Still no progress report - trace-2 not complete
        assert len(progress_events) == 0
        assert len(ready_events) == 0

        # Complete second trace
        emitter.emit(Emitter.TRACE_WRITTEN, "trace-2", "rec-1", 10)
        await asyncio.sleep(0.3)

        # Now progress report should be emitted
        assert len(ready_events) == 0
        assert len(progress_events) == 1
        recording_id, start_time, end_time, trace_map, total_bytes = progress_events[0]
        assert recording_id == "rec-1"
        assert set(trace_map.keys()) == {"trace-1", "trace-2"}
        assert total_bytes == 20
        # Verify time bounds are reasonable (both created close to now)
        assert start_time <= end_time
    finally:
        emitter.remove_listener(Emitter.READY_FOR_UPLOAD, ready_handler)
        emitter.remove_listener(Emitter.PROGRESS_REPORT, progress_handler)


@pytest.mark.asyncio
async def test_trace_written_waits_for_all_traces_before_progress_report(
    state_manager, emitter: Emitter
) -> None:
    """Test that progress report waits for all traces before emitting."""
    _, store = state_manager
    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    updated_at = datetime(2024, 1, 2, tzinfo=timezone.utc)
    trace_written = _make_trace(
        "trace-written",
        "rec-1",
        write_status=TraceWriteStatus.WRITTEN,
        bytes_written=8,
        total_bytes=8,
        created_at=created_at,
        last_updated=updated_at,
    )
    trace_pending = _make_trace(
        "trace-pending",
        "rec-1",
        write_status=TraceWriteStatus.INITIALIZING,
        bytes_written=None,
        total_bytes=8,
        created_at=created_at,
        last_updated=updated_at,
    )
    store._traces_by_id["trace-written"] = trace_written
    store._traces_by_id["trace-pending"] = trace_pending
    store._traces_by_recording["rec-1"] = [trace_written, trace_pending]
    store._expected_trace_count["rec-1"] = 2
    store._expected_trace_count_reported["rec-1"] = True

    progress_events: list[tuple] = []

    def progress_handler(*args) -> None:
        progress_events.append(args)

    emitter.on(Emitter.PROGRESS_REPORT, progress_handler)
    try:
        emitter.emit(Emitter.TRACE_WRITTEN, "trace-written", "rec-1", 8)
        await asyncio.sleep(0.3)

        assert progress_events == []
    finally:
        emitter.remove_listener(Emitter.PROGRESS_REPORT, progress_handler)


@pytest.mark.asyncio
async def test_recording_completion_isolated_across_recordings(
    state_manager, emitter: Emitter
) -> None:
    """Test that recordings complete independently of each other."""
    _, store = state_manager
    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    updated_at = datetime(2024, 1, 2, tzinfo=timezone.utc)
    trace_a = _make_trace(
        "trace-a",
        "rec-a",
        write_status=TraceWriteStatus.INITIALIZING,
        bytes_written=None,
        created_at=created_at,
        last_updated=updated_at,
    )
    trace_b1 = _make_trace(
        "trace-b1",
        "rec-b",
        write_status=TraceWriteStatus.WRITTEN,
        bytes_written=10,
        total_bytes=10,
        created_at=created_at,
        last_updated=updated_at,
    )
    trace_b2 = _make_trace(
        "trace-b2",
        "rec-b",
        write_status=TraceWriteStatus.INITIALIZING,
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
    store._expected_trace_count_reported["rec-b"] = True
    store._expected_trace_count["rec-b"] = 2

    progress_events: list[tuple] = []

    def progress_handler(*args) -> None:
        progress_events.append(args)

    emitter.on(Emitter.PROGRESS_REPORT, progress_handler)
    try:
        emitter.emit(Emitter.IS_CONNECTED, True)
        await asyncio.sleep(0.1)

        # Stop rec-b so stopped_at gets set on its traces
        emitter.emit(Emitter.STOP_RECORDING_REQUESTED, "rec-b")
        await asyncio.sleep(0.1)

        emitter.emit(Emitter.TRACE_WRITTEN, "trace-b2", "rec-b", 10)
        await asyncio.sleep(0.3)

        assert len(progress_events) == 1
        recording_id, _, _, trace_map, _ = progress_events[0]
        assert recording_id == "rec-b"
        assert set(trace_map.keys()) == {"trace-b1", "trace-b2"}
    finally:
        emitter.remove_listener(Emitter.PROGRESS_REPORT, progress_handler)


@pytest.mark.asyncio
async def test_upload_failed_does_not_block_other_recordings(
    state_manager, emitter: Emitter
) -> None:
    """Test that upload failure in one recording doesn't block others."""
    _, store = state_manager
    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    updated_at = datetime(2024, 1, 2, tzinfo=timezone.utc)
    trace_a = _make_trace(
        "trace-a",
        "rec-a",
        write_status=TraceWriteStatus.INITIALIZING,
        bytes_written=None,
        created_at=created_at,
        last_updated=updated_at,
    )
    trace_b = _make_trace(
        "trace-b",
        "rec-b",
        write_status=TraceWriteStatus.INITIALIZING,
        bytes_written=None,
        created_at=created_at,
        last_updated=updated_at,
    )
    store._traces_by_id["trace-a"] = trace_a
    store._traces_by_id["trace-b"] = trace_b
    store._traces_by_recording["rec-a"] = [trace_a]
    store._traces_by_recording["rec-b"] = [trace_b]
    store._expected_trace_count_reported["rec-b"] = True

    ready_events: list[tuple] = []

    def ready_handler(*args) -> None:
        ready_events.append(args)

    emitter.on(Emitter.READY_FOR_UPLOAD, ready_handler)
    try:
        emitter.emit(Emitter.IS_CONNECTED, True)
        await asyncio.sleep(0.1)

        emitter.emit(
            Emitter.UPLOAD_FAILED,
            "trace-a",
            0,
            TraceErrorCode.DISK_FULL,
            "disk full",
        )
        await asyncio.sleep(0.1)

        emitter.emit(Emitter.TRACE_WRITTEN, "trace-b", "rec-b", 10)
        await asyncio.sleep(0.3)

        assert ready_events == []
    finally:
        emitter.remove_listener(Emitter.READY_FOR_UPLOAD, ready_handler)


@pytest.mark.asyncio
async def test_is_connected_emits_ready_for_due_retries_event_suite(
    state_manager, emitter: Emitter
) -> None:
    """When we come online, traces that are eligible should be re-queued."""
    _, store = state_manager

    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    updated_at = datetime(2024, 1, 2, tzinfo=timezone.utc)

    due_trace = _make_trace(
        "trace-due",
        "rec-due",
        write_status=TraceWriteStatus.WRITTEN,
        bytes_written=10,
        total_bytes=10,
        bytes_uploaded=3,
        num_upload_attempts=1,
        next_retry_at=datetime(2024, 1, 1, 0, 0, 0),
        created_at=created_at,
        last_updated=updated_at,
    )

    store._traces_by_id["trace-due"] = due_trace
    store._traces_by_recording["rec-due"] = [due_trace]

    # IMPORTANT: StateManager likely queries find_ready_traces()
    store.ready_traces = [due_trace]

    ready_events: list[tuple] = []
    ready_evt = asyncio.Event()

    def ready_handler(*args) -> None:
        ready_events.append(args)
        if args[0] == "trace-due":
            ready_evt.set()

    emitter.on(Emitter.READY_FOR_UPLOAD, ready_handler)
    try:
        emitter.emit(Emitter.IS_CONNECTED, True)
        await asyncio.wait_for(ready_evt.wait(), timeout=2.0)

        assert len(ready_events) == 1
        assert ready_events[0][:2] == ("trace-due", "rec-due")
    finally:
        emitter.remove_listener(Emitter.READY_FOR_UPLOAD, ready_handler)


@pytest.mark.asyncio
async def test_is_connected_emits_ready_even_when_next_retry_at_in_future_event_suite(
    state_manager, emitter: Emitter
) -> None:
    """IS_CONNECTED emits READY for traces returned by find_ready_traces()."""
    _, store = state_manager

    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    updated_at = datetime(2024, 1, 2, tzinfo=timezone.utc)

    trace = _make_trace(
        "trace-not-due",
        "rec-not-due",
        write_status=TraceWriteStatus.WRITTEN,
        bytes_written=10,
        total_bytes=10,
        bytes_uploaded=1,
        num_upload_attempts=1,
        next_retry_at=datetime(2099, 1, 1, 0, 0, 0),
        created_at=created_at,
        last_updated=updated_at,
    )

    store._traces_by_id["trace-not-due"] = trace
    store._traces_by_recording["rec-not-due"] = [trace]
    store.ready_traces = [trace]

    ready_events: list[tuple] = []
    ready_evt = asyncio.Event()

    def ready_handler(*args) -> None:
        ready_events.append(args)
        if args[0] == "trace-not-due":
            ready_evt.set()

    emitter.on(Emitter.READY_FOR_UPLOAD, ready_handler)
    try:
        emitter.emit(Emitter.IS_CONNECTED, True)
        await asyncio.wait_for(ready_evt.wait(), timeout=2.0)
    finally:
        emitter.remove_listener(Emitter.READY_FOR_UPLOAD, ready_handler)

    assert len(ready_events) == 1
    assert ready_events[0][0] == "trace-not-due"


@pytest.mark.asyncio
async def test_is_connected_reconciles_failed_trace_with_payload_and_emits_ready(
    state_manager,
    emitter: Emitter,
) -> None:
    _, store = state_manager

    created_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=5)
    failed_trace = _make_trace(
        "trace-failed-recoverable",
        "rec-failed-recoverable",
        write_status=TraceWriteStatus.WRITTEN,
        registration_status=TraceRegistrationStatus.REGISTERED,
        upload_status=TraceUploadStatus.FAILED,
        bytes_written=10,
        total_bytes=10,
        bytes_uploaded=4,
        num_upload_attempts=1,
        created_at=created_at,
        last_updated=created_at,
    )

    store._traces_by_id[failed_trace.trace_id] = failed_trace
    store._traces_by_recording[failed_trace.recording_id] = [failed_trace]

    ready_events: list[tuple] = []
    ready_event = asyncio.Event()

    def ready_handler(*args) -> None:
        ready_events.append(args)
        if args[0] == "trace-failed-recoverable":
            ready_event.set()

    emitter.on(Emitter.READY_FOR_UPLOAD, ready_handler)
    try:
        emitter.emit(Emitter.IS_CONNECTED, True)
        await asyncio.wait_for(ready_event.wait(), timeout=2.0)
    finally:
        emitter.remove_listener(Emitter.READY_FOR_UPLOAD, ready_handler)

    recovered = store._traces_by_id["trace-failed-recoverable"]
    assert recovered.upload_status == TraceUploadStatus.PENDING
    assert recovered.bytes_uploaded == 0
    assert recovered.num_upload_attempts == 0
    assert recovered.error_code is None
    assert recovered.error_message is None

    assert len(ready_events) == 1
    assert ready_events[0][:2] == (
        "trace-failed-recoverable",
        "rec-failed-recoverable",
    )


@pytest.mark.asyncio
async def test_is_connected_deletes_failed_trace_when_payload_missing(
    state_manager,
    emitter: Emitter,
) -> None:
    _, store = state_manager

    created_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=5)
    unrecoverable = _make_trace(
        "trace-failed-missing-payload",
        "rec-failed-missing-payload",
        write_status=TraceWriteStatus.WRITTEN,
        registration_status=TraceRegistrationStatus.REGISTERED,
        upload_status=TraceUploadStatus.FAILED,
        bytes_written=0,
        total_bytes=0,
        bytes_uploaded=0,
        num_upload_attempts=1,
        created_at=created_at,
        last_updated=created_at,
    )
    unrecoverable = replace(unrecoverable, path=None)

    store._traces_by_id[unrecoverable.trace_id] = unrecoverable
    store._traces_by_recording[unrecoverable.recording_id] = [unrecoverable]

    ready_events: list[tuple] = []

    def ready_handler(*args) -> None:
        ready_events.append(args)

    emitter.on(Emitter.READY_FOR_UPLOAD, ready_handler)
    try:
        emitter.emit(Emitter.IS_CONNECTED, True)
        await asyncio.sleep(0.2)
    finally:
        emitter.remove_listener(Emitter.READY_FOR_UPLOAD, ready_handler)

    assert "trace-failed-missing-payload" in store.deleted
    assert ready_events == []


@pytest.mark.asyncio
async def test_upload_failed_does_not_record_error_and_does_not_emit_ready_event_suite(
    state_manager, emitter: Emitter
) -> None:
    """UPLOAD_FAILED should not call record_error and not emit READY_FOR_UPLOAD."""
    _, store = state_manager

    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trace = _make_trace(
        "trace-fail-payload",
        "rec-fail-payload",
        created_at=created_at,
        last_updated=created_at,
    )
    store._traces_by_id["trace-fail-payload"] = trace
    store._traces_by_recording["rec-fail-payload"] = [trace]

    ready_events: list[tuple] = []

    def ready_handler(*args) -> None:
        ready_events.append(args)

    emitter.on(Emitter.READY_FOR_UPLOAD, ready_handler)
    try:
        emitter.emit(
            Emitter.UPLOAD_FAILED,
            "trace-fail-payload",
            7,
            TraceErrorCode.NETWORK_ERROR,
            "net down",
        )
        await asyncio.sleep(0.2)
    finally:
        emitter.remove_listener(Emitter.READY_FOR_UPLOAD, ready_handler)

    assert store.errors == []
    assert ready_events == []


@pytest.mark.asyncio
async def test_mark_traces_registration_failed_requeues_traces_for_registration(
    state_manager,
) -> None:
    manager, store = state_manager

    created_at = datetime.now(timezone.utc)

    trace = _make_trace(
        "trace-reg-fail",
        "rec-reg-fail",
        write_status=TraceWriteStatus.WRITTEN,
        registration_status=TraceRegistrationStatus.REGISTERING,
        upload_status=TraceUploadStatus.PENDING,
        bytes_written=10,
        total_bytes=10,
        created_at=created_at,
        last_updated=created_at,
    )

    store._traces_by_id[trace.trace_id] = trace
    store._traces_by_recording[trace.recording_id] = [trace]

    await manager.mark_traces_registration_failed(
        ["trace-reg-fail"],
        "backend registration failed",
    )

    updated = store._traces_by_id["trace-reg-fail"]
    assert updated.registration_status == TraceRegistrationStatus.PENDING

    claimed = await manager.claim_traces_for_registration(limit=10, max_wait_s=0)
    assert [candidate.trace_id for candidate in claimed] == ["trace-reg-fail"]
