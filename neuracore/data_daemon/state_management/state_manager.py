"""State manager facade for trace state operations."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from neuracore.data_daemon.const import (
    UPLOAD_MAX_RETRIES,
    UPLOAD_RETRY_BASE_SECONDS,
    UPLOAD_RETRY_MAX_SECONDS,
)
from neuracore.data_daemon.event_emitter import Emitter, get_emitter
from neuracore.data_daemon.models import (
    DataType,
    ProgressReportStatus,
    TraceErrorCode,
    TraceRecord,
    TraceStatus,
)

from .state_store import StateStore

logger = logging.getLogger(__name__)


class StateManager:
    """Domain-facing API for trace state."""

    def __init__(self, store: StateStore) -> None:
        """Initialize with a persistence backend."""
        self._store = store
        self._reporting_recordings: set[str] = set()

        self._emitter = get_emitter()

        self._emitter.on(Emitter.TRACE_WRITTEN, self._handle_trace_written)
        self._emitter.on(Emitter.START_TRACE, self._handle_start_trace)
        self._emitter.on(Emitter.UPLOAD_COMPLETE, self.handle_upload_complete)
        self._emitter.on(Emitter.UPLOADED_BYTES, self.update_bytes_uploaded)
        self._emitter.on(Emitter.UPLOAD_FAILED, self.handle_upload_failed)
        self._emitter.on(Emitter.STOP_RECORDING, self.handle_stop_recording)
        self._emitter.on(Emitter.IS_CONNECTED, self.handle_is_connected)
        self._emitter.on(Emitter.PROGRESS_REPORTED, self.mark_progress_as_reported)
        self._emitter.on(
            Emitter.PROGRESS_REPORT_FAILED, self.handle_progress_report_error
        )

    async def _handle_start_trace(
        self,
        trace_id: str,
        recording_id: str,
        data_type: DataType,
        data_type_name: str,
        robot_instance: int,
        dataset_id: str | None,
        dataset_name: str | None,
        robot_name: str | None,
        robot_id: str | None,
        path: str,
        total_bytes: int | None = None,
    ) -> None:
        """Handle START_TRACE event - upsert trace metadata.

        State transitions:
        - If trace doesn't exist: creates with INITIALIZING status
        - If trace exists with PENDING_BYTES: transitions to WRITTEN
        - If status is WRITTEN after upsert: finalizes trace for upload
        """
        trace = await self._store.upsert_trace_metadata(
            trace_id=trace_id,
            recording_id=recording_id,
            data_type=data_type,
            data_type_name=data_type_name,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            robot_name=robot_name,
            robot_id=robot_id,
            robot_instance=robot_instance,
            path=path,
            total_bytes=total_bytes,
        )
        if trace.status == TraceStatus.WRITTEN:
            await self._finalize_trace(trace)

    async def handle_stop_recording(self, recording_id: str) -> None:
        """Handle a stop recording event from the data bridge.

        This function is called when the data bridge wants to stop all traces
        for a given recording. It will emit an event to the RDEM to
        stop all traces for the recording.

        Args:
            recording_id (str): unique identifier for the recording.
        """
        self._emitter.emit(Emitter.STOP_ALL_TRACES_FOR_RECORDING, recording_id)
        await self._store.set_stopped_ats(recording_id)

    async def handle_is_connected(self, is_connected: bool) -> None:
        """Handle a connection status event from the data bridge."""
        if not is_connected:
            return

        traces = await self._store.find_ready_traces()
        for trace in traces:
            await self._emit_ready_for_upload_from_trace(trace)

        due = await self._store.find_traces_ready_for_reupload()
        for trace in due:
            await self._emit_ready_for_upload_from_trace(trace)

    async def _emit_ready_for_upload_from_trace(self, trace: TraceRecord) -> None:
        await self._store.update_status(trace.trace_id, TraceStatus.UPLOADING)
        self._emitter.emit(
            Emitter.READY_FOR_UPLOAD,
            trace.trace_id,
            trace.recording_id,
            trace.path,
            trace.data_type,
            trace.data_type_name,
            trace.bytes_uploaded,
        )

    async def handle_upload_complete(self, trace_id: str) -> None:
        """Handle an upload complete event from an uploader.

        This function is called when an uploader completes an upload.
        It emits an event to delete the trace from the database and then
        deletes the trace record from the database.
        """
        # Trigger file deletion
        trace_record = await self._store.get_trace(trace_id)
        if trace_record is None:
            logger.warning("Trace record not found: %s", trace_id)
            return

        self._emitter.emit(
            Emitter.DELETE_TRACE,
            trace_record.recording_id,
            trace_id,
            trace_record.data_type,
        )
        # Delete db entry
        await self._store.delete_trace(trace_id)

    async def _handle_trace_written(
        self, trace_id: str, recording_id: str, bytes_written: int
    ) -> None:
        """Handle TRACE_WRITTEN event - upsert trace bytes.

        State transitions:
        - If trace doesn't exist: creates with PENDING_BYTES status
        - If trace exists with INITIALIZING: transitions to WRITTEN
        - If status is WRITTEN after upsert: finalizes trace for upload
        """
        trace = await self._store.upsert_trace_bytes(
            trace_id=trace_id,
            recording_id=recording_id,
            bytes_written=bytes_written,
        )
        if trace.status == TraceStatus.WRITTEN:
            await self._finalize_trace(trace)

    async def _finalize_trace(self, trace: TraceRecord) -> None:
        """Finalize a WRITTEN trace and emit READY_FOR_UPLOAD.

        Transitions trace from WRITTEN to UPLOADING and emits
        READY_FOR_UPLOAD event for the uploader. Uses atomic CAS
        to ensure only one handler finalizes when racing.
        """
        transitioned = await self._store.update_status(
            trace.trace_id, TraceStatus.UPLOADING
        )
        if not transitioned:
            return

        self._emitter.emit(
            Emitter.READY_FOR_UPLOAD,
            trace.trace_id,
            trace.recording_id,
            trace.path,
            trace.data_type,
            trace.data_type_name,
            trace.bytes_uploaded,
        )

        traces = await self._store.find_traces_by_recording_id(trace.recording_id)
        await self._emit_progress_report_if_recording_stopped(traces)

    async def _emit_progress_report_if_recording_stopped(
        self, traces: list[TraceRecord]
    ) -> None:
        if not traces:
            return

        recording_id = traces[0].recording_id

        if recording_id in self._reporting_recordings:
            return

        if any(
            trace.progress_reported == ProgressReportStatus.REPORTED for trace in traces
        ):
            return

        if not all(
            trace.data_type is not None and trace.bytes_written is not None
            for trace in traces
        ):
            return

        self._reporting_recordings.add(recording_id)
        start_time, end_time = self._find_recording_start_and_end(traces)
        asyncio.create_task(self._emit_progress_report(start_time, end_time, traces))

    async def _emit_progress_report(
        self, start_time: float, end_time: float, traces: list[TraceRecord]
    ) -> None:
        """Emit progress report event asynchronously."""
        await asyncio.sleep(0)
        self._emitter.emit(Emitter.PROGRESS_REPORT, start_time, end_time, traces)

    async def mark_progress_as_reported(self, recording_id: str) -> None:
        """Mark a recording as progress-reported.

        Args:
            recording_id (str): unique identifier for the recording.
        """
        self._reporting_recordings.discard(recording_id)
        await self._store.mark_recording_reported(recording_id)

    def _find_recording_start_and_end(
        self, traces: list[TraceRecord]
    ) -> tuple[float, float]:
        earliest_start = traces[0].created_at
        latest_end = traces[0].last_updated
        for trace in traces[1:]:
            if trace.created_at < earliest_start:
                earliest_start = trace.created_at
            if trace.last_updated > latest_end:
                latest_end = trace.last_updated
        return earliest_start.timestamp(), latest_end.timestamp()

    async def update_bytes_uploaded(self, trace_id: str, bytes_uploaded: int) -> None:
        """Increment uploaded byte count for a trace."""
        await self._store.update_bytes_uploaded(trace_id, bytes_uploaded)

    async def update_status(
        self, trace_id: str, status: TraceStatus, *, error_message: str | None = None
    ) -> None:
        """Update the status and optional error message for a trace."""
        await self._store.update_status(
            trace_id,
            status,
            error_message=error_message,
        )

    async def handle_upload_failed(
        self,
        trace_id: str,
        bytes_uploaded: int,
        status: TraceStatus,
        error_code: TraceErrorCode,
        error_message: str,
    ) -> None:
        """Handle an upload failed event from an uploader.

        Args:
            trace_id: unique identifier for the trace.
            bytes_uploaded: latest uploaded byte count for this trace.
            status: trace status reported at failure time.
            error_code: error code describing the failure type.
            error_message: human readable failure message.
        """
        await self.update_bytes_uploaded(trace_id, bytes_uploaded)

        trace = await self._store.get_trace(trace_id)
        if trace is None:
            logger.warning("Trace record not found: %s", trace_id)
            return

        next_attempt = int(getattr(trace, "num_upload_attempts", 0)) + 1

        if next_attempt >= UPLOAD_MAX_RETRIES:
            await self._store.mark_retry_exhausted(
                trace_id,
                error_code=error_code,
                error_message=error_message,
            )
            return

        backoff = UPLOAD_RETRY_BASE_SECONDS * (2 ** (next_attempt - 1))
        if backoff > UPLOAD_RETRY_MAX_SECONDS:
            backoff = UPLOAD_RETRY_MAX_SECONDS

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        next_retry_at = now + timedelta(seconds=float(backoff))

        await self._store.schedule_retry(
            trace_id,
            next_retry_at=next_retry_at,
            error_code=error_code,
            error_message=error_message,
        )

        loop = asyncio.get_running_loop()
        loop.call_later(
            float(backoff),
            lambda: asyncio.create_task(self._retry_emit(trace_id)),
        )

    async def _retry_emit(self, trace_id: str) -> None:
        """Emit READY_FOR_UPLOAD when a trace is due for retry."""
        trace = await self._store.get_trace(trace_id)
        if trace is None:
            return
        if trace.status != TraceStatus.WRITTEN:
            return
        if trace.next_retry_at is not None:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            if trace.next_retry_at > now:
                delay = (trace.next_retry_at - now).total_seconds()
                loop = asyncio.get_running_loop()
                loop.call_later(
                    float(delay),
                    lambda: asyncio.create_task(self._retry_emit(trace_id)),
                )
                return
        await self._emit_ready_for_upload_from_trace(trace)

    async def handle_progress_report_error(
        self, recording_id: str, error_message: str
    ) -> None:
        """Handle a progress report error event from an uploader.

        Record an error for each trace associated with the recording.

        Args:
            recording_id (str): Unique identifier for the recording.
            error_message (str): Error message associated with
            the progress report error.
        """
        self._reporting_recordings.discard(recording_id)
        traces = await self._store.find_traces_by_recording_id(recording_id)
        if not traces:
            return
        for trace in traces:
            await self._record_error(
                trace.trace_id,
                error_message,
                error_code=TraceErrorCode.PROGRESS_REPORT_ERROR,
                status=TraceStatus.FAILED,
            )

    async def _record_error(
        self,
        trace_id: str,
        error_message: str,
        error_code: TraceErrorCode | None = TraceErrorCode.UNKNOWN,
        status: TraceStatus = TraceStatus.FAILED,
    ) -> None:
        """Record an error for a trace.

        Args:
            trace_id: str
                Trace ID of the trace to record the error for.
            error_message: str
                Error message of the error.
            error_code: TraceErrorCode | None, optional
                Error code of the error, by default None.
            status: TraceStatus, optional
                Status to set for the trace after recording the error,
                by default TraceStatus.FAILED.
        """
        await self._store.record_error(
            trace_id,
            error_message,
            error_code,
            status,
        )

    async def delete_trace(self, trace_id: str) -> None:
        """Delete a trace record."""
        await self._store.delete_trace(trace_id)
