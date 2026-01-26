"""State manager facade for trace state operations."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

from neuracore.data_daemon.event_emitter import Emitter, get_emitter
from neuracore.data_daemon.models import (
    DataType,
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
        self._is_connected = False

        self._emitter = get_emitter()

        # From RDEM
        self._emitter.on(Emitter.TRACE_WRITTEN, self._handle_trace_written)
        self._emitter.on(Emitter.START_TRACE, self.create_trace)

        # From uploader
        self._emitter.on(Emitter.UPLOAD_COMPLETE, self.handle_upload_complete)
        self._emitter.on(Emitter.UPLOADED_BYTES, self.update_bytes_uploaded)
        self._emitter.on(Emitter.UPLOAD_FAILED, self.handle_upload_failed)

        # From the data bridge
        self._emitter.on(Emitter.STOP_RECORDING, self.handle_stop_recording)

        # From connection manager
        self._emitter.on(Emitter.IS_CONNECTED, self.handle_is_connected)

        # From progress report service
        self._emitter.on(Emitter.PROGRESS_REPORTED, self.mark_progress_as_reported)
        self._emitter.on(
            Emitter.PROGRESS_REPORT_FAILED, self.handle_progress_report_error
        )

    async def create_trace(
        self,
        trace_id: str,
        recording_id: str,
        data_type: DataType,
        data_type_name: str,
        robot_instance: int,
        dataset_id: str | None = None,
        dataset_name: str | None = None,
        robot_name: str | None = None,
        robot_id: str | None = None,
        *,
        path: str,
        total_bytes: int | None = None,
    ) -> None:
        """Create a trace record.

        This function is called when a producer wants to create a new trace.
        It will create or update a trace record in the database and emit an event
        to the uploader to trigger an upload.

        Args:
            trace_id (str): unique identifier for the trace.
            recording_id (str): unique identifier for the recording.
            data_type (DataType): type of data being recorded.
            data_type_name (str | None): name of the data type.
            dataset_id (str | None): unique identifier for the dataset.
            dataset_name (str | None): name of the dataset.
            robot_name (str | None): name of the robot.
            robot_id (str | None): unique identifier for the robot.
            robot_instance (int): instance of the robot.
            path (str): path to the trace file.
            total_bytes (int | None): total number of bytes in the trace file.
        """
        await self._store.create_trace(
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
        if not self._is_connected:
            return
        traces = await self._store.find_traces_by_recording_id(recording_id)
        await self._emit_progress_report_if_recording_stopped(traces)

    async def handle_is_connected(self, is_connected: bool) -> None:
        """Handle a connection status event from the data bridge.

        If the connection is lost, do nothing. If the connection is established,
        find all ready traces and sort them by their created time. Then, emit
        READY_FOR_UPLOAD events for each trace. Finally, find all traces that
        have not been marked as progress-reported and check if all traces
        are written for each recording. If all traces are written, emit a
        PROGRESS_REPORT event for the recording.
        """
        self._is_connected = is_connected
        if not is_connected:
            return

        # Find/sort ready traces end trigger upload
        traces = await self._store.find_ready_traces()
        traces.sort(
            key=lambda trace: (
                trace.status != TraceStatus.UPLOADING,
                trace.created_at,
            )
        )
        for trace in traces:
            self._emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                trace.trace_id,
                trace.recording_id,
                trace.path,
                trace.data_type,
                trace.data_type_name,
                trace.bytes_uploaded,
            )
        # Find if all traces are written for a recording to trigger progress report
        unreported_traces = await self._store.find_unreported_traces()
        traces_by_recording: dict[str, list[TraceRecord]] = {}
        for trace in unreported_traces:
            traces_by_recording.setdefault(trace.recording_id, []).append(trace)
        for _, recording_traces in traces_by_recording.items():
            if all(trace.ready_for_upload == 1 for trace in recording_traces):
                await self._emit_progress_report_if_recording_stopped(recording_traces)

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
        """Handle a trace written event from a producer.

        This function is called when a producer completes writing a trace.
        It updates the trace record in the database and emits an event to
        the uploader to trigger an upload.

        Additionally, it will trigger a progress report event if all traces
        for the recording are written.
        """
        # Update db
        await self._store.mark_trace_as_written(trace_id, bytes_written)

        trace_record = await self._store.get_trace(trace_id)

        if not trace_record:
            logger.warning("Trace record not found: %s", trace_id)
            return

        if not self._is_connected:
            return

        # Emit event to uploader
        self._emitter.emit(
            Emitter.READY_FOR_UPLOAD,
            trace_id,
            trace_record.recording_id,
            trace_record.path,
            trace_record.data_type,
            trace_record.data_type_name,
            trace_record.bytes_uploaded,
        )

        traces = await self._store.find_traces_by_recording_id(recording_id)
        await self._emit_progress_report_if_recording_stopped(traces)

    async def _emit_progress_report_if_recording_stopped(
        self, traces: list[TraceRecord]
    ) -> None:
        if not traces:
            return
        if any(trace.progress_reported == 1 for trace in traces):
            return
        if not all(
            trace.status == TraceStatus.WRITTEN
            and trace.total_bytes == trace.bytes_written
            and trace.ready_for_upload == 1
            for trace in traces
        ):
            return
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

    async def claim_ready_traces(self, limit: int = 50) -> list[Mapping[str, Any]]:
        """Claim ready traces for upload and mark them in-progress."""
        return await self._store.claim_ready_traces(limit)

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
        """Handle an upload failed event from an uploader."""
        await self._record_error(
            trace_id,
            error_code=error_code,
            error_message=error_message,
            status=status,
        )
        await self.update_bytes_uploaded(trace_id, bytes_uploaded)

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
