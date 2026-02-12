"""State manager facade for trace state operations."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy.exc import OperationalError

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

    _TRACE_WRITTEN_LOCK_MAX_RETRIES = 5
    _TRACE_WRITTEN_LOCK_BASE_DELAY_S = 0.1
    _TRACE_WRITTEN_LOCK_MAX_DELAY_S = 2.0
    _START_TRACE_LOCK_MAX_RETRIES = 5
    _START_TRACE_LOCK_BASE_DELAY_S = 0.1
    _START_TRACE_LOCK_MAX_DELAY_S = 2.0
    _FAILED_TRACE_MAX_AGE_S = 60 * 60 * 4  # 4 hours

    def __init__(self, store: StateStore) -> None:
        """Initialize with a persistence backend."""
        self._store = store
        self._reporting_recordings: set[str] = set()
        self._trace_written_retry_queue: asyncio.Queue[tuple[str, str, int, int]] = (
            asyncio.Queue()
        )
        self._trace_written_retry_task: asyncio.Task | None = None
        self._start_trace_retry_queue: asyncio.Queue[
            tuple[
                str,
                str,
                DataType,
                str,
                int,
                str | None,
                str | None,
                str | None,
                str | None,
                str,
                int,
            ]
        ] = asyncio.Queue()
        self._start_trace_retry_task: asyncio.Task | None = None
        self._retry_emit_handles: dict[str, asyncio.Handle] = {}

        self._emitter = get_emitter()

        self._emitter.on(Emitter.START_TRACE, self._handle_start_trace)
        self._emitter.on(Emitter.TRACE_WRITTEN, self._handle_trace_written)
        self._emitter.on(Emitter.UPLOAD_STARTED, self.handle_upload_started)
        self._emitter.on(Emitter.UPLOADED_BYTES, self.update_bytes_uploaded)
        self._emitter.on(Emitter.UPLOAD_COMPLETE, self.handle_upload_complete)
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
    ) -> None:
        """Handle START_TRACE event - upsert trace metadata.

        State transitions:
        - If trace doesn't exist: creates with INITIALIZING status
        - If trace exists with PENDING_METADATA: transitions to WRITTEN
        - If status is WRITTEN after upsert: finalizes trace for upload
        """
        logger.info(
            "START_TRACE received: trace=%s recording=%s data_type=%s path=%s",
            trace_id,
            recording_id,
            data_type,
            path,
        )
        try:
            await self._process_start_trace(
                trace_id,
                recording_id,
                data_type,
                data_type_name,
                robot_instance,
                dataset_id,
                dataset_name,
                robot_name,
                robot_id,
                path,
            )
        except OperationalError as exc:
            if "database is locked" not in str(exc).lower():
                raise
            logger.warning(
                "START_TRACE DB locked (trace=%s, recording=%s). Queueing retry.",
                trace_id,
                recording_id,
            )
            await self._enqueue_start_trace_retry(
                trace_id,
                recording_id,
                data_type,
                data_type_name,
                robot_instance,
                dataset_id,
                dataset_name,
                robot_name,
                robot_id,
                path,
            )

    async def _process_start_trace(
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
    ) -> None:
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
        )
        logger.info(
            "Trace metadata upsert complete: trace=%s status=%s bytes_written=%s",
            trace_id,
            trace.status,
            trace.bytes_written,
        )
        if trace.status == TraceStatus.WRITTEN:
            logger.info(
                "Trace %s is WRITTEN after metadata upsert, finalizing", trace_id
            )
            await self._finalize_trace(trace)

    async def _enqueue_start_trace_retry(
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
    ) -> None:
        self._ensure_start_trace_retry_worker()
        await self._start_trace_retry_queue.put((
            trace_id,
            recording_id,
            data_type,
            data_type_name,
            robot_instance,
            dataset_id,
            dataset_name,
            robot_name,
            robot_id,
            path,
            1,
        ))

    def _ensure_start_trace_retry_worker(self) -> None:
        if self._start_trace_retry_task is not None:
            return
        loop = asyncio.get_running_loop()
        self._start_trace_retry_task = loop.create_task(
            self._start_trace_retry_worker()
        )

    async def _start_trace_retry_worker(self) -> None:
        while True:
            (
                trace_id,
                recording_id,
                data_type,
                data_type_name,
                robot_instance,
                dataset_id,
                dataset_name,
                robot_name,
                robot_id,
                path,
                attempt,
            ) = await self._start_trace_retry_queue.get()
            try:
                await self._process_start_trace(
                    trace_id,
                    recording_id,
                    data_type,
                    data_type_name,
                    robot_instance,
                    dataset_id,
                    dataset_name,
                    robot_name,
                    robot_id,
                    path,
                )
            except OperationalError as exc:
                if (
                    "database is locked" in str(exc).lower()
                    and attempt < self._START_TRACE_LOCK_MAX_RETRIES
                ):
                    delay = min(
                        self._START_TRACE_LOCK_BASE_DELAY_S * (2 ** (attempt - 1)),
                        self._START_TRACE_LOCK_MAX_DELAY_S,
                    )
                    logger.warning(
                        "START_TRACE retry %s/%s after DB lock (trace=%s) in %.2fs",
                        attempt,
                        self._START_TRACE_LOCK_MAX_RETRIES,
                        trace_id,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    await self._start_trace_retry_queue.put((
                        trace_id,
                        recording_id,
                        data_type,
                        data_type_name,
                        robot_instance,
                        dataset_id,
                        dataset_name,
                        robot_name,
                        robot_id,
                        path,
                        attempt + 1,
                    ))
                else:
                    logger.exception(
                        "START_TRACE failed after retries (trace=%s)", trace_id
                    )
            finally:
                self._start_trace_retry_queue.task_done()

    async def handle_stop_recording(self, recording_id: str) -> None:
        """Handle a stop recording event from the data bridge.

        This function is called when the data bridge wants to stop all traces
        for a given recording. It will emit an event to the RDEM to
        stop all traces for the recording.

        Args:
            recording_id (str): unique identifier for the recording.
        """
        logger.info("STOP_RECORDING received (recording_id=%s)", recording_id)
        self._emitter.emit(Emitter.STOP_ALL_TRACES_FOR_RECORDING, recording_id)
        logger.info(
            "Emitted STOP_ALL_TRACES_FOR_RECORDING (recording_id=%s)",
            recording_id,
        )
        await self._store.set_stopped_ats(recording_id)

    async def handle_is_connected(self, is_connected: bool) -> None:
        """Handle a connection status event from the data bridge."""
        if not is_connected:
            return

        await self._reconcile_failed_traces()
        await self.restore_retry_schedules()
        traces = await self._store.find_ready_traces()
        logger.info("Connection restored, %s traces ready for upload", len(traces))
        for trace in traces:
            await self._emit_ready_for_upload_from_trace(trace)

    def _schedule_retry_emit(self, trace_id: str, delay_s: float) -> None:
        """Schedule (or reschedule) retry emission for a trace."""
        existing = self._retry_emit_handles.pop(trace_id, None)
        if existing is not None and not existing.cancelled():
            existing.cancel()

        loop = asyncio.get_running_loop()
        handle = loop.call_later(
            float(delay_s),
            lambda: asyncio.create_task(self._retry_emit(trace_id)),
        )
        self._retry_emit_handles[trace_id] = handle

    async def restore_retry_schedules(self) -> None:
        """Restore persisted retry timers into in-memory loop scheduling."""
        traces = await self._store.find_retry_scheduled_traces()
        if not traces:
            return

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        for trace in traces:
            if trace.next_retry_at is None:
                continue
            delay_s = (trace.next_retry_at - now).total_seconds()
            if delay_s <= 0:
                continue
            self._schedule_retry_emit(trace.trace_id, delay_s)

    async def _emit_ready_for_upload_from_trace(self, trace: TraceRecord) -> None:
        logger.info(
            "Emitting READY_FOR_UPLOAD for trace %s (recording=%s, bytes_uploaded=%s)",
            trace.trace_id,
            trace.recording_id,
            trace.bytes_uploaded,
        )
        self._emitter.emit(
            Emitter.READY_FOR_UPLOAD,
            trace.trace_id,
            trace.recording_id,
            trace.path,
            trace.data_type,
            trace.data_type_name,
            trace.bytes_uploaded,
        )

    async def _reconcile_failed_traces(self) -> None:
        failed_traces = await self._store.find_failed_traces()
        if not failed_traces:
            return

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cutoff = now - timedelta(seconds=self._FAILED_TRACE_MAX_AGE_S)

        for trace in failed_traces:
            too_old = trace.created_at < cutoff
            retries_exhausted = (
                getattr(trace, "num_upload_attempts", 0) >= UPLOAD_MAX_RETRIES
            )
            has_uploadable_payload = (
                trace.total_bytes is not None
                and trace.total_bytes > 0
                and trace.path is not None
                and trace.data_type is not None
            )

            if retries_exhausted or too_old or not has_uploadable_payload:
                logger.warning(
                    "Deleting failed trace (trace=%s, recording=%s, retries=%s, "
                    "too_old=%s, uploadable=%s)",
                    trace.trace_id,
                    trace.recording_id,
                    getattr(trace, "num_upload_attempts", 0),
                    too_old,
                    has_uploadable_payload,
                )
                if trace.data_type is not None:
                    self._emitter.emit(
                        Emitter.DELETE_TRACE,
                        trace.recording_id,
                        trace.trace_id,
                        trace.data_type,
                    )
                await self._store.delete_trace(trace.trace_id)
                continue

            logger.info(
                "Resetting failed trace for retry (trace=%s, recording=%s)",
                trace.trace_id,
                trace.recording_id,
            )
            await self._store.reset_failed_trace_for_retry(trace.trace_id)
            refreshed = await self._store.get_trace(trace.trace_id)
            if refreshed is not None and refreshed.status == TraceStatus.WRITTEN:
                await self._emit_ready_for_upload_from_trace(refreshed)

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

        logger.info(
            "Upload complete for trace %s, deleting trace files and record",
            trace_id,
        )
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
        - If trace doesn't exist: creates with PENDING_METADATA status
        - If trace exists with INITIALIZING: transitions to WRITTEN
        - If status is WRITTEN after upsert: finalizes trace for upload
        """
        try:
            await self._process_trace_written(trace_id, recording_id, bytes_written)
        except OperationalError as exc:
            if "database is locked" not in str(exc).lower():
                raise
            logger.warning(
                "TRACE_WRITTEN DB locked (trace=%s, recording=%s). Queueing retry.",
                trace_id,
                recording_id,
            )
            await self._enqueue_trace_written_retry(
                trace_id, recording_id, bytes_written
            )

    async def _process_trace_written(
        self, trace_id: str, recording_id: str, bytes_written: int
    ) -> None:
        trace = await self._store.upsert_trace_bytes(
            trace_id=trace_id,
            recording_id=recording_id,
            bytes_written=bytes_written,
        )
        logger.info(
            "TRACE_WRITTEN received: trace=%s recording=%s bytes_written=%s status=%s",
            trace_id,
            recording_id,
            bytes_written,
            trace.status,
        )
        if trace.status == TraceStatus.WRITTEN:
            logger.info("Trace %s is WRITTEN after bytes update, finalizing", trace_id)
            await self._finalize_trace(trace)

    async def _enqueue_trace_written_retry(
        self, trace_id: str, recording_id: str, bytes_written: int
    ) -> None:
        self._ensure_trace_written_retry_worker()
        await self._trace_written_retry_queue.put(
            (trace_id, recording_id, bytes_written, 1)
        )

    def _ensure_trace_written_retry_worker(self) -> None:
        if self._trace_written_retry_task is not None:
            return
        loop = asyncio.get_running_loop()
        self._trace_written_retry_task = loop.create_task(
            self._trace_written_retry_worker()
        )

    async def _trace_written_retry_worker(self) -> None:
        while True:
            trace_id, recording_id, bytes_written, attempt = (
                await self._trace_written_retry_queue.get()
            )
            try:
                await self._process_trace_written(trace_id, recording_id, bytes_written)
            except OperationalError as exc:
                if (
                    "database is locked" in str(exc).lower()
                    and attempt < self._TRACE_WRITTEN_LOCK_MAX_RETRIES
                ):
                    delay = min(
                        self._TRACE_WRITTEN_LOCK_BASE_DELAY_S * (2 ** (attempt - 1)),
                        self._TRACE_WRITTEN_LOCK_MAX_DELAY_S,
                    )
                    logger.warning(
                        "TRACE_WRITTEN retry %s/%s after DB lock (trace=%s) in %.2fs",
                        attempt,
                        self._TRACE_WRITTEN_LOCK_MAX_RETRIES,
                        trace_id,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    await self._trace_written_retry_queue.put(
                        (trace_id, recording_id, bytes_written, attempt + 1)
                    )
                else:
                    logger.exception(
                        "TRACE_WRITTEN failed after retries (trace=%s)", trace_id
                    )
            finally:
                self._trace_written_retry_queue.task_done()

    async def _finalize_trace(self, trace: TraceRecord) -> None:
        """Finalize a WRITTEN trace and emit READY_FOR_UPLOAD.

        Emits READY_FOR_UPLOAD event for the uploader.
        """
        logger.info(
            "Finalizing trace %s for upload (recording=%s, path=%s)",
            trace.trace_id,
            trace.recording_id,
            trace.path,
        )
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

    async def handle_upload_started(self, trace_id: str) -> None:
        """Mark a trace as uploading once the uploader starts."""
        logger.info("Upload started for trace %s", trace_id)
        await self._store.update_status(trace_id, TraceStatus.UPLOADING)

    async def _emit_progress_report_if_recording_stopped(
        self, traces: list[TraceRecord]
    ) -> None:
        """Check if all traces are ready for progress reporting and emit a progress report.

        If the traces are from the same recording, check if they are all WRITTEN or COMPLETE.
        If so, add the recording to the reporting recordings list and schedule a progress report.
        """
        if not traces:
            return

        recording_id = traces[0].recording_id

        if not self._validate_traces_ready_for_reporting(traces, recording_id):
            return

        self._reporting_recordings.add(recording_id)
        start_time, end_time = self._find_recording_start_and_end(traces)
        asyncio.create_task(self._emit_progress_report(start_time, end_time, traces))

    def _validate_traces_ready_for_reporting(
        self, traces: list[TraceRecord], recording_id: str
    ) -> bool:
        """Validate that all traces are ready for progress reporting."""
        # Currently reporting
        if recording_id in self._reporting_recordings:
            return False
        # Already reported
        if all(
            trace.progress_reported == ProgressReportStatus.REPORTED for trace in traces
        ):
            logger.warning(
                "Progress already reported for recording %s, despite trace finalizing later, skipping report",
                recording_id,
                exc_info=True,
            )
            return False

        if not all(trace.stopped_at is not None for trace in traces):
            logger.warning(
                "Not all traces have stopped at, skipping report", exc_info=True
            )
            return False

        return True

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
        logger.debug(
            "Updating uploaded bytes for trace %s: %s", trace_id, bytes_uploaded
        )
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
        error_code: TraceErrorCode,
        error_message: str,
    ) -> None:
        """Handle an upload failed event from an uploader.

        Args:
            trace_id: unique identifier for the trace.
            bytes_uploaded: latest uploaded byte count for this trace.
            error_code: error code describing the failure type.
            error_message: human readable failure message.
        """
        logger.warning(
            "Upload failed for trace %s (bytes_uploaded=%s, code=%s, error=%s)",
            trace_id,
            bytes_uploaded,
            error_code,
            error_message,
        )
        await self.update_bytes_uploaded(trace_id, bytes_uploaded)

        trace = await self._store.get_trace(trace_id)
        if trace is None:
            logger.warning("Trace record not found: %s", trace_id)
            return

        next_attempt = int(getattr(trace, "num_upload_attempts", 0)) + 1

        if next_attempt >= UPLOAD_MAX_RETRIES:
            logger.error(
                "Upload retries exhausted for trace %s after %s attempts",
                trace_id,
                next_attempt,
            )
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
        logger.info(
            "Scheduled retry for trace %s in %s seconds (attempt %s)",
            trace_id,
            backoff,
            next_attempt,
        )

        loop = asyncio.get_running_loop()
        loop.call_later(
            float(backoff),
            lambda: asyncio.create_task(self._retry_emit(trace_id)),
        )

    async def _retry_emit(self, trace_id: str) -> None:
        """Emit READY_FOR_UPLOAD when a trace is due for retry."""
        self._retry_emit_handles.pop(trace_id, None)

        trace = await self._store.get_trace(trace_id)
        if trace is None:
            return

        if trace.status not in {TraceStatus.RETRYING, TraceStatus.WRITTEN}:
            return

        if trace.next_retry_at is not None:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            if trace.next_retry_at > now:
                delay = (trace.next_retry_at - now).total_seconds()
                self._schedule_retry_emit(trace_id, float(delay))
                return

        logger.info("Retrying upload for trace %s", trace_id)
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
