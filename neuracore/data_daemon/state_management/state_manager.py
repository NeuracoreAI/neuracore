"""State manager facade for trace state operations."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import aiohttp

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.const import (
    API_URL,
    BACKEND_API_MAX_BACKOFF_SECONDS,
    BACKEND_API_MAX_RETRIES,
    BACKEND_API_RETRYABLE_STATUS_CODES,
    COMPLETED_RECORDING_RETENTION_HOURS,
    UPLOAD_MAX_RETRIES,
    UPLOAD_RETRY_BASE_SECONDS,
    UPLOAD_RETRY_MAX_SECONDS,
)
from neuracore.data_daemon.event_emitter import Emitter, get_emitter
from neuracore.data_daemon.models import (
    DataType,
    TraceErrorCode,
    TraceRecord,
    TraceRegistrationStatus,
    TraceUploadStatus,
    TraceWriteStatus,
)
from neuracore.data_daemon.registration_management.registration_manager import (
    RegistrationCandidate,
)

from .state_store import StateStore

logger = logging.getLogger(__name__)


class StateManager:
    """Domain-facing API for trace state."""

    _FAILED_TRACE_MAX_AGE_S = 60 * 60 * 4  # 4 hours
    _EMPTY_RECORDING_MAX_AGE_HOURS = 24

    def __init__(self, store: StateStore, config: DaemonConfig | None = None) -> None:
        """Initialize with a persistence backend."""
        self._store = store
        self._config = config

        self.expected_trace_count_reporting: dict[str, bool] = {}

        self._emitter = get_emitter()

        self._emitter.on(Emitter.START_TRACE, self._handle_start_trace)
        self._emitter.on(Emitter.TRACE_WRITTEN, self._handle_trace_written)
        self._emitter.on(Emitter.UPLOAD_STARTED, self.handle_upload_started)
        self._emitter.on(Emitter.UPLOADED_BYTES, self.update_bytes_uploaded)
        self._emitter.on(Emitter.UPLOAD_COMPLETE, self.handle_upload_complete)
        self._emitter.on(Emitter.UPLOAD_FAILED, self.handle_upload_failed)
        self._emitter.on(
            Emitter.STOP_RECORDING_REQUESTED, self.handle_stop_recording_requested
        )
        self._emitter.on(Emitter.STOP_RECORDING, self.handle_stop_recording)
        self._emitter.on(Emitter.IS_CONNECTED, self.handle_is_connected)
        self._emitter.on(Emitter.PROGRESS_REPORTED, self.mark_progress_as_reported)
        self._emitter.on(
            Emitter.PROGRESS_REPORT_FAILED, self.handle_progress_report_error
        )
        self._emitter.on(
            Emitter.SET_EXPECTED_TRACE_COUNT, self._handle_set_expected_trace_count
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
        """
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
        # NOTE
        # Here traces be written to disk but without metadata.
        # So either this will just add the metadata of the written file,
        # ...or create entry
        await self._store.upsert_trace_metadata(
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

    async def handle_stop_recording_requested(self, recording_id: str) -> None:
        """Handle phase-1 stop event from the data bridge.

        Records stop time immediately when the producer requests stop.
        """
        await self._store.set_stopped_at(recording_id)

    async def handle_stop_recording(self, recording_id: str) -> None:
        """Handle phase-2 stop event from the data bridge.

        This event is emitted only after the bridge commits close for the
        recording; at this point RDM can flush/close traces for that recording.
        """
        # Flush RDM states
        self._emitter.emit(Emitter.STOP_ALL_TRACES_FOR_RECORDING, recording_id)

        # Set stopped at if previously missed event
        if not await self._store.is_recording_stopped(recording_id):
            await self._store.set_stopped_at(recording_id)

        await self._emit_progress_report_if_last_trace_written(recording_id)

    async def handle_is_connected(self, is_connected: bool) -> None:
        """Handle a connection status event from the data bridge."""
        if not is_connected:
            return

        # Reconcile in flight states on reconnect
        reset_count = await self._store.reset_retrying_to_written()
        if reset_count:
            logger.info(
                "Reset transient upload statuses on reconnect (count=%d)", reset_count
            )
        reset_reporting_count = (
            await self._store.reset_reporting_recordings_to_pending()
        )
        if reset_reporting_count:
            logger.info(
                "Reset reporting recordings to pending on reconnect (count=%d)",
                reset_reporting_count,
            )
        await self._store.reconcile_recordings_from_traces()
        pruned_count = await self._store.prune_old_empty_recordings(
            self._EMPTY_RECORDING_MAX_AGE_HOURS
        )
        if pruned_count:
            logger.info(
                (
                    "Pruned stale empty recordings on reconnect "
                    "(count=%d, max_age_hours=%d)"
                ),
                pruned_count,
                self._EMPTY_RECORDING_MAX_AGE_HOURS,
            )
        await self._resume_reporting_work_on_reconnect()

        # delete recording older the expiration window
        deleted_count = await self._store.delete_expired_completed_recordings(
            COMPLETED_RECORDING_RETENTION_HOURS
        )
        if deleted_count:
            logger.info(
                "Deleted expired completed recordings on reconnect "
                "(count=%d, max_age_hours=%d)",
                deleted_count,
                COMPLETED_RECORDING_RETENTION_HOURS,
            )

        # Wake registration worker after connectivity is restored.
        self._emit_trace_registration_available()
        await self._reconcile_failed_traces()

        traces = await self._store.find_ready_traces()
        for trace in traces:
            await self._emit_ready_for_upload_from_trace(trace)

    async def recover_startup_state(self) -> None:
        """Run one-time startup recovery that does not require connectivity."""
        traces = await self._store.list_traces()
        recording_ids = sorted({str(trace.recording_id) for trace in traces})
        if not recording_ids:
            return

        for recording_id in recording_ids:
            if not await self._store.is_recording_stopped(recording_id):
                await self._store.set_stopped_at(recording_id)

    async def _resume_reporting_work_on_reconnect(self) -> None:
        """Resume backend-dependent reporting tasks when connectivity returns."""
        traces = await self._store.list_traces()
        recording_ids = sorted({str(trace.recording_id) for trace in traces})
        for recording_id in recording_ids:
            if not await self._store.is_expected_trace_count_reported(recording_id):
                trace_count = await self._store.count_traces_for_recording(recording_id)
                await self._store.set_expected_trace_count(recording_id, trace_count)
                await self._set_expected_trace_count(recording_id, trace_count)

            await self._emit_progress_report_if_last_trace_written(recording_id)

    async def _emit_ready_for_upload_from_trace(
        self,
        trace: TraceRecord,
        session_uris: dict[str, str] | None = None,
    ) -> None:
        if (
            trace.path is None
            or trace.data_type is None
            or trace.data_type_name is None
        ):
            logger.warning(
                "Skipping READY_FOR_UPLOAD for trace %s: incomplete metadata",
                trace.trace_id,
            )
            return
        self._emitter.emit(
            Emitter.READY_FOR_UPLOAD,
            trace.trace_id,
            trace.recording_id,
            trace.path,
            trace.data_type,
            trace.data_type_name,
            trace.bytes_uploaded,
            session_uris,
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
                if trace.data_type is not None:
                    self._emitter.emit(
                        Emitter.DELETE_TRACE,
                        trace.recording_id,
                        trace.trace_id,
                        trace.data_type,
                    )
                await self.delete_trace(trace.trace_id)
                continue

            await self._store.reset_failed_trace_for_retry(trace.trace_id)
            refreshed = await self._store.get_trace(trace.trace_id)
            if (
                refreshed is not None
                and refreshed.write_status == TraceWriteStatus.WRITTEN
                and refreshed.registration_status == TraceRegistrationStatus.REGISTERED
                and refreshed.upload_status == TraceUploadStatus.PENDING
            ):
                await self._emit_ready_for_upload_from_trace(refreshed)

    async def handle_upload_complete(self, trace_id: str) -> None:
        """Handle an upload complete event from an uploader.

        This function is called when an uploader completes an upload.
        Local trace data is deleted immediately, while DB metadata can be
        retained until progress reporting completes.
        """
        await self._process_upload_complete(trace_id)

    async def _process_upload_complete(self, trace_id: str) -> None:
        trace_record = await self._store.get_trace(trace_id)
        if trace_record is None:
            logger.warning("Trace record not found: %s", trace_id)
            return

        await self._store.update_upload_status(trace_id, TraceUploadStatus.UPLOADED)
        await self._store.increment_uploaded_trace_count(trace_record.recording_id)
        await self._emit_progress_report_if_last_trace_written(
            trace_record.recording_id
        )

        # Always delete local trace data after upload
        # DB metadata may be kept until progress reporting total_bytes.
        if trace_record.data_type is not None:
            self._emitter.emit(
                Emitter.DELETE_TRACE,
                trace_record.recording_id,
                trace_id,
                trace_record.data_type,
            )
        # Traces will remain until progress reporting completes with status uploaded
        # these can be deleted if progress has bee reported
        if await self._store.recording_has_reported_progress(trace_record.recording_id):
            await self._store.delete_uploaded_traces_for_recording(
                trace_record.recording_id
            )

    async def _handle_trace_written(
        self, trace_id: str, recording_id: str, bytes_written: int
    ) -> None:
        """Handle TRACE_WRITTEN event - upsert trace bytes.

        State transitions:
        - If trace doesn't exist: creates with PENDING_METADATA status
        - If trace exists with INITIALIZING: transitions to WRITTEN
        """
        try:
            await self._process_trace_written(trace_id, recording_id, bytes_written)
        except Exception:
            logger.exception("Failed to process TRACE_WRITTEN event")
            raise

    async def _process_trace_written(
        self, trace_id: str, recording_id: str, bytes_written: int
    ) -> None:
        trace = await self._store.upsert_trace_bytes(
            trace_id=trace_id,
            recording_id=recording_id,
            bytes_written=bytes_written,
        )
        if trace.write_status == TraceWriteStatus.WRITTEN:
            await self._finalize_trace(trace)

    async def handle_upload_started(self, trace_id: str) -> None:
        """Mark a trace as uploading once the uploader starts."""
        await self._store.update_upload_status(trace_id, TraceUploadStatus.UPLOADING)

    def _emit_trace_registration_available(self) -> None:
        self._emitter.emit(Emitter.TRACE_REGISTRATION_AVAILABLE)

    async def _emit_progress_report_if_last_trace_written(
        self, recording_id: str
    ) -> None:
        """Emit progress report when the final expected trace is written.

        Progress reporting should wait until all expected traces for the
        recording are fully written so total bytes and trace map are complete.

        Args:
            recording_id (str): The ID of the recording to check.
        """
        if await self._store.recording_has_reported_progress(recording_id):
            return
        if not await self._store.is_expected_trace_count_reported(recording_id):
            return

        expected_trace_count = await self._store.get_expected_trace_count(recording_id)
        if expected_trace_count is None:
            return

        written_trace_count = await self._store.count_traces_for_recording(recording_id)
        if written_trace_count < int(expected_trace_count):
            return

        snapshot = await self._store.get_progress_report_snapshot(recording_id)
        if snapshot is None:
            return
        start_time, end_time, trace_map, total_bytes = snapshot

        marked_reported_recordings = await self._store.mark_recording_reporting(
            recording_id
        )
        if not marked_reported_recordings:
            return
        self._emitter.emit(
            Emitter.PROGRESS_REPORT,
            recording_id,
            start_time,
            end_time,
            trace_map,
            total_bytes,
        )

    async def _handle_set_expected_trace_count(
        self, recording_id: str, expected_trace_count: int
    ) -> None:
        """Handle an expected trace count update event.

        When an expected trace count is updated, this method updates the
        local expected trace count and schedules a progress report if the
        recording has been stopped and progress reporting has not been
        claimed.

        Args:
            recording_id (str): The ID of the recording to update.
            expected_trace_count (int): The new expected trace count.
        """
        await self._store.set_expected_trace_count(recording_id, expected_trace_count)
        await self._set_expected_trace_count(recording_id, expected_trace_count)
        self._emit_trace_registration_available()

        # Possibly all channels drained before expected trace count set
        await self._emit_progress_report_if_last_trace_written(recording_id)

    async def _finalize_trace(self, trace: TraceRecord) -> None:
        """Finalize trace for upload."""
        # If recording stopped at not set, set it
        if not await self._store.is_recording_stopped(trace.recording_id):
            await self._store.set_stopped_at(trace.recording_id)

        # Trace finished writing but buffer states not flushed
        # so possibly still unsure of expected_trace_count at this stage
        if not await self._store.is_expected_trace_count_reported(trace.recording_id):
            return

        # Start registering loop if not started yet
        self._emit_trace_registration_available()

        # Possibly last trace to be written, so check if progress can be reported
        await self._emit_progress_report_if_last_trace_written(trace.recording_id)

    async def _set_expected_trace_count(
        self, recording_id: str, expected_trace_count: int
    ) -> bool:
        """Post expected trace count for a recording to the backend."""
        if not recording_id:
            return False
        if recording_id in self.expected_trace_count_reporting:
            return False

        if self._config and self._config.offline:
            return False

        self.expected_trace_count_reporting[recording_id] = True
        loop = asyncio.get_running_loop()
        auth = get_auth()

        try:
            try:
                org_id = await loop.run_in_executor(None, get_current_org)
                headers = await loop.run_in_executor(None, auth.get_headers)
            except Exception:
                logger.exception(
                    "Failed preparing expected trace count request for recording %s",
                    recording_id,
                )
                return False

            url = (
                f"{API_URL}/org/{org_id}/recording/{recording_id}/expected-trace-count"
            )
            payload = {
                "expected_trace_count": int(expected_trace_count),
            }
            last_error: str | None = None

            async with aiohttp.ClientSession() as session:
                for attempt in range(BACKEND_API_MAX_RETRIES):
                    try:
                        async with session.put(
                            url,
                            json=payload,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as response:
                            if response.status < 400:
                                await self._store.mark_expected_trace_count_reported(
                                    recording_id
                                )
                                return True
                            if response.status == 401:
                                await loop.run_in_executor(None, auth.login)
                                headers = await loop.run_in_executor(
                                    None, auth.get_headers
                                )
                                continue

                            error_text = await response.text()
                            last_error = f"HTTP {response.status}: {error_text}"
                            logger.warning(
                                (
                                    "Expected trace count post failed "
                                    "(attempt %d/%d) for %s: %s"
                                ),
                                attempt + 1,
                                BACKEND_API_MAX_RETRIES,
                                recording_id,
                                last_error,
                            )
                            if (
                                response.status
                                not in BACKEND_API_RETRYABLE_STATUS_CODES
                            ):
                                break
                    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                        last_error = str(exc)
                        logger.warning(
                            (
                                "Expected trace count request failed "
                                "(attempt %d/%d) for %s: %s"
                            ),
                            attempt + 1,
                            BACKEND_API_MAX_RETRIES,
                            recording_id,
                            exc,
                        )

                    if attempt < BACKEND_API_MAX_RETRIES - 1:
                        delay = min(2**attempt, BACKEND_API_MAX_BACKOFF_SECONDS)
                        await asyncio.sleep(delay)

            logger.error(
                "Failed to post expected trace count for recording %s: %s",
                recording_id,
                last_error or "unknown error",
            )
            return False
        finally:
            self.expected_trace_count_reporting.pop(recording_id, None)

    async def mark_progress_as_reported(self, recording_id: str) -> None:
        """Mark a recording as progress-reported.

        Args:
            recording_id (str): unique identifier for the recording.
        """
        await self._store.mark_recording_reported(recording_id)
        if await self._store.is_expected_trace_count_reported(recording_id):
            await self._store.delete_uploaded_traces_for_recording(recording_id)

    async def update_bytes_uploaded(self, trace_id: str, bytes_uploaded: int) -> None:
        """Increment uploaded byte count for a trace."""
        await self._store.update_bytes_uploaded(trace_id, bytes_uploaded)

    async def mark_traces_registering(self, trace_ids: list[str]) -> list[str]:
        """Batch mark traces as currently registering with the backend."""
        if not trace_ids:
            return []
        return await self._store.mark_traces_as_registering(trace_ids)

    async def mark_traces_registered(self, trace_ids: list[str]) -> list[str]:
        """Batch mark traces as registered with the backend."""
        if not trace_ids:
            return []
        return await self._store.mark_traces_as_registered(trace_ids)

    async def claim_traces_for_registration(
        self, limit: int = 200, max_wait_s: float = 1
    ) -> list[RegistrationCandidate]:
        """Claim registration-eligible traces from state storage."""
        records = await self._store.claim_traces_for_registration(limit, max_wait_s)
        candidates: list[RegistrationCandidate] = []
        skipped_missing_data_type = 0
        for trace in records:
            if trace.data_type is None or trace.data_type_name is None:
                logger.warning(
                    "Skipping registration claim for trace %s: missing data_type",
                    trace.trace_id,
                )
                skipped_missing_data_type += 1
                continue
            candidates.append(
                RegistrationCandidate(
                    trace_id=trace.trace_id,
                    recording_id=trace.recording_id,
                    data_type=trace.data_type,
                    data_type_name=trace.data_type_name,
                )
            )
        return candidates

    async def mark_traces_registration_failed(
        self, trace_ids: list[str], error_message: str
    ) -> None:
        """Mark traces as registration-retry pending after a failed batch."""
        if not trace_ids:
            return
        logger.warning(
            (
                "StateManager marking registration failed traces "
                "(count=%d, sample_ids=%s, error=%s)"
            ),
            len(trace_ids),
            trace_ids[:5],
            error_message,
        )
        for trace_id in trace_ids:
            await self._store.update_registration_status(
                trace_id, TraceRegistrationStatus.RETRYING
            )
            await self._store.update_registration_status(
                trace_id, TraceRegistrationStatus.PENDING
            )
            await self._store.record_error(
                trace_id,
                error_message=error_message,
                error_code=TraceErrorCode.UNKNOWN,
            )
        self._emit_trace_registration_available()

    async def emit_ready_for_upload(
        self,
        trace_ids: list[str],
        upload_session_uris: dict[str, dict[str, str]] | None = None,
    ) -> None:
        """Emit READY_FOR_UPLOAD for trace IDs that are upload-eligible."""
        if not trace_ids:
            return
        for trace_id in trace_ids:
            trace = await self._store.get_trace(trace_id)
            if trace is None:
                logger.warning(
                    "Cannot emit READY_FOR_UPLOAD: trace %s missing from store",
                    trace_id,
                )
                continue
            trace_uris = (
                upload_session_uris.get(trace_id) if upload_session_uris else None
            )
            await self._emit_ready_for_upload_from_trace(trace, trace_uris)

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
        if trace.upload_status != TraceUploadStatus.RETRYING:
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
        await self._store.mark_recording_pending(recording_id)
        logger.error(
            "Progress report failed for recording %s: %s",
            recording_id,
            error_message,
        )
        traces = await self._store.find_traces_by_recording_id(recording_id)
        if not traces:
            return
        for trace in traces:
            await self._store.record_error(
                trace.trace_id,
                error_message,
                error_code=TraceErrorCode.PROGRESS_REPORT_ERROR,
            )

    async def _record_error(
        self,
        trace_id: str,
        error_message: str,
        error_code: TraceErrorCode | None = TraceErrorCode.UNKNOWN,
    ) -> None:
        """Record an error for a trace.

        Args:
            trace_id: str
                Trace ID of the trace to record the error for.
            error_message: str
                Error message of the error.
            error_code: TraceErrorCode | None, optional
                Error code of the error, by default None.
        """
        await self._store.update_upload_status(trace_id, TraceUploadStatus.FAILED)
        await self._store.record_error(
            trace_id,
            error_message,
            error_code,
        )

    async def delete_trace(self, trace_id: str) -> None:
        """Delete a trace record."""
        await self._store.delete_trace(trace_id)
