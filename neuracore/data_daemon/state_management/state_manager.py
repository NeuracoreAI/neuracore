"""State manager facade for trace state operations."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import aiohttp
from sqlalchemy.exc import OperationalError

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.data_daemon.const import (
    API_URL,
    BACKEND_API_MAX_BACKOFF_SECONDS,
    BACKEND_API_MAX_RETRIES,
    BACKEND_API_RETRYABLE_STATUS_CODES,
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

    _TRACE_WRITTEN_LOCK_MAX_RETRIES = 5
    _TRACE_WRITTEN_LOCK_BASE_DELAY_S = 0.1
    _TRACE_WRITTEN_LOCK_MAX_DELAY_S = 2.0
    _START_TRACE_LOCK_MAX_RETRIES = 5
    _START_TRACE_LOCK_BASE_DELAY_S = 0.1
    _START_TRACE_LOCK_MAX_DELAY_S = 2.0
    _FAILED_TRACE_MAX_AGE_S = 60 * 60 * 4  # 4 hours
    _EMPTY_RECORDING_MAX_AGE_HOURS = 24

    def __init__(self, store: StateStore) -> None:
        """Initialize with a persistence backend."""
        self._store = store
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
        # NOTE
        # Here traces be written to disk but without metadata.
        # So either this will just add the metadata of the written file,
        # ...or create entry
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
        if trace.write_status == TraceWriteStatus.WRITTEN:
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
            finally:
                self._start_trace_retry_queue.task_done()

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
        self._emitter.emit(Emitter.STOP_ALL_TRACES_FOR_RECORDING, recording_id)
        if not await self._store.is_recording_stopped(recording_id):
            await self._store.set_stopped_at(recording_id)
        await self._emit_progress_report_if_recording_stopped(recording_id)

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
                "Pruned stale empty recordings on reconnect (count=%d, max_age_hours=%d)",
                pruned_count,
                self._EMPTY_RECORDING_MAX_AGE_HOURS,
            )

        # Wake registration worker after connectivity is restored.
        self._emit_trace_registration_available()
        await self._reconcile_failed_traces()

        traces = await self._store.find_ready_traces()
        for trace in traces:
            await self._emit_ready_for_upload_from_trace(trace)

    async def _emit_ready_for_upload_from_trace(self, trace: TraceRecord) -> None:
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
        trace_record = await self._store.get_trace(trace_id)
        if trace_record is None:
            logger.warning("Trace record not found: %s", trace_id)
            return

        await self._store.update_upload_status(trace_id, TraceUploadStatus.UPLOADED)

        # Always delete local trace data after upload
        # DB metadata may be kept until progress reporting total_bytes.
        if trace_record.data_type is not None:
            self._emitter.emit(
                Emitter.DELETE_TRACE,
                trace_record.recording_id,
                trace_id,
                trace_record.data_type,
            )

        # Delete traces kept but marked as uploaded after progress reporting.
        if await self._delete_uploaded_traces_if_progress_reported(
            trace_record.recording_id
        ):
            return

        await self._emit_progress_report_if_recording_stopped(trace_record.recording_id)

    async def _delete_uploaded_traces_if_progress_reported(
        self, recording_id: str
    ) -> bool:
        """Delete uploaded trace metadata when reporting is already complete."""
        if not await self._store.recording_has_reported_progress(recording_id):
            return False
        await self._store.delete_uploaded_traces_for_recording(recording_id)
        return True

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
        if trace.write_status == TraceWriteStatus.WRITTEN:
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
                    await asyncio.sleep(delay)
                    await self._trace_written_retry_queue.put(
                        (trace_id, recording_id, bytes_written, attempt + 1)
                    )
            finally:
                self._trace_written_retry_queue.task_done()

    async def _finalize_trace(self, trace: TraceRecord) -> None:
        """Finalize trace for upload."""
        # Start registering loop
        self._emit_trace_registration_available()

        # If recording stopped at not set, set it
        if not await self._store.is_recording_stopped(trace.recording_id):
            await self._store.set_stopped_at(trace.recording_id)

        trace_count = await self._store.count_traces_for_recording(trace.recording_id)
        expected_reported = await self._store.is_expected_trace_count_reported(
            trace.recording_id
        )
        # Make expected trace count call (on first opportunity only)
        # 1 = reported, 0 = Not reported
        if not expected_reported:
            await self._store.set_expected_trace_count(trace.recording_id, trace_count)
            asyncio.create_task(
                self._set_expected_trace_count(trace.recording_id, trace_count)
            )

        # Possibly last trace to be written, so check if so and can report progress
        await self._emit_progress_report_if_recording_stopped(trace.recording_id)

    async def handle_upload_started(self, trace_id: str) -> None:
        """Mark a trace as uploading once the uploader starts."""
        await self._store.update_upload_status(trace_id, TraceUploadStatus.UPLOADING)

    def _emit_trace_registration_available(self) -> None:
        self._emitter.emit(Emitter.TRACE_REGISTRATION_AVAILABLE)

    async def _emit_progress_report_if_recording_stopped(
        self, recording_id: str
    ) -> None:
        if not await self._store.is_recording_stopped(recording_id):
            return

        if await self._store.recording_has_reported_progress(recording_id):
            return

        traces = await self._store.find_traces_by_recording_id(recording_id)
        if not traces:
            return
        if not self._is_progress_report_eligible(traces):
            return
        claimed = await self._store.mark_recording_reporting(recording_id)
        if not claimed:
            return
        self._schedule_progress_report(recording_id, traces)

    def _is_progress_report_eligible(self, traces: list[TraceRecord]) -> bool:
        """Return True when traces have complete, consistent byte totals."""
        for trace in traces:
            if trace.data_type is None:
                return False
            if trace.bytes_written is None:
                return False
            if trace.total_bytes is None:
                return False
            if trace.bytes_written != trace.total_bytes:
                return False
        return True

    def _schedule_progress_report(
        self, recording_id: str, traces: list[TraceRecord]
    ) -> None:
        """Schedule async progress reporting for a recording."""
        start_time, end_time = self._find_recording_start_and_end(traces)
        asyncio.create_task(self._emit_progress_report(start_time, end_time, traces))

    async def _emit_progress_report(
        self, start_time: float, end_time: float, traces: list[TraceRecord]
    ) -> None:
        """Emit progress report event asynchronously."""
        self._emitter.emit(Emitter.PROGRESS_REPORT, start_time, end_time, traces)

    async def _set_expected_trace_count(
        self, recording_id: str, expected_trace_count: int
    ) -> None:
        """Post expected trace count for a recording to the backend."""
        if not recording_id:
            return
        if recording_id in self.expected_trace_count_reporting:
            return

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
                return

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
                                return
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
        finally:
            self.expected_trace_count_reporting.pop(recording_id, None)

    async def mark_progress_as_reported(self, recording_id: str) -> None:
        """Mark a recording as progress-reported.

        Args:
            recording_id (str): unique identifier for the recording.
        """
        await self._store.mark_recording_reported(recording_id)
        await self._store.delete_uploaded_traces_for_recording(recording_id)

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

    async def mark_traces_registering(self, trace_ids: list[str]) -> list[str]:
        """Batch mark traces as currently registering with the backend."""
        if not trace_ids:
            return []

        updated = await self._store.mark_traces_as_registering(trace_ids)
        logger.info(
            "StateManager marked traces REGISTERING (requested=%d, updated=%d)",
            len(trace_ids),
            len(updated),
        )
        return updated

    async def mark_traces_registered(self, trace_ids: list[str]) -> list[str]:
        """Batch mark traces as registered with the backend."""
        if not trace_ids:
            return []

        updated = await self._store.mark_traces_as_registered(trace_ids)
        logger.info(
            "StateManager marked traces REGISTERED (requested=%d, updated=%d)",
            len(trace_ids),
            len(updated),
        )
        return updated

    async def claim_traces_for_registration(
        self, limit: int = 200, max_wait_s: float = 1
    ) -> list[RegistrationCandidate]:
        """Claim registration-eligible traces from state storage."""
        records = await self._store.claim_traces_for_registration(limit, max_wait_s)
        if records:
            logger.info(
                "StateManager claim_traces_for_registration returned %d records (limit=%d, max_wait_s=%.2f)",
                len(records),
                limit,
                max_wait_s,
            )
        else:
            logger.debug(
                "StateManager claim_traces_for_registration returned 0 records (limit=%d, max_wait_s=%.2f)",
                limit,
                max_wait_s,
            )
        candidates: list[RegistrationCandidate] = []
        skipped_missing_data_type = 0
        for trace in records:
            if trace.data_type is None:
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
                )
            )
        if candidates or skipped_missing_data_type:
            logger.info(
                "StateManager prepared %d registration candidates (skipped_missing_data_type=%d)",
                len(candidates),
                skipped_missing_data_type,
            )
        else:
            logger.debug("StateManager prepared 0 registration candidates")
        return candidates

    async def mark_traces_registration_failed(
        self, trace_ids: list[str], error_message: str
    ) -> None:
        """Mark traces as registration-retry pending after a failed batch."""
        if not trace_ids:
            return
        logger.warning(
            "StateManager marking registration failed traces (count=%d, sample_ids=%s, error=%s)",
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

    async def emit_ready_for_upload(self, trace_ids: list[str]) -> None:
        """Emit READY_FOR_UPLOAD for trace IDs that are upload-eligible."""
        if not trace_ids:
            return
        logger.info(
            "StateManager emitting READY_FOR_UPLOAD for trace ids (count=%d, sample_ids=%s)",
            len(trace_ids),
            trace_ids[:5],
        )
        for trace_id in trace_ids:
            trace = await self._store.get_trace(trace_id)
            if trace is None:
                logger.warning(
                    "Cannot emit READY_FOR_UPLOAD: trace %s missing from store",
                    trace_id,
                )
                continue
            await self._emit_ready_for_upload_from_trace(trace)

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
