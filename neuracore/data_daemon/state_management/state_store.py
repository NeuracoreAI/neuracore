"""Protocol for trace state persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from neuracore.data_daemon.models import (
    DataType,
    TraceErrorCode,
    TraceRecord,
    TraceRegistrationStatus,
    TraceUploadStatus,
    TraceWriteStatus,
)


class StateStore(Protocol):
    """Persistence interface for trace state."""

    async def set_stopped_at(self, recording_id: str) -> None:
        """Set recording-level stopped_at for a recording."""
        ...

    async def get_trace(self, trace_id: str) -> TraceRecord | None:
        """Get a trace record by ID."""
        ...

    async def find_traces_by_recording_id(self, recording_id: str) -> list[TraceRecord]:
        """Return all traces for a given recording ID."""
        ...

    async def get_progress_report_snapshot(
        self, recording_id: str
    ) -> tuple[float, float, dict[str, int], int] | None:
        """Return an immutable progress snapshot for one recording.

        Returns None when there are no traces or when any trace is ineligible
        (missing metadata/byte totals).
        """
        ...

    async def list_traces(self) -> list[TraceRecord]:
        """Return all trace records."""
        ...

    async def update_bytes_uploaded(self, trace_id: str, bytes_uploaded: int) -> None:
        """Increment uploaded byte count for a trace."""
        ...

    async def find_ready_traces(self) -> list[TraceRecord]:
        """Return all traces marked as ready for upload."""
        ...

    async def claim_traces_for_registration(
        self, limit: int, max_wait_s: float
    ) -> list[TraceRecord]:
        """Claim traces for registration using size-or-age policy.

        Claims immediately when enough traces are available to fill `limit`.
        Otherwise, claims only traces older than `max_wait_s` based on
        `last_updated`.
        """
        ...

    async def find_unreported_traces(self) -> list[TraceRecord]:
        """Return all traces that have not been progress-reported."""
        ...

    async def mark_recording_reported(self, recording_id: str) -> None:
        """Mark a recording as progress-reported."""
        ...

    async def mark_recording_reporting(self, recording_id: str) -> bool:
        """Atomically move recording progress state from PENDING to REPORTING."""
        ...

    async def mark_recording_pending(self, recording_id: str) -> None:
        """Set recording progress state to PENDING."""
        ...

    async def reset_reporting_recordings_to_pending(self) -> int:
        """Reset in-flight REPORTING rows to PENDING and return affected count."""
        ...

    async def recording_has_reported_progress(self, recording_id: str) -> bool:
        """Return True when recording progress status is REPORTED."""
        ...

    async def delete_uploaded_traces_for_recording(self, recording_id: str) -> int:
        """Delete all UPLOADED traces for one recording and return deleted count."""
        ...

    async def delete_traces_for_recording(self, recording_id: str) -> int:
        """Delete all remaining trace rows for a recording and return deleted count."""
        ...

    async def delete_expired_completed_recordings(self, max_age_hours: int) -> int:
        """Delete completed recording rows older than the retention window."""
        ...

    async def list_recording_ids_with_stopped_traces(self) -> list[str]:
        """Return recording IDs that already have at least one stopped trace."""
        ...

    async def reconcile_recordings_from_traces(self) -> None:
        """Rebuild recording rows from trace rows (startup reconciliation)."""
        ...

    async def prune_old_empty_recordings(self, max_age_hours: int) -> int:
        """Delete recordings with no traces and age older than threshold hours."""
        ...

    async def is_recording_stopped(self, recording_id: str) -> bool:
        """Return True when recording has stopped_at set."""
        ...

    async def is_expected_trace_count_reported(self, recording_id: str) -> bool:
        """Return True when expected trace count has been reported for recording."""
        ...

    async def get_expected_trace_count(self, recording_id: str) -> int | None:
        """Return expected trace count for a recording, if any."""
        ...

    async def count_traces_for_recording(self, recording_id: str) -> int:
        """Return trace count for a recording."""
        ...

    async def set_expected_trace_count(
        self, recording_id: str, expected_trace_count: int
    ) -> None:
        """Persist expected trace count for a recording."""
        ...

    async def mark_expected_trace_count_reported(self, recording_id: str) -> None:
        """Mark a recording's expected trace count as reported."""
        ...

    async def find_failed_traces(self) -> list[TraceRecord]:
        """Return all traces marked as FAILED."""
        ...

    async def reset_failed_trace_for_retry(self, trace_id: str) -> None:
        """Reset a failed trace back to WRITTEN for retry."""
        ...

    async def update_write_status(
        self, trace_id: str, write_status: TraceWriteStatus
    ) -> None:
        """Update write lifecycle status for a trace."""
        ...

    async def update_registration_status(
        self, trace_id: str, registration_status: TraceRegistrationStatus
    ) -> None:
        """Update registration lifecycle status for a trace."""
        ...

    async def mark_traces_as_registering(self, trace_ids: list[str]) -> list[str]:
        """Batch mark traces as registering.

        Returns trace_ids that were actually updated.
        """
        ...

    async def mark_traces_as_registered(self, trace_ids: list[str]) -> list[str]:
        """Batch mark traces as registered.

        Returns trace_ids that were actually updated.
        """
        ...

    async def update_upload_status(
        self, trace_id: str, upload_status: TraceUploadStatus
    ) -> None:
        """Update upload lifecycle status for a trace."""
        ...

    async def increment_uploaded_trace_count(self, recording_id: str) -> None:
        """Increment recording-level uploaded trace count."""
        ...

    async def record_error(
        self,
        trace_id: str,
        error_message: str,
        error_code: TraceErrorCode | None = None,
    ) -> None:
        """Record a standardized error for a trace."""
        ...

    async def delete_trace(self, trace_id: str) -> None:
        """Delete a trace record."""
        ...

    async def init_async_store(self) -> None:
        """Apply pragmas and ensure schema."""

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
        """Insert or update trace with metadata from START_TRACE.

        State transitions:
        - If trace doesn't exist: creates with INITIALIZING status
        - If trace exists with PENDING_METADATA: transitions to WRITTEN
        - If trace exists with other status: updates metadata only

        Returns the trace record after upsert.
        """
        ...

    async def upsert_trace_bytes(
        self,
        trace_id: str,
        recording_id: str,
        bytes_written: int,
    ) -> TraceRecord:
        """Insert or update trace with bytes from TRACE_WRITTEN.

        State transitions:
        - If trace doesn't exist: creates with PENDING_METADATA status
        - If trace exists with INITIALIZING: transitions to WRITTEN
        - If trace exists with other status: updates bytes only

        Returns the trace record after upsert.
        """
        ...

    async def schedule_retry(
        self,
        trace_id: str,
        *,
        next_retry_at: datetime,
        error_code: TraceErrorCode,
        error_message: str,
    ) -> int:
        """Schedule next upload retry and persist failure details."""
        ...

    async def mark_retry_exhausted(
        self,
        trace_id: str,
        *,
        error_code: TraceErrorCode,
        error_message: str,
    ) -> int:
        """Mark retries exhausted and persist final failure details."""
        ...

    async def reset_retrying_to_written(self) -> int:
        """Reset RETRYING/UPLOADING traces back to upload PENDING."""
        ...
