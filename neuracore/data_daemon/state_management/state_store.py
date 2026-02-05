"""Protocol for trace state persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from neuracore.data_daemon.models import (
    DataType,
    TraceErrorCode,
    TraceRecord,
    TraceStatus,
)


class StateStore(Protocol):
    """Persistence interface for trace state."""

    async def set_stopped_ats(self, recording_id: str) -> None:
        """Set the end time for all traces for a recording."""
        ...

    async def get_trace(self, trace_id: str) -> TraceRecord | None:
        """Get a trace record by ID."""
        ...

    async def find_traces_by_recording_id(self, recording_id: str) -> list[TraceRecord]:
        """Return all traces for a given recording ID."""
        ...

    def list_traces(self) -> list[TraceRecord]:
        """Return all trace records."""
        ...

    async def update_bytes_uploaded(self, trace_id: str, bytes_uploaded: int) -> None:
        """Increment uploaded byte count for a trace."""
        ...

    async def find_ready_traces(self) -> list[TraceRecord]:
        """Return all traces marked as ready for upload."""
        ...

    async def find_unreported_traces(self) -> list[TraceRecord]:
        """Return all traces that have not been progress-reported."""
        ...

    async def mark_recording_reported(self, recording_id: str) -> None:
        """Mark a recording as progress-reported."""
        ...

    async def find_failed_traces(self) -> list[TraceRecord]:
        """Return all traces marked as FAILED."""
        ...

    async def reset_failed_trace_for_retry(self, trace_id: str) -> None:
        """Reset a failed trace back to WRITTEN for retry."""
        ...

    async def update_status(
        self,
        trace_id: str,
        status: TraceStatus,
        *,
        error_message: str | None = None,
    ) -> bool:
        """Update the status and optional error message for a trace.

        Returns True if the status was changed, False if already at target status.
        Raises ValueError for invalid transitions or missing trace.
        """
        ...

    async def record_error(
        self,
        trace_id: str,
        error_message: str,
        error_code: TraceErrorCode | None = None,
        status: TraceStatus = TraceStatus.FAILED,
    ) -> None:
        """Record a standardized error for a trace."""
        ...

    async def delete_trace(self, trace_id: str) -> None:
        """Delete a trace record."""
        ...

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
        """Reset RETRYING/UPLOADING traces back to WRITTEN (preserve retry schedule)."""
        ...
