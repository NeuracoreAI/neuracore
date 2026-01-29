"""Protocol for trace state persistence."""

from __future__ import annotations

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

    async def create_trace(
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
    ) -> None:
        """Create or update a trace record."""
        ...

    async def get_trace(self, trace_id: str) -> TraceRecord | None:
        """Get a trace record by ID."""
        ...

    async def find_traces_by_recording_id(self, recording_id: str) -> list[TraceRecord]:
        """Return all traces for a given recording ID."""
        ...

    async def update_bytes_uploaded(self, trace_id: str, bytes_uploaded: int) -> None:
        """Increment uploaded byte count for a trace."""
        ...

    async def mark_trace_as_written(self, trace_id: str, bytes_written: int) -> None:
        """Finalize writing for a trace and mark it ready for upload."""
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

    async def update_status(
        self,
        trace_id: str,
        status: TraceStatus,
        *,
        error_message: str | None = None,
    ) -> None:
        """Update the status and optional error message for a trace."""
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
        total_bytes: int | None = None,
    ) -> TraceRecord:
        """Insert or update trace with metadata from START_TRACE.

        Creates trace in PENDING if new, updates metadata fields if exists.
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

        Creates trace in PENDING if new, updates bytes_written if exists.
        Returns the trace record after upsert.
        """
        ...
