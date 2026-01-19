"""Protocol for trace state persistence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from neuracore.data_daemon.models import (
    DataType,
    TraceErrorCode,
    TraceRecord,
    TraceStatus,
)


class StateStore(Protocol):
    """Persistence interface for trace state."""

    def set_stopped_ats(self, recording_id: str) -> None:
        """Set the end time for all traces for a recording."""
        ...

    def create_trace(
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

    def get_trace(self, trace_id: str) -> TraceRecord | None:
        """Get a trace record by ID."""
        ...

    def find_traces_by_recording_id(self, recording_id: str) -> list[TraceRecord]:
        """Return all traces for a given recording ID."""
        ...

    def update_bytes_uploaded(self, trace_id: str, bytes_uploaded: int) -> None:
        """Increment uploaded byte count for a trace."""
        ...

    def mark_trace_as_written(self, trace_id: str, bytes_written: int) -> None:
        """Finalize writing for a trace and mark it ready for upload."""
        ...

    def find_ready_traces(self) -> list[TraceRecord]:
        """Return all traces marked as ready for upload."""
        ...

    def find_unreported_traces(self) -> list[TraceRecord]:
        """Return all traces that have not been progress-reported."""
        ...

    def claim_ready_traces(self, limit: int = 50) -> list[Mapping[str, Any]]:
        """Claim ready traces for upload and mark them in-progress."""
        ...

    def mark_recording_reported(self, recording_id: str) -> None:
        """Mark a recording as progress-reported."""
        ...

    def update_status(
        self,
        trace_id: str,
        status: TraceStatus,
        *,
        error_message: str | None = None,
    ) -> None:
        """Update the status and optional error message for a trace."""
        ...

    def record_error(
        self,
        trace_id: str,
        error_message: str,
        error_code: TraceErrorCode | None = None,
        status: TraceStatus = TraceStatus.FAILED,
    ) -> None:
        """Record a standardized error for a trace."""
        ...

    def delete_trace(self, trace_id: str) -> None:
        """Delete a trace record."""
        ...
