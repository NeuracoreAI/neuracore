"""SQLite-backed trace state store."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import case, delete, or_, select, text, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from neuracore.data_daemon.models import (
    DataType,
    ProgressReportStatus,
    TraceErrorCode,
    TraceRecord,
    TraceStatus,
)

from .state_store import StateStore
from .tables import metadata, traces

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


_ALLOWED_PREVIOUS_STATUSES: dict[TraceStatus, set[TraceStatus]] = {
    TraceStatus.WRITTEN: {
        TraceStatus.INITIALIZING,
        TraceStatus.PENDING_METADATA,
        TraceStatus.RETRYING,
    },
    TraceStatus.UPLOADING: {
        TraceStatus.WRITTEN,
        TraceStatus.PAUSED,
        TraceStatus.RETRYING,
    },
    TraceStatus.UPLOADED: {TraceStatus.UPLOADING},
    TraceStatus.PAUSED: {TraceStatus.UPLOADING},
}


class SqliteStateStore(StateStore):
    """SQLite StateStore for trace state only."""

    def __init__(self, db_path: Path) -> None:
        """Initialize the SQLite engine and ensure schema."""
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._engine: AsyncEngine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            future=True,
        )

    async def init_async_store(self) -> None:
        """Apply pragmas and ensure schema."""
        await self._apply_pragmas()
        await self._ensure_schema()

    async def _apply_pragmas(self) -> None:
        """Apply database pragmas for better performance.

        Sets the journal mode to WAL (Write-Ahead Logging) to ensure that
        database changes are written to disk immediately. Sets the synchronous
        mode to NORMAL to prevent the database from blocking on disk I/O.
        """
        async with self._engine.begin() as conn:
            await conn.execute(text("PRAGMA journal_mode=WAL;"))
            await conn.execute(text("PRAGMA synchronous=NORMAL;"))

    async def _ensure_schema(self) -> None:
        """Ensures that the database schema is created.

        Calls :meth:`sqlalchemy.Meta.create_all` on the :attr:`_engine` to create
        the database schema if it does not already exist.
        """
        async with self._engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def find_retry_scheduled_traces(self) -> list[TraceRecord]:
        """Return WRITTEN traces with persisted next_retry_at values."""
        async with self._engine.begin() as conn:
            rows = (
                (
                    await conn.execute(
                        select(traces)
                        .where(traces.c.status == TraceStatus.WRITTEN)
                        .where(traces.c.next_retry_at.is_not(None))
                        .order_by(traces.c.next_retry_at.asc())
                    )
                )
                .mappings()
                .all()
            )
        return [TraceRecord.from_row(dict(row)) for row in rows]

    async def set_stopped_ats(self, recording_id: str) -> None:
        """Set the end stopped_at for all traces associated with a recording.

        Updates the stopped_at of all traces associated with a recording to the
        current UTC time.

        Args:
            recording_id (str): The unique identifier for the recording.
        """
        async with self._engine.begin() as conn:
            await conn.execute(
                update(traces)
                .where(traces.c.recording_id == recording_id)
                .values(stopped_at=_utc_now())
            )

    async def update_bytes_uploaded(self, trace_id: str, bytes_uploaded: int) -> None:
        """Increment the number of bytes uploaded for a trace.

        Args:
            trace_id (str): unique identifier for the trace.
            bytes_uploaded (int): number of bytes uploaded.
        """
        now = _utc_now()
        async with self._engine.begin() as conn:
            await conn.execute(
                update(traces)
                .where(traces.c.trace_id == trace_id)
                .values(
                    bytes_uploaded=int(bytes_uploaded),
                    last_updated=now,
                )
            )

    async def get_trace(self, trace_id: str) -> TraceRecord | None:
        """Return a trace record by ID.

        Args:
            trace_id (str): Unique identifier for the trace.

        Returns:
            TraceRecord | None: The trace record if it exists, otherwise None.
        """
        async with self._engine.begin() as conn:
            row = (
                (
                    await conn.execute(
                        select(traces).where(traces.c.trace_id == trace_id)
                    )
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            return None
        return TraceRecord.from_row(dict(row))

    async def find_traces_by_recording_id(self, recording_id: str) -> list[TraceRecord]:
        """Return all traces associated with a recording ID.

        Args:
            recording_id (str): Unique identifier for the recording.

        Returns:
            list[TraceRecord]: A list of trace records associated with the recording ID.
        """
        async with self._engine.begin() as conn:
            rows = (
                (
                    await conn.execute(
                        select(traces).where(traces.c.recording_id == recording_id)
                    )
                )
                .mappings()
                .all()
            )
        return [TraceRecord.from_row(dict(row)) for row in rows]

    def list_traces(self) -> list[TraceRecord]:
        """Return all trace records."""
        with self._engine.begin() as conn:
            rows = conn.execute(select(traces)).mappings().all()
        return [TraceRecord.from_row(dict(row)) for row in rows]

    async def update_status(
        self,
        trace_id: str,
        status: TraceStatus,
        error_message: str | None = None,
    ) -> bool:
        """Update the status and optional error message for a trace.

        Args:
            trace_id (str): Unique identifier for the trace.
            status (TraceStatus): New status for the trace.
            error_message (str | None): Optional error message to
            associate with the trace.

        Returns:
            True if status was changed, False if already at target status.

        Raises:
            ValueError: If trace not found or invalid transition.
        """
        now = _utc_now()
        values: dict[str, Any] = {"status": status, "last_updated": now}
        if error_message is not None:
            values["error_message"] = error_message
        allowed_previous = _ALLOWED_PREVIOUS_STATUSES.get(status, set())
        async with self._engine.begin() as conn:
            # No previous enforcement for FAILED status
            if status == TraceStatus.FAILED:
                result = await conn.execute(
                    update(traces).where(traces.c.trace_id == trace_id).values(**values)
                )
            elif allowed_previous:
                result = await conn.execute(
                    update(traces)
                    .where(traces.c.trace_id == trace_id)
                    .where(traces.c.status.in_(allowed_previous))
                    .values(**values)
                )
            else:
                result = await conn.execute(
                    update(traces)
                    .where(traces.c.trace_id == trace_id)
                    .where(traces.c.status == status)
                    .values(**values)
                )
            if result.rowcount == 0:
                current = (
                    await conn.execute(
                        select(traces.c.status).where(traces.c.trace_id == trace_id)
                    )
                ).scalar_one_or_none()
                if current is None:
                    raise ValueError(f"Trace not found: {trace_id}")
                if current == status:
                    logger.debug(
                        "Trace %s already has status %s (no-op)", trace_id, status
                    )
                    return False
                raise ValueError(
                    f"Invalid status transition {current} -> {status} for {trace_id}"
                )
            return True

    async def record_error(
        self,
        trace_id: str,
        error_message: str,
        error_code: TraceErrorCode | None = None,
        status: TraceStatus = TraceStatus.FAILED,
    ) -> None:
        """Record a standardized error for a trace.

        Args:
            trace_id (str): Unique identifier for the trace.
            error_message (str): Error message of the error.
            error_code (TraceErrorCode | None): Error code of the
            error, by default None.
            status (TraceStatus): Status to set for the trace after
            recording the error, by default TraceStatus.FAILED.
        """
        now = _utc_now()
        async with self._engine.begin() as conn:
            await conn.execute(
                update(traces)
                .where(traces.c.trace_id == trace_id)
                .values(
                    status=status,
                    error_message=error_message,
                    error_code=error_code.value if error_code else None,
                    last_updated=now,
                )
            )

    async def delete_trace(self, trace_id: str) -> None:
        """Delete a trace record.

        Args:
            trace_id (str): Unique identifier for the trace to delete.
        """
        async with self._engine.begin() as conn:
            await conn.execute(delete(traces).where(traces.c.trace_id == trace_id))

    async def find_ready_traces(self) -> list[TraceRecord]:
        """Return traces ready to start an upload attempt.

        Args:
            None

        Returns:
            list[TraceRecord]: Traces eligible for upload.
        """
        now = _utc_now()

        async with self._engine.begin() as conn:
            rows = (
                (
                    await conn.execute(
                        select(traces)
                        .where(traces.c.status == TraceStatus.WRITTEN)
                        .where(
                            or_(
                                # First time trying to upload
                                traces.c.next_retry_at.is_(None),
                                # Retry upload
                                traces.c.next_retry_at <= now,
                            )
                        )
                        .order_by(traces.c.created_at.asc())
                    )
                )
                .mappings()
                .all()
            )

        return [TraceRecord.from_row(dict(row)) for row in rows]

    async def find_unreported_traces(self) -> list[TraceRecord]:
        """Return all traces that have not been progress-reported."""
        async with self._engine.begin() as conn:
            rows = (
                (
                    await conn.execute(
                        select(traces).where(
                            traces.c.progress_reported == ProgressReportStatus.PENDING
                        )
                    )
                )
                .mappings()
                .all()
            )
        return [TraceRecord.from_row(dict(row)) for row in rows]

    async def find_failed_traces(self) -> list[TraceRecord]:
        """Return all traces marked as FAILED."""
        async with self._engine.begin() as conn:
            rows = (
                (
                    await conn.execute(
                        select(traces).where(traces.c.status == TraceStatus.FAILED)
                    )
                )
                .mappings()
                .all()
            )
        return [TraceRecord.from_row(dict(row)) for row in rows]

    async def mark_recording_reported(self, recording_id: str) -> None:
        """Mark a recording as progress-reported."""
        now = _utc_now()
        async with self._engine.begin() as conn:
            await conn.execute(
                update(traces)
                .where(traces.c.recording_id == recording_id)
                .values(
                    progress_reported=ProgressReportStatus.REPORTED, last_updated=now
                )
            )

    async def reset_failed_trace_for_retry(self, trace_id: str) -> None:
        """Reset a failed trace back to WRITTEN for retry."""
        now = _utc_now()
        async with self._engine.begin() as conn:
            await conn.execute(
                update(traces)
                .where(traces.c.trace_id == trace_id)
                .values(
                    status=TraceStatus.WRITTEN,
                    error_code=None,
                    error_message=None,
                    next_retry_at=None,
                    num_upload_attempts=0,
                    bytes_uploaded=0,
                    last_updated=now,
                )
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
        """Insert or update trace with metadata from START_TRACE.

        State transitions:
        - If trace doesn't exist: creates with INITIALIZING status
        - If trace exists with PENDING_METADATA: transitions to WRITTEN
        - If trace exists with other status: updates metadata only

        Returns the trace record after upsert.
        """
        now = _utc_now()
        stmt = insert(traces).values(
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
            total_bytes=None,
            status=TraceStatus.INITIALIZING,
            bytes_uploaded=0,
            progress_reported=ProgressReportStatus.PENDING,
            created_at=now,
            last_updated=now,
        )
        update_set: dict[str, Any] = {
            "data_type": data_type,
            "data_type_name": data_type_name,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "robot_name": robot_name,
            "robot_id": robot_id,
            "robot_instance": robot_instance,
            "path": path,
            "last_updated": now,
            # If trace_written received before metadata,
            # and entry exists, set status to WRITTEN
            "status": case(
                (
                    traces.c.status == TraceStatus.PENDING_METADATA,
                    TraceStatus.WRITTEN,
                ),
                else_=traces.c.status,
            ),
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["trace_id"],
            set_=update_set,
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)
            row = (
                (
                    await conn.execute(
                        select(traces).where(traces.c.trace_id == trace_id)
                    )
                )
                .mappings()
                .one()
            )
        return TraceRecord.from_row(dict(row))

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
        now = _utc_now()
        stmt = insert(traces).values(
            trace_id=trace_id,
            recording_id=recording_id,
            bytes_written=bytes_written,
            total_bytes=bytes_written,
            status=TraceStatus.PENDING_METADATA,
            bytes_uploaded=0,
            progress_reported=ProgressReportStatus.PENDING,
            created_at=now,
            last_updated=now,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["trace_id"],
            set_={
                "bytes_written": bytes_written,
                "total_bytes": bytes_written,
                "last_updated": now,
                "status": case(
                    (
                        traces.c.status == TraceStatus.INITIALIZING,
                        TraceStatus.WRITTEN,
                    ),
                    else_=traces.c.status,
                ),
            },
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)
            row = (
                (
                    await conn.execute(
                        select(traces).where(traces.c.trace_id == trace_id)
                    )
                )
                .mappings()
                .one()
            )
        return TraceRecord.from_row(dict(row))

    async def schedule_retry(
        self,
        trace_id: str,
        *,
        next_retry_at: datetime,
        error_code: TraceErrorCode,
        error_message: str,
    ) -> int:
        """Schedule a retry for a failed upload attempt.

        Args:
            trace_id: Unique identifier for the trace.
            next_retry_at: When the next retry is due (naive UTC).
            error_code: Error code describing the failure type.
            error_message: Human-readable failure message.

        Returns:
            int: Updated num_upload_attempts value.

        Raises:
            ValueError: If the trace does not exist.
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        async with self._engine.begin() as conn:
            await conn.execute(
                update(traces)
                .where(traces.c.trace_id == trace_id)
                .values(
                    status=TraceStatus.RETRYING,
                    error_code=error_code.value,
                    error_message=error_message,
                    next_retry_at=next_retry_at,
                    num_upload_attempts=traces.c.num_upload_attempts + 1,
                    last_updated=now,
                )
            )
            attempts = (
                await conn.execute(
                    select(traces.c.num_upload_attempts).where(
                        traces.c.trace_id == trace_id
                    )
                )
            ).scalar_one_or_none()
        if attempts is None:
            raise ValueError(f"Trace not found: {trace_id}")
        return int(attempts)

    async def mark_retry_exhausted(
        self,
        trace_id: str,
        *,
        error_code: TraceErrorCode,
        error_message: str,
    ) -> int:
        """Mark a trace as permanently failed due to exhausted retries.

        Args:
            trace_id: Unique identifier for the trace.
            error_code: Error code describing the failure type.
            error_message: Human-readable failure message.

        Returns:
            int: Updated num_upload_attempts value.

        Raises:
            ValueError: If the trace does not exist.
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        async with self._engine.begin() as conn:
            await conn.execute(
                update(traces)
                .where(traces.c.trace_id == trace_id)
                .values(
                    status=TraceStatus.FAILED,
                    error_code=error_code.value,
                    error_message=error_message,
                    next_retry_at=None,
                    num_upload_attempts=traces.c.num_upload_attempts + 1,
                    last_updated=now,
                )
            )
            attempts = (
                await conn.execute(
                    select(traces.c.num_upload_attempts).where(
                        traces.c.trace_id == trace_id
                    )
                )
            ).scalar_one_or_none()
        if attempts is None:
            raise ValueError(f"Trace not found: {trace_id}")
        return int(attempts)

    async def reset_retrying_to_written(self) -> int:
        """Reset RETRYING/UPLOADING traces back to WRITTEN (preserve retry schedule)."""
        now = _utc_now()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                update(traces)
                .where(
                    traces.c.status.in_((TraceStatus.RETRYING, TraceStatus.UPLOADING))
                )
                .values(status=TraceStatus.WRITTEN, last_updated=now)
            )
        return int(result.rowcount or 0)

    async def close(self) -> None:
        """Close the database connection and dispose of the engine.

        This must be called before the event loop closes to prevent
        aiosqlite worker thread exceptions.
        """
        await self._engine.dispose()
