"""SQLite-backed trace state store."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import delete, insert, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from neuracore.data_daemon.models import (
    DataType,
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
    TraceStatus.WRITING: {TraceStatus.PENDING},
    TraceStatus.WRITTEN: {TraceStatus.PENDING, TraceStatus.WRITING},
    TraceStatus.UPLOADING: {TraceStatus.WRITTEN, TraceStatus.PAUSED},
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
        """Create a trace record.

        This function creates a trace record or updates an existing trace record if
        a trace with the same trace ID already exists.

        Args:
            trace_id (str): Unique identifier for the trace.
            recording_id (str): Unique identifier for the recording.
            data_type (DataType): Type of data being recorded.
            path (str): Path to the trace file.
            data_type_name (str | None): Name of the data type.
            dataset_id (str | None): Unique identifier for the dataset.
            dataset_name (str | None): Name of the dataset.
            robot_name (str | None): Name of the robot.
            robot_id (str | None): Unique identifier for the robot.
            total_bytes (int | None): Total number of bytes in the trace file.

        Returns:
            None
        """
        now = _utc_now()
        async with self._engine.begin() as conn:
            try:
                await conn.execute(
                    insert(traces).values(
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
                        status=TraceStatus.PENDING,
                        bytes_written=0,
                        bytes_uploaded=0,
                        ready_for_upload=0,
                        progress_reported=0,
                        error_code=None,
                        error_message=None,
                        created_at=now,
                        last_updated=now,
                    )
                )
            except IntegrityError:
                await conn.execute(
                    update(traces)
                    .where(traces.c.trace_id == trace_id)
                    .values(
                        recording_id=recording_id,
                        data_type=data_type,
                        data_type_name=data_type_name,
                        path=path,
                        total_bytes=total_bytes,
                        last_updated=now,
                    )
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

    async def update_status(
        self,
        trace_id: str,
        status: TraceStatus,
        error_message: str | None = None,
    ) -> None:
        """Update the status and optional error message for a trace.

        Args:
            trace_id (str): Unique identifier for the trace.
            status (TraceStatus): New status for the trace.
            error_message (str | None): Optional error message to
            associate with the trace.
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
                    logger.warning(
                        f"Failed to update trace status: Trace {trace_id} "
                        f"already has status {status}"
                    )
                    return
                raise ValueError(
                    f"Invalid status transition {current} -> {status} for {trace_id}"
                )

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

    async def mark_trace_as_written(self, trace_id: str, bytes_written: int) -> None:
        """Mark a trace as written, ready for upload.

        Args:
            trace_id (str): Unique identifier for the trace.
            bytes_written (int): Number of bytes written to the trace.

        Updates the trace record in the database to mark it as written
        and ready for upload.
        """
        async with self._engine.begin() as conn:
            result = await conn.execute(
                update(traces)
                .where(traces.c.trace_id == trace_id)
                .where(
                    traces.c.status.in_(
                        [TraceStatus.PENDING, TraceStatus.WRITING, TraceStatus.WRITTEN]
                    )
                )
                .values(
                    status=TraceStatus.WRITTEN,
                    last_updated=_utc_now(),
                    total_bytes=bytes_written,
                    ready_for_upload=1,
                    bytes_written=bytes_written,
                )
            )
            if result.rowcount == 0:
                current = (
                    await conn.execute(
                        select(traces.c.status).where(traces.c.trace_id == trace_id)
                    )
                ).scalar_one_or_none()
                if current is None:
                    raise ValueError(f"Trace not found: {trace_id}")
                raise ValueError(
                    f"Invalid status transition {current} -> {TraceStatus.WRITTEN} "
                    f"for {trace_id}"
                )

    async def find_ready_traces(self) -> list[TraceRecord]:
        """Return all traces marked as ready for upload."""
        async with self._engine.begin() as conn:
            rows = (
                (
                    await conn.execute(
                        select(traces).where(traces.c.ready_for_upload == 1)
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
                        select(traces).where(traces.c.progress_reported == 0)
                    )
                )
                .mappings()
                .all()
            )
        return [TraceRecord.from_row(dict(row)) for row in rows]

    async def claim_ready_traces(self, limit: int = 50) -> list[Mapping[str, Any]]:
        """Claim ready traces for upload and mark them in-progress."""
        async with self._engine.begin() as conn:
            rows = (
                (
                    await conn.execute(
                        select(traces)
                        .where(
                            (traces.c.ready_for_upload == 1)
                            & (traces.c.status == TraceStatus.WRITTEN)
                        )
                        .order_by(traces.c.last_updated.asc())
                        .limit(int(limit))
                    )
                )
                .mappings()
                .all()
            )
            if not rows:
                return []
            trace_ids = [row["trace_id"] for row in rows]
            now = _utc_now()
            await conn.execute(
                update(traces)
                .where(traces.c.trace_id.in_(trace_ids))
                .where(traces.c.ready_for_upload == 1)
                .where(traces.c.status == TraceStatus.WRITTEN)
                .values(
                    ready_for_upload=0,
                    status=TraceStatus.UPLOADING,
                    last_updated=now,
                )
            )
            updated_rows = (
                (
                    await conn.execute(
                        select(traces)
                        .where(traces.c.trace_id.in_(trace_ids))
                        .where(traces.c.status == TraceStatus.UPLOADING)
                        .where(traces.c.last_updated == now)
                    )
                )
                .mappings()
                .all()
            )
            return [dict(row) for row in updated_rows]

    async def mark_recording_reported(self, recording_id: str) -> None:
        """Mark a recording as progress-reported."""
        now = _utc_now()
        async with self._engine.begin() as conn:
            await conn.execute(
                update(traces)
                .where(traces.c.recording_id == recording_id)
                .values(progress_reported=1, last_updated=now)
            )

    async def set_external_trace_id(
        self, trace_id: str, external_trace_id: str
    ) -> None:
        """Store the external trace ID for a trace.

        Args:
            trace_id: Internal trace identifier.
            external_trace_id: Backend-generated trace ID to persist.
        """
        now = _utc_now()
        async with self._engine.begin() as conn:
            await conn.execute(
                update(traces)
                .where(traces.c.trace_id == trace_id)
                .values(external_trace_id=external_trace_id, last_updated=now)
            )

    async def close(self) -> None:
        """Close the database connection and dispose of the engine.

        This must be called before the event loop closes to prevent
        aiosqlite worker thread exceptions.
        """
        await self._engine.dispose()
