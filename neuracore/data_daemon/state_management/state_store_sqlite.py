"""SQLite-backed trace state store."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import case, delete, func, or_, select, text, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from neuracore.data_daemon.models import (
    DataType,
    ProgressReportStatus,
    RecordingProgressSnapshot,
    TraceErrorCode,
    TraceRegistrationStatus,
    TraceRecord,
    TraceUploadStatus,
    TraceWriteStatus,
)

from .state_store import StateStore
from .tables import metadata, traces

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


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
                .where(traces.c.stopped_at.is_(None))
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

    async def list_traces(self) -> list[TraceRecord]:
        """Return all trace records."""
        async with self._engine.begin() as conn:
            rows = (await conn.execute(select(traces))).mappings().all()
        return [TraceRecord.from_row(dict(row)) for row in rows]

    async def _update_lifecycle_status_column(
        self, trace_id: str, *, column_name: str, value: Any
    ) -> None:
        """Update one lifecycle status column for a trace."""
        now = _utc_now()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                update(traces)
                .where(traces.c.trace_id == trace_id)
                .values({column_name: value, "last_updated": now})
            )
            if not result.rowcount:
                raise ValueError(f"Trace not found: {trace_id}")

    async def update_write_status(
        self, trace_id: str, write_status: TraceWriteStatus
    ) -> None:
        """Update write lifecycle status for a trace."""
        await self._update_lifecycle_status_column(
            trace_id, column_name="write_status", value=write_status
        )

    async def update_registration_status(
        self, trace_id: str, registration_status: TraceRegistrationStatus
    ) -> None:
        """Update registration lifecycle status for a trace."""
        await self._update_lifecycle_status_column(
            trace_id, column_name="registration_status", value=registration_status
        )

    async def _mark_traces_registration_status(
        self, trace_ids: list[str], registration_status: TraceRegistrationStatus
    ) -> list[str]:
        """Batch set registration lifecycle status for traces."""
        if not trace_ids:
            return []

        # Keep only IDs that currently exist; caller can drop the rest from
        # subsequent workflow steps.
        unique_ids = list(dict.fromkeys(trace_ids))
        async with self._engine.begin() as conn:
            existing_rows = (
                (
                    await conn.execute(
                        select(traces.c.trace_id).where(
                            traces.c.trace_id.in_(unique_ids)
                        )
                    )
                )
                .scalars()
                .all()
            )
        existing_ids = [str(trace_id) for trace_id in existing_rows]
        if not existing_ids:
            return []

        now = _utc_now()
        async with self._engine.begin() as conn:
            await conn.execute(
                update(traces)
                .where(traces.c.trace_id.in_(existing_ids))
                .values(
                    registration_status=registration_status,
                    last_updated=now,
                )
            )
        return existing_ids

    async def mark_traces_as_registering(self, trace_ids: list[str]) -> list[str]:
        """Batch mark traces as registering."""
        return await self._mark_traces_registration_status(
            trace_ids, TraceRegistrationStatus.REGISTERING
        )

    async def mark_traces_as_registered(self, trace_ids: list[str]) -> list[str]:
        """Batch mark traces as registered."""
        return await self._mark_traces_registration_status(
            trace_ids, TraceRegistrationStatus.REGISTERED
        )

    async def update_upload_status(
        self, trace_id: str, upload_status: TraceUploadStatus
    ) -> None:
        """Update upload lifecycle status for a trace."""
        await self._update_lifecycle_status_column(
            trace_id, column_name="upload_status", value=upload_status
        )

    async def record_error(
        self,
        trace_id: str,
        error_message: str,
        error_code: TraceErrorCode | None = None,
    ) -> None:
        """Record a standardized error for a trace.

        Args:
            trace_id (str): Unique identifier for the trace.
            error_message (str): Error message of the error.
            error_code (TraceErrorCode | None): Error code of the
            error, by default None.
        """
        now = _utc_now()
        async with self._engine.begin() as conn:
            await conn.execute(
                update(traces)
                .where(traces.c.trace_id == trace_id)
                .values(
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
                        .where(traces.c.write_status == TraceWriteStatus.WRITTEN)
                        .where(
                            traces.c.registration_status
                            == TraceRegistrationStatus.REGISTERED
                        )
                        .where(
                            traces.c.upload_status.in_(
                                (
                                    TraceUploadStatus.PENDING,
                                    TraceUploadStatus.RETRYING,
                                )
                            )
                        )
                        .where(traces.c.path.is_not(None))
                        .where(traces.c.data_type.is_not(None))
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

    async def claim_traces_for_registration(
        self, limit: int = 200, max_wait_s: float = 1
    ) -> list[TraceRecord]:
        """Claim traces ready for registration by transitioning to REGISTERING.

        Selection criteria:
        - write_status == WRITTEN
        - registration_status == PENDING
        Ordered by created_at ascending.
        Claim policy:
        - if at least `limit` candidates exist: claim immediately
        - otherwise: claim only candidates older than `max_wait_s`
          using `last_updated` as "became ready" timestamp
        """
        if limit <= 0 or max_wait_s < 0:
            return []

        now = _utc_now()
        async with self._engine.begin() as conn:
            candidate_rows = (
                (
                    await conn.execute(
                        select(traces.c.trace_id, traces.c.last_updated)
                        .where(traces.c.write_status == TraceWriteStatus.WRITTEN)
                        .where(
                            traces.c.registration_status
                            == TraceRegistrationStatus.PENDING
                        )
                        .order_by(traces.c.created_at.asc())
                        .limit(int(limit))
                    )
                )
                .all()
            )
            logger.debug(
                "claim_traces_for_registration fetched %d candidate rows (limit=%d, max_wait_s=%.2f)",
                len(candidate_rows),
                limit,
                max_wait_s,
            )

            if len(candidate_rows) >= int(limit):
                candidate_ids = [str(row[0]) for row in candidate_rows[: int(limit)]]
            else:
                cutoff = now - timedelta(seconds=float(max_wait_s))
                candidate_ids = [
                    str(trace_id)
                    for trace_id, last_updated in candidate_rows
                    if last_updated is not None and last_updated <= cutoff
                ]
            if not candidate_ids:
                logger.debug("claim_traces_for_registration selected no claimable ids")
                return []
            logger.info(
                "claim_traces_for_registration claiming %d traces (sample_ids=%s)",
                len(candidate_ids),
                candidate_ids[:5],
            )

            await conn.execute(
                update(traces)
                .where(traces.c.trace_id.in_(candidate_ids))
                .where(traces.c.registration_status == TraceRegistrationStatus.PENDING)
                .values(
                    registration_status=TraceRegistrationStatus.REGISTERING,
                    last_updated=now,
                )
            )

            claimed_rows = (
                (
                    await conn.execute(
                        select(traces)
                        .where(traces.c.trace_id.in_(candidate_ids))
                        .where(
                            traces.c.registration_status
                            == TraceRegistrationStatus.REGISTERING
                        )
                        .where(traces.c.last_updated == now)
                    )
                )
                .mappings()
                .all()
            )
            logger.info(
                "claim_traces_for_registration claimed %d rows",
                len(claimed_rows),
            )

        return [TraceRecord.from_row(dict(row)) for row in claimed_rows]

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
                        select(traces).where(
                            traces.c.upload_status == TraceUploadStatus.FAILED
                        )
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

    async def get_recording_progress_snapshot(
        self, recording_id: str
    ) -> RecordingProgressSnapshot | None:
        """Return aggregate progress-report gate state for one recording."""
        async with self._engine.begin() as conn:
            row = (
                (
                    await conn.execute(
                        select(
                            func.count().label("trace_count"),
                            func.sum(
                                case(
                                    (
                                        traces.c.progress_reported
                                        == ProgressReportStatus.REPORTED,
                                        1,
                                    ),
                                    else_=0,
                                )
                            ).label("reported_count"),
                            func.sum(
                                case((traces.c.stopped_at.is_(None), 1), else_=0)
                            ).label("missing_stopped_at_count"),
                            func.sum(
                                case((traces.c.data_type.is_(None), 1), else_=0)
                            ).label("missing_data_type_count"),
                            func.sum(
                                case((traces.c.bytes_written.is_(None), 1), else_=0)
                            ).label("missing_bytes_written_count"),
                            func.sum(
                                case((traces.c.total_bytes.is_(None), 1), else_=0)
                            ).label("missing_total_bytes_count"),
                            func.sum(
                                case(
                                    (
                                        traces.c.bytes_written.is_not(None)
                                        & traces.c.total_bytes.is_not(None)
                                        & (traces.c.bytes_written != traces.c.total_bytes),
                                        1,
                                    ),
                                    else_=0,
                                )
                            ).label("mismatched_bytes_count"),
                        ).where(traces.c.recording_id == recording_id)
                    )
                )
                .mappings()
                .one()
            )

        trace_count = int(row["trace_count"] or 0)
        if trace_count == 0:
            return None

        return RecordingProgressSnapshot(
            recording_id=recording_id,
            trace_count=trace_count,
            reported_count=int(row["reported_count"] or 0),
            missing_stopped_at_count=int(row["missing_stopped_at_count"] or 0),
            missing_data_type_count=int(row["missing_data_type_count"] or 0),
            missing_bytes_written_count=int(row["missing_bytes_written_count"] or 0),
            missing_total_bytes_count=int(row["missing_total_bytes_count"] or 0),
            mismatched_bytes_count=int(row["mismatched_bytes_count"] or 0),
        )

    async def recording_has_reported_progress(self, recording_id: str) -> bool:
        """Return True when any trace in recording is already REPORTED."""
        async with self._engine.begin() as conn:
            row = (
                (
                    await conn.execute(
                        select(traces.c.trace_id)
                        .where(traces.c.recording_id == recording_id)
                        .where(
                            traces.c.progress_reported == ProgressReportStatus.REPORTED
                        )
                        .limit(1)
                    )
                )
                .scalar_one_or_none()
            )
        return row is not None

    async def delete_uploaded_traces_for_recording(self, recording_id: str) -> int:
        """Delete all UPLOADED traces for one recording and return deleted count."""
        async with self._engine.begin() as conn:
            result = await conn.execute(
                delete(traces)
                .where(traces.c.recording_id == recording_id)
                .where(traces.c.upload_status == TraceUploadStatus.UPLOADED)
            )
        return int(result.rowcount or 0)

    async def count_traces_for_recording(self, recording_id: str) -> int:
        """Return count of traces associated with recording."""
        async with self._engine.begin() as conn:
            count = (
                (
                    await conn.execute(
                        select(func.count()).where(
                            traces.c.recording_id == recording_id
                        )
                    )
                )
                .scalar_one()
            )
        return int(count)

    async def list_recording_ids_with_stopped_traces(self) -> list[str]:
        """Return recording IDs that already have at least one stopped trace."""
        async with self._engine.begin() as conn:
            rows = (
                (
                    await conn.execute(
                        select(traces.c.recording_id)
                        .where(traces.c.stopped_at.is_not(None))
                        .where(
                            traces.c.progress_reported == ProgressReportStatus.PENDING
                        )
                        .distinct()
                    )
                )
                .scalars()
                .all()
            )
        return [str(recording_id) for recording_id in rows]

    async def mark_expected_trace_count_reported(self, recording_id: str) -> None:
        """Mark a recording's expected trace count as reported."""
        now = _utc_now()
        async with self._engine.begin() as conn:
            await conn.execute(
                update(traces)
                .where(traces.c.recording_id == recording_id)
                .values(expected_trace_count_reported=1, last_updated=now)
            )

    async def reset_failed_trace_for_retry(self, trace_id: str) -> None:
        """Reset a failed trace back to WRITTEN for retry."""
        now = _utc_now()
        async with self._engine.begin() as conn:
            await conn.execute(
                update(traces)
                .where(traces.c.trace_id == trace_id)
                .values(
                    write_status=TraceWriteStatus.WRITTEN,
                    upload_status=TraceUploadStatus.PENDING,
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
            write_status=TraceWriteStatus.INITIALIZING,
            registration_status=TraceRegistrationStatus.PENDING,
            upload_status=TraceUploadStatus.PENDING,
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
            # and entry exists, set status/write_status to WRITTEN
            "write_status": case(
                (
                    traces.c.write_status == TraceWriteStatus.PENDING_METADATA,
                    TraceWriteStatus.WRITTEN,
                ),
                else_=traces.c.write_status,
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
            write_status=TraceWriteStatus.PENDING_METADATA,
            registration_status=TraceRegistrationStatus.PENDING,
            upload_status=TraceUploadStatus.PENDING,
            bytes_uploaded=0,
            progress_reported=ProgressReportStatus.PENDING,
            created_at=now,
            last_updated=now,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["trace_id"],
            set_={
                "bytes_written": bytes_written,
                "total_bytes": case(
                    (
                        traces.c.total_bytes.is_(None),
                        bytes_written,
                    ),
                    else_=traces.c.total_bytes,
                ),
                "last_updated": now,
                "write_status": case(
                    (
                        traces.c.write_status == TraceWriteStatus.INITIALIZING,
                        TraceWriteStatus.WRITTEN,
                    ),
                    else_=traces.c.write_status,
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
                    upload_status=TraceUploadStatus.RETRYING,
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
                    upload_status=TraceUploadStatus.FAILED,
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
        """Reset RETRYING/UPLOADING traces back to upload PENDING."""
        now = _utc_now()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                update(traces)
                .where(
                    traces.c.upload_status.in_(
                        (TraceUploadStatus.RETRYING, TraceUploadStatus.UPLOADING)
                    )
                )
                .values(
                    upload_status=TraceUploadStatus.PENDING,
                    last_updated=now,
                )
            )
        return int(result.rowcount or 0)

    async def close(self) -> None:
        """Close the database connection and dispose of the engine.

        This must be called before the event loop closes to prevent
        aiosqlite worker thread exceptions.
        """
        await self._engine.dispose()
