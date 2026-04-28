"""SQLite-backed trace state store."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import and_, case, delete, func, or_, select, text, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from neuracore.data_daemon.models import (
    DataType,
    ProgressReportStatus,
    RecordingStopReportStatus,
    TraceErrorCode,
    TraceRecord,
    TraceRegistrationStatus,
    TraceUploadStatus,
    TraceWriteStatus,
)

from .state_store import StateStore
from .tables import metadata, recordings, traces

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _parse_progress_status(value: Any) -> ProgressReportStatus:
    """Normalize DB/SQLAlchemy progress status values to enum."""
    if isinstance(value, ProgressReportStatus):
        return value
    if value is None:
        return ProgressReportStatus.PENDING
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("ProgressReportStatus."):
            raw = raw.split(".", 1)[1]
        return ProgressReportStatus(raw.lower())
    return ProgressReportStatus(str(value))


def _parse_stop_report_status(value: Any) -> RecordingStopReportStatus:
    """Normalize DB/SQLAlchemy stop-report status values to enum."""
    if isinstance(value, RecordingStopReportStatus):
        return value
    if value is None:
        return RecordingStopReportStatus.PENDING
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("RecordingStopReportStatus."):
            raw = raw.split(".", 1)[1]
        return RecordingStopReportStatus(raw.lower())
    return RecordingStopReportStatus(str(value))


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
        self._write_semaphore = asyncio.Semaphore(1)

    @asynccontextmanager
    async def _write_lock(self) -> AsyncIterator[AsyncConnection]:
        async with self._write_semaphore:
            async with self._engine.begin() as conn:
                yield conn

    async def init_async_store(self) -> None:
        """Apply pragmas and ensure schema."""
        await self._apply_pragmas()
        await self._ensure_schema()
        await self.reconcile_recordings_from_traces()

    async def _apply_pragmas(self) -> None:
        """Apply database pragmas for better performance.

        Sets the journal mode to WAL (Write-Ahead Logging) to ensure that
        database changes are written to disk immediately. Sets the synchronous
        mode to NORMAL to prevent the database from blocking on disk I/O.
        """
        async with self._write_lock() as conn:
            await conn.execute(text("PRAGMA journal_mode=WAL;"))
            await conn.execute(text("PRAGMA synchronous=NORMAL;"))
            await conn.execute(text("PRAGMA busy_timeout=1000;"))

    async def _ensure_schema(self) -> None:
        """Ensures that the database schema is created.

        Calls :meth:`sqlalchemy.Meta.create_all` on the :attr:`_engine` to create
        the database schema if it does not already exist.
        """
        async with self._write_lock() as conn:
            await self._migrate_legacy_traces_schema_if_needed(conn)
            await conn.run_sync(metadata.create_all)
            await self._ensure_recordings_trace_count_col(conn)
            await self._ensure_recordings_org_id_col(conn)
            await self._ensure_recordings_stop_report_columns(conn)

    async def _table_exists(self, conn: AsyncConnection, table_name: str) -> bool:
        """Return True when a SQLite table exists."""
        result = await conn.execute(
            text(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = :table_name LIMIT 1"
            ),
            {"table_name": table_name},
        )
        return result.scalar_one_or_none() is not None

    async def _table_column_names(
        self, conn: AsyncConnection, table_name: str
    ) -> set[str]:
        """Return all column names for a SQLite table."""
        rows = (
            (await conn.execute(text(f"PRAGMA table_info({table_name})")))  # noqa: S608
            .mappings()
            .all()
        )
        return {str(row["name"]) for row in rows}

    async def _migrate_legacy_traces_schema_if_needed(
        self, conn: AsyncConnection
    ) -> None:
        """Migrate legacy single-table trace schema to current schema.

        Legacy schema has only `traces` table and includes `status`,
        `progress_reported`, `expected_trace_count_reported`, and `stopped_at`
        columns directly on trace rows.
        """
        if not await self._table_exists(conn, "traces"):
            return
        trace_columns = await self._table_column_names(conn, "traces")
        if "status" not in trace_columns:
            return

        logger.info("Migrating legacy trace schema to recordings + lifecycle columns")
        await conn.execute(text("DROP TABLE IF EXISTS traces_migrated"))
        await conn.execute(text("DROP TABLE IF EXISTS recordings_migrated"))
        await conn.execute(
            text(
                """
                CREATE TABLE traces_migrated (
                    trace_id TEXT PRIMARY KEY,
                    write_status TEXT NOT NULL,
                    registration_status TEXT NOT NULL,
                    upload_status TEXT NOT NULL,
                    recording_id TEXT NOT NULL,
                    data_type TEXT,
                    data_type_name TEXT,
                    dataset_id TEXT,
                    dataset_name TEXT,
                    robot_name TEXT,
                    robot_id TEXT,
                    robot_instance INTEGER,
                    path TEXT,
                    bytes_written INTEGER,
                    total_bytes INTEGER,
                    bytes_uploaded INTEGER DEFAULT 0,
                    error_code TEXT,
                    error_message TEXT,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_updated DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    num_upload_attempts INTEGER NOT NULL DEFAULT 0,
                    next_retry_at DATETIME
                )
                """
            )
        )
        await conn.execute(
            text(
                """
                CREATE TABLE recordings_migrated (
                    recording_id TEXT PRIMARY KEY,
                    org_id TEXT,
                    expected_trace_count INTEGER NOT NULL DEFAULT 0,
                    trace_count INTEGER NOT NULL DEFAULT 0,
                    expected_trace_count_reported INTEGER NOT NULL DEFAULT 0,
                    uploaded_trace_count INTEGER NOT NULL DEFAULT 0,
                    progress_reported TEXT NOT NULL DEFAULT 'pending',
                    stop_report_status TEXT NOT NULL DEFAULT 'pending',
                    stopped_at DATETIME,
                    stop_reported_at DATETIME,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_updated DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        await conn.execute(
            text(
                """
                INSERT INTO traces_migrated (
                    trace_id,
                    write_status,
                    registration_status,
                    upload_status,
                    recording_id,
                    data_type,
                    data_type_name,
                    dataset_id,
                    dataset_name,
                    robot_name,
                    robot_id,
                    robot_instance,
                    path,
                    bytes_written,
                    total_bytes,
                    bytes_uploaded,
                    error_code,
                    error_message,
                    created_at,
                    last_updated,
                    num_upload_attempts,
                    next_retry_at
                )
                SELECT
                    trace_id,
                    CASE
                        WHEN status IN ('initializing', 'pending_metadata', 'written')
                            THEN status
                        WHEN status = 'failed'
                             AND (total_bytes IS NULL OR total_bytes <= 0)
                            THEN 'failed'
                        WHEN status IN (
                            'uploading', 'retrying', 'paused', 'uploaded', 'failed'
                        )
                            THEN 'written'
                        ELSE 'pending'
                    END AS write_status,
                    CASE
                        WHEN status IN (
                            'uploading', 'retrying', 'paused', 'uploaded', 'failed'
                        )
                             AND total_bytes IS NOT NULL
                             AND total_bytes > 0
                            THEN 'registered'
                        ELSE 'pending'
                    END AS registration_status,
                    CASE
                        WHEN status IN (
                            'uploading', 'retrying', 'paused', 'uploaded', 'failed'
                        )
                            THEN status
                        ELSE 'pending'
                    END AS upload_status,
                    recording_id,
                    data_type,
                    data_type_name,
                    dataset_id,
                    dataset_name,
                    robot_name,
                    robot_id,
                    robot_instance,
                    path,
                    bytes_written,
                    total_bytes,
                    COALESCE(bytes_uploaded, 0),
                    error_code,
                    error_message,
                    COALESCE(created_at, CURRENT_TIMESTAMP),
                    COALESCE(last_updated, CURRENT_TIMESTAMP),
                    COALESCE(num_upload_attempts, 0),
                    next_retry_at
                FROM traces
                """
            )
        )
        await conn.execute(
            text(
                """
                INSERT INTO recordings_migrated (
                    recording_id,
                    org_id,
                    expected_trace_count,
                    trace_count,
                    expected_trace_count_reported,
                    uploaded_trace_count,
                    progress_reported,
                    stop_report_status,
                    stopped_at,
                    stop_reported_at,
                    created_at,
                    last_updated
                )
                SELECT
                    recording_id,
                    NULL AS org_id,
                    SUM(
                        CASE
                            WHEN total_bytes IS NOT NULL
                                 AND total_bytes > 0
                                 AND status IN (
                                     'written',
                                     'uploading',
                                     'retrying',
                                     'paused',
                                     'uploaded',
                                     'failed'
                                 )
                                THEN 1
                            ELSE 0
                        END
                    ) AS expected_trace_count,
                    SUM(
                        CASE
                            WHEN total_bytes IS NOT NULL
                                 AND total_bytes > 0
                                 AND status IN (
                                     'written',
                                     'uploading',
                                     'retrying',
                                     'paused',
                                     'uploaded',
                                     'failed'
                                 )
                                THEN 1
                            ELSE 0
                        END
                    ) AS trace_count,
                    MAX(COALESCE(expected_trace_count_reported, 0))
                        AS expected_trace_count_reported,
                    SUM(CASE WHEN status = 'uploaded' THEN 1 ELSE 0 END)
                        AS uploaded_trace_count,
                    CASE
                        WHEN MIN(CASE
                            WHEN progress_reported = 'reported' THEN 1
                            ELSE 0
                        END) = 1
                            THEN 'reported'
                        WHEN MAX(CASE
                            WHEN progress_reported = 'reporting' THEN 1
                            ELSE 0
                        END) = 1
                            THEN 'reporting'
                        ELSE 'pending'
                    END AS progress_reported,
                    'pending' AS stop_report_status,
                    MAX(stopped_at) AS stopped_at,
                    NULL AS stop_reported_at,
                    COALESCE(MIN(created_at), CURRENT_TIMESTAMP) AS created_at,
                    COALESCE(MAX(last_updated), CURRENT_TIMESTAMP) AS last_updated
                FROM traces
                GROUP BY recording_id
                """
            )
        )
        await conn.execute(text("DROP TABLE traces"))
        await conn.execute(text("DROP TABLE IF EXISTS recordings"))
        await conn.execute(text("ALTER TABLE traces_migrated RENAME TO traces"))
        await conn.execute(text("ALTER TABLE recordings_migrated RENAME TO recordings"))
        logger.info("Legacy trace schema migration complete")

    async def _ensure_recordings_trace_count_col(self, conn: AsyncConnection) -> None:
        """Add recordings.trace_count for pre-column databases."""
        if not await self._table_exists(conn, "recordings"):
            return
        recording_columns = await self._table_column_names(conn, "recordings")
        if "trace_count" in recording_columns:
            return
        logger.info("Adding missing recordings.trace_count column")
        await conn.execute(
            text(
                "ALTER TABLE recordings "
                "ADD COLUMN trace_count INTEGER NOT NULL DEFAULT 0"
            )
        )
        await conn.execute(
            text(
                """
                UPDATE recordings
                SET trace_count = (
                    SELECT COUNT(*)
                    FROM traces
                    WHERE traces.recording_id = recordings.recording_id
                      AND traces.write_status = 'written'
                      AND traces.total_bytes IS NOT NULL
                      AND traces.total_bytes > 0
                )
                """
            )
        )

    async def _ensure_recordings_org_id_col(self, conn: AsyncConnection) -> None:
        """Add recordings.org_id for pre-column databases."""
        if not await self._table_exists(conn, "recordings"):
            return
        recording_columns = await self._table_column_names(conn, "recordings")
        if "org_id" in recording_columns:
            return
        logger.info("Adding missing recordings.org_id column")
        await conn.execute(text("ALTER TABLE recordings " "ADD COLUMN org_id TEXT"))

    async def _ensure_recordings_stop_report_columns(
        self, conn: AsyncConnection
    ) -> None:
        """Add stop-reporting columns for pre-column databases."""
        if not await self._table_exists(conn, "recordings"):
            return
        recording_columns = await self._table_column_names(conn, "recordings")
        if "stop_report_status" not in recording_columns:
            logger.info("Adding missing recordings.stop_report_status column")
            await conn.execute(
                text(
                    "ALTER TABLE recordings "
                    "ADD COLUMN stop_report_status TEXT NOT NULL DEFAULT 'pending'"
                )
            )
        if "stop_reported_at" not in recording_columns:
            logger.info("Adding missing recordings.stop_reported_at column")
            await conn.execute(
                text("ALTER TABLE recordings " "ADD COLUMN stop_reported_at DATETIME")
            )

    @staticmethod
    def _recording_row_insert_query(recording_id: str, now: datetime) -> Any:
        return (
            insert(recordings)
            .values(
                recording_id=recording_id,
                org_id=None,
                expected_trace_count=0,
                trace_count=0,
                expected_trace_count_reported=0,
                uploaded_trace_count=0,
                progress_reported=ProgressReportStatus.PENDING,
                stop_report_status=RecordingStopReportStatus.PENDING,
                stopped_at=None,
                stop_reported_at=None,
                created_at=now,
                last_updated=now,
            )
            .on_conflict_do_nothing(index_elements=["recording_id"])
        )

    async def _ensure_recording_row(self, recording_id: str) -> None:
        """Ensure a recording row exists for the given recording_id."""
        now = _utc_now()
        async with self._write_lock() as conn:
            await conn.execute(self._recording_row_insert_query(recording_id, now))

    async def _refresh_recording_counters(self, recording_id: str) -> None:
        """Refresh aggregate trace counters for one recording."""
        now = _utc_now()
        async with self._write_lock() as conn:
            row = (
                (
                    await conn.execute(
                        select(
                            func.count().label("trace_count"),
                            func.sum(
                                case(
                                    (
                                        traces.c.upload_status
                                        == TraceUploadStatus.UPLOADED,
                                        1,
                                    ),
                                    else_=0,
                                )
                            ).label("uploaded_trace_count"),
                        ).where(traces.c.recording_id == recording_id)
                    )
                )
                .mappings()
                .one()
            )
            trace_count = int(row["trace_count"] or 0)
            uploaded_trace_count = int(row["uploaded_trace_count"] or 0)
            if trace_count == 0:
                await conn.execute(
                    delete(recordings).where(recordings.c.recording_id == recording_id)
                )
                return
            existing_expected = (
                await conn.execute(
                    select(recordings.c.expected_trace_count).where(
                        recordings.c.recording_id == recording_id
                    )
                )
            ).scalar_one_or_none()
            expected_trace_count = max(int(existing_expected or 0), trace_count)
            await conn.execute(
                update(recordings)
                .where(recordings.c.recording_id == recording_id)
                .values(
                    expected_trace_count=expected_trace_count,
                    trace_count=trace_count,
                    uploaded_trace_count=uploaded_trace_count,
                    last_updated=now,
                )
            )

    async def reconcile_recordings_from_traces(self) -> None:
        """Rebuild trace-backed recording aggregates without deleting retained rows."""
        now = _utc_now()
        async with self._write_lock() as conn:
            recording_rows = (
                (await conn.execute(select(func.distinct(traces.c.recording_id))))
                .scalars()
                .all()
            )

            for recording_id_raw in recording_rows:
                recording_id = str(recording_id_raw)

                existing_row = (
                    (
                        await conn.execute(
                            select(
                                recordings.c.recording_id,
                                recordings.c.org_id,
                                recordings.c.expected_trace_count,
                                recordings.c.expected_trace_count_reported,
                                recordings.c.uploaded_trace_count,
                                recordings.c.progress_reported,
                                recordings.c.stop_report_status,
                                recordings.c.stopped_at,
                                recordings.c.stop_reported_at,
                                recordings.c.created_at,
                            ).where(recordings.c.recording_id == recording_id)
                        )
                    )
                    .mappings()
                    .one_or_none()
                )

                uploaded_trace_count = int(
                    (
                        await conn.execute(
                            select(
                                func.sum(
                                    case(
                                        (
                                            traces.c.upload_status
                                            == TraceUploadStatus.UPLOADED,
                                            1,
                                        ),
                                        else_=0,
                                    )
                                )
                            ).where(traces.c.recording_id == recording_id)
                        )
                    ).scalar_one_or_none()
                    or 0
                )

                trace_count = int(
                    (
                        await conn.execute(
                            select(func.count()).where(
                                traces.c.recording_id == recording_id,
                                traces.c.write_status == TraceWriteStatus.WRITTEN,
                                traces.c.total_bytes.is_not(None),
                                traces.c.total_bytes > 0,
                            )
                        )
                    ).scalar_one()
                )

                expected_trace_count = trace_count

                existing_expected_trace_count = (
                    int(existing_row["expected_trace_count"] or 0)
                    if existing_row is not None
                    else 0
                )
                existing_uploaded_trace_count = (
                    int(existing_row["uploaded_trace_count"] or 0)
                    if existing_row is not None
                    else 0
                )
                expected_reported = (
                    int(existing_row["expected_trace_count_reported"] or 0)
                    if existing_row is not None
                    else 0
                )
                progress_reported_value = (
                    existing_row["progress_reported"]
                    if existing_row is not None
                    else ProgressReportStatus.PENDING
                )
                stopped_at_value = (
                    existing_row["stopped_at"] if existing_row is not None else None
                )
                stop_report_status_value = (
                    existing_row["stop_report_status"]
                    if existing_row is not None
                    else RecordingStopReportStatus.PENDING
                )
                stop_reported_at_value = (
                    existing_row["stop_reported_at"]
                    if existing_row is not None
                    else None
                )
                created_at_value = (
                    existing_row["created_at"] if existing_row is not None else now
                )

                org_id_value = (
                    existing_row["org_id"] if existing_row is not None else None
                )

                await conn.execute(
                    insert(recordings)
                    .values(
                        recording_id=recording_id,
                        org_id=org_id_value,
                        expected_trace_count=max(
                            expected_trace_count,
                            existing_expected_trace_count,
                        ),
                        trace_count=trace_count,
                        expected_trace_count_reported=expected_reported,
                        uploaded_trace_count=max(
                            uploaded_trace_count,
                            existing_uploaded_trace_count,
                        ),
                        progress_reported=_parse_progress_status(
                            progress_reported_value
                        ),
                        stop_report_status=_parse_stop_report_status(
                            stop_report_status_value
                        ),
                        stopped_at=stopped_at_value,
                        stop_reported_at=stop_reported_at_value,
                        created_at=created_at_value,
                        last_updated=now,
                    )
                    .on_conflict_do_update(
                        index_elements=["recording_id"],
                        set_={
                            "org_id": org_id_value,
                            "expected_trace_count": max(
                                expected_trace_count,
                                existing_expected_trace_count,
                            ),
                            "trace_count": trace_count,
                            "expected_trace_count_reported": expected_reported,
                            "uploaded_trace_count": max(
                                uploaded_trace_count,
                                existing_uploaded_trace_count,
                            ),
                            "progress_reported": _parse_progress_status(
                                progress_reported_value
                            ),
                            "stop_report_status": _parse_stop_report_status(
                                stop_report_status_value
                            ),
                            "stopped_at": stopped_at_value,
                            "stop_reported_at": stop_reported_at_value,
                            "last_updated": now,
                        },
                    )
                )

    async def prune_old_empty_recordings(self, max_age_hours: int) -> int:
        """Delete recordings with no traces and age older than threshold hours."""
        if max_age_hours < 0:
            return 0
        cutoff = _utc_now() - timedelta(hours=max_age_hours)
        async with self._write_lock() as conn:
            result = await conn.execute(
                delete(recordings)
                .where(recordings.c.last_updated <= cutoff)
                .where(
                    ~select(traces.c.recording_id)
                    .where(traces.c.recording_id == recordings.c.recording_id)
                    .exists()
                )
            )
        return int(result.rowcount or 0)

    async def is_recording_stopped(self, recording_id: str) -> bool:
        """Return True when recording has stopped_at set."""
        async with self._engine.begin() as conn:
            value = (
                await conn.execute(
                    select(recordings.c.stopped_at).where(
                        recordings.c.recording_id == recording_id
                    )
                )
            ).scalar_one_or_none()
        return value is not None

    async def is_expected_trace_count_reported(self, recording_id: str) -> bool:
        """Return True when expected trace count has been reported for recording."""
        async with self._engine.begin() as conn:
            value = (
                await conn.execute(
                    select(recordings.c.expected_trace_count_reported).where(
                        recordings.c.recording_id == recording_id
                    )
                )
            ).scalar_one_or_none()
        return int(value or 0) == 1

    async def get_expected_trace_count(self, recording_id: str) -> int | None:
        """Return persisted expected trace count for a recording."""
        async with self._engine.begin() as conn:
            row = (
                await conn.execute(
                    select(recordings.c.expected_trace_count).where(
                        recordings.c.recording_id == recording_id
                    )
                )
            ).scalar_one_or_none()
        if row is None:
            return None
        return int(row or 0)

    async def count_traces_for_recording(self, recording_id: str) -> int:
        """Return count of written traces with data for a recording."""
        async with self._engine.begin() as conn:
            trace_count = (
                await conn.execute(
                    select(func.count()).where(
                        traces.c.recording_id == recording_id,
                        traces.c.write_status == TraceWriteStatus.WRITTEN,
                        traces.c.total_bytes.is_not(None),
                        traces.c.total_bytes > 0,
                    )
                )
            ).scalar_one()
        return int(trace_count or 0)

    async def set_expected_trace_count(
        self, recording_id: str, expected_trace_count: int
    ) -> None:
        """Set expected trace count and mark it as needing report."""
        now = _utc_now()
        await self._ensure_recording_row(recording_id)
        async with self._write_lock() as conn:
            current_expected = (
                await conn.execute(
                    select(recordings.c.expected_trace_count).where(
                        recordings.c.recording_id == recording_id
                    )
                )
            ).scalar_one_or_none()
            next_expected = max(int(current_expected or 0), int(expected_trace_count))
            await conn.execute(
                update(recordings)
                .where(recordings.c.recording_id == recording_id)
                .values(
                    expected_trace_count=next_expected,
                    expected_trace_count_reported=0,
                    last_updated=now,
                )
            )

    async def set_stopped_at(self, recording_id: str) -> None:
        """Set recording-level stopped_at for a recording."""
        now = _utc_now()
        await self._ensure_recording_row(recording_id)
        async with self._write_lock() as conn:
            await conn.execute(
                update(recordings)
                .where(recordings.c.recording_id == recording_id)
                .values(stopped_at=now, last_updated=now)
            )

    async def get_recording_stopped_at(self, recording_id: str) -> datetime | None:
        """Return the persisted recording-level stopped_at timestamp."""
        async with self._engine.begin() as conn:
            value = (
                await conn.execute(
                    select(recordings.c.stopped_at).where(
                        recordings.c.recording_id == recording_id
                    )
                )
            ).scalar_one_or_none()
        return value

    async def set_recording_org_id(self, recording_id: str, org_id: str) -> None:
        """Backfill org_id for a recording when it becomes known."""
        now = _utc_now()
        await self._ensure_recording_row(recording_id)
        async with self._write_lock() as conn:
            await conn.execute(
                update(recordings)
                .where(recordings.c.recording_id == recording_id)
                .where(recordings.c.org_id.is_(None))
                .values(
                    org_id=org_id,
                    last_updated=now,
                )
            )

    async def get_recording_org_id(self, recording_id: str) -> str | None:
        """Return the persisted org_id for a recording."""
        async with self._engine.begin() as conn:
            value = (
                await conn.execute(
                    select(recordings.c.org_id).where(
                        recordings.c.recording_id == recording_id
                    )
                )
            ).scalar_one_or_none()
        return str(value) if value is not None else None

    async def update_bytes_uploaded(self, trace_id: str, bytes_uploaded: int) -> None:
        """Increment the number of bytes uploaded for a trace.

        Args:
            trace_id (str): unique identifier for the trace.
            bytes_uploaded (int): number of bytes uploaded.
        """
        now = _utc_now()
        async with self._write_lock() as conn:
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

    async def get_progress_report_snapshot(
        self, recording_id: str
    ) -> tuple[float, float, dict[str, int], int] | None:
        """Return progress-report snapshot derived from DB queries."""
        async with self._engine.begin() as conn:
            aggregate_row = (
                (
                    await conn.execute(
                        select(
                            func.count().label("row_count"),
                            func.sum(
                                case(
                                    (
                                        and_(
                                            traces.c.data_type.is_not(None),
                                            traces.c.bytes_written.is_not(None),
                                            traces.c.total_bytes.is_not(None),
                                            traces.c.bytes_written
                                            == traces.c.total_bytes,
                                        ),
                                        1,
                                    ),
                                    else_=0,
                                )
                            ).label("eligible_count"),
                            func.min(traces.c.created_at).label("start_at"),
                            func.max(traces.c.last_updated).label("end_at"),
                            func.sum(traces.c.total_bytes).label("total_bytes"),
                        ).where(traces.c.recording_id == recording_id)
                    )
                )
                .mappings()
                .one()
            )

            row_count = int(aggregate_row["row_count"] or 0)
            eligible_count = int(aggregate_row["eligible_count"] or 0)
            start_at = aggregate_row["start_at"]
            end_at = aggregate_row["end_at"]
            total_bytes = int(aggregate_row["total_bytes"] or 0)
            if row_count == 0 or eligible_count != row_count:
                return None
            if start_at is None or end_at is None:
                return None

            rows = (
                (
                    await conn.execute(
                        select(traces.c.trace_id, traces.c.total_bytes).where(
                            traces.c.recording_id == recording_id
                        )
                    )
                )
                .mappings()
                .all()
            )

        trace_map: dict[str, int] = {}
        for row in rows:
            trace_id = str(row["trace_id"])
            trace_total_bytes = row["total_bytes"]
            if trace_total_bytes is None:
                return None
            trace_map[trace_id] = int(trace_total_bytes)
        if not trace_map:
            return None

        return (
            start_at.timestamp(),
            end_at.timestamp(),
            trace_map,
            total_bytes,
        )

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
        async with self._write_lock() as conn:
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
        async with self._write_lock() as conn:
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
        async with self._write_lock() as conn:
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

    async def increment_uploaded_trace_count(self, recording_id: str) -> None:
        """Increment uploaded trace count for a recording."""
        now = _utc_now()
        await self._ensure_recording_row(recording_id)
        async with self._write_lock() as conn:
            await conn.execute(
                update(recordings)
                .where(recordings.c.recording_id == recording_id)
                .values(
                    uploaded_trace_count=recordings.c.uploaded_trace_count + 1,
                    last_updated=now,
                )
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
        async with self._write_lock() as conn:
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
        async with self._write_lock() as conn:
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
                            traces.c.upload_status.in_((
                                TraceUploadStatus.PENDING,
                                TraceUploadStatus.RETRYING,
                            ))
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
        async with self._write_lock() as conn:
            candidate_rows = (
                await conn.execute(
                    select(traces.c.trace_id, traces.c.last_updated)
                    .select_from(
                        traces.join(
                            recordings,
                            recordings.c.recording_id == traces.c.recording_id,
                        )
                    )
                    .where(traces.c.write_status == TraceWriteStatus.WRITTEN)
                    .where(
                        traces.c.registration_status == TraceRegistrationStatus.PENDING
                    )
                    .where(recordings.c.expected_trace_count_reported == 1)
                    .order_by(traces.c.created_at.asc())
                    .limit(int(limit))
                )
            ).all()
            logger.debug(
                (
                    "claim_traces_for_registration fetched %d candidate rows "
                    "(limit=%d, max_wait_s=%.2f)"
                ),
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
            logger.debug(
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

            rows_to_Register = (
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
            logger.debug(
                "claim_traces_for_registration claimed %d rows",
                len(rows_to_Register),
            )

        return [TraceRecord.from_row(dict(row)) for row in rows_to_Register]

    async def find_unreported_traces(self) -> list[TraceRecord]:
        """Return all traces that have not been progress-reported."""
        async with self._engine.begin() as conn:
            rows = (
                (
                    await conn.execute(
                        select(traces)
                        .select_from(
                            traces.join(
                                recordings,
                                recordings.c.recording_id == traces.c.recording_id,
                            )
                        )
                        .where(
                            recordings.c.progress_reported
                            == ProgressReportStatus.PENDING
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
        await self._ensure_recording_row(recording_id)
        async with self._write_lock() as conn:
            await conn.execute(
                update(recordings)
                .where(recordings.c.recording_id == recording_id)
                .values(
                    progress_reported=ProgressReportStatus.REPORTED,
                    last_updated=now,
                )
            )

    async def mark_recording_reporting(self, recording_id: str) -> bool:
        """Atomically move recording progress state from PENDING to REPORTING."""
        now = _utc_now()
        await self._ensure_recording_row(recording_id)
        async with self._write_lock() as conn:
            result = await conn.execute(
                update(recordings)
                .where(recordings.c.recording_id == recording_id)
                .where(recordings.c.progress_reported == ProgressReportStatus.PENDING)
                .values(
                    progress_reported=ProgressReportStatus.REPORTING,
                    last_updated=now,
                )
            )
        return int(result.rowcount or 0) > 0

    async def mark_recording_pending(self, recording_id: str) -> None:
        """Set recording progress state to PENDING."""
        now = _utc_now()
        await self._ensure_recording_row(recording_id)
        async with self._write_lock() as conn:
            await conn.execute(
                update(recordings)
                .where(recordings.c.recording_id == recording_id)
                .values(
                    progress_reported=ProgressReportStatus.PENDING,
                    last_updated=now,
                )
            )

    async def mark_recording_stop_report_pending(self, recording_id: str) -> None:
        """Set stop-report state to PENDING without regressing REPORTED rows."""
        now = _utc_now()
        await self._ensure_recording_row(recording_id)
        async with self._write_lock() as conn:
            await conn.execute(
                update(recordings)
                .where(recordings.c.recording_id == recording_id)
                .where(
                    recordings.c.stop_report_status
                    != RecordingStopReportStatus.REPORTED
                )
                .values(
                    stop_report_status=RecordingStopReportStatus.PENDING,
                    stop_reported_at=None,
                    last_updated=now,
                )
            )

    async def claim_recording_stop_report(self, recording_id: str) -> bool:
        """Atomically move stop-report state from PENDING to REPORTING."""
        now = _utc_now()
        await self._ensure_recording_row(recording_id)
        async with self._write_lock() as conn:
            result = await conn.execute(
                update(recordings)
                .where(recordings.c.recording_id == recording_id)
                .where(
                    recordings.c.stop_report_status == RecordingStopReportStatus.PENDING
                )
                .values(
                    stop_report_status=RecordingStopReportStatus.REPORTING,
                    last_updated=now,
                )
            )
        return int(result.rowcount or 0) > 0

    async def mark_recording_stop_reported(self, recording_id: str) -> None:
        """Mark a recording stop as backend-reported."""
        now = _utc_now()
        await self._ensure_recording_row(recording_id)
        async with self._write_lock() as conn:
            await conn.execute(
                update(recordings)
                .where(recordings.c.recording_id == recording_id)
                .values(
                    stop_report_status=RecordingStopReportStatus.REPORTED,
                    stop_reported_at=now,
                    last_updated=now,
                )
            )

    async def reset_recording_stop_reporting_to_pending(self) -> int:
        """Reset in-flight stop-report rows to PENDING and return affected count."""
        now = _utc_now()
        async with self._write_lock() as conn:
            result = await conn.execute(
                update(recordings)
                .where(
                    recordings.c.stop_report_status
                    == RecordingStopReportStatus.REPORTING
                )
                .values(
                    stop_report_status=RecordingStopReportStatus.PENDING,
                    stop_reported_at=None,
                    last_updated=now,
                )
            )
        return int(result.rowcount or 0)

    async def recording_stop_has_been_reported(self, recording_id: str) -> bool:
        """Return True when recording stop-report status is REPORTED."""
        async with self._engine.begin() as conn:
            value = (
                await conn.execute(
                    select(recordings.c.stop_report_status).where(
                        recordings.c.recording_id == recording_id
                    )
                )
            ).scalar_one_or_none()
        return _parse_stop_report_status(value) == RecordingStopReportStatus.REPORTED

    async def list_pending_recording_stop_reports(self) -> list[str]:
        """Return recording IDs whose stop still needs backend reporting."""
        async with self._engine.begin() as conn:
            rows = (
                (
                    await conn.execute(
                        select(recordings.c.recording_id).where(
                            recordings.c.stop_report_status
                            == RecordingStopReportStatus.PENDING
                        )
                    )
                )
                .scalars()
                .all()
            )
        return [str(recording_id) for recording_id in rows]

    async def reset_reporting_recordings_to_pending(self) -> int:
        """Reset in-flight REPORTING rows to PENDING and return affected count."""
        now = _utc_now()
        async with self._write_lock() as conn:
            result = await conn.execute(
                update(recordings)
                .where(recordings.c.progress_reported == ProgressReportStatus.REPORTING)
                .values(
                    progress_reported=ProgressReportStatus.PENDING,
                    last_updated=now,
                )
            )
        return int(result.rowcount or 0)

    async def recording_has_reported_progress(self, recording_id: str) -> bool:
        """Return True when recording progress status is REPORTED."""
        async with self._engine.begin() as conn:
            progress_value = (
                await conn.execute(
                    select(recordings.c.progress_reported).where(
                        recordings.c.recording_id == recording_id
                    )
                )
            ).scalar_one_or_none()
        return _parse_progress_status(progress_value) == ProgressReportStatus.REPORTED

    async def delete_uploaded_traces_for_recording(self, recording_id: str) -> int:
        """Delete all UPLOADED traces for one recording and return deleted count."""
        async with self._write_lock() as conn:
            result = await conn.execute(
                delete(traces)
                .where(traces.c.recording_id == recording_id)
                .where(traces.c.upload_status == TraceUploadStatus.UPLOADED)
            )
        return int(result.rowcount or 0)

    async def delete_traces_for_recording(self, recording_id: str) -> int:
        """Delete all trace rows for a recording and return deleted count."""
        async with self._write_lock() as conn:
            result = await conn.execute(
                delete(traces).where(traces.c.recording_id == recording_id)
            )
        return int(result.rowcount or 0)

    async def delete_expired_completed_recordings(self, max_age_hours: int) -> int:
        """Delete completed recording rows older than the retention window."""
        if max_age_hours < 0:
            return 0

        cutoff = _utc_now() - timedelta(hours=max_age_hours)

        async with self._write_lock() as conn:
            result = await conn.execute(
                delete(recordings).where(
                    recordings.c.progress_reported == ProgressReportStatus.REPORTED,
                    recordings.c.expected_trace_count
                    == recordings.c.uploaded_trace_count,
                    recordings.c.stopped_at.is_not(None),
                    recordings.c.stopped_at <= cutoff,
                )
            )

        return int(result.rowcount or 0)

    async def list_recording_ids_with_stopped_traces(self) -> list[str]:
        """Return recording IDs that already have at least one stopped trace."""
        async with self._engine.begin() as conn:
            rows = (
                (
                    await conn.execute(
                        select(recordings.c.recording_id).where(
                            recordings.c.stopped_at.is_not(None)
                        )
                    )
                )
                .scalars()
                .all()
            )
        return [str(recording_id) for recording_id in rows]

    async def mark_expected_trace_count_reported(self, recording_id: str) -> None:
        """Mark a recording's expected trace count as reported."""
        now = _utc_now()
        await self._ensure_recording_row(recording_id)
        async with self._write_lock() as conn:
            await conn.execute(
                update(recordings)
                .where(recordings.c.recording_id == recording_id)
                .values(expected_trace_count_reported=1, last_updated=now)
            )

    async def reset_failed_trace_for_retry(self, trace_id: str) -> None:
        """Reset a failed trace back to WRITTEN for retry."""
        now = _utc_now()
        async with self._write_lock() as conn:
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

    async def _increment_trace_count_if_new_write_with_data(
        self,
        *,
        conn: AsyncConnection,
        recording_id: str,
        previous_write_status: Any,
        current_write_status: Any,
        current_total_bytes: Any,
        now: datetime,
    ) -> None:
        """Increment expected count when a trace first becomes WRITTEN with data."""
        if previous_write_status == TraceWriteStatus.WRITTEN:
            return
        if current_write_status != TraceWriteStatus.WRITTEN:
            return
        if current_total_bytes is None or int(current_total_bytes) <= 0:
            return

        await conn.execute(self._recording_row_insert_query(recording_id, now))
        await conn.execute(
            update(recordings)
            .where(recordings.c.recording_id == recording_id)
            .values(
                trace_count=recordings.c.trace_count + 1,
                last_updated=now,
            )
        )

    async def _execute_trace_upsert_and_update_counters(
        self,
        *,
        conn: AsyncConnection,
        stmt: Any,
        trace_id: str,
        recording_id: str,
        now: datetime,
    ) -> TraceRecord:
        """Run trace upsert and recording counter updates in one transaction."""
        previous_write_status = (
            await conn.execute(
                select(traces.c.write_status).where(traces.c.trace_id == trace_id)
            )
        ).scalar_one_or_none()
        await conn.execute(stmt)
        row = (
            (await conn.execute(select(traces).where(traces.c.trace_id == trace_id)))
            .mappings()
            .one()
        )
        await self._increment_trace_count_if_new_write_with_data(
            conn=conn,
            recording_id=recording_id,
            previous_write_status=previous_write_status,
            current_write_status=row["write_status"],
            current_total_bytes=row["total_bytes"],
            now=now,
        )
        return TraceRecord.from_row(dict(row))

    async def _execute_trace_write_progress_upsert(
        self,
        *,
        conn: AsyncConnection,
        stmt: Any,
        trace_id: str,
        recording_id: str,
        now: datetime,
    ) -> TraceRecord:
        """Run TRACE_WRITE_PROGRESS upsert and create recording row if needed."""
        previous_write_status = (
            await conn.execute(
                select(traces.c.write_status).where(traces.c.trace_id == trace_id)
            )
        ).scalar_one_or_none()
        await conn.execute(stmt)
        row = (
            (await conn.execute(select(traces).where(traces.c.trace_id == trace_id)))
            .mappings()
            .one()
        )
        if row["write_status"] == TraceWriteStatus.WRITING:
            await conn.execute(self._recording_row_insert_query(recording_id, now))
        await self._increment_trace_count_if_new_write_with_data(
            conn=conn,
            recording_id=recording_id,
            previous_write_status=previous_write_status,
            current_write_status=row["write_status"],
            current_total_bytes=row["total_bytes"],
            now=now,
        )
        return TraceRecord.from_row(dict(row))

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
        async with self._write_lock() as conn:
            return await self._execute_trace_upsert_and_update_counters(
                conn=conn,
                stmt=stmt,
                trace_id=trace_id,
                recording_id=recording_id,
                now=now,
            )

    async def upsert_trace_bytes(
        self,
        trace_id: str,
        recording_id: str,
        bytes_written: int,
    ) -> TraceRecord:
        """Insert or update trace with bytes from TRACE_WRITTEN.

        State transitions:
        - If trace doesn't exist: creates with PENDING_METADATA status
        - If trace exists with INITIALIZING/WRITING: transitions to WRITTEN
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
                        traces.c.write_status.in_((
                            TraceWriteStatus.INITIALIZING,
                            TraceWriteStatus.WRITING,
                        )),
                        TraceWriteStatus.WRITTEN,
                    ),
                    else_=traces.c.write_status,
                ),
            },
        )
        async with self._write_lock() as conn:
            return await self._execute_trace_upsert_and_update_counters(
                conn=conn,
                stmt=stmt,
                trace_id=trace_id,
                recording_id=recording_id,
                now=now,
            )

    async def upsert_trace_write_progress(
        self,
        trace_id: str,
        recording_id: str,
        bytes_written: int,
    ) -> TraceRecord:
        """Insert or update trace write progress from TRACE_WRITE_PROGRESS.

        State transitions:
        - If trace doesn't exist: creates with WRITING status
        - If trace exists with INITIALIZING/PENDING_METADATA: transitions to WRITING
        - If trace exists with WRITTEN/FAILED: preserves terminal status

        Returns the trace record after upsert.
        """
        now = _utc_now()
        stmt = insert(traces).values(
            trace_id=trace_id,
            recording_id=recording_id,
            bytes_written=bytes_written,
            total_bytes=None,
            write_status=TraceWriteStatus.WRITING,
            registration_status=TraceRegistrationStatus.PENDING,
            upload_status=TraceUploadStatus.PENDING,
            bytes_uploaded=0,
            created_at=now,
            last_updated=now,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["trace_id"],
            set_={
                "bytes_written": case(
                    (
                        traces.c.bytes_written.is_(None),
                        bytes_written,
                    ),
                    else_=func.max(traces.c.bytes_written, bytes_written),
                ),
                "last_updated": now,
                "write_status": case(
                    (
                        traces.c.write_status.in_((
                            TraceWriteStatus.PENDING,
                            TraceWriteStatus.INITIALIZING,
                            TraceWriteStatus.PENDING_METADATA,
                            TraceWriteStatus.WRITING,
                        )),
                        TraceWriteStatus.WRITING,
                    ),
                    else_=traces.c.write_status,
                ),
            },
        )
        async with self._write_lock() as conn:
            return await self._execute_trace_write_progress_upsert(
                conn=conn,
                stmt=stmt,
                trace_id=trace_id,
                recording_id=recording_id,
                now=now,
            )

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
        async with self._write_lock() as conn:
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
        async with self._write_lock() as conn:
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
        async with self._write_lock() as conn:
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
