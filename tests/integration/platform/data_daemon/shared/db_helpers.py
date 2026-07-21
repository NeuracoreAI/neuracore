"""SQLite query primitives and polling helpers.

Provides the low-level building blocks used by :mod:`assertions` and
by test files that need to inspect the daemon's state DB or wait for
asynchronous events.

Contents:

- **DB store helpers** — :class:`DaemonDbStore` centralises SQLite
    connections and common ``SELECT`` patterns used across the test helpers.
- **DB query primitives** — thin wrappers around common ``SELECT`` patterns
    (:func:`fetch_all_rows`, :func:`fetch_recording`,
    :func:`fetch_all_traces`) used by both verification assertions and
    DB-readiness polls.
- **High-level DB fetchers** — :func:`fetch_trace_registration_stats`,
  :func:`fetch_expected_trace_count_reported`,
  :func:`fetch_recording_online_verification_stats`,
  :func:`fetch_recording_trace_upload_stats`.
- **Wait helpers** — :func:`wait_for_dataset_ready`,
  :func:`wait_for_recording_to_exist_in_db`,
  :func:`wait_for_offline_db_ready`, :func:`wait_for_all_traces_written`,
    :func:`wait_for_upload_complete_in_db`.
"""

from __future__ import annotations

import dataclasses
import logging
import sqlite3
import time
from collections.abc import Callable, Iterable
from contextlib import closing
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from tests.integration.platform.data_daemon.shared.test_case.build_test_case_context import (  # noqa: E501
        ContextResult,
    )

import neuracore as nc
from neuracore.data_daemon.helpers import get_daemon_db_path
from neuracore.data_daemon.rust_selection import is_rust_daemon_enabled
from tests.integration.platform.data_daemon.shared.db_constants import (
    COLUMN_EXPECTED_TRACE_COUNT,
    COLUMN_EXPECTED_TRACE_COUNT_REPORTED,
    COLUMN_LAST_UPDATED,
    COLUMN_PROGRESS_REPORTED,
    COLUMN_RECORDING_ID,
    COLUMN_RECORDING_INDEX,
    COLUMN_REGISTRATION_STATUS,
    COLUMN_ROBOT_ID,
    COLUMN_ROBOT_INSTANCE,
    COLUMN_STOPPED_AT,
    COLUMN_TRACE_ID,
    COLUMN_UPLOAD_STATUS,
    COLUMN_WRITE_STATUS,
    RECORDING_PROGRESS_REPORTED,
    RECORDINGS_TABLE,
    STAT_EXPECTED_TRACE_COUNT,
    STAT_EXPECTED_TRACE_COUNT_REPORTED,
    STAT_NON_PENDING_REGISTRATION_TRACES,
    STAT_PROGRESS_REPORTED,
    STAT_REGISTERED_TRACES,
    STAT_TOTAL_TRACES,
    STAT_UPLOAD_PROGRESS_TRACES,
    STAT_UPLOADED_TRACES,
    TRACE_REGISTRATION_PENDING,
    TRACE_REGISTRATION_REGISTERED,
    TRACE_UPLOAD_DETAIL_COLUMNS,
    TRACE_UPLOAD_INTEGER_COLUMNS,
    TRACE_UPLOAD_PROGRESS_STATUSES,
    TRACE_UPLOAD_STAT_REGISTRATION_STATUS_COUNTS,
    TRACE_UPLOAD_STAT_TRACE_ROWS,
    TRACE_UPLOAD_STAT_UPLOAD_STATUS_COUNTS,
    TRACE_UPLOAD_STAT_WRITE_STATUS_COUNTS,
    TRACE_UPLOAD_UPLOADED,
    TRACE_WRITE_WRITTEN,
    TRACES_TABLE,
)
from tests.integration.platform.data_daemon.shared.disk_helpers import (
    list_recording_ids_on_disk,
    list_recording_indexes_on_disk,
    normalize_recording_ids,
    normalize_recording_indexes,
)
from tests.integration.platform.data_daemon.shared.process_control import Timer
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    MAX_TIME_TO_START_S,
)

logger = logging.getLogger(__name__)


def _recording_correlation_column() -> str:
    """Return the recordings/traces correlation column for the active daemon.

    Rust daemon: recordings are keyed by the local INTEGER ``recording_index``
    (the cloud ``recording_id`` is a separate, nullable column). Legacy Python
    daemon: recordings are keyed by the cloud ``recording_id`` (TEXT PK), which
    is also the traces foreign key.
    """
    return COLUMN_RECORDING_INDEX if is_rust_daemon_enabled() else COLUMN_RECORDING_ID


class DaemonDbStore:
    """Encapsulate SQLite access patterns for daemon integration tests."""

    def __init__(self, db_path_provider: Callable[[], Any] = get_daemon_db_path):
        self._db_path_provider = db_path_provider

    @staticmethod
    def _table_name(table: str) -> str:
        return str(table)

    def connect(self) -> sqlite3.Connection:
        """Return a read-write SQLite connection to the active daemon state DB."""
        conn = sqlite3.connect(str(self._db_path_provider()))
        conn.row_factory = sqlite3.Row
        return conn

    def connect_read_only(self) -> sqlite3.Connection:
        """Return a read-only URI-mode connection to the active daemon state DB."""
        conn = sqlite3.connect(
            f"file:{self._db_path_provider()}?mode=ro",
            uri=True,
        )
        conn.row_factory = sqlite3.Row
        return conn

    def fetch_all_rows(self, table: str) -> list[dict[str, Any]]:
        """Return every row from the named table in the daemon state DB."""
        table_name = self._table_name(table)
        with closing(self.connect()) as conn:
            rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()  # noqa: S608
        return [dict(row) for row in rows]

    def list_tables(self) -> set[str]:
        """Return the names of all user tables currently present in the daemon DB."""
        with closing(self.connect()) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        return {str(row[0]) for row in rows}

    def table_exists(
        self,
        conn: sqlite3.Connection,
        table: str,
    ) -> bool:
        """Return ``True`` when ``table`` exists in the SQLite database."""
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            (self._table_name(table),),
        ).fetchone()
        return row is not None

    def table_columns(
        self,
        conn: sqlite3.Connection,
        table: str,
    ) -> set[str]:
        """Return the column names for the given SQLite table."""
        table_name = self._table_name(table)
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()  # noqa: S608
        return {str(row[1]) for row in rows}

    def fetch_recording(self, recording_key: int | str) -> dict[str, Any] | None:
        """Return the recording row for ``recording_key`` if it exists.

        ``recording_key`` is the daemon's correlation key for the active mode:
        the local INTEGER ``recording_index`` under the Rust daemon, or the cloud
        ``recording_id`` (TEXT) under the legacy Python daemon.
        """
        correlation_column = _recording_correlation_column()
        with closing(self.connect()) as conn:
            row = conn.execute(
                f"SELECT * FROM {RECORDINGS_TABLE} " f"WHERE {correlation_column} = ?",
                (recording_key,),
            ).fetchone()
        return dict(row) if row is not None else None

    def fetch_recordings_for_source(
        self,
        robot_id: str,
        robot_instance: int,
    ) -> list[dict[str, Any]]:
        """Return recording rows for one source ordered by ``recording_index``.

        Source identity is ``(robot_id, robot_instance)``. This mirrors the
        daemon's own source lookup (``idx_recordings_source``) and is how tests
        correlate a worker's recordings to daemon-minted ``recording_index``
        values without seeing the local handle.
        """
        with closing(self.connect()) as conn:
            rows = conn.execute(
                f"SELECT * FROM {RECORDINGS_TABLE} "
                f"WHERE {COLUMN_ROBOT_ID} = ? AND {COLUMN_ROBOT_INSTANCE} = ? "
                f"ORDER BY {COLUMN_RECORDING_INDEX}",
                (robot_id, robot_instance),
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_all_traces(
        self,
        recording_key: int | str,
        *,
        columns: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return trace rows for one recording, limited to available columns.

        Traces join to recordings on the active correlation column: the integer
        ``recording_index`` under the Rust daemon, or the cloud ``recording_id``
        under the legacy Python daemon.
        """
        correlation_column = _recording_correlation_column()
        with closing(self.connect_read_only()) as conn:
            if not self.table_exists(conn, TRACES_TABLE):
                return []

            trace_columns = self.table_columns(conn, TRACES_TABLE)
            if correlation_column not in trace_columns:
                return []

            if columns is None:
                selected_columns = sorted(trace_columns)
            else:
                selected_columns = [col for col in columns if col in trace_columns]

            if not selected_columns:
                return []

            order_by_columns = [
                col
                for col in (COLUMN_LAST_UPDATED, COLUMN_TRACE_ID)
                if col in selected_columns
            ]
            order_by_clause = (
                f" ORDER BY {', '.join(order_by_columns)}" if order_by_columns else ""
            )
            rows = conn.execute(
                f"SELECT {', '.join(selected_columns)} "
                f"FROM {TRACES_TABLE} "
                f"WHERE {correlation_column} = ?{order_by_clause}",
                (recording_key,),
            ).fetchall()

        return [{column: row[column] for column in selected_columns} for row in rows]


_TEST_STORE = DaemonDbStore()


def connect_daemon_db() -> sqlite3.Connection:
    """Return a read-write SQLite connection to the active daemon state DB."""
    return _TEST_STORE.connect()


def connect_daemon_db_read_only() -> sqlite3.Connection:
    """Return a read-only URI-mode connection to the active daemon state DB."""
    return _TEST_STORE.connect_read_only()


def fetch_all_rows(table: str) -> list[dict[str, Any]]:
    """Return every row from the named table in the daemon state DB."""
    return _TEST_STORE.fetch_all_rows(table)


def list_tables() -> set[str]:
    """Return the names of all user tables currently present in the daemon DB."""
    return _TEST_STORE.list_tables()


def sqlite_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """Return ``True`` when ``table`` exists in the SQLite database."""
    return _TEST_STORE.table_exists(conn, table)


def sqlite_table_columns(
    conn: sqlite3.Connection,
    table: str,
) -> set[str]:
    """Return the column names for the given SQLite table."""
    return _TEST_STORE.table_columns(conn, table)


def fetch_recording(recording_key: int | str) -> dict[str, Any] | None:
    """Return the recording row for the active-mode correlation key, if present."""
    return _TEST_STORE.fetch_recording(recording_key)


def fetch_recordings_for_source(
    robot_id: str,
    robot_instance: int,
) -> list[dict[str, Any]]:
    """Return recording rows for one source ordered by ``recording_index``."""
    return _TEST_STORE.fetch_recordings_for_source(robot_id, robot_instance)


def fetch_all_traces(
    recording_key: int | str,
    *,
    columns: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Return trace rows for one recording, limited to available columns."""
    return _TEST_STORE.fetch_all_traces(recording_key, columns=columns)


def wait_for_recording_index_for_source(
    robot_id: str,
    robot_instance: int,
    *,
    after_index: int = 0,
    timeout_s: float = 30.0,
    poll_interval_s: float = 0.1,
) -> int:
    """Block until the daemon assigns a recording with index > ``after_index``.

    Returns the highest ``recording_index`` for ``(robot_id, robot_instance)``
    once it exceeds ``after_index`` — i.e. the recording just started. The daemon
    assigns ``recording_index`` the moment it sees ``StartRecording``, so a short
    poll covers the window between the producer's ``start_recording`` call and the
    daemon persisting the row.

    Gating on a *new, higher* index rather than a raw row count keeps this robust
    to the recording reaper, which deletes a recording's row once it is fully
    uploaded: the just-started recording is always the highest-indexed row for the
    source and is never reaped before the next recording starts, but older
    fully-uploaded rows can vanish mid-session, so the row *count* is not
    monotonic. Pass the previously-resolved index as ``after_index`` (``0`` for
    the first recording of a source).

    Raises:
        TimeoutError: If no recording with index > ``after_index`` appears in time.
    """
    deadline = time.monotonic() + timeout_s
    newest: int | None = None
    while True:
        try:
            rows = fetch_recordings_for_source(robot_id, robot_instance)
        except sqlite3.OperationalError:
            # Daemon has created the DB file but not yet the schema; retry.
            rows = []
        indices = [int(row[COLUMN_RECORDING_INDEX]) for row in rows]
        newest = max(indices) if indices else None
        if newest is not None and newest > after_index:
            return newest
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Daemon did not assign a new recording_index for source "
                f"({robot_id}, {robot_instance}); expected an index greater than "
                f"{after_index}, newest seen {newest} after {timeout_s}s."
            )
        time.sleep(poll_interval_s)


def assert_offline_recordings_pending(results: list[ContextResult]) -> None:
    """Assert offline recordings exist by ``recording_index`` with cloud id NULL.

    Rust daemon only: in offline mode the daemon assigns ``recording_index``
    immediately but the cloud ``recording_id`` stays NULL until the daemon is
    online and ``/recording/start`` lands. This confirms that contract per
    ``recording_index`` captured by each worker. Under the legacy daemon the
    cloud ``recording_id`` is the recording's identity from the moment recording
    starts, so this offline-NULL contract does not apply and the check is a
    no-op.

    Raises:
        AssertionError: If a recording row is missing or already has a cloud id.
    """
    if not is_rust_daemon_enabled():
        return
    for result in results:
        rows_by_index = {
            row[COLUMN_RECORDING_INDEX]: row
            for row in fetch_recordings_for_source(result.source[0], result.source[1])
        }
        for recording_index in result.recording_indexes:
            row = rows_by_index.get(recording_index)
            assert row is not None, (
                f"Offline recording_index {recording_index} for source "
                f"{result.source} has no recordings row"
            )
            assert row.get(COLUMN_RECORDING_ID) is None, (
                f"Offline recording_index {recording_index} unexpectedly already "
                f"has cloud recording_id={row.get(COLUMN_RECORDING_ID)!r}"
            )


def resolve_cloud_recording_ids(
    results: list[ContextResult],
    *,
    timeout_s: float = 60.0,
) -> list[ContextResult]:
    """Return copies of *results* with cloud ``recording_ids`` resolved per mode.

    Legacy Python daemon: ``nc.start_recording()`` already returns the cloud
    ``recording_id``, so ``result.recording_ids`` is authoritative and the
    results are returned unchanged.

    Rust daemon: the cloud ``recording_id`` is populated asynchronously by the
    start-notifier once the daemon is online, so the captured ids may have been
    NULL at record time. This reads the daemon DB directly by ``recording_index``
    (which the recording worker already captured) and waits until each row's
    cloud ``recording_id`` is filled in, so cloud verification (which matches the
    dataset's ``recording.id``) has the correct ids.

    Resolving from the DB rather than ``nc.get_cloud_recording_id`` is required
    here: recordings are made in worker subprocesses, so the verifying process
    has no active recording context to resolve against — only the daemon-assigned
    ``recording_index`` values carried on each result.

    Raises:
        AssertionError: If any recording's cloud id is not populated in time.
    """
    if not is_rust_daemon_enabled():
        return results

    resolved: list[ContextResult] = []
    for result in results:
        cloud_ids: list[str] = []
        for recording_index in result.recording_indexes:
            cloud_id = _wait_for_cloud_recording_id(
                recording_index, timeout_s=timeout_s
            )
            assert cloud_id, (
                f"Cloud recording_id never populated for recording_index "
                f"{recording_index} (robot {result.robot_name!r}) within "
                f"{timeout_s}s"
            )
            cloud_ids.append(cloud_id)
        resolved.append(dataclasses.replace(result, recording_ids=cloud_ids))
    return resolved


def _wait_for_cloud_recording_id(
    recording_index: int,
    *,
    timeout_s: float,
    poll_interval_s: float = 0.5,
) -> str | None:
    """Poll the daemon DB until ``recording_index`` has a cloud id, or time out."""
    deadline = time.monotonic() + timeout_s
    while True:
        row = fetch_recording(recording_index)
        if row is not None and row.get(COLUMN_RECORDING_ID):
            return str(row[COLUMN_RECORDING_ID])
        if time.monotonic() >= deadline:
            return None
        time.sleep(poll_interval_s)


# ---------------------------------------------------------------------------
# High-level DB fetchers
# ---------------------------------------------------------------------------


def fetch_trace_registration_stats(recording_key: int | str) -> tuple[int, int]:
    """Return ``(total_traces, non_pending_traces)`` for a recording.

    Queries the ``traces`` table and counts all rows for ``recording_key``
    plus the subset whose ``registration_status`` is not ``"pending"``.

    Args:
        recording_key: The active-mode correlation key (``recording_index``
            under the Rust daemon, cloud ``recording_id`` under the legacy
            daemon) to query.

    Returns:
        A two-tuple of ``(total_trace_count, non_pending_registration_count)``.
    """
    traces = fetch_all_traces(
        recording_key,
        columns=[COLUMN_REGISTRATION_STATUS],
    )
    non_pending = sum(
        1
        for trace in traces
        if trace.get(COLUMN_REGISTRATION_STATUS) != TRACE_REGISTRATION_PENDING
    )
    return len(traces), non_pending


def fetch_expected_trace_count_reported(recording_key: int | str) -> int | None:
    """Return the ``expected_trace_count_reported`` value for a recording row.

    Args:
        recording_key: The active-mode correlation key (``recording_index``
            under the Rust daemon, cloud ``recording_id`` under the legacy
            daemon) to look up.

    Returns:
        The integer value of ``expected_trace_count_reported``, or ``None``
        when the recording row is not found.
    """
    row = fetch_recording(recording_key)
    if row is None or row.get(COLUMN_EXPECTED_TRACE_COUNT_REPORTED) is None:
        return None
    return int(row[COLUMN_EXPECTED_TRACE_COUNT_REPORTED])


def fetch_recording_online_verification_stats(
    recording_key: int | str,
) -> dict[str, int | str | None]:
    """Fetch a comprehensive set of online-verification stats for one recording.

    Queries both the ``recordings`` and ``traces`` tables with column-existence
    guards so the function works even on older DB schemas that may be missing
    some columns.  Handles missing DB files gracefully by returning a zeroed-out
    default dict.

    Args:
        recording_key: The active-mode correlation key (``recording_index``
            under the Rust daemon, cloud ``recording_id`` under the legacy
            daemon) to inspect.

    Returns:
        A dict with the following keys:

        - ``expected_trace_count`` — value from the recordings row, or
          ``None`` when absent.
        - ``expected_trace_count_reported`` — ``int`` from the recordings row.
        - ``progress_reported`` — string status from the recordings row.
        - ``total_traces`` — total number of trace rows for this recording.
        - ``non_pending_registration_traces`` — traces with
          ``registration_status != 'pending'``.
        - ``registered_traces`` — traces with
          ``registration_status == 'registered'``.
        - ``upload_progress_traces`` — traces with ``upload_status`` in
          ``{'queued', 'uploading', 'uploaded'}``.
        - ``uploaded_traces`` — traces with ``upload_status == 'uploaded'``.
    """
    default_stats: dict[str, int | str | None] = {
        STAT_EXPECTED_TRACE_COUNT: None,
        STAT_EXPECTED_TRACE_COUNT_REPORTED: None,
        STAT_PROGRESS_REPORTED: None,
        STAT_TOTAL_TRACES: 0,
        STAT_NON_PENDING_REGISTRATION_TRACES: 0,
        STAT_REGISTERED_TRACES: 0,
        STAT_UPLOAD_PROGRESS_TRACES: 0,
        STAT_UPLOADED_TRACES: 0,
    }

    try:
        recording_row = fetch_recording(recording_key)
        traces = fetch_all_traces(
            recording_key,
            columns=[COLUMN_REGISTRATION_STATUS, COLUMN_UPLOAD_STATUS],
        )
    except sqlite3.OperationalError:
        return default_stats

    expected_trace_count = _optional_int(
        recording_row,
        COLUMN_EXPECTED_TRACE_COUNT,
    )
    expected_trace_count_reported = _optional_int(
        recording_row,
        COLUMN_EXPECTED_TRACE_COUNT_REPORTED,
    )
    progress_reported = _optional_str(
        recording_row,
        COLUMN_PROGRESS_REPORTED,
    )
    non_pending_registration_traces = sum(
        1
        for trace in traces
        if trace.get(COLUMN_REGISTRATION_STATUS)
        not in {None, TRACE_REGISTRATION_PENDING}
    )
    registered_traces = sum(
        1
        for trace in traces
        if trace.get(COLUMN_REGISTRATION_STATUS) == TRACE_REGISTRATION_REGISTERED
    )
    upload_progress_traces = sum(
        1
        for trace in traces
        if trace.get(COLUMN_UPLOAD_STATUS) in TRACE_UPLOAD_PROGRESS_STATUSES
    )
    uploaded_traces = sum(
        1
        for trace in traces
        if trace.get(COLUMN_UPLOAD_STATUS) == TRACE_UPLOAD_UPLOADED
    )

    return {
        STAT_EXPECTED_TRACE_COUNT: expected_trace_count,
        STAT_EXPECTED_TRACE_COUNT_REPORTED: expected_trace_count_reported,
        STAT_PROGRESS_REPORTED: progress_reported,
        STAT_TOTAL_TRACES: len(traces),
        STAT_NON_PENDING_REGISTRATION_TRACES: non_pending_registration_traces,
        STAT_REGISTERED_TRACES: registered_traces,
        STAT_UPLOAD_PROGRESS_TRACES: upload_progress_traces,
        STAT_UPLOADED_TRACES: uploaded_traces,
    }


def fetch_recording_trace_upload_stats(
    recording_key: int | str,
) -> dict[str, object]:
    """Fetch per-trace upload and registration status counts for a recording.

    Returns aggregate counts broken down by ``write_status``,
    ``registration_status``, and ``upload_status``, plus a full list of
    per-trace row snapshots.  Handles missing tables and columns gracefully.

    Args:
        recording_key: The active-mode correlation key (``recording_index``
            under the Rust daemon, cloud ``recording_id`` under the legacy
            daemon) to inspect.

    Returns:
        A dict containing:

        - ``write_status_counts`` — ``{status: count}`` for ``write_status``.
        - ``registration_status_counts`` — ``{status: count}`` for
          ``registration_status``.
        - ``upload_status_counts`` — ``{status: count}`` for ``upload_status``.
        - ``trace_rows`` — list of per-trace dicts with selected columns.
    """
    default_result: dict[str, object] = {
        TRACE_UPLOAD_STAT_WRITE_STATUS_COUNTS: {},
        TRACE_UPLOAD_STAT_REGISTRATION_STATUS_COUNTS: {},
        TRACE_UPLOAD_STAT_UPLOAD_STATUS_COUNTS: {},
        TRACE_UPLOAD_STAT_TRACE_ROWS: [],
    }

    selected_columns = list(TRACE_UPLOAD_DETAIL_COLUMNS)

    try:
        rows = fetch_all_traces(recording_key, columns=selected_columns)
    except sqlite3.OperationalError:
        return default_result

    write_status_counts: dict[str, int] = {}
    registration_status_counts: dict[str, int] = {}
    upload_status_counts: dict[str, int] = {}
    trace_rows: list[dict[str, object]] = []

    for row in rows:
        trace_row: dict[str, object] = {}
        for col in selected_columns:
            if col not in row:
                continue

            value = row[col]
            if col in TRACE_UPLOAD_INTEGER_COLUMNS:
                trace_row[col] = int(value) if value is not None else None
            else:
                trace_row[col] = value
        trace_rows.append(trace_row)

        if COLUMN_WRITE_STATUS in trace_row:
            status = str(trace_row[COLUMN_WRITE_STATUS])
            write_status_counts[status] = write_status_counts.get(status, 0) + 1
        if COLUMN_REGISTRATION_STATUS in trace_row:
            status = str(trace_row[COLUMN_REGISTRATION_STATUS])
            registration_status_counts[status] = (
                registration_status_counts.get(status, 0) + 1
            )
        if COLUMN_UPLOAD_STATUS in trace_row:
            status = str(trace_row[COLUMN_UPLOAD_STATUS])
            upload_status_counts[status] = upload_status_counts.get(status, 0) + 1

    return {
        TRACE_UPLOAD_STAT_WRITE_STATUS_COUNTS: write_status_counts,
        TRACE_UPLOAD_STAT_REGISTRATION_STATUS_COUNTS: registration_status_counts,
        TRACE_UPLOAD_STAT_UPLOAD_STATUS_COUNTS: upload_status_counts,
        TRACE_UPLOAD_STAT_TRACE_ROWS: trace_rows,
    }


# ---------------------------------------------------------------------------
# Wait helpers
# ---------------------------------------------------------------------------


def poll_until_condition(
    condition: Callable[[], bool],
    *,
    timeout_s: float,
    poll_interval_s: float,
    timeout_message: str,
) -> None:
    """Poll ``condition`` until it returns ``True`` or timeout elapses.

    Args:
        condition: Callable returning ``True`` once the target condition is met.
        timeout_s: Maximum time in seconds to keep polling.
        poll_interval_s: Delay between polls in seconds.
        timeout_message: Message used when the timeout is reached.

    Raises:
        TimeoutError: If the condition is not met before ``timeout_s``.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(poll_interval_s)
    raise TimeoutError(timeout_message)


def wait_for_dataset_ready(
    dataset_name: str,
    expected_recording_count: int = 1,
    timeout_s: float = 120.0,
    poll_interval_s: float = 1.5,
) -> None:
    """Block until the named dataset contains the expected number of recordings.

    Polls :func:`neuracore.get_dataset` until ``len(dataset) ==
    expected_recording_count``. Does not verify DB upload state or cloud
    finalization — those are separate concerns.

    Args:
        dataset_name: Name of the dataset to poll.
        expected_recording_count: Exact number of recordings to wait for.
        timeout_s: Maximum time to wait in seconds before raising.
        poll_interval_s: Seconds between successive polls.

    Raises:
        TimeoutError: If the dataset does not have ``expected_recording_count``
            recordings within ``timeout_s`` seconds.
    """
    wait_start = time.perf_counter()
    last_error: Exception | None = None
    recording_count: int | None = None
    while True:
        elapsed_s = time.perf_counter() - wait_start
        try:
            dataset = nc.get_dataset(dataset_name)
            recording_count = len(dataset)
            if recording_count == expected_recording_count:
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc

        if elapsed_s >= timeout_s:
            raise TimeoutError(
                f"Timed out waiting for dataset '{dataset_name}' to have "
                f"{expected_recording_count} recording(s) after {timeout_s}s. "
                f"Has {recording_count if recording_count is not None else 0} "
                f"recording(s)."
            ) from last_error

        time.sleep(min(poll_interval_s, max(0.0, timeout_s - elapsed_s)))


def wait_for_recordings_finalized(
    dataset_name: str,
    recording_ids: set[str],
    *,
    timeout_s: float = 120.0,
    poll_interval_s: float = 2.0,
) -> None:
    """Block until every recording in ``recording_ids`` has a finalized end_time.

    The backend updates a recording's ``end_time`` asynchronously.  This helper
    re-fetches the dataset on each poll and returns once every expected
    recording exists and has a non-null ``end_time``.  Duration correctness is
    checked later by the structural verification pass so failures surface
    immediately instead of polling for a duration value that is already final.

    Args:
        dataset_name: Name of the dataset to poll.
        recording_ids: Set of recording IDs that must all be finalized.
        timeout_s: Maximum time to wait before raising.
        poll_interval_s: Seconds between successive polls.

    Raises:
        TimeoutError: If not all recordings are finalized within ``timeout_s``.
    """
    deadline = time.monotonic() + timeout_s
    last_missing = set(recording_ids)
    last_without_end_time: set[str] = set()

    while time.monotonic() < deadline:
        try:
            dataset = nc.get_dataset(dataset_name)
        except Exception:  # noqa: BLE001
            time.sleep(poll_interval_s)
            continue

        seen: set[str] = set()
        without_end_time: set[str] = set()
        for recording in dataset:
            rec_id = str(recording.id)
            if rec_id not in recording_ids:
                continue
            seen.add(rec_id)
            if recording.end_time is None:
                without_end_time.add(rec_id)

        missing = recording_ids - seen
        if not missing and not without_end_time:
            return

        last_missing = missing
        last_without_end_time = without_end_time
        time.sleep(min(poll_interval_s, max(0.0, deadline - time.monotonic())))

    raise TimeoutError(
        f"Recordings in dataset '{dataset_name}' did not finalize within "
        f"{timeout_s}s. Missing: {sorted(last_missing)}. "
        f"Without end_time: {sorted(last_without_end_time)}"
    )


def wait_for_offline_db_ready(
    timeout_s: float = MAX_TIME_TO_START_S,
    *,
    expected_recording_keys: Iterable[int | str] | None = None,
) -> None:
    """Block until the offline daemon's SQLite DB schema is ready for queries.

    The daemon can create the SQLite file before finishing schema
    initialisation, so tests that query ``recordings``/``traces`` immediately
    after startup may race and fail with ``no such table``.  This helper polls
    until both tables exist.

    Args:
        timeout_s: Maximum seconds to wait before raising.
        expected_recording_keys: When supplied, also waits until at least one of
            the expected recording directories exists on disk, preventing a
            false-positive ``ready`` result from an empty-but-initialised DB.
            Keys are ``recording_index`` values under the Rust daemon and cloud
            ``recording_id`` strings under the legacy daemon.

    Raises:
        AssertionError: If the DB is not ready within ``timeout_s`` seconds.
    """
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    required_tables = {
        RECORDINGS_TABLE,
        TRACES_TABLE,
    }
    if is_rust_daemon_enabled():
        target_recording_keys: set[int] | set[str] = normalize_recording_indexes(
            expected_recording_keys
        )
        list_on_disk: Callable[[], set[int]] | Callable[[], set[str]] = (
            list_recording_indexes_on_disk
        )
    else:
        target_recording_keys = normalize_recording_ids(
            None
            if expected_recording_keys is None
            else (str(key) for key in expected_recording_keys)
        )
        list_on_disk = list_recording_ids_on_disk

    with Timer(timeout_s, label="daemon.offline_db_ready", always_log=True):
        while time.monotonic() < deadline:
            db_path = get_daemon_db_path()
            if not db_path.exists():
                time.sleep(0.1)
                continue

            if not target_recording_keys and not list_on_disk():
                time.sleep(0.1)
                continue

            try:
                existing_tables = list_tables()
            except sqlite3.OperationalError as exc:
                last_error = exc
                time.sleep(0.1)
                continue

            if required_tables.issubset(existing_tables):
                return

            time.sleep(0.1)

    db_path = get_daemon_db_path()
    existing_tables: set[str] = set()
    try:
        if db_path.exists():
            existing_tables = list_tables()
    except sqlite3.OperationalError as exc:
        last_error = exc

    raise AssertionError(
        "Offline daemon DB did not become ready within "
        f"{timeout_s}s. db_path={db_path} exists={db_path.exists()} "
        f"recordings_on_disk={sorted(list_on_disk())} "
        f"tables={sorted(existing_tables)} last_error={last_error!r}"
    )


def wait_for_all_traces_written(
    timeout_s: float = 120.0,
    *,
    results: list,
) -> None:
    """Block until every trace for every recording in *results* has been written.

    Recordings are correlated by the active daemon's key: the daemon-assigned
    INTEGER ``recording_index`` under the Rust daemon (the on-disk directory name
    and the traces join key, since the cloud ``recording_id`` is nullable / minted
    async), or the cloud ``recording_id`` under the legacy Python daemon. The
    recordings root directory is the source of truth for which keys to check —
    this catches recordings the daemon started that the client-side results may
    not reflect (e.g. an ``already_started`` race on reconnect).

    Blocks until all of the following are true for every recording in scope:

    - A matching row exists in the DB with ``stopped_at`` set.
    - Every recording has at least one trace row.
    - Every trace row has ``write_status == 'written'``.

    Args:
        timeout_s: Maximum seconds to wait before raising.
        results: List of :class:`~build_test_case_context.ContextResult` objects
            whose recording keys scope the check.

    Raises:
        AssertionError: If the condition is not met within ``timeout_s``.
    """
    min_poll_interval_s = 0.05
    max_poll_interval_s = 1.0

    use_rust = is_rust_daemon_enabled()
    correlation_column = COLUMN_RECORDING_INDEX if use_rust else COLUMN_RECORDING_ID

    def _raw_keys() -> list:
        if use_rust:
            return [key for result in results for key in result.recording_indexes]
        return [key for result in results for key in result.recording_ids]

    if use_rust:
        expected_keys: set[int] | set[str] = normalize_recording_indexes(_raw_keys())
        list_keys_on_disk: Callable[[], set[int]] | Callable[[], set[str]] = (
            list_recording_indexes_on_disk
        )
    else:
        expected_keys = normalize_recording_ids(_raw_keys())
        list_keys_on_disk = list_recording_ids_on_disk

    deadline = time.monotonic() + timeout_s
    poll_interval_s = min_poll_interval_s
    last_state: tuple[int, int, int, int, int] | None = None

    def _sleep_for_next_poll(*, progress_made: bool) -> None:
        nonlocal poll_interval_s
        now = time.monotonic()
        if now >= deadline:
            return

        if progress_made:
            poll_interval_s = min_poll_interval_s
            return

        sleep_s = min(poll_interval_s, max(0.0, deadline - now))
        if sleep_s > 0:
            time.sleep(sleep_s)
        poll_interval_s = min(max_poll_interval_s, poll_interval_s * 2)

    wait_for_offline_db_ready(
        timeout_s=max(0.0, deadline - time.monotonic()),
        expected_recording_keys=expected_keys,
    )
    while time.monotonic() < deadline:
        recording_keys = expected_keys or list_keys_on_disk()
        if not recording_keys:
            _sleep_for_next_poll(progress_made=False)
            continue

        try:
            recordings = {
                row[correlation_column]: row
                for row in fetch_all_rows(RECORDINGS_TABLE)
                if row[correlation_column] in recording_keys
            }
            traces = [
                trace
                for trace in fetch_all_rows(TRACES_TABLE)
                if trace[correlation_column] in recording_keys
            ]
        except sqlite3.OperationalError:
            _sleep_for_next_poll(progress_made=False)
            continue

        traces_by_recording: dict[Any, list[dict[str, Any]]] = {}
        for trace in traces:
            traces_by_recording.setdefault(trace[correlation_column], []).append(trace)

        stopped_count = sum(
            1 for row in recordings.values() if row[COLUMN_STOPPED_AT] is not None
        )
        written_count = sum(
            1 for trace in traces if trace[COLUMN_WRITE_STATUS] == TRACE_WRITE_WRITTEN
        )
        current_state = (
            len(recording_keys),
            len(recordings),
            stopped_count,
            len(traces),
            written_count,
        )
        progress_made = current_state != last_state
        last_state = current_state

        if len(recordings) < len(recording_keys):
            _sleep_for_next_poll(progress_made=progress_made)
            continue

        if stopped_count < len(recordings):
            _sleep_for_next_poll(progress_made=progress_made)
            continue

        # Intentionally weakened to "every recording has >= 1 trace row": the
        # Rust schema dropped the `trace_count` column, so we can no longer
        # assert an exact per-recording trace count here.
        all_have_traces = all(
            len(traces_by_recording.get(recording_key, [])) > 0
            for recording_key in recording_keys
        )
        if not all_have_traces:
            _sleep_for_next_poll(progress_made=progress_made)
            continue

        if written_count == len(traces):
            return
        _sleep_for_next_poll(progress_made=progress_made)

    recording_keys = expected_keys or list_keys_on_disk()
    try:
        recordings = fetch_all_rows(RECORDINGS_TABLE)
        traces = fetch_all_rows(TRACES_TABLE)
    except sqlite3.OperationalError as exc:
        raise AssertionError(
            f"Daemon DB was still not queryable after waiting {timeout_s}s: {exc}"
        ) from exc
    unfinished = [
        {
            COLUMN_TRACE_ID: t[COLUMN_TRACE_ID],
            correlation_column: t[correlation_column],
            COLUMN_WRITE_STATUS: t[COLUMN_WRITE_STATUS],
        }
        for t in traces
        if t[correlation_column] in recording_keys
        and t[COLUMN_WRITE_STATUS] != TRACE_WRITE_WRITTEN
    ]
    missing_in_db = sorted(
        recording_keys - {row[correlation_column] for row in recordings}
    )
    not_stopped = sorted(
        row[correlation_column]
        for row in recordings
        if row[correlation_column] in recording_keys and row[COLUMN_STOPPED_AT] is None
    )
    recordings_without_traces = sorted(
        recording_key
        for recording_key in recording_keys
        if not any(trace[correlation_column] == recording_key for trace in traces)
    )
    all_raw_keys = _raw_keys()
    duplicate_keys = sorted(
        {key for key in all_raw_keys if all_raw_keys.count(key) > 1}
    )
    raise AssertionError(
        f"Daemon did not finish writing all traces within {timeout_s}s.\n"
        f"  Duplicate recording keys across contexts: {duplicate_keys}\n"
        f"  Recording keys on disk with no DB row: {missing_in_db}\n"
        f"  Recordings not yet stopped (stopped_at is NULL): {not_stopped}\n"
        f"  Recordings with no trace rows: {recordings_without_traces}\n"
        f"  Traces still in non-written state ({len(unfinished)}):\n"
        + "\n".join(f"    {t}" for t in unfinished)
    )


def assert_recording_db_statuses(
    recording_index: int,
    *,
    check_cloud_statuses: bool = False,
) -> None:
    """Assert that all DB trace statuses for a recording are in terminal states.

    The shared core checks that every trace has ``write_status == 'written'``.
    When ``check_cloud_statuses=True`` the function additionally asserts that
    every trace has ``registration_status == 'registered'`` and
    ``upload_status == 'uploaded'``, which are only reachable after a full
    online upload cycle.

    Args:
        recording_index: The local ``recording_index`` to inspect.
        check_cloud_statuses: When ``True``, also assert registration and
            upload statuses in addition to write status.  Pass this for cloud
            (online) test cases only.

    Raises:
        AssertionError: When any trace has an unexpected status value.
    """
    try:
        traces = fetch_all_traces(
            recording_index,
            columns=[
                COLUMN_TRACE_ID,
                COLUMN_WRITE_STATUS,
                COLUMN_REGISTRATION_STATUS,
                COLUMN_UPLOAD_STATUS,
            ],
        )
    except sqlite3.OperationalError as exc:
        raise AssertionError(
            f"Cannot query traces for recording_index {recording_index}: {exc}"
        ) from exc

    assert traces, f"No trace rows found in DB for recording_index {recording_index}"

    non_written = [
        {
            COLUMN_TRACE_ID: t[COLUMN_TRACE_ID],
            COLUMN_WRITE_STATUS: t[COLUMN_WRITE_STATUS],
        }
        for t in traces
        if t.get(COLUMN_WRITE_STATUS) != TRACE_WRITE_WRITTEN
    ]
    assert not non_written, (
        f"Recording {recording_index}: traces not in 'written' state "
        f"({len(non_written)}/{len(traces)}):\n"
        + "\n".join(f"  {t}" for t in non_written)
    )

    if not check_cloud_statuses:
        return

    non_registered = [
        {
            COLUMN_TRACE_ID: t[COLUMN_TRACE_ID],
            COLUMN_REGISTRATION_STATUS: t[COLUMN_REGISTRATION_STATUS],
        }
        for t in traces
        if t.get(COLUMN_REGISTRATION_STATUS) != TRACE_REGISTRATION_REGISTERED
    ]
    assert not non_registered, (
        f"Recording {recording_index}: traces not in 'registered' state "
        f"({len(non_registered)}/{len(traces)}):\n"
        + "\n".join(f"  {t}" for t in non_registered)
    )

    non_uploaded = [
        {
            COLUMN_TRACE_ID: t[COLUMN_TRACE_ID],
            COLUMN_UPLOAD_STATUS: t[COLUMN_UPLOAD_STATUS],
        }
        for t in traces
        if t.get(COLUMN_UPLOAD_STATUS) != TRACE_UPLOAD_UPLOADED
    ]
    assert not non_uploaded, (
        f"Recording {recording_index}: traces not in 'uploaded' state "
        f"({len(non_uploaded)}/{len(traces)}):\n"
        + "\n".join(f"  {t}" for t in non_uploaded)
    )


def wait_for_upload_complete_in_db(
    recording_key: int | str,
    timeout_s: float = 90.0,
) -> None:
    """Block until all known traces for a recording are uploaded per the daemon DB.

    Polls :func:`fetch_recording_online_verification_stats` and returns once
    the daemon's local SQLite state shows upload complete. This helper adapts
    to slow but advancing uploads:

    - Poll interval uses exponential backoff while no progress is observed.
    - Poll interval resets immediately when progress is observed.
    - Timeout is exponentially extended on progress, up to a capped multiplier.

    Completion is defined as DB-local state only — either every trace row has
    ``upload_status == 'uploaded'``, or ``progress_reported == 'reported'``
    (traces already deleted by daemon after acknowledgement). This does NOT
    verify that data is present in the cloud.

    Args:
        recording_key: The active-mode correlation key (``recording_index``
            under the Rust daemon, cloud ``recording_id`` under the legacy
            daemon) to wait on.
        timeout_s: Base no-progress timeout in seconds.

    Raises:
        pytest failure: If upload completion is not observed before timeout.
    """
    min_poll_interval_s = 0.1
    max_poll_interval_s = 2.0
    max_timeout_backoff_factor = 8.0

    poll_interval_s = min_poll_interval_s
    progress_timeout_s = timeout_s
    deadline = time.monotonic() + progress_timeout_s
    last_state: tuple[int | str | None, ...] | None = None

    while time.monotonic() < deadline:
        stats = fetch_recording_online_verification_stats(recording_key)
        if _is_online_upload_complete(stats):
            return

        current_state: tuple[int | str | None, ...] = (
            stats[STAT_EXPECTED_TRACE_COUNT],
            stats[STAT_TOTAL_TRACES],
            stats[STAT_NON_PENDING_REGISTRATION_TRACES],
            stats[STAT_REGISTERED_TRACES],
            stats[STAT_UPLOAD_PROGRESS_TRACES],
            stats[STAT_UPLOADED_TRACES],
            stats[STAT_PROGRESS_REPORTED],
        )

        progress_made = last_state is not None and current_state != last_state
        last_state = current_state

        if progress_made:
            progress_timeout_s = min(
                timeout_s * max_timeout_backoff_factor,
                progress_timeout_s * 2,
            )
            deadline = time.monotonic() + progress_timeout_s
            poll_interval_s = min_poll_interval_s
            continue

        now = time.monotonic()
        sleep_s = min(poll_interval_s, max(0.0, deadline - now))
        if sleep_s > 0:
            time.sleep(sleep_s)
        poll_interval_s = min(max_poll_interval_s, poll_interval_s * 2)

    stats = fetch_recording_online_verification_stats(recording_key)
    trace_upload_stats = fetch_recording_trace_upload_stats(recording_key)
    pytest.fail(
        "Online upload did not complete for recording "
        f"{recording_key!r} within {progress_timeout_s:.1f}s of last progress; "
        f"stats={stats}; trace_upload_stats={trace_upload_stats}"
    )


def _optional_int(
    row: dict[str, Any] | None,
    key: str,
) -> int | None:
    """Return ``row[key]`` coerced to ``int`` when present."""
    if row is None or row.get(key) is None:
        return None
    return int(row[key])


def _optional_str(
    row: dict[str, Any] | None,
    key: str,
) -> str | None:
    """Return ``row[key]`` coerced to ``str`` when present."""
    if row is None or row.get(key) is None:
        return None
    return str(row[key])


def _is_online_upload_complete(stats: dict[str, int | str | None]) -> bool:
    """Return ``True`` once all DB traces are in the uploaded state.

    Two terminal states are recognised:

    1. Traces still present: ``total_traces > 0`` and every trace has
       ``upload_status == 'uploaded'``.
    2. Traces already deleted: the daemon deletes trace rows after the progress
       report is acknowledged by the backend, so ``total_traces == 0`` combined
       with ``progress_reported == 'reported'`` is an equally valid completion
       signal.
    """
    progress_reported = stats[STAT_PROGRESS_REPORTED]
    if progress_reported == RECORDING_PROGRESS_REPORTED:
        return True

    total_traces = stats[STAT_TOTAL_TRACES]
    uploaded_traces = stats[STAT_UPLOADED_TRACES]
    return (
        isinstance(total_traces, int)
        and total_traces > 0
        and isinstance(uploaded_traces, int)
        and uploaded_traces == total_traces
    )
