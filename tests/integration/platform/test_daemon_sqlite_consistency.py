"""Daemon SQLite consistency checks.

Each test drives data through the full nc.login() → daemon → SQLite path
(no cloud upload) and then directly queries the daemon's SQLite database to
assert structural invariants.

All tests use the offline daemon profile so that traces are written to disk
but registration and upload are never attempted.  This isolates the write-path
and lets us assert stable state immediately after recording stops.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from typing import Any

import numpy as np
from recording_playback_shared import (
    run_minimal_recording_flow,
    use_offline_daemon_profile,
    wait_for_recording_to_exist_in_db,
)

import neuracore as nc
from neuracore.data_daemon.helpers import get_daemon_db_path

# ---------------------------------------------------------------------------
# Raw SQLite helpers – query the live daemon DB without any ORM layer
# ---------------------------------------------------------------------------


def _connect() -> sqlite3.Connection:
    """Return a read-only sqlite3 connection to the daemon DB."""
    db_path = get_daemon_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_all_traces(recording_id: str | None = None) -> list[dict[str, Any]]:
    """Return all trace rows, optionally filtered by recording_id."""
    with _connect() as conn:
        if recording_id is not None:
            rows = conn.execute(
                "SELECT * FROM traces WHERE recording_id = ?", (recording_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM traces").fetchall()
    return [dict(row) for row in rows]


def _fetch_all_recordings() -> list[dict[str, Any]]:
    """Return all recording rows."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM recordings").fetchall()
    return [dict(row) for row in rows]


def _fetch_recording(recording_id: str) -> dict[str, Any] | None:
    """Return a single recording row, or None if not found."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM recordings WHERE recording_id = ?", (recording_id,)
        ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Invariant assertion helpers
# ---------------------------------------------------------------------------


def assert_no_orphaned_traces() -> None:
    """Every trace must reference a recording_id that exists in recordings."""
    all_traces = _fetch_all_traces()
    all_recordings = _fetch_all_recordings()
    known_ids = {rec["recording_id"] for rec in all_recordings}
    for trace in all_traces:
        assert trace["recording_id"] in known_ids, (
            f"Trace {trace['trace_id']} references unknown recording "
            f"'{trace['recording_id']}'"
        )


def assert_no_duplicate_trace_ids() -> None:
    """All trace_ids across the database must be unique."""
    all_traces = _fetch_all_traces()
    seen: set[str] = set()
    for trace in all_traces:
        assert trace["trace_id"] not in seen, f"Duplicate trace_id: {trace['trace_id']}"
        seen.add(trace["trace_id"])


def assert_trace_count_matches_rows(recording_id: str) -> None:
    """recordings.trace_count must equal the actual number of trace rows."""
    rec = _fetch_recording(recording_id)
    assert rec is not None, f"Recording '{recording_id}' not found in recordings table"
    actual = len(_fetch_all_traces(recording_id))
    assert rec["trace_count"] == actual, (
        f"Recording '{recording_id}': trace_count={rec['trace_count']} "
        f"but actual rows={actual}"
    )


def assert_written_traces_have_bytes(recording_id: str) -> None:
    """Every WRITTEN trace must have bytes_written > 0."""
    for trace in _fetch_all_traces(recording_id):
        if trace["write_status"] == "written":
            assert trace["bytes_written"] is not None and trace["bytes_written"] > 0, (
                f"Trace {trace['trace_id']} is 'written' but bytes_written="
                f"{trace['bytes_written']}"
            )


def assert_bytes_uploaded_not_exceeds_total(recording_id: str) -> None:
    """bytes_uploaded must not exceed total_bytes where total_bytes is set."""
    for trace in _fetch_all_traces(recording_id):
        total = trace["total_bytes"]
        uploaded = trace["bytes_uploaded"] or 0
        if total is not None and total > 0:
            assert uploaded <= total, (
                f"Trace {trace['trace_id']} bytes_uploaded={uploaded} > "
                f"total_bytes={total}"
            )


def assert_no_in_flight_upload_status(recording_id: str) -> None:
    """In offline mode no trace should reach UPLOADING or RETRYING."""
    for trace in _fetch_all_traces(recording_id):
        assert trace["upload_status"] not in ("uploading", "retrying"), (
            f"Trace {trace['trace_id']} has unexpected upload_status="
            f"{trace['upload_status']} in offline mode"
        )


def assert_registration_status_pending(recording_id: str) -> None:
    """In offline mode, all traces must have registration_status = 'pending'."""
    for trace in _fetch_all_traces(recording_id):
        assert trace["registration_status"] == "pending", (
            f"Trace {trace['trace_id']} has registration_status="
            f"{trace['registration_status']} (expected 'pending' in offline mode)"
        )


def assert_all_invariants(recording_id: str) -> None:
    """Run every consistency check for a given recording."""
    assert_no_duplicate_trace_ids()
    assert_no_orphaned_traces()
    assert_trace_count_matches_rows(recording_id)
    assert_written_traces_have_bytes(recording_id)
    assert_bytes_uploaded_not_exceeds_total(recording_id)
    assert_no_in_flight_upload_status(recording_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDaemonSqliteConsistency:
    """Consistency checks on the daemon SQLite DB after recording flows."""

    def test_recording_row_created_after_stop(self) -> None:
        """A recordings row must exist once the daemon processes the recording."""
        with use_offline_daemon_profile():
            recording_id = run_minimal_recording_flow(
                label_prefix="consistency.row_created"
            )
            wait_for_recording_to_exist_in_db(recording_id)

            recording = _fetch_recording(recording_id)
            assert (
                recording is not None
            ), f"Expected recordings row for '{recording_id}'"

    def test_trace_count_matches_actual_rows(self) -> None:
        """recordings.trace_count must equal the real number of trace rows."""
        with use_offline_daemon_profile():
            recording_id = run_minimal_recording_flow(
                label_prefix="consistency.trace_count"
            )
            wait_for_recording_to_exist_in_db(recording_id)

            assert_trace_count_matches_rows(recording_id)

    def test_no_orphaned_traces(self) -> None:
        """No trace may reference a recording_id absent from the recordings table."""
        with use_offline_daemon_profile():
            recording_id = run_minimal_recording_flow(
                label_prefix="consistency.orphan_check"
            )
            wait_for_recording_to_exist_in_db(recording_id)

            assert_no_orphaned_traces()

    def test_no_duplicate_trace_ids(self) -> None:
        """All trace_ids in the database must be globally unique."""
        with use_offline_daemon_profile():
            recording_id = run_minimal_recording_flow(label_prefix="consistency.dedup")
            wait_for_recording_to_exist_in_db(recording_id)

            assert_no_duplicate_trace_ids()

    def test_written_traces_have_positive_bytes(self) -> None:
        """Every trace in write_status='written' must have bytes_written > 0."""
        with use_offline_daemon_profile():
            recording_id = run_minimal_recording_flow(
                label_prefix="consistency.bytes_written"
            )
            wait_for_recording_to_exist_in_db(recording_id)

            assert_written_traces_have_bytes(recording_id)

    def test_offline_mode_no_upload_states(self) -> None:
        """In offline mode traces must not enter UPLOADING or RETRYING state."""
        with use_offline_daemon_profile():
            recording_id = run_minimal_recording_flow(
                label_prefix="consistency.upload_states"
            )
            wait_for_recording_to_exist_in_db(recording_id)

            assert_no_in_flight_upload_status(recording_id)

    def test_offline_mode_registration_stays_pending(self) -> None:
        """In offline mode all traces must have registration_status='pending'."""
        with use_offline_daemon_profile():
            recording_id = run_minimal_recording_flow(
                label_prefix="consistency.reg_pending"
            )
            wait_for_recording_to_exist_in_db(recording_id)

            assert_registration_status_pending(recording_id)

    def test_full_invariant_suite_single_recording(self) -> None:
        """All invariants pass together for a single minimal recording."""
        with use_offline_daemon_profile():
            recording_id = run_minimal_recording_flow(
                label_prefix="consistency.full_single"
            )
            wait_for_recording_to_exist_in_db(recording_id)

            assert_all_invariants(recording_id)

    def test_full_invariant_suite_multiple_recordings(self) -> None:
        """All invariants hold when multiple recordings exist in the same DB."""
        with use_offline_daemon_profile():
            robot_name = f"consistency_multi_robot_{uuid.uuid4().hex[:8]}"
            dataset_name = f"consistency_multi_ds_{uuid.uuid4().hex[:8]}"

            nc.login()
            robot = nc.connect_robot(robot_name, overwrite=False)
            nc.create_dataset(dataset_name)

            recording_ids: list[str] = []
            for frame_index in range(3):
                nc.start_recording(robot_name=robot_name)
                recording_id = robot.get_current_recording_id()
                assert recording_id is not None
                recording_ids.append(str(recording_id))

                nc.log_custom_1d(
                    "marker",
                    np.array([float(frame_index)], dtype=np.float32),
                    robot_name=robot_name,
                    timestamp=float(frame_index),
                )
                nc.stop_recording(robot_name=robot_name, wait=False)

            for recording_id in recording_ids:
                wait_for_recording_to_exist_in_db(recording_id)

            for recording_id in recording_ids:
                assert_all_invariants(recording_id)

            # Cross-recording: counts must be per-recording, not summed globally
            for recording_id in recording_ids:
                rec = _fetch_recording(recording_id)
                assert rec is not None
                actual = len(_fetch_all_traces(recording_id))
                assert rec["trace_count"] == actual, (
                    f"Cross-recording trace_count mismatch for '{recording_id}': "
                    f"stored={rec['trace_count']}, actual={actual}"
                )

    def test_expected_trace_count_not_reported_in_offline_mode(self) -> None:
        """In offline mode expected_trace_count_reported must be 0 (not reported)."""
        with use_offline_daemon_profile():
            recording_id = run_minimal_recording_flow(
                label_prefix="consistency.etc_not_reported"
            )
            wait_for_recording_to_exist_in_db(recording_id)

            recording = _fetch_recording(recording_id)
            assert recording is not None
            assert recording["expected_trace_count_reported"] == 0, (
                f"expected_trace_count_reported={recording['expected_trace_count_reported']}"
                " but should be 0 in offline mode"
            )

    def test_stopped_at_set_after_recording_stops(self) -> None:
        """recordings.stopped_at must be non-NULL once the recording has stopped."""
        with use_offline_daemon_profile():
            recording_id = run_minimal_recording_flow(
                label_prefix="consistency.stopped_at"
            )
            wait_for_recording_to_exist_in_db(recording_id)

            # Give the daemon a moment to process the stop event
            deadline = time.time() + 15.0
            stopped_at = None
            while time.time() < deadline:
                recording = _fetch_recording(recording_id)
                if recording and recording["stopped_at"] is not None:
                    stopped_at = recording["stopped_at"]
                    break
                time.sleep(0.25)

            assert stopped_at is not None, (
                f"recordings.stopped_at is still NULL for '{recording_id}' "
                "after recording stopped"
            )

    def test_trace_path_populated_for_written_traces(self) -> None:
        """Every WRITTEN trace must have a non-empty path column."""
        with use_offline_daemon_profile():
            recording_id = run_minimal_recording_flow(
                label_prefix="consistency.trace_path"
            )
            wait_for_recording_to_exist_in_db(recording_id)

            for trace in _fetch_all_traces(recording_id):
                if trace["write_status"] == "written":
                    assert (
                        trace["path"] is not None and trace["path"] != ""
                    ), f"Trace {trace['trace_id']} is 'written' but path is empty"
