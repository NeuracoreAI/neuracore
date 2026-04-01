"""Offline recording matrix tests.

Each parametrized case records data via the offline daemon profile, then
verifies local state: SQLite invariants, trace files on disk, and recording
metadata.  No data is ever uploaded.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest
from recording_playback_matrix_config import (
    MatrixDimensionConfig,
    RecordingPlaybackMatrixCase,
    assert_context_mode,
    build_matrix_cases,
    case_id,
    run_case_contexts,
)
from recording_playback_shared import (
    OFFLINE_DB_PATH,
    OFFLINE_RECORDINGS_ROOT,
    TEST_DAEMON_STATE_DIR,
    daemon_cleanup,
    use_offline_daemon_profile,
)

from neuracore.data_daemon.helpers import get_daemon_db_path

logger = logging.getLogger(__name__)

TRACE_JSON_NAME = "trace.json"
VIDEO_TRACE_DATA_TYPES = {"RGB_IMAGES", "DEPTH_IMAGES"}
VIDEO_TRACE_FILENAMES = {TRACE_JSON_NAME, "lossy.mp4", "lossless.mp4"}

OFFLINE_DIMENSION_CONFIG = MatrixDimensionConfig(
    durations_sec=(50,),
    parallel_contexts_options=(1, 2),
    recording_counts=(10,),
    joint_counts=(10, 30),
    producer_channel_options=("synchronous", "per_thread"),
    video_counts=(0, 1),
    image_dimensions=((64, 64), (1920, 1080)),
)

OFFLINE_MATRIX_CASES = build_matrix_cases(OFFLINE_DIMENSION_CONFIG)


# ---------------------------------------------------------------------------
# SQLite helpers (mirrors test_daemon_sqlite_consistency.py)
# ---------------------------------------------------------------------------


def _connect() -> sqlite3.Connection:
    """Return a read-only sqlite3 connection to the daemon DB."""
    db_path = get_daemon_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_all_traces(recording_id: str) -> list[dict[str, Any]]:
    """Return all trace rows for a recording."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM traces WHERE recording_id = ?", (recording_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def _fetch_recording(recording_id: str) -> dict[str, Any] | None:
    """Return a single recording row, or None if not found."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM recordings WHERE recording_id = ?", (recording_id,)
        ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Generic DB helpers
# ---------------------------------------------------------------------------


def _fetch_all(table: str) -> list[dict[str, Any]]:
    """Return all rows from *table* in the daemon DB."""
    with _connect() as conn:
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()  # noqa: S608
    return [dict(row) for row in rows]


def _recording_ids_on_disk() -> set[str]:
    """Return recording IDs that exist as directories under the recordings root."""
    if not OFFLINE_RECORDINGS_ROOT.exists():
        return set()
    return {child.name for child in OFFLINE_RECORDINGS_ROOT.iterdir() if child.is_dir()}


def _expected_trace_files(trace: dict[str, Any]) -> set[Path]:
    """Return the expected on-disk files for a trace row."""
    trace_dir = Path(trace["path"]).resolve()
    if trace.get("data_type") in VIDEO_TRACE_DATA_TYPES:
        return {trace_dir / filename for filename in VIDEO_TRACE_FILENAMES}
    return {trace_dir / TRACE_JSON_NAME}


# ---------------------------------------------------------------------------
# Offline verification
# ---------------------------------------------------------------------------


def _assert_offline_db_invariants(
    *,
    results: list[dict[str, object]],
) -> None:
    """Assert offline-only DB invariants for all recordings in *results*.

    Checks that no upload or registration activity occurred:
    - ``registration_status == 'pending'`` on every trace
    - ``upload_status`` is not ``'uploading'`` or ``'retrying'``
    - ``bytes_uploaded <= total_bytes`` where both are set
    - ``expected_trace_count_reported == 0`` on every recording row
    """
    for result in results:
        for recording_id in result["recording_ids"]:
            recording = _fetch_recording(recording_id)
            assert (
                recording is not None
            ), f"Recording '{recording_id}' missing from recordings table"
            assert recording["expected_trace_count_reported"] == 0, (
                f"Recording '{recording_id}': expected_trace_count_reported="
                f"{recording['expected_trace_count_reported']} but should be 0 "
                "in offline mode"
            )

            for trace in _fetch_all_traces(recording_id):
                assert trace["registration_status"] == "pending", (
                    f"Trace {trace['trace_id']}: registration_status="
                    f"{trace['registration_status']} "
                    "(expected 'pending' in offline mode)"
                )
                assert trace["upload_status"] not in ("uploading", "retrying"), (
                    f"Trace {trace['trace_id']}: unexpected upload_status="
                    f"{trace['upload_status']} in offline mode"
                )
                total_bytes = trace["total_bytes"]
                uploaded = trace["bytes_uploaded"] or 0
                if total_bytes is not None and total_bytes > 0:
                    assert uploaded <= total_bytes, (
                        f"Trace {trace['trace_id']}: bytes_uploaded={uploaded} > "
                        f"total_bytes={total_bytes}"
                    )


def wait_for_all_traces_written(
    timeout_s: float = 30.0,
) -> None:
    """Poll until every trace for every recording on disk is written.

    Uses the recordings root directory as the source of truth for which
    recording IDs exist, rather than the results list.  This catches
    recordings the daemon started that the client-side results may not
    reflect (e.g. due to the 'already started' race on reconnect).

    Blocks until:
    - All recording IDs on disk have a matching row in the DB with stopped_at set.
    - Every recording's trace_count matches the number of trace rows in the DB.
    - Every trace row has write_status='written'.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        recording_ids = _recording_ids_on_disk()
        if not recording_ids:
            time.sleep(0.25)
            continue

        recordings = {
            row["recording_id"]: row
            for row in _fetch_all("recordings")
            if row["recording_id"] in recording_ids
        }
        traces = [
            trace
            for trace in _fetch_all("traces")
            if trace["recording_id"] in recording_ids
        ]
        traces_by_recording: dict[str, list[dict[str, Any]]] = {}
        for trace in traces:
            traces_by_recording.setdefault(trace["recording_id"], []).append(trace)

        if len(recordings) < len(recording_ids):
            time.sleep(0.25)
            continue

        all_stopped = all(row["stopped_at"] is not None for row in recordings.values())
        if not all_stopped:
            time.sleep(0.25)
            continue

        any_recording_has_no_traces = any(
            row["trace_count"] == 0 for row in recordings.values()
        )
        if any_recording_has_no_traces:
            time.sleep(0.25)
            continue

        counts_match = all(
            row["trace_count"] == len(traces_by_recording.get(rid, []))
            for rid, row in recordings.items()
        )
        if not counts_match:
            time.sleep(0.25)
            continue

        if all(trace["write_status"] == "written" for trace in traces):
            return
        time.sleep(0.25)

    recording_ids = _recording_ids_on_disk()
    recordings = _fetch_all("recordings")
    traces = _fetch_all("traces")
    unfinished = [
        {
            "trace_id": t["trace_id"],
            "recording_id": t["recording_id"],
            "write_status": t["write_status"],
        }
        for t in traces
        if t["recording_id"] in recording_ids and t["write_status"] != "written"
    ]
    missing_in_db = sorted(recording_ids - {row["recording_id"] for row in recordings})
    not_stopped = sorted(
        row["recording_id"]
        for row in recordings
        if row["recording_id"] in recording_ids and row["stopped_at"] is None
    )
    raise AssertionError(
        f"Daemon did not finish writing all traces within {timeout_s}s.\n"
        f"  Recordings on disk with no DB row: {missing_in_db}\n"
        f"  Recordings not yet stopped (stopped_at is NULL): {not_stopped}\n"
        f"  Traces still in non-written state ({len(unfinished)}):\n"
        + "\n".join(f"    {t}" for t in unfinished)
    )


def assert_db_contents(
    results: list[dict[str, object]],
    case: RecordingPlaybackMatrixCase,
    *,
    state_db_action: str = "preserve",
) -> None:
    """Assert the offline DB state matches expectations for *state_db_action*.

    ``"preserve"`` (default): DB exists with one recordings row per started
    recording, no orphaned traces, and every recording has at least one trace.

    ``"empty"``: DB file exists but both tables are empty (daemon cleanup
    cleared rows without deleting the file).

    ``"delete"``: DB file and its WAL/SHM sidecars are all absent.
    """
    if state_db_action == "delete":
        for candidate in (
            OFFLINE_DB_PATH,
            OFFLINE_DB_PATH.with_suffix(OFFLINE_DB_PATH.suffix + ".wal"),
            OFFLINE_DB_PATH.with_suffix(OFFLINE_DB_PATH.suffix + ".shm"),
        ):
            assert (
                not candidate.exists()
            ), f"DB artefact was not removed after cleanup: {candidate}"
        return

    assert (
        OFFLINE_DB_PATH.exists()
    ), f"Expected DB to exist after daemon stopped: {OFFLINE_DB_PATH}"

    recordings = _fetch_all("recordings")
    traces = _fetch_all("traces")

    if state_db_action == "empty":
        assert not recordings, (
            f"Expected empty recordings table after 'empty' action but found "
            f"{len(recordings)} row(s)"
        )
        assert not traces, (
            f"Expected empty traces table after 'empty' action but found "
            f"{len(traces)} row(s)"
        )
        return

    expected_recording_ids: set[str] = set()
    for result in results:
        for recording_id in result["recording_ids"]:
            expected_recording_ids.add(str(recording_id))

    actual_recording_ids = {row["recording_id"] for row in recordings}
    assert expected_recording_ids == actual_recording_ids, (
        f"DB recordings mismatch.\n"
        f"  Expected: {sorted(expected_recording_ids)}\n"
        f"  Actual:   {sorted(actual_recording_ids)}"
    )

    orphaned = [
        trace for trace in traces if trace["recording_id"] not in actual_recording_ids
    ]
    assert not orphaned, (
        f"{len(orphaned)} trace(s) in DB reference a recording_id with no matching "
        f"recordings row — orphaned trace IDs: "
        f"{sorted(t['trace_id'] for t in orphaned)}"
    )

    recordings_without_traces = expected_recording_ids - {
        trace["recording_id"] for trace in traces
    }
    assert not recordings_without_traces, (
        f"{len(recordings_without_traces)} recording(s) have no trace rows in DB — "
        f"recording IDs: {sorted(recordings_without_traces)}"
    )


def assert_disk_traces() -> None:
    """Assert that trace files on disk match exactly what the DB says was written.

    Checks:
    - The recordings root exists (traces were flushed to disk).
    - No trace row is in a non-terminal write_status after daemon shutdown
      ('initializing' or 'pending_metadata' indicate the daemon stopped before
      completing the DB write).
        - Every trace row with write_status='written' has the expected non-empty
            artifact files at its recorded path.
    - No files exist under the recordings root that are not referenced by any
      trace row in the DB (no ghost files).
    """
    assert (
        OFFLINE_RECORDINGS_ROOT.exists()
    ), f"Recordings root missing after daemon stopped: {OFFLINE_RECORDINGS_ROOT}"

    traces = _fetch_all("traces")

    non_terminal = [
        trace
        for trace in traces
        if trace["write_status"] in ("initializing", "pending_metadata")
    ]
    assert not non_terminal, (
        f"{len(non_terminal)} trace(s) still in a non-terminal write_status after "
        f"daemon shutdown — the daemon stopped before completing its DB writes.\n"
        f"  Affected traces:\n"
        + "\n".join(
            f"    trace_id={t['trace_id']}  recording_id={t['recording_id']}"
            f"  write_status={t['write_status']}"
            for t in non_terminal
        )
    )

    db_paths: set[Path] = set()
    for trace in traces:
        if not trace.get("path"):
            continue

        trace_dir = Path(trace["path"])
        expected_files = _expected_trace_files(trace)

        if trace["write_status"] == "written":
            assert trace_dir.is_dir(), (
                f"Trace {trace['trace_id']} (recording {trace['recording_id']}): "
                "DB write_status='written' but trace directory not found on "
                f"disk: {trace_dir}"
            )
            for expected_file in expected_files:
                assert expected_file.exists(), (
                    f"Trace {trace['trace_id']} (recording {trace['recording_id']}): "
                    f"DB write_status='written' but expected file not found on disk: "
                    f"{expected_file}"
                )
                assert expected_file.stat().st_size > 0, (
                    f"Trace {trace['trace_id']} (recording {trace['recording_id']}): "
                    f"expected file exists but is empty: {expected_file}"
                )

        db_paths.update(expected_files)

    disk_files = {
        path.resolve() for path in OFFLINE_RECORDINGS_ROOT.rglob("*") if path.is_file()
    }
    ghost_files = disk_files - db_paths
    if ghost_files:
        ghost_recording_ids = {
            Path(path).parts[-4] for path in (str(p) for p in ghost_files)
        }
        recordings = _fetch_all("recordings")
        recordings_for_ghosts = [
            row for row in recordings if row["recording_id"] in ghost_recording_ids
        ]
        traces_for_ghosts = [
            trace for trace in traces if trace["recording_id"] in ghost_recording_ids
        ]
        sample_ghost = next(iter(ghost_files))
        sample_db_path = next(iter(db_paths)) if db_paths else None
        assert not ghost_files, (
            f"{len(ghost_files)} file(s) on disk have no matching trace row in DB.\n"
            f"  Sample ghost disk path: {sample_ghost}\n"
            f"  Sample DB path:         {sample_db_path}\n"
            f"  Ghost recording IDs: {sorted(ghost_recording_ids)}\n"
            f"  DB recordings rows for those IDs ({len(recordings_for_ghosts)}):\n"
            + "\n".join(
                f"    recording_id={r['recording_id']}  trace_count={r['trace_count']}"
                f"  stopped_at={r['stopped_at']}"
                for r in recordings_for_ghosts
            )
            + f"\n  DB trace rows for those IDs ({len(traces_for_ghosts)}):\n"
            + "\n".join(
                f"    trace_id={t['trace_id']}  write_status={t['write_status']}"
                f"  path={t.get('path')}"
                for t in traces_for_ghosts[:5]
            )
            + (
                f"\n    ... and {len(traces_for_ghosts) - 5} more"
                if len(traces_for_ghosts) > 5
                else ""
            )
        )


def _reset_daemon_state(
    *,
    kill_daemon: bool = True,
    state_db_action: str = "empty",
) -> None:
    """Kill the daemon and wipe the entire test_data_daemon_state directory.

    Args:
        kill_daemon: Whether to kill the daemon before cleaning up.
        state_db_action: Passed to ``daemon_cleanup`` when ``kill_daemon`` is
            True.  One of ``"preserve"``, ``"empty"``, or ``"delete"``.  Has
            no effect when ``kill_daemon`` is False because the state directory
            is removed unconditionally by ``shutil.rmtree`` below.
    """
    if kill_daemon:
        daemon_cleanup(state_db_action=state_db_action)
    if state_db_action != "preserve" and TEST_DAEMON_STATE_DIR.exists():
        shutil.rmtree(TEST_DAEMON_STATE_DIR, ignore_errors=True)


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", OFFLINE_MATRIX_CASES, ids=case_id)
def test_recording_and_playback_matrix_offline(
    case: RecordingPlaybackMatrixCase,
    daemon_offline_env,
    reset_matrix_timer_stats,
    log_run_analysis_on_teardown,
) -> None:
    """Record offline and verify local SQLite and RDM file integrity."""
    _reset_daemon_state(kill_daemon=case.kill_daemon_between_tests)
    try:
        with use_offline_daemon_profile():
            try:
                results = run_case_contexts(case, wait=False)
            except Exception:
                log_run_analysis_on_teardown(case, [])
                raise
            log_run_analysis_on_teardown(case, results)
            assert_context_mode(case, results)
            wait_for_all_traces_written()
            _assert_offline_db_invariants(results=results)
            assert_db_contents(results, case)
            assert_disk_traces()
    finally:
        _reset_daemon_state(kill_daemon=case.kill_daemon_between_tests)
