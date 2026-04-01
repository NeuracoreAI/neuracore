"""Daemon isolation tests for the recording matrix.

Each parametrised case verifies that the daemon runs in a fully isolated
state: before and after the test there must be no leftover processes, files,
sockets, DB artefacts, recordings, or residual producer sub-processes.

During the test exactly one daemon PID should be alive. Offline cases keep the
daemon alive for the duration of the profile context; online cases keep it
alive through recording and upload verification.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from recording_playback_matrix_config import (
    MatrixDimensionConfig,
    RecordingPlaybackMatrixCase,
    assert_context_mode,
    build_context_specs,
    build_matrix_cases,
    case_id,
    has_configured_org,
    run_case_contexts,
)
from recording_playback_shared import (
    daemon_cleanup,
    fetch_recording_recovery_stats,
    get_runner_pids,
    use_offline_daemon_profile,
    wait_for_online_recovery,
)
from test_recording_and_playback_matrix_offline import (
    _assert_offline_db_invariants,
    _reset_daemon_state,
    assert_db_contents,
    assert_disk_traces,
    wait_for_all_traces_written,
)
from test_recording_and_playback_matrix_online import _verify_all_results

import neuracore as nc
from neuracore.data_daemon.const import SOCKET_PATH
from neuracore.data_daemon.helpers import (
    get_daemon_db_path,
    get_daemon_pid_path,
    get_daemon_recordings_root_path,
)
from neuracore.data_daemon.lifecycle.daemon_lifecycle import (
    ensure_daemon_running,
    pid_is_running,
    read_pid_from_file,
)

# ---------------------------------------------------------------------------
# Minimal dimension config – one representative case per isolation check
# ---------------------------------------------------------------------------

OFFLINE_ISOLATION_DIMENSION_CONFIG = MatrixDimensionConfig(
    durations_sec=(50,),
    parallel_contexts_options=(1, 2),
    recording_counts=(10,),
    joint_counts=(10, 30),
    producer_channel_options=("synchronous", "per_thread"),
    video_counts=(0, 1),
    image_dimensions=((64, 64), (1920, 1080)),
    kill_daemon_between_tests=True,
    state_db_action="empty",
)
ONLINE_ISOLATION_DIMENSION_CONFIG = MatrixDimensionConfig(
    durations_sec=(50,),
    parallel_contexts_options=(1,),
    recording_counts=(2,),
    joint_counts=(10,),
    producer_channel_options=("synchronous",),
    video_counts=(1,),
    image_dimensions=((64, 64),),
    kill_daemon_between_tests=True,
    state_db_action="preserve",
)
OFFLINE_ISOLATION_CASES = build_matrix_cases(OFFLINE_ISOLATION_DIMENSION_CONFIG)
ONLINE_ISOLATION_CASES = build_matrix_cases(ONLINE_ISOLATION_DIMENSION_CONFIG)


# ---------------------------------------------------------------------------
# Isolation assertion helpers
# ---------------------------------------------------------------------------


def _live_daemon_pids() -> set[int]:
    """Return PIDs of all live daemon processes (runner and PID-file)."""
    pid_path = get_daemon_pid_path()
    pids: set[int] = set(get_runner_pids())
    stored_pid = read_pid_from_file(pid_path)
    if stored_pid is not None and pid_is_running(stored_pid):
        pids.add(stored_pid)
    return pids


def _assert_no_daemon_pids() -> None:
    """Fail if any daemon PIDs are still running."""
    live = _live_daemon_pids()
    assert not live, (
        f"Daemon processes still running after expected shutdown — "
        f"PIDs: {sorted(live)}"
    )


def _assert_exactly_one_daemon_pid() -> int:
    """Fail unless exactly one daemon PID is alive, then return it."""
    live = _live_daemon_pids()
    assert len(live) == 1, (
        f"Expected exactly one daemon PID after startup but found {len(live)} — "
        f"PIDs: {sorted(live)}"
    )
    return next(iter(live))


def _assert_no_pid_file() -> None:
    """Fail if the daemon PID file still exists on disk."""
    pid_path = get_daemon_pid_path()
    assert (
        not pid_path.exists()
    ), f"PID file was not cleaned up after daemon shutdown: {pid_path}"


def _assert_socket_unlinked() -> None:
    """Fail if the daemon Unix socket path still exists."""
    socket_path = Path(str(SOCKET_PATH))
    assert (
        not socket_path.exists()
    ), f"Unix socket was not unlinked after daemon shutdown: {socket_path}"


def _assert_db_absent() -> None:
    """Fail if the active daemon DB file or its WAL/SHM sidecars still exist."""
    db_path = get_daemon_db_path()
    for candidate in (
        db_path,
        db_path.with_suffix(db_path.suffix + ".wal"),
        db_path.with_suffix(db_path.suffix + ".shm"),
    ):
        assert (
            not candidate.exists()
        ), f"DB artefact was not removed after cleanup: {candidate}"


def _assert_recordings_folder_absent() -> None:
    """Fail if the active daemon recordings root directory still exists."""
    recordings_root = get_daemon_recordings_root_path()
    assert (
        not recordings_root.exists()
    ), f"Recordings folder still present: {recordings_root}"


def _assert_no_producer_processes() -> None:
    """Fail if any runner/producer subprocesses are still in the process table."""
    runner_pids = get_runner_pids()
    assert not runner_pids, (
        f"Producer/runner subprocesses still alive after test — "
        f"PIDs: {sorted(runner_pids)}"
    )


def _snapshot_daemon_disk_state() -> dict[str, object]:
    """Capture the current on-disk state of all daemon-managed paths.

    Returns a dict with existence flags and file sizes for the DB, its
    WAL/SHM sidecars, the PID file, the socket path, and every file under
    the recordings root.  Used to verify the disk is in the same state
    before and after a test.
    """
    pid_path = get_daemon_pid_path()
    db_path = get_daemon_db_path()
    recordings_root = get_daemon_recordings_root_path()
    socket_path = Path(str(SOCKET_PATH))

    def _file_info(path: Path) -> dict[str, object] | None:
        if not path.exists():
            return None
        return {"size": path.stat().st_size}

    recordings_files: dict[str, int] = {}
    if recordings_root.exists():
        for file_path in recordings_root.rglob("*"):
            if file_path.is_file():
                recordings_files[str(file_path)] = file_path.stat().st_size

    db_wal = db_path.with_suffix(db_path.suffix + ".wal")
    db_shm = db_path.with_suffix(db_path.suffix + ".shm")

    return {
        "db": _file_info(db_path),
        "db_wal": _file_info(db_wal),
        "db_shm": _file_info(db_shm),
        "pid_file": _file_info(pid_path),
        "socket": socket_path.exists(),
        "recordings_files": recordings_files,
    }


def _reset_generic_daemon_state(*, state_db_action: str = "delete") -> None:
    """Kill the daemon and remove the active DB and recordings root."""
    daemon_cleanup(state_db_action=state_db_action)
    recordings_root = get_daemon_recordings_root_path()
    if state_db_action != "preserve" and recordings_root.exists():
        shutil.rmtree(recordings_root, ignore_errors=True)


def _assert_online_recovery_invariants(results: list[dict[str, object]]) -> None:
    """Assert that each recording shows online recovery progress in the daemon DB."""
    for result in results:
        for recording_id in result["recording_ids"]:
            wait_for_online_recovery(str(recording_id))
            stats = fetch_recording_recovery_stats(str(recording_id))
            made_progress = (
                stats["non_pending_registration_traces"] > 0
                or stats["upload_progress_traces"] > 0
                or stats["uploaded_traces"] > 0
                or (stats["expected_trace_count"] or 0) > 0
                or stats["progress_reported"] == "reported"
            )
            assert made_progress, (
                "Online recovery showed no registration/upload progress for "
                f"recording {recording_id}; stats={stats}"
            )


def _assert_disk_state_unchanged(
    before: dict[str, object], after: dict[str, object]
) -> None:
    """Assert the daemon disk state is identical before and after a test."""
    mismatches: list[str] = []
    for key in ("db", "db_wal", "db_shm", "pid_file"):
        if before[key] != after[key]:
            mismatches.append(f"  {key}: before={before[key]!r}  after={after[key]!r}")
    if before["socket"] != after["socket"]:
        mismatches.append(
            f"  socket: before={before['socket']}  after={after['socket']}"
        )
    before_recordings = before["recordings_files"]
    after_recordings = after["recordings_files"]
    if before_recordings != after_recordings:
        added = set(after_recordings) - set(before_recordings)
        removed = set(before_recordings) - set(after_recordings)
        if added:
            mismatches.append(f"  recordings added: {sorted(added)}")
        if removed:
            mismatches.append(f"  recordings removed: {sorted(removed)}")
    assert not mismatches, (
        "On-disk daemon state is not identical before and after the test "
        "(test left artefacts behind):\n" + "\n".join(mismatches)
    )


def assert_fully_isolated(state_db_action: str = "delete") -> None:
    """Assert every isolation invariant in one call.

    When *state_db_action* is ``"preserve"`` the DB file and recordings folder
    are intentionally left on disk for post-mortem inspection, so the DB-absent
    and recordings-folder-absent checks are skipped.
    """
    _assert_no_daemon_pids()
    _assert_no_pid_file()
    _assert_socket_unlinked()
    if state_db_action != "preserve":
        _assert_db_absent()
        _assert_recordings_folder_absent()
    _assert_no_producer_processes()


# ---------------------------------------------------------------------------
# Parametrised isolation test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", OFFLINE_ISOLATION_CASES, ids=case_id)
def test_daemon_isolation_around_matrix_offline(
    case: RecordingPlaybackMatrixCase,
    daemon_offline_env,
    reset_matrix_timer_stats,
    log_run_analysis_on_teardown,
) -> None:
    """Verify full daemon isolation before and after each offline matrix case.

    Assertion thread
    ----------------
    1. Pre-conditions – environment is clean before the daemon starts:
       - No daemon PIDs alive (runner processes + PID file)
       - PID file absent
       - Unix socket path unlinked
       - DB file and WAL/SHM sidecars absent
       - Recordings folder absent
       - No residual producer sub-processes

    2. Daemon starts – exactly one daemon PID is alive and stays alive for the
       duration of the profile context.

    3. Workload runs – ``run_case_contexts`` drives recordings; ``wait=False``
       means stop_recording returns before the daemon finishes writing.

    4. Traces written – poll until every recording on disk has ``stopped_at``
       set, ``trace_count`` matches the DB row count, and every trace row has
       ``write_status='written'``.

    5. Offline DB invariants – while the daemon is still running and the DB is
       live, assert the offline-specific guarantees:
       - ``registration_status == 'pending'`` on every trace (no registration
         attempted)
       - ``upload_status`` not ``'uploading'`` or ``'retrying'``
       - ``bytes_uploaded <= total_bytes`` where both are present
       - ``expected_trace_count_reported == 0`` on every recording row

    6. Daemon still alive – the PID from step 2 is still running, confirming
       the daemon did not crash during the workload.

    7. Daemon terminates – the profile context exits, stopping the daemon.
       Assert no daemon PIDs remain and no producer sub-processes linger.

    8. DB and disk consistency – after shutdown:
       - DB contains exactly the expected recording IDs, no orphaned traces,
         and every recording has at least one trace.
       - Every ``write_status='written'`` trace has a non-empty file at its
         recorded path; no ghost files exist on disk.

    9. Post-conditions – after ``_reset_daemon_state`` cleans up (daemon killed,
       state directory removed via ``shutil.rmtree``), the full isolation
       invariant set from step 1 holds again.  DB absence is a consequence of
       the directory removal, not a daemon guarantee — the daemon's own DB
       consistency was verified in step 8.  The on-disk state snapshot must
       also be identical to before the test.
    """
    # 1. Pre-conditions — always do a full clean at the start regardless of
    # case.state_db_action value.
    _reset_daemon_state(kill_daemon=True)
    assert_fully_isolated()
    disk_state_before = _snapshot_daemon_disk_state()

    results: list[dict[str, object]] = []
    try:
        with use_offline_daemon_profile():
            # 2. Daemon starts
            daemon_pid = _assert_exactly_one_daemon_pid()
            assert pid_is_running(
                daemon_pid
            ), f"Daemon PID {daemon_pid} is not running after profile startup"

            # 3. Workload runs
            results = run_case_contexts(case, wait=False)
            log_run_analysis_on_teardown(case, results)

            # 4. Traces written
            wait_for_all_traces_written()

            # 5. Offline DB invariants
            _assert_offline_db_invariants(results=results)

            # 6. Daemon still alive
            assert pid_is_running(
                daemon_pid
            ), f"Daemon PID {daemon_pid} died unexpectedly during the test"

        # 7. Daemon terminates
        _assert_no_daemon_pids()
        _assert_no_producer_processes()

        # 8. DB and disk consistency
        # Validate the completed run against the preserved DB contents before
        # the final cleanup path applies case.state_db_action.
        assert_db_contents(results, case, state_db_action="preserve")
        assert_disk_traces()

    finally:
        _reset_daemon_state(kill_daemon=True, state_db_action=case.state_db_action)

    # 9. Post-conditions
    assert_fully_isolated(state_db_action=case.state_db_action)
    if case.state_db_action != "preserve":
        _assert_disk_state_unchanged(disk_state_before, _snapshot_daemon_disk_state())


@pytest.mark.parametrize("case", ONLINE_ISOLATION_CASES, ids=case_id)
def test_daemon_isolation_around_matrix_online(
    case: RecordingPlaybackMatrixCase,
    dataset_cleanup,
    reset_matrix_timer_stats,
    log_run_analysis_on_teardown,
) -> None:
    """Verify full daemon isolation before and after each online matrix case.

    Assertion thread
    ----------------
    1. Pre-conditions – environment is clean before the daemon starts:
       - No daemon PIDs alive (runner processes + PID file)
       - PID file absent
       - Unix socket path unlinked
       - DB file and WAL/SHM sidecars absent
       - Recordings folder absent
       - No residual producer sub-processes

    2. Daemon starts – explicitly launch the online daemon and assert exactly
       one daemon PID is alive.

     3. Workload runs – ``run_case_contexts`` drives online recordings with
         ``wait=False`` so the daemon continues upload/recovery work after
         ``stop_recording`` returns.

    4. Online recovery invariants – while the daemon is still running, every
       recording must show registration/upload progress or completed progress
       reporting in the daemon DB.

    5. Daemon still alive – the PID from step 2 remains alive through the
       upload and playback verification path.

    6. Playback verified – the dataset becomes ready on the platform and every
       uploaded recording can be downloaded and validated.

    7. Post-conditions – after explicit cleanup, the full isolation invariant
       set from step 1 holds again and the on-disk daemon state matches the
       pre-test snapshot.
    """
    nc.login()
    if not has_configured_org():
        pytest.skip(
            "Recording/playback matrix tests require NEURACORE_ORG_ID"
            " or a saved current organization."
        )

    wait_for_upload = False

    _reset_generic_daemon_state()
    assert_fully_isolated()
    disk_state_before = _snapshot_daemon_disk_state()

    specs = build_context_specs(case, wait=wait_for_upload)
    dataset_cleanup(str(specs[0]["dataset_name"]))

    results: list[dict[str, object]] = []
    try:
        daemon_pid = ensure_daemon_running(timeout_s=10.0)
        assert daemon_pid == _assert_exactly_one_daemon_pid(), (
            "Daemon PID mismatch after explicit online startup: "
            f"ensure_daemon_running returned {daemon_pid}"
        )
        assert pid_is_running(
            daemon_pid
        ), f"Daemon PID {daemon_pid} is not running after online startup"

        results = run_case_contexts(case, wait=wait_for_upload, specs=specs)
        log_run_analysis_on_teardown(case, results)
        assert_context_mode(case, results)

        _assert_online_recovery_invariants(results)

        assert pid_is_running(
            daemon_pid
        ), f"Daemon PID {daemon_pid} died during online recovery"

        nc.login()
        _verify_all_results(results=results, case=case)

        assert pid_is_running(
            daemon_pid
        ), f"Daemon PID {daemon_pid} died before online verification completed"
    finally:
        _reset_generic_daemon_state()

    assert_fully_isolated()
    _assert_disk_state_unchanged(disk_state_before, _snapshot_daemon_disk_state())
