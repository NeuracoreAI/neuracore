from __future__ import annotations

import time
import uuid
from collections.abc import Callable

import pytest

import neuracore as nc
from tests.integration.platform.data_daemon.daemon_test_cases import (
    PRE_NETWORK_INTEGRITY_CASES,
)
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
    verify_cloud_results,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    ObservedRecordingUploads,
    latching_upload_observer,
    resolve_cloud_recording_ids,
    wait_for_recording_index_for_source,
    wait_for_upload_complete_in_db,
)
from tests.integration.platform.data_daemon.shared.disk_helpers import (
    assert_rgb_trace_survived_boundary,
)
from tests.integration.platform.data_daemon.shared.runners import (
    online_daemon_running,
    split_video_process_running,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestBatch,
    DataDaemonTestCase,
    case_ids,
    has_configured_org,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case_context import (  # noqa: E501
    ContextResult,
    build_context_specs,
    create_testing_dataset_name,
    run_case_contexts,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    DETAIL_REALISTIC,
    MAX_TIME_TO_START_S,
    PACING_BURST_VIDEO,
    PRODUCER_CONTINUOUS,
    STOP_METHOD_CLI,
    STORAGE_STATE_DELETE,
)
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    scoped_storage_state,
    set_case_analysis_report,
)

_CASES = DataDaemonTestBatch(
    cases=PRE_NETWORK_INTEGRITY_CASES,
    storage_state_action=STORAGE_STATE_DELETE,
    stop_method=STOP_METHOD_CLI,
    producer_pacing=PACING_BURST_VIDEO,
    producer_channels=PRODUCER_CONTINUOUS,
).as_cases()


def _assert_online_verification_invariants(
    results: list[ContextResult],
    *,
    observed: ObservedRecordingUploads,
    timeout_seconds: float = 30.0,
) -> None:
    """Block until every recording in *results* has reached ``upload_complete``
    in the platform DB.  Must be called before cloud frame verification so
    that downloaded data reflects the fully-committed upload state.

    Upload completion is tracked in the daemon DB by the local
    ``recording_index`` correlation key.

    Recordings whose completion *observed* already latched during the record
    phase are satisfied: their rows have since been reclaimed by the recording
    reaper, so there is nothing left here to poll. Everything else is waited on
    exactly as before — not being complete yet is precisely what makes a
    recording ineligible for reclamation, so its rows are still there.
    """
    for result in results:
        for recording_index in result.recording_indexes:
            if observed.is_complete(recording_index):
                continue
            wait_for_upload_complete_in_db(recording_index, timeout_s=timeout_seconds)


@pytest.mark.parametrize("case", _CASES, ids=case_ids(_CASES))
def test_cloud_data_integrity(
    case: DataDaemonTestCase,
    clear_daemon_timer_stats,
    request: pytest.FixtureRequest,
    test_wall_timer: Callable[[], float],
) -> None:
    """Record data in online mode and verify cloud-side data integrity.

    Extends pre-network integrity (local disk timestamps + SQLite write status)
    by confirming the upload is correct on the platform side.

    - asserts no leftover daemon state before starting (isolation pre-condition)
    - records all context specs against the live platform
    - waits for every recording to reach ``upload_complete`` in the daemon DB
    - asserts exactly one daemon PID throughout
    - structural pass: verifies recording duration, byte size, and robot ID
      from the cloud (no sync required)
    - data pass: synchronises the dataset and validates per-episode frame
      counts and joint values against what was recorded
    - asserts no residual processes, files, sockets, or DB artefacts remain
      (isolation post-condition)
    """
    if not has_configured_org():
        pytest.skip(
            "Recording/playback matrix tests require NEURACORE_ORG_ID"
            " or a saved current organization."
        )

    dataset_name = create_testing_dataset_name(case)
    specs = build_context_specs(case, dataset_name=dataset_name)
    results: list[ContextResult] = []

    with scoped_storage_state(case, dataset_name=dataset_name):
        try:
            with online_daemon_running():
                assert_exactly_one_daemon_pid()
                with latching_upload_observer() as observed:
                    results = run_case_contexts(case, specs=specs)
                _assert_online_verification_invariants(results, observed=observed)
                results = resolve_cloud_recording_ids(results, observed=observed)
                verify_cloud_results(results=results, case=case)

        finally:
            set_case_analysis_report(
                request=request,
                case=case,
                results=results,
                test_wall_s=test_wall_timer(),
            )


def test_split_process_video_survives_recording_boundary() -> None:
    """A video producer in a different OS process than the recording owner
    must not lose its tail chunk at the recording boundary.

    Reproduces the real-world topology behind the RGB-tail-chunk-orphaned bug:
    one process owns ``start_recording``/``stop_recording`` for a robot while
    a separate process — sharing the same source, connected independently via
    :func:`~runners.split_video_process_running` — owns that robot's RGB
    video. The daemon's per-producer flush-marker tracking (see
    ``rust/data_daemon/src/pipeline/dispatcher.rs``) exists so the
    video-less lifecycle process's flush marker cannot vouch for the video
    process's still-open writer barrier.

    Requires ``online_daemon_running()``: the video-only process never calls
    ``start_recording`` itself, so it only learns a recording is active via
    the recording-state manager's SSE notification stream, which offline
    daemons never emit — see ``behavioural_correctness/test_multi_process_producers.py``
    for the same online-only constraint on a cross-process producer.

    Duration and resolution are pinned to the original bug report's shape:
    640x480x3 bytes/frame crosses the writer's 256 MiB chunk-flush threshold
    at frame 292 of 300 (30fps x 10s), so a chunk seals mid-recording — the
    daemon has already attributed this window's video to the producer process
    before its tail chunk (frames 292-300) is announced after the stop.

    Asserts the RGB trace reaches ``write_status='written'``, has the full
    expected frame count, and that its last on-disk frame does not trail the
    recording's nominal end by more than a couple of video frame intervals —
    the assertion a truncated tail chunk fails and an exact-count check alone
    could miss for a producer that just delivers slightly fewer frames.
    """
    if not has_configured_org():
        pytest.skip(
            "Recording/playback matrix tests require NEURACORE_ORG_ID"
            " or a saved current organization."
        )

    run_id = uuid.uuid4().hex[:10]
    dataset_name = f"split_video_dataset_{run_id}"
    robot_name = f"split_video_robot_{run_id}"
    case = DataDaemonTestCase(
        duration_sec=10,
        joint_count=0,
        video_count=1,
        image_width=640,
        image_height=480,
        video_fps=30,
        video_detail=DETAIL_REALISTIC,
        storage_state_action=STORAGE_STATE_DELETE,
    )

    robot = None
    with scoped_storage_state(case, dataset_name=dataset_name):
        try:
            with online_daemon_running():
                assert_exactly_one_daemon_pid()
                nc.create_dataset(dataset_name)

                with split_video_process_running(
                    robot_name=robot_name,
                    dataset_name=dataset_name,
                    camera_name="camera_0",
                    case=case,
                ):
                    robot = nc.connect_robot(robot_name, overwrite=False)
                    nc.start_recording(robot_name=robot_name, timestamp=0.0)
                    recording_index = wait_for_recording_index_for_source(
                        str(robot.id),
                        int(robot.instance),
                        timeout_s=MAX_TIME_TO_START_S,
                    )
                    time.sleep(case.duration_sec)
                    nc.stop_recording(
                        robot_name=robot_name,
                        wait=True,
                        timestamp=float(case.duration_sec),
                    )
                    time.sleep(2.0)

                assert_rgb_trace_survived_boundary(recording_index, case)
        finally:
            if robot is not None:
                robot.close()
