"""Behavioural correctness tests for cancel-recording flows.

Verifies that cancelling a recording discards all logged data, and that a
valid recording survives a cancel either side of it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import pytest

import neuracore as nc
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
    assert_post_test_storage_state,
    verify_cloud_results,
)
from tests.integration.platform.data_daemon.shared.db_constants import (
    COLUMN_RECORDING_INDEX,
    COLUMN_UPLOAD_STATUS,
    TRACE_UPLOAD_UPLOADED,
    TRACES_TABLE,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    fetch_all_rows,
    fetch_recording,
    wait_for_recording_index_for_source,
)
from tests.integration.platform.data_daemon.shared.disk_helpers import (
    list_recording_indexes_on_disk,
)
from tests.integration.platform.data_daemon.shared.process_control import Timer
from tests.integration.platform.data_daemon.shared.runners import online_daemon_running
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestBatch,
    DataDaemonTestCase,
    PerThread,
    Synchronous,
    case_ids,
    has_configured_org,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    CONTROL_REMOTE,
    CONTROL_SPLIT_PROCESS,
    MAX_TIME_TO_START_S,
    camera_names,
    joint_names_for_count,
)
from tests.integration.platform.data_daemon.shared.test_case.context_spec import (
    ContextResult,
    build_context_specs,
)
from tests.integration.platform.data_daemon.shared.test_case.context_worker import (
    create_testing_dataset_name,
    log_frames,
)
from tests.integration.platform.data_daemon.shared.test_case.producers import (
    make_producer_session,
)
from tests.integration.platform.data_daemon.shared.test_case.recording_control import (
    await_gate,
    make_recording_controller,
)
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    scoped_storage_state,
    set_case_analysis_report,
)

logger = logging.getLogger(__name__)

_CLEANUP_CASES = DataDaemonTestBatch(
    cases=(
        Synchronous(
            duration_sec=5,
            joint_count=4,
            video_count=1,
            image_width=64,
            image_height=64,
        ),
        PerThread(
            duration_sec=5,
            joint_count=4,
            video_count=1,
            image_width=64,
            image_height=64,
            recording_control=CONTROL_REMOTE,
        ),
    ),
).as_cases()
_NEIGHBOR_CASES = DataDaemonTestBatch(
    cases=(
        Synchronous(
            duration_sec=5,
            joint_count=4,
            video_count=1,
            image_width=64,
            image_height=64,
        ),
        Synchronous(
            duration_sec=5,
            joint_count=4,
            video_count=1,
            image_width=64,
            image_height=64,
            recording_control=CONTROL_SPLIT_PROCESS,
        ),
    ),
).as_cases()


def _await_window_open(robot: object) -> None:
    """Wait for the daemon to report the window open.

    A local start returns before the daemon has applied it, so the window is
    not open the instant the call comes back.
    """
    await_gate(
        robot,
        open_gate=True,
        deadline=time.time() + MAX_TIME_TO_START_S,
        label="cancel.open_gate_wait",
        overdue="the recording to cancel never opened",
        assert_deadline=False,
    )


_CANCEL_CLEANUP_TIMEOUT_S = 75.0


def _trace_rows(recording_index: int) -> list[dict[str, Any]]:
    """Return every trace still tied to the recording, including orphans."""
    return [
        row
        for row in fetch_all_rows(TRACES_TABLE)
        if int(row[COLUMN_RECORDING_INDEX]) == recording_index
    ]


def _await_materialized_recording(recording_index: int) -> None:
    """Prove the test is cancelling real, not-yet-uploaded local state."""
    deadline = time.monotonic() + MAX_TIME_TO_START_S
    last_state: tuple[bool, int, bool, bool] | None = None
    while time.monotonic() < deadline:
        recording_exists = fetch_recording(recording_index) is not None
        traces = _trace_rows(recording_index)
        on_disk = recording_index in list_recording_indexes_on_disk()
        has_pending_trace = any(
            trace[COLUMN_UPLOAD_STATUS] != TRACE_UPLOAD_UPLOADED for trace in traces
        )
        last_state = (recording_exists, len(traces), on_disk, has_pending_trace)
        if recording_exists and traces and on_disk and has_pending_trace:
            return
        time.sleep(0.1)

    raise AssertionError(
        f"Recording {recording_index} never materialized before cancel; "
        f"last state was {last_state}"
    )


def _await_recording_fully_reclaimed(recording_index: int) -> None:
    """Wait until the cancelled recording is absent from DB and disk."""
    deadline = time.monotonic() + _CANCEL_CLEANUP_TIMEOUT_S
    last_state: tuple[dict[str, Any] | None, list[dict[str, Any]], bool] | None = None
    while time.monotonic() < deadline:
        recording = fetch_recording(recording_index)
        traces = _trace_rows(recording_index)
        on_disk = recording_index in list_recording_indexes_on_disk()
        last_state = (recording, traces, on_disk)
        if recording is None and not traces and not on_disk:
            return
        time.sleep(0.25)

    recording, traces, on_disk = last_state or (None, [], False)
    raise AssertionError(
        f"Cancelled recording {recording_index} was not fully reclaimed within "
        f"{_CANCEL_CLEANUP_TIMEOUT_S}s: recording_row={recording!r}, "
        f"trace_rows={traces!r}, on_disk={on_disk}"
    )


@pytest.mark.parametrize("case", _CLEANUP_CASES, ids=case_ids(_CLEANUP_CASES))
def test_cancel_recording_produces_no_data(
    case: DataDaemonTestCase,
    clear_daemon_timer_stats,
    request: pytest.FixtureRequest,
    test_wall_timer: Callable[[], float],
) -> None:
    """Verify that cancelling a recording discards all logged data.

    Runs both real paths: a producer-side SDK cancel and a backend-side cancel
    delivered to the daemon by the recording notification stream.
    """
    if not has_configured_org():
        pytest.skip(
            "Cancel-recording behavioural tests require NEURACORE_ORG_ID"
            " or a saved current organization."
        )

    dataset_name = create_testing_dataset_name(case)
    specs = build_context_specs(case, dataset_name=dataset_name)
    spec = specs[0]
    robot_name = spec.robot_name
    controller = None

    try:
        with scoped_storage_state(case, specs):
            with online_daemon_running():
                assert_exactly_one_daemon_pid()

                with Timer(
                    MAX_TIME_TO_START_S, label="nc.create_dataset", always_log=True
                ):
                    nc.create_dataset(dataset_name, description="Cancel recording test")
                with Timer(
                    MAX_TIME_TO_START_S, label="nc.connect_robot", always_log=True
                ):
                    robot = nc.connect_robot(robot_name, overwrite=False)

                controller = make_recording_controller(spec, robot=robot)
                producer = make_producer_session(spec, marker_name="marker_cancel")
                producer.start()
                try:
                    controller.open(time.time())
                    _await_window_open(robot)

                    producer.run_recording(0)

                    # A backend-announced window is materialized lazily by its
                    # first datum. Resolve its row only after publishing data;
                    # waiting before that deadlocks the remote-control case.
                    recording_index = wait_for_recording_index_for_source(
                        str(robot.id),
                        int(robot.instance),
                    )
                    _await_materialized_recording(recording_index)
                    controller.cancel(time.time())
                finally:
                    producer.finish()
                _await_recording_fully_reclaimed(recording_index)

                with Timer(
                    MAX_TIME_TO_START_S,
                    label="nc.get_dataset",
                    always_log=True,
                    assert_deadline=False,
                ):
                    dataset = nc.get_dataset(dataset_name)
                assert (
                    len(dataset) == 0
                ), f"Expected 0 recordings after cancel, got {len(dataset)}"
    finally:
        if controller is not None:
            controller.shutdown()
        set_case_analysis_report(
            request=request,
            case=case,
            results=[],
            test_wall_s=test_wall_timer(),
        )

    assert_post_test_storage_state(case.storage_state_action)


@pytest.mark.parametrize("case", _NEIGHBOR_CASES, ids=case_ids(_NEIGHBOR_CASES))
@pytest.mark.parametrize("gap_s", [0, 10], ids=["no_gap", "10s_gap"])
def test_cancel_either_side_of_a_valid_recording(
    gap_s: int,
    case: DataDaemonTestCase,
    clear_daemon_timer_stats,
    request: pytest.FixtureRequest,
    test_wall_timer: Callable[[], float],
) -> None:
    """Verify a cancel discards its own window and neither of its neighbours.

    Two variants are tested: resuming immediately (gap_s=0) and after a 10s
    pause (gap_s=10) to cover both tight and relaxed timing paths. Under split
    control this is the sharper of the two cancel tests: a successor window
    opens right behind a cancelled one, which is exactly the shape that lets an
    unqualified control call reach the wrong recording.

    A third window is then opened and cancelled behind the valid recording.
    Stopping a recording does not finish it: the daemon retains its window while
    the tail of its data drains, and a cancel arriving in that interval names
    only the recording in progress. So the valid one must come through whole —
    every frame, not merely present in the dataset — while the cancelled window
    leaves nothing behind. That window logs nothing: an empty one is destroyed
    exactly like a full one.
    """
    if not has_configured_org():
        pytest.skip(
            "Cancel-recording behavioural tests require NEURACORE_ORG_ID"
            " or a saved current organization."
        )

    dataset_name = create_testing_dataset_name(case)
    specs = build_context_specs(case, dataset_name=dataset_name)
    spec = specs[0]
    robot_name = spec.robot_name
    results: list[ContextResult] = []
    controller = None

    try:
        with scoped_storage_state(case, specs):
            with online_daemon_running():
                assert_exactly_one_daemon_pid()

                with Timer(
                    MAX_TIME_TO_START_S, label="nc.create_dataset", always_log=True
                ):
                    nc.create_dataset(
                        dataset_name,
                        description=f"Cancel-then-resume test gap={gap_s}s",
                    )
                with Timer(
                    MAX_TIME_TO_START_S, label="nc.connect_robot", always_log=True
                ):
                    robot = nc.connect_robot(robot_name, overwrite=False)

                controller = make_recording_controller(spec, robot=robot)

                # --- cancelled recording ---
                controller.open(time.time())
                _await_window_open(robot)

                log_frames(
                    spec, robot=robot, recording_index=0, marker_name="marker_cancelled"
                )

                controller.cancel(time.time())

                if gap_s > 0:
                    logger.info("Waiting %ds between cancel and next recording", gap_s)
                    time.sleep(gap_s)

                # --- valid recording ---
                recording_capture_start_s = time.time()
                recording_capture_stop_s = recording_capture_start_s + case.duration_sec
                opened = controller.open(recording_capture_start_s)
                wall_started_at = opened.settled_at
                resumed_recording_id = robot.get_cloud_recording_id()
                assert resumed_recording_id is not None

                log_frames(
                    spec, robot=robot, recording_index=0, marker_name="marker_resume"
                )

                controller.close(recording_capture_stop_s)
                wall_stopped_at = time.time()

                # --- cancelled window behind the valid recording ---
                controller.open(time.time())
                _await_window_open(robot)
                controller.cancel(time.time())

                results = [
                    ContextResult(
                        dataset_name=dataset_name,
                        recording_ids=[resumed_recording_id],
                        robot_name=robot_name,
                        joint_names=joint_names_for_count(spec.case.joint_count),
                        camera_names=camera_names(spec.case.video_count),
                        joint_frame_count=spec.expected_joint_frames,
                        video_frame_count=spec.expected_video_frames,
                        joint_fps=spec.case.joint_fps,
                        video_fps=spec.case.video_fps,
                        duration_sec=case.duration_sec,
                        timestamp_start_s=spec.timestamp_start_s,
                        timestamp_end_s=spec.timestamp_start_s + case.duration_sec,
                        marker_names=["marker_resume"],
                        has_video=bool(spec.case.video_count),
                        context_index=0,
                        wall_started_at=wall_started_at,
                        wall_stopped_at=wall_stopped_at,
                        random_phase=case.random_phase,
                    )
                ]
                verify_cloud_results(results=results, case=case)
    finally:
        if controller is not None:
            controller.shutdown()
        set_case_analysis_report(
            request=request,
            case=case,
            results=results,
            label_prefix="no_gap" if gap_s == 0 else f"{gap_s}s_gap",
            test_wall_s=test_wall_timer(),
        )
