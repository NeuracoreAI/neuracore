"""Behavioural correctness tests for cancel-recording flows.

Verifies that cancelling a recording discards all logged data, and that a
valid recording can follow immediately after a cancelled one.
"""

from __future__ import annotations

import logging
import time

import pytest

import neuracore as nc
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
    verify_cloud_results,
)
from tests.integration.platform.data_daemon.shared.process_control import Timer
from tests.integration.platform.data_daemon.shared.runners import online_daemon_running
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestBatch,
    DataDaemonTestCase,
    camera_names,
    case_ids,
    has_configured_org,
    joint_names_for_count,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case_context import (  # noqa: E501
    ContextResult,
    build_context_specs,
    create_testing_dataset_name,
    log_frames,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    MAX_TIME_TO_START_S,
    STOP_RECORDING_OVERHEAD_PER_SEC,
    TIMESTAMP_MODE_REAL,
)
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    scoped_storage_state,
    set_case_analysis_report,
)

logger = logging.getLogger(__name__)

_CASES = DataDaemonTestBatch(
    cases=(
        DataDaemonTestCase(
            duration_sec=5,
            joint_count=4,
            video_count=1,
            image_width=64,
            image_height=64,
        ),
        DataDaemonTestCase(
            duration_sec=5,
            joint_count=4,
            video_count=1,
            image_width=64,
            image_height=64,
            timestamp_mode=TIMESTAMP_MODE_REAL,
        ),
    ),
).as_cases()


@pytest.mark.parametrize("case", _CASES, ids=case_ids(_CASES))
def test_cancel_recording_produces_no_data(
    case: DataDaemonTestCase,
    clear_daemon_timer_stats,
    request: pytest.FixtureRequest,
) -> None:
    """Verify that cancelling a recording discards all logged data."""
    if not has_configured_org():
        pytest.skip(
            "Cancel-recording behavioural tests require NEURACORE_ORG_ID"
            " or a saved current organization."
        )

    dataset_name = create_testing_dataset_name(case)
    specs = build_context_specs(case, dataset_name=dataset_name)
    spec = specs[0]
    robot_name = spec.robot_name

    with scoped_storage_state(case, dataset_name=dataset_name):
        try:
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

                with Timer(
                    MAX_TIME_TO_START_S, label="nc.start_recording", always_log=True
                ):
                    nc.start_recording(robot_name=robot_name)
                cancelled_recording_id = robot.get_current_recording_id()
                assert cancelled_recording_id is not None

                log_frames(spec, recording_index=0, marker_name="marker_cancel")

                with Timer(
                    case.duration_sec * STOP_RECORDING_OVERHEAD_PER_SEC,
                    label="nc.cancel_recording",
                    always_log=True,
                    assert_limit=False,
                ):
                    nc.cancel_recording(robot_name=robot_name)

                time.sleep(5)

                with Timer(
                    MAX_TIME_TO_START_S,
                    label="nc.get_dataset",
                    always_log=True,
                    assert_limit=False,
                ):
                    dataset = nc.get_dataset(dataset_name)
                assert (
                    len(dataset) == 0
                ), f"Expected 0 recordings after cancel, got {len(dataset)}"
        finally:
            set_case_analysis_report(
                request=request,
                case=case,
                results=[],
            )


@pytest.mark.parametrize("case", _CASES, ids=case_ids(_CASES))
@pytest.mark.parametrize("gap_s", [0, 10], ids=["no_gap", "10s_gap"])
def test_cancel_then_start_new_recording(
    gap_s: int,
    case: DataDaemonTestCase,
    clear_daemon_timer_stats,
    request: pytest.FixtureRequest,
) -> None:
    """Verify a valid recording succeeds after cancelling a prior one.

    Two variants are tested: resuming immediately (gap_s=0) and after a 10s
    pause (gap_s=10) to cover both tight and relaxed timing paths.
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

    with scoped_storage_state(case, dataset_name=dataset_name):
        try:
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

                # --- cancelled recording ---
                with Timer(
                    MAX_TIME_TO_START_S, label="nc.start_recording", always_log=True
                ):
                    nc.start_recording(robot_name=robot_name)
                cancelled_recording_id = robot.get_current_recording_id()
                assert cancelled_recording_id is not None

                log_frames(spec, recording_index=0, marker_name="marker_cancelled")

                with Timer(
                    case.duration_sec * STOP_RECORDING_OVERHEAD_PER_SEC,
                    label="nc.cancel_recording",
                    always_log=True,
                    assert_limit=False,
                ):
                    nc.cancel_recording(robot_name=robot_name)

                if gap_s > 0:
                    logger.info("Waiting %ds between cancel and next recording", gap_s)
                    time.sleep(gap_s)

                # --- valid recording ---
                with Timer(
                    MAX_TIME_TO_START_S, label="nc.start_recording", always_log=True
                ):
                    nc.start_recording(robot_name=robot_name)
                resumed_recording_id = robot.get_current_recording_id()
                assert resumed_recording_id is not None

                log_frames(spec, recording_index=0, marker_name="marker_resume")

                with Timer(
                    case.duration_sec * STOP_RECORDING_OVERHEAD_PER_SEC,
                    label="nc.stop_recording",
                    always_log=True,
                    assert_limit=False,
                ):
                    nc.stop_recording(robot_name=robot_name, wait=True)

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
                        wall_started_at=None,
                        wall_stopped_at=0.0,
                        timestamp_mode=case.timestamp_mode,
                    )
                ]
                verify_cloud_results(results=results, case=case)
        finally:
            set_case_analysis_report(
                request=request,
                case=case,
                results=results,
            )
