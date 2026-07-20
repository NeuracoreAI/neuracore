"""Behavioural tests for shared-slot reopen recovery through real daemon flows."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import pytest

import neuracore as nc
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
    assert_post_test_storage_state,
    verify_cloud_results,
)
from tests.integration.platform.data_daemon.shared.process_control import Timer
from tests.integration.platform.data_daemon.shared.runners import online_daemon_running
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestCase,
    camera_names,
    has_configured_org,
    joint_names_for_count,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case_context import (  # noqa: E501
    ContextResult,
    build_context_specs,
    create_testing_dataset_name,
    encode_frame_number,
    log_frames,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    MAX_TIME_TO_START_S,
    STOP_RECORDING_OVERHEAD_PER_SEC,
)
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    scoped_storage_state,
    set_case_analysis_report,
)

logger = logging.getLogger(__name__)


_CASE = DataDaemonTestCase(
    duration_sec=5,
    joint_count=4,
    video_count=1,
    image_width=64,
    image_height=64,
)


def test_shared_slot_reopen_after_stalled_descriptor_uploads_next_recording(
    monkeypatch: pytest.MonkeyPatch,
    clear_daemon_timer_stats,
    request: pytest.FixtureRequest,
    test_wall_timer: Callable[[], float],
) -> None:
    """Force the shared-slot reopen race, then verify uploaded cloud data.

    The daemon subprocess delays the first shared-slot descriptor long enough for
    the same producer to cancel and immediately reopen shared slots. The reopen
    path must abandon the old pending descriptor, avoid getting stuck on the old
    sequence number, and still upload the following valid recording through the
    normal cloud verification path.
    """
    if not has_configured_org():
        pytest.skip(
            "Shared-slot reopen behavioural test requires NEURACORE_ORG_ID"
            " or a saved current organization."
        )

    monkeypatch.setenv("NCD_SHARED_SLOT_REOPEN_DRAIN_TIMEOUT_S", "0.05")
    monkeypatch.setenv("NCD_TEST_SHARED_SLOT_DESCRIPTOR_DELAY_ONCE_S", "2.0")

    case = _CASE
    dataset_name = create_testing_dataset_name(case)
    specs = build_context_specs(case, dataset_name=dataset_name)
    spec = specs[0]
    robot_name = spec.robot_name
    camera_name = camera_names(spec.case.video_count)[0]
    results: list[ContextResult] = []

    try:
        with scoped_storage_state(case, dataset_name=dataset_name):
            with online_daemon_running():
                assert_exactly_one_daemon_pid()

                with Timer(
                    MAX_TIME_TO_START_S, label="nc.create_dataset", always_log=True
                ):
                    nc.create_dataset(
                        dataset_name,
                        description="Shared-slot reopen recovery upload test",
                    )
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

                nc.log_rgb(
                    camera_name,
                    encode_frame_number(
                        0,
                        spec.case.image_width,
                        spec.case.image_height,
                    ),
                    robot_name=robot_name,
                    timestamp=0.0,
                )
                time.sleep(0.2)

                with Timer(
                    case.duration_sec * STOP_RECORDING_OVERHEAD_PER_SEC,
                    label="nc.cancel_recording",
                    always_log=True,
                    assert_deadline=False,
                ):
                    nc.cancel_recording(robot_name=robot_name)

                logger.info(
                    "Starting replacement recording immediately after forced "
                    "shared-slot descriptor stall"
                )
                wall_started_at = time.time()
                with Timer(
                    MAX_TIME_TO_START_S, label="nc.start_recording", always_log=True
                ):
                    nc.start_recording(robot_name=robot_name)
                resumed_recording_id = robot.get_cloud_recording_id()
                assert resumed_recording_id is not None

                log_frames(spec, recording_index=0, marker_name="marker_reopen")

                with Timer(
                    case.duration_sec * STOP_RECORDING_OVERHEAD_PER_SEC,
                    label="nc.stop_recording",
                    always_log=True,
                    assert_deadline=False,
                ):
                    nc.stop_recording(robot_name=robot_name, wait=True)
                wall_stopped_at = time.time()

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
                        duration_sec=case.duration_sec + 1,
                        timestamp_start_s=spec.timestamp_start_s,
                        timestamp_end_s=spec.timestamp_start_s + case.duration_sec,
                        marker_names=["marker_reopen"],
                        has_video=bool(spec.case.video_count),
                        context_index=0,
                        wall_started_at=wall_started_at,
                        wall_stopped_at=wall_stopped_at,
                        timestamp_mode=case.timestamp_mode,
                    )
                ]
                verify_cloud_results(results=results, case=case)
    finally:
        set_case_analysis_report(
            request=request,
            case=case,
            results=results,
            label_prefix="shared_slot_reopen",
            test_wall_s=test_wall_timer(),
        )

    assert_post_test_storage_state(case.storage_state_action)
