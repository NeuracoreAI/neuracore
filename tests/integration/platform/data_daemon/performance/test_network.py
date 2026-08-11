from __future__ import annotations

import time
import uuid

import pytest

import neuracore as nc
from tests.integration.platform.data_daemon.daemon_test_cases import (
    NETWORK_PERFORMANCE_CASES,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    wait_for_dataset_ready,
)
from tests.integration.platform.data_daemon.shared.process_control import Timer
from tests.integration.platform.data_daemon.shared.runners import (
    online_daemon_running,
    split_video_process_running,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestBatch,
    DataDaemonTestCase,
    case_ids,
    case_timeout_seconds,
    has_configured_org,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case_context import (  # noqa: E501
    ContextResult,
    build_context_specs,
    create_testing_dataset_name,
    run_case_contexts,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    DETAIL_FLAT,
    STOP_METHOD_CLI,
    STORAGE_STATE_DELETE,
)
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    cloud_resource_deleter,
    cloud_resource_names,
    scoped_test_dir_state,
)

# Cloud performance covers both nowait and wait=True because upload/registration
# progress is asynchronous and both stop-recording modes must remain valid.
CASES = DataDaemonTestBatch(
    cases=NETWORK_PERFORMANCE_CASES,
    storage_state_action=STORAGE_STATE_DELETE,
    stop_method=STOP_METHOD_CLI,
).as_cases()


@pytest.mark.parametrize("case", CASES, ids=case_ids(CASES))
def test_cloud_upload_and_readiness_performance(
    case: DataDaemonTestCase,
    clear_daemon_timer_stats,
    performance_report,
) -> None:
    """Record a high-volume online workload and verify cloud upload timing.

    Focused on performance — does not perform per-frame data verification.

    - records all context specs against the live platform at high volume
    - asserts the stop-recording mode (wait vs nowait) is correctly reflected
    - waits for the dataset to become ready on the platform within the case
      timing budget (``case_timeout_seconds``)
    - asserts the expected number of recordings are present in the dataset
    """
    if not has_configured_org():
        pytest.skip(
            "Online performance tests require NEURACORE_ORG_ID"
            " or a saved current organization."
        )
    dataset_name = create_testing_dataset_name(case)
    specs = build_context_specs(case, dataset_name=dataset_name, assert_deadline=True)
    cloud_names = cloud_resource_names(specs)
    with performance_report(case, dataset_name=dataset_name) as report:
        results: list[ContextResult] = []
        with (
            cloud_resource_deleter(*cloud_names),
            scoped_test_dir_state(case),
        ):
            with online_daemon_running():
                with report.step("Record workload and stop recordings"):
                    with Timer(
                        case_timeout_seconds(case),
                        label="performance.recording_contexts",
                        always_log=True,
                        assert_deadline=False,
                    ):
                        results = report.capture_results(
                            run_case_contexts(case, specs=specs)
                        )
                with report.step("Wait for cloud dataset readiness"):
                    with Timer(
                        case_timeout_seconds(case),
                        label="performance.dataset_ready_wait",
                        always_log=True,
                        assert_deadline=False,
                    ):
                        wait_for_dataset_ready(
                            results[0].dataset_name,
                            expected_recording_count=case.recording_count,
                            timeout_s=case_timeout_seconds(case),
                        )


def test_split_process_video_upload_and_readiness_performance(
    clear_daemon_timer_stats,
    performance_report,
) -> None:
    """Record a split-process video workload and verify cloud upload timing.

    Performance counterpart to data_integrity's
    ``test_split_process_video_survives_recording_boundary``: exercises the
    same topology — a video producer in a separate OS process from the
    recording owner, via :func:`~runners.split_video_process_running` — but
    measures cloud upload/readiness timing instead of per-frame data
    integrity. Focused on performance — does not perform per-frame data
    verification.

    - runs the video-only producer in its own OS process while this process
      owns ``start_recording``/``stop_recording`` for the same robot
    - waits for the dataset to become ready on the platform within the case
      timing budget (``case_timeout_seconds``)
    - asserts the expected number of recordings are present in the dataset
    """
    if not has_configured_org():
        pytest.skip(
            "Online performance tests require NEURACORE_ORG_ID"
            " or a saved current organization."
        )

    run_id = uuid.uuid4().hex[:10]
    dataset_name = f"split_video_perf_dataset_{run_id}"
    robot_name = f"split_video_perf_robot_{run_id}"
    case = DataDaemonTestCase(
        duration_sec=10,
        joint_count=0,
        video_count=1,
        image_width=640,
        image_height=480,
        video_fps=30,
        # Flat frames: this measures upload/readiness timing, and the
        # realistic frame bank's build cost would only move the numbers.
        video_detail=DETAIL_FLAT,
        storage_state_action=STORAGE_STATE_DELETE,
    )

    robot = None
    with performance_report(case, dataset_name=dataset_name) as report:
        with (
            cloud_resource_deleter(dataset_name, [robot_name]),
            scoped_test_dir_state(case),
        ):
            try:
                with online_daemon_running():
                    nc.create_dataset(dataset_name)
                    with report.step(
                        "Record split-process workload and stop recording"
                    ):
                        with Timer(
                            case_timeout_seconds(case),
                            label="performance.split_process_recording",
                            always_log=True,
                            assert_deadline=False,
                        ):
                            with split_video_process_running(
                                robot_name=robot_name,
                                dataset_name=dataset_name,
                                camera_name="camera_0",
                                case=case,
                            ):
                                robot = nc.connect_robot(robot_name, overwrite=False)
                                nc.start_recording(robot_name=robot_name, timestamp=0.0)
                                time.sleep(case.duration_sec)
                                nc.stop_recording(
                                    robot_name=robot_name,
                                    wait=True,
                                    timestamp=float(case.duration_sec),
                                )
                    with report.step("Wait for cloud dataset readiness"):
                        with Timer(
                            case_timeout_seconds(case),
                            label="performance.split_process_dataset_ready_wait",
                            always_log=True,
                            assert_deadline=False,
                        ):
                            wait_for_dataset_ready(
                                dataset_name,
                                expected_recording_count=case.recording_count,
                                timeout_s=case_timeout_seconds(case),
                            )
            finally:
                if robot is not None:
                    robot.close()
