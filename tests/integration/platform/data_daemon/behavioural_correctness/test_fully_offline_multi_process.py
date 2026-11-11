import os
import time
from collections.abc import Callable, Iterator

import pytest

import neuracore as nc
from neuracore.data_daemon.offline_robots import offline_robot_dir
from tests.integration.platform.data_daemon.shared.assertions import (
    verify_cloud_results,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    assert_offline_recordings_hold_local_ids,
    assert_offline_recordings_pending,
    latching_upload_observer,
    resolve_cloud_recording_ids,
    wait_for_all_traces_written,
)
from tests.integration.platform.data_daemon.shared.network import (
    NetworkSwitch,
    switchable_network,
)
from tests.integration.platform.data_daemon.shared.reporting import report_step
from tests.integration.platform.data_daemon.shared.runners import online_daemon_running
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestBatch,
    DataDaemonTestCase,
    ProcessPerLimbPerCamera,
    case_ids,
    has_configured_org,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    DETAIL_FLAT,
    PACING_BURST_VIDEO,
)
from tests.integration.platform.data_daemon.shared.test_case.context_spec import (
    ContextResult,
    build_context_specs,
)
from tests.integration.platform.data_daemon.shared.test_case.context_worker import (
    create_testing_dataset_name,
    run_case_contexts,
)
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    scoped_storage_state,
    set_case_analysis_report,
)

_CASES = DataDaemonTestBatch(
    cases=(
        ProcessPerLimbPerCamera(
            duration_sec=5,
            recording_count=2,
            joint_count=8,
            video_count=1,
            image_width=64,
            image_height=64,
            video_fps=30,
            producer_pacing=PACING_BURST_VIDEO,
            video_detail=DETAIL_FLAT,
        ),
    ),
).as_cases()


@pytest.fixture(autouse=True)
def login_parent_process() -> Iterator[None]:
    """Replace the suite's online login: this host has no network at the start."""
    yield
    if os.environ.get("NCD_PRESERVE_TEST_LOGIN") != "1":
        nc.logout()


@pytest.fixture
def network() -> Iterator[NetworkSwitch]:
    with switchable_network() as switch:
        yield switch


@pytest.mark.parametrize("case", _CASES, ids=case_ids(_CASES))
def test_multi_process_recording_made_offline_uploads_once_online(
    case: DataDaemonTestCase,
    network: NetworkSwitch,
    clear_daemon_timer_stats,
    request: pytest.FixtureRequest,
    test_wall_timer: Callable[[], float],
) -> None:
    """Record from several processes with no network, then reconnect.

    - every process logs in, connects the robot and selects the dataset
      with the network down
    - the owner opens and closes the recordings; limb and camera children
      log into them
    - every trace is written locally and no recording has a cloud id
    - each recording keeps the local robot and dataset ids and their names
    - every process worked out the same local robot id
    - the network comes back with the same daemon still running
    - the robot is registered and its saved files removed
    - every recording uploads; the test process logs in again and every
      recording verifies against the cloud
    """
    if not has_configured_org():
        pytest.skip("Needs NEURACORE_ORG_ID or a saved current organization.")

    dataset_name = create_testing_dataset_name(case)
    specs = build_context_specs(case, dataset_name=dataset_name)
    results: list[ContextResult] = []

    try:
        with scoped_storage_state(case, specs), online_daemon_running():
            with report_step("Record with no network"):
                results = run_case_contexts(case, specs=specs)
                wait_for_all_traces_written(results=results)
                assert_offline_recordings_pending(results)
                assert_offline_recordings_hold_local_ids(results)
                robot_ids = {result.source[0] for result in results}
                assert len(robot_ids) == 1, robot_ids

            with report_step("Reconnect and upload"):
                network.go_online()
                with latching_upload_observer() as observed:
                    recording_indexes = [
                        recording_index
                        for result in results
                        for recording_index in result.recording_indexes
                    ]
                    observed.wait_for_all_complete(recording_indexes, timeout_s=180.0)
                results = resolve_cloud_recording_ids(results, observed=observed)
                _wait_for_robot_sent(robot_ids.pop())
                # Back online, the script logs in again, as the offline login
                # told it to; until then its session is still offline.
                nc.login()
                verify_cloud_results(results=results, case=case)
    finally:
        set_case_analysis_report(
            request=request,
            case=case,
            results=results,
            test_wall_s=test_wall_timer(),
        )


def _wait_for_robot_sent(robot_id: str, timeout_s: float = 60.0) -> None:
    """Wait for the daemon to register the robot saved offline.

    The daemon deletes the robot's saved files once the backend has it.
    """
    deadline = time.monotonic() + timeout_s
    while offline_robot_dir(robot_id).exists():
        assert time.monotonic() < deadline, f"robot {robot_id} was never sent"
        time.sleep(0.5)
