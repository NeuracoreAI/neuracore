import time
from collections.abc import Callable, Iterator

import pytest

import neuracore as nc
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
    verify_cloud_results,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    assert_offline_recordings_pending,
    assert_recordings_wait_on_unconfirmed_rows,
    fetch_robot_row,
    latching_upload_observer,
    resolve_cloud_recording_ids,
    wait_for_all_traces_written,
)
from tests.integration.platform.data_daemon.shared.network import (
    NetworkSwitch,
    switchable_network,
)
from tests.integration.platform.data_daemon.shared.reporting import report_step
from tests.integration.platform.data_daemon.shared.runners import (
    offline_daemon_running,
    online_daemon_running,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestBatch,
    DataDaemonTestCase,
    ProcessPerLimbPerCamera,
    Synchronous,
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
        Synchronous(
            duration_sec=5,
            joint_count=4,
        ),
        Synchronous(
            duration_sec=5,
            joint_count=4,
            video_count=1,
            image_width=64,
            image_height=64,
        ),
    ),
).as_cases()

# Every process logs in, connects the robot and selects the dataset with no
# network, then the network comes back with the same daemon running. The
# daemon's offline profile above only keeps the daemon off the server; this
# takes the network from the SDK too.
_FULLY_OFFLINE_CASES = DataDaemonTestBatch(
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


@pytest.mark.parametrize("case", _CASES, ids=case_ids(_CASES))
def test_offline_pending_data_recovers_when_online(
    case: DataDaemonTestCase,
    clear_daemon_timer_stats,
    request: pytest.FixtureRequest,
    test_wall_timer: Callable[[], float],
) -> None:
    """Verify offline recordings are correctly uploaded when the daemon goes online.

    - records all context specs via the offline daemon profile
    - waits for all traces to reach ``write_status == 'written'`` in SQLite
    - stops the offline daemon, preserving local artefacts for recovery
    - restarts the daemon in online mode
    - waits for every recording to reach ``upload_complete`` in the daemon DB
    - performs structural and per-episode frame verification via the cloud
    """
    if not has_configured_org():
        pytest.skip(
            "Offline-to-online behavioural tests require NEURACORE_ORG_ID"
            " or a saved current organization."
        )

    dataset_name = create_testing_dataset_name(case)
    specs = build_context_specs(case, dataset_name=dataset_name)
    results: list[ContextResult] = []

    try:
        with scoped_storage_state(case, specs):
            with offline_daemon_running():
                assert_exactly_one_daemon_pid()
                results = run_case_contexts(case, specs=specs)
                wait_for_all_traces_written(results=results)
                assert_offline_recordings_pending(results)
            # offline_daemon_running() stops the daemon on exit, preserving
            # offline artefacts for the online recovery phase below.

            with online_daemon_running():
                with latching_upload_observer() as observed:
                    recording_indexes = [
                        recording_index
                        for result in results
                        for recording_index in result.recording_indexes
                    ]
                    observed.wait_for_all_complete(recording_indexes, timeout_s=90.0)

                results = resolve_cloud_recording_ids(results, observed=observed)
                verify_cloud_results(results=results, case=case)

    finally:
        set_case_analysis_report(
            request=request,
            case=case,
            results=results,
            test_wall_s=test_wall_timer(),
        )


@pytest.fixture
def network() -> Iterator[NetworkSwitch]:
    with switchable_network() as switch:
        yield switch


@pytest.mark.no_network_at_start
@pytest.mark.parametrize(
    "case", _FULLY_OFFLINE_CASES, ids=case_ids(_FULLY_OFFLINE_CASES)
)
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
    - each recording points at the robot and dataset rows, not yet confirmed
    - every process connected the same robot row
    - the network comes back with the same daemon still running
    - the robot is registered and its saved package sent
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
                assert_recordings_wait_on_unconfirmed_rows(results)
                # Every process connected the one robot row.
                robot_rows = {result.source[0] for result in results}
                assert len(robot_rows) == 1, robot_rows

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
                _wait_for_robot_registered(int(robot_rows.pop()))
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


def _wait_for_robot_registered(robot_index: int, timeout_s: float = 60.0) -> None:
    """Wait for the daemon to register the robot connected offline.

    Its row is confirmed, and its saved package sent and dropped, once the
    server has it.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        robot = fetch_robot_row(robot_index)
        if (
            robot is not None
            and robot["status"] == "done"
            and not robot["package_path"]
        ):
            return
        assert (
            time.monotonic() < deadline
        ), f"robot row {robot_index} was never registered"
        time.sleep(0.5)
