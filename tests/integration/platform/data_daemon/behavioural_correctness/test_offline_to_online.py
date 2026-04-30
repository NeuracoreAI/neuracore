import time

import pytest

from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
    verify_cloud_results,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    wait_for_all_traces_written,
    wait_for_upload_complete_in_db,
)
from tests.integration.platform.data_daemon.shared.runners import (
    offline_daemon_running,
    online_daemon_running,
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
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    scoped_storage_state,
    set_case_analysis_report,
)

_CASES = DataDaemonTestBatch(
    cases=(
        DataDaemonTestCase(
            duration_sec=5,
            joint_count=4,
        ),
        DataDaemonTestCase(
            duration_sec=5,
            joint_count=4,
            video_count=1,
            image_width=64,
            image_height=64,
        ),
    ),
).as_cases()


@pytest.mark.parametrize("case", _CASES, ids=case_ids(_CASES))
def test_offline_pending_data_recovers_when_online(
    case: DataDaemonTestCase,
    clear_daemon_timer_stats,
    request: pytest.FixtureRequest,
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

    start_time = time.perf_counter()
    dataset_name = create_testing_dataset_name(case)
    specs = build_context_specs(case, dataset_name=dataset_name)
    results: list[ContextResult] = []

    try:
        with scoped_storage_state(case, dataset_name=dataset_name):
            with offline_daemon_running():
                assert_exactly_one_daemon_pid()
                results = run_case_contexts(case, specs=specs)
                wait_for_all_traces_written(results=results)
            # offline_daemon_running() stops the daemon on exit, preserving
            # offline artefacts for the online recovery phase below.

            with online_daemon_running():
                for result in results:
                    for recording_id in result.recording_ids:
                        wait_for_upload_complete_in_db(str(recording_id))

                verify_cloud_results(results=results, case=case)

    finally:
        set_case_analysis_report(
            request=request,
            case=case,
            results=results,
            test_wall_s=time.perf_counter() - start_time,
        )
