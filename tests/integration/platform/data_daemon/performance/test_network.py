from __future__ import annotations

import pytest

import neuracore as nc
from tests.integration.platform.data_daemon.daemon_test_cases import (
    NETWORK_PERFORMANCE_CASES,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    wait_for_dataset_ready,
)
from tests.integration.platform.data_daemon.shared.runners import online_daemon_running
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
    STOP_METHOD_CLI,
    STORAGE_STATE_DELETE,
)
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    scoped_storage_state,
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
    log_run_analysis_on_teardown,
) -> None:
    """Record a high-volume online workload and verify cloud upload timing.

    Focused on performance — does not perform per-frame data verification.

    - records all context specs against the live platform at high volume
    - asserts the stop-recording mode (wait vs nowait) is correctly reflected
    - waits for the dataset to become ready on the platform within the case
      timing budget (``case_timeout_seconds``)
    - asserts the expected number of recordings are present in the dataset
    """
    nc.login()
    if not has_configured_org():
        pytest.skip(
            "Online performance tests require NEURACORE_ORG_ID"
            " or a saved current organization."
        )

    dataset_name = create_testing_dataset_name(case)
    specs = build_context_specs(case, dataset_name=dataset_name)
    results: list[ContextResult] = []

    print(f"TEST: start case={case_ids([case])[0]} dataset={dataset_name}", flush=True)

    with scoped_storage_state(case, dataset_name=dataset_name):
        print("TEST: entered scoped_storage_state", flush=True)
        try:
            print("TEST: before online_daemon_running", flush=True)
            with online_daemon_running():
                print("TEST: entered online_daemon_running", flush=True)

                print("TEST: before run_case_contexts", flush=True)
                results = run_case_contexts(case, specs=specs)
                wait_for_dataset_ready(
                    results[0].dataset_name,
                    expected_recording_count=case.recording_count,
                    timeout_s=case_timeout_seconds(case),
                )
                print("TEST: after wait_for_dataset_ready", flush=True)
        finally:
            print("TEST: before log_run_analysis_on_teardown", flush=True)
            log_run_analysis_on_teardown(case, results)
            print("TEST: after log_run_analysis_on_teardown", flush=True)

    print("TEST: after scoped_storage_state", flush=True)
