from __future__ import annotations

from collections.abc import Callable

import pytest

from tests.integration.platform.data_daemon.daemon_test_cases import (
    NETWORK_PERFORMANCE_CASES,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    resolve_cloud_recording_ids,
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

DATASET_READY_STALL_TIMEOUT_S = 10 * 60.0

# Cloud performance covers both nowait and wait=True because upload/registration
# progress is asynchronous and both stop-recording modes must remain valid.
CASES = DataDaemonTestBatch(
    cases=NETWORK_PERFORMANCE_CASES,
    storage_state_action=STORAGE_STATE_DELETE,
    stop_method=STOP_METHOD_CLI,
    requires_rust_daemon=True,
).as_cases()


@pytest.mark.parametrize("case", CASES, ids=case_ids(CASES))
def test_cloud_upload_and_readiness_performance(
    case: DataDaemonTestCase,
    clear_daemon_timer_stats,
    log_run_analysis_on_teardown,
    test_wall_timer: Callable[[], float],
) -> None:
    """Record a high-volume online workload and verify cloud upload timing.

    Focused on performance — does not perform per-frame data verification.

    - records all context specs against the live platform at high volume
    - asserts the stop-recording mode (wait vs nowait) is correctly reflected
    - resolves the expected cloud recording IDs
    - waits for every expected recording to become visible in the dataset
    - fails early if dataset readiness stops making progress
    """
    if not has_configured_org():
        pytest.skip(
            "Online performance tests require NEURACORE_ORG_ID"
            " or a saved current organization."
        )

    dataset_name = create_testing_dataset_name(case)
    specs = build_context_specs(case, dataset_name=dataset_name, assert_deadline=True)
    results: list[ContextResult] = []

    with scoped_storage_state(case, dataset_name=dataset_name):
        try:
            with online_daemon_running():
                results = run_case_contexts(case, specs=specs)
                results = resolve_cloud_recording_ids(results)

                expected_recording_ids = {
                    recording_id
                    for result in results
                    for recording_id in result.recording_ids
                    if recording_id
                }
                assert len(expected_recording_ids) == case.recording_count, (
                    "Expected "
                    f"{case.recording_count} unique cloud recording IDs, "
                    f"resolved {len(expected_recording_ids)}: "
                    f"{sorted(expected_recording_ids)}"
                )

                wait_for_dataset_ready(
                    results[0].dataset_name,
                    expected_recording_ids=expected_recording_ids,
                    timeout_s=case_timeout_seconds(case),
                    stall_timeout_s=DATASET_READY_STALL_TIMEOUT_S,
                )
        finally:
            log_run_analysis_on_teardown(
                case,
                results,
                test_wall_s=test_wall_timer(),
            )
