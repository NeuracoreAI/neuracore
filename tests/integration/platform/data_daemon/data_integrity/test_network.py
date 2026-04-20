from __future__ import annotations

import pytest

from tests.integration.platform.data_daemon.daemon_test_cases import (
    NETWORK_INTEGRITY_CASES,
)
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
    verify_cloud_results,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    wait_for_upload_complete_in_db,
)
from tests.integration.platform.data_daemon.shared.runners import online_daemon_running
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
    run_and_assert_case_contexts,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    STOP_METHOD_CLI,
    STORAGE_STATE_PRESERVE,
)
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    scoped_storage_state,
    set_case_analysis_report,
)

_CASES = DataDaemonTestBatch(
    cases=NETWORK_INTEGRITY_CASES,
    storage_state_action=STORAGE_STATE_PRESERVE,
    stop_method=STOP_METHOD_CLI,
).as_cases()


def _assert_online_verification_invariants(
    results: list[ContextResult],
    *,
    timeout_seconds: float = 30.0,
) -> None:
    """Block until every recording in *results* has reached ``upload_complete``
    in the platform DB.  Must be called before cloud frame verification so
    that downloaded data reflects the fully-committed upload state.
    """
    for result in results:
        for recording_id in result.recording_ids:
            wait_for_upload_complete_in_db(str(recording_id), timeout_s=timeout_seconds)


# ---------------------------------------------------------------------------
# Isolation and integrity parametrized test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _CASES, ids=case_ids(_CASES))
def test_cloud_data_integrity(
    case: DataDaemonTestCase,
    clear_daemon_timer_stats,
    request: pytest.FixtureRequest,
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
                results = run_and_assert_case_contexts(case, specs=specs)
                _assert_online_verification_invariants(results)
                assert_exactly_one_daemon_pid()
                verify_cloud_results(results=results, case=case)

        finally:
            set_case_analysis_report(
                request=request,
                case=case,
                results=results,
            )
