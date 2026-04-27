from __future__ import annotations

import pytest

from tests.integration.platform.data_daemon.daemon_test_cases import (
    PRE_NETWORK_PERFORMANCE_CASES,
)
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
)
from tests.integration.platform.data_daemon.shared.runners import offline_daemon_running
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestBatch,
    DataDaemonTestCase,
    case_ids,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case_context import (  # noqa: E501
    create_testing_dataset_name,
    run_case_contexts,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    STOP_METHOD_CLI,
    STORAGE_STATE_PRESERVE,
)
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    scoped_storage_state,
)

_CASES = DataDaemonTestBatch(
    cases=PRE_NETWORK_PERFORMANCE_CASES,
    storage_state_action=STORAGE_STATE_PRESERVE,
    stop_method=STOP_METHOD_CLI,
).as_cases()


@pytest.mark.parametrize("case", _CASES, ids=case_ids(_CASES))
def test_disk_db_write_performance(
    case: DataDaemonTestCase,
    clear_daemon_timer_stats,
    log_run_analysis_on_teardown,
) -> None:
    """Record a high-volume offline workload and verify trace write timing.

    Focused on performance — does not upload data or perform cloud verification.

    - records all context specs via the offline daemon profile at high volume
    - asserts all traces are written to disk within the case timing budget
    - asserts per-context frame counts and recording structure are correct
    """

    dataset_name = create_testing_dataset_name(case)
    with scoped_storage_state(case, dataset_name=dataset_name):
        with offline_daemon_running():
            results = []
            try:
                assert_exactly_one_daemon_pid()
                results = run_case_contexts(case, wait_for_traces=True)
            finally:
                log_run_analysis_on_teardown(case, results)
