from __future__ import annotations

import pytest

from tests.integration.platform.data_daemon.daemon_test_cases import PERFORMANCE_CASES
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
)
from tests.integration.platform.data_daemon.shared.runners import daemon_running_for
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestBatch,
    DataDaemonTestCase,
    case_ids,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    STOP_METHOD_CLI,
    STORAGE_STATE_DELETE,
)
from tests.integration.platform.data_daemon.shared.test_case.context_spec import (
    build_context_specs,
)
from tests.integration.platform.data_daemon.shared.test_case.context_worker import (
    create_testing_dataset_name,
    run_case_contexts,
)
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    cloud_resource_deleter,
    cloud_resource_names,
    scoped_test_dir_state,
)

# Performance workloads target the daemon's throughput ceiling.
_CASES = DataDaemonTestBatch(
    cases=PERFORMANCE_CASES,
    storage_state_action=STORAGE_STATE_DELETE,
    stop_method=STOP_METHOD_CLI,
).as_cases()


@pytest.mark.parametrize("case", _CASES, ids=case_ids(_CASES))
def test_disk_db_write_performance(
    case: DataDaemonTestCase,
    clear_daemon_timer_stats,
    performance_report,
) -> None:
    """Record a high-volume workload and verify trace write timing.

    Reads nothing from the platform: the cloud performance test owns upload and
    readiness timing.

    - records all context specs at high volume, on the daemon the case needs
      (see :func:`daemon_running_for`)
    - asserts all traces are written to disk within the case timing budget
    - asserts per-context frame counts and recording structure are correct

    A case logging from a non-owning process has to record online, so its write
    timing carries upload contention: read it against itself over time, not
    against the offline cases.
    """
    if case.wait:
        pytest.skip(
            "wait=True blocks stop_recording until every trace has uploaded, and "
            "its budget is sized from the upload cost (see "
            "ContextCaseSpec.stop_recording_sla_s). Nothing here waits on an "
            "upload it does not measure — the cloud performance test owns this "
            "axis."
        )

    dataset_name = create_testing_dataset_name(case)
    specs = build_context_specs(case, dataset_name=dataset_name, assert_deadline=True)
    cloud_names = cloud_resource_names(specs)
    with performance_report(case, dataset_name=dataset_name) as report:
        with (
            cloud_resource_deleter(*cloud_names),
            scoped_test_dir_state(case),
        ):
            with daemon_running_for(case):
                assert_exactly_one_daemon_pid()
                report.capture_results(
                    run_case_contexts(case, specs=specs, wait_for_traces=True)
                )
