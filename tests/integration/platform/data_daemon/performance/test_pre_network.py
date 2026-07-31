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
    scoped_storage_state,
)

# Performance workloads target the Rust daemon's throughput ceiling; the legacy
# Python daemon cannot sustain them, so the whole suite is Rust-only.
#
# Flat frame content: this suite's budgets are calibrated against solid-fill
# frames. STOP_RECORDING_UPLOAD_SLA_PER_VIDEO_PIXEL_S in particular is a
# per-*pixel* cost, which realistic content raises by two orders of magnitude, and
# the long 1080p cases would write hundreds of GB. Encoder realism is the
# integrity suites' job; throughput scaling is this one's.
_CASES = DataDaemonTestBatch(
    cases=PRE_NETWORK_PERFORMANCE_CASES,
    storage_state_action=STORAGE_STATE_DELETE,
    stop_method=STOP_METHOD_CLI,
    requires_rust_daemon=True,
    video_detail=DETAIL_FLAT,
).as_cases()


@pytest.mark.parametrize("case", _CASES, ids=case_ids(_CASES))
def test_disk_db_write_performance(
    case: DataDaemonTestCase,
    clear_daemon_timer_stats,
    performance_report,
) -> None:
    """Record a high-volume offline workload and verify trace write timing.

    Focused on performance — does not upload data or perform cloud verification.

    - records all context specs via the offline daemon profile at high volume
    - asserts all traces are written to disk within the case timing budget
    - asserts per-context frame counts and recording structure are correct
    """

    dataset_name = create_testing_dataset_name(case)
    specs = build_context_specs(case, dataset_name=dataset_name, assert_deadline=True)
    with performance_report(case, dataset_name=dataset_name) as report:
        with scoped_storage_state(case, dataset_name=dataset_name):
            with offline_daemon_running():
                assert_exactly_one_daemon_pid()
                report.capture_results(
                    run_case_contexts(case, specs=specs, wait_for_traces=True)
                )
