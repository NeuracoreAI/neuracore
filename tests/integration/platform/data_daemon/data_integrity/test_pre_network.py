from __future__ import annotations

import time

import pytest

from tests.integration.platform.data_daemon.daemon_test_cases import (
    PRE_NETWORK_INTEGRITY_CASES,
)
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
    assert_no_daemon_pids,
    assert_no_producer_processes,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    wait_for_all_traces_written,
)
from tests.integration.platform.data_daemon.shared.disk_helpers import (
    assert_disk_recording_properties,
)
from tests.integration.platform.data_daemon.shared.runners import offline_daemon_running
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestBatch,
    DataDaemonTestCase,
    case_id,
    case_ids,
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
    setup_per_test_artifact_dirs,
)

CASES = DataDaemonTestBatch(
    cases=PRE_NETWORK_INTEGRITY_CASES,
    storage_state_action=STORAGE_STATE_PRESERVE,
    stop_method=STOP_METHOD_CLI,
    # timestamp_mode=TIMESTAMP_MODE_REAL,
).as_cases()

# ---------------------------------------------------------------------------
# Isolation and integrity parametrized test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=case_ids(CASES))
def test_disk_db_data_integrity(
    case: DataDaemonTestCase,
    daemon_offline_env,
    clear_daemon_timer_stats,
    request: pytest.FixtureRequest,
) -> None:
    """Record data in offline mode and verify local disk and DB state.

    No data is uploaded to the platform.  Complements the network integrity
    test, which additionally verifies the cloud-side upload.

    - asserts no leftover daemon state before starting (isolation pre-condition)
    - records all context specs via the offline daemon profile
    - waits for all traces to reach ``write_status == 'written'`` in SQLite
    - validates on-disk trace timestamps fall within the expected recording
      window for every frame of every recording
    - asserts daemon and producer processes exit cleanly after stop
    - asserts no residual processes, files, sockets, or DB artefacts remain
      (isolation post-condition)
    """
    if case.preserve_artifacts_per_test:
        setup_per_test_artifact_dirs(case_id(case))

    results: list[ContextResult] = []
    daemon_shutdown_s: float | None = None
    cleanup_started_at: float = time.perf_counter()
    dataset_name = create_testing_dataset_name(case)

    with scoped_storage_state(case, dataset_name=dataset_name):
        try:
            with offline_daemon_running():
                assert_exactly_one_daemon_pid()
                specs = build_context_specs(case, dataset_name=dataset_name)
                results = run_and_assert_case_contexts(case, specs=specs)
                wait_for_all_traces_written(results=results)
                assert_disk_recording_properties(results)
                # Mark the start of shutdown — stop_daemon() runs in the
                # offline_daemon_running() finally block as the with exits.
                profile_shutdown_started_at = time.perf_counter()
            daemon_shutdown_s = time.perf_counter() - profile_shutdown_started_at

            assert_no_daemon_pids()
            assert_no_producer_processes()
            cleanup_started_at = time.perf_counter()

        finally:
            set_case_analysis_report(
                request=request,
                case=case,
                results=results,
                daemon_shutdown_s=daemon_shutdown_s,
                final_cleanup_s=time.perf_counter() - cleanup_started_at,
            )
