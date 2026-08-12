from __future__ import annotations

from collections.abc import Callable

import pytest

from tests.integration.platform.data_daemon.daemon_test_cases import (
    PRE_NETWORK_INTEGRITY_CASES,
)
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    wait_for_all_traces_written,
)
from tests.integration.platform.data_daemon.shared.disk_helpers import (
    assert_disk_frame_codes,
    assert_disk_recording_properties,
    assert_video_artifacts,
)
from tests.integration.platform.data_daemon.shared.runners import offline_daemon_running
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestBatch,
    DataDaemonTestCase,
    case_id,
    case_ids,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    STOP_METHOD_CLI,
    STORAGE_STATE_DELETE,
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
    cloud_resource_deleter,
    cloud_resource_names,
    scoped_test_dir_state,
    set_case_analysis_report,
    setup_per_test_artifact_dirs,
)

CASES = DataDaemonTestBatch(
    cases=PRE_NETWORK_INTEGRITY_CASES,
    storage_state_action=STORAGE_STATE_DELETE,
    stop_method=STOP_METHOD_CLI,
).as_cases()

# ---------------------------------------------------------------------------
# Isolation and integrity parametrized test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=case_ids(CASES))
def test_disk_db_data_integrity(
    case: DataDaemonTestCase,
    clear_daemon_timer_stats,
    request: pytest.FixtureRequest,
    test_wall_timer: Callable[[], float],
) -> None:
    """Record data in offline mode and verify local disk and DB state.

    No data is uploaded to the platform.  Complements the network integrity
    test, which additionally verifies the cloud-side upload.

    - asserts no leftover daemon state before starting (isolation pre-condition)
    - records all context specs via the offline daemon profile
    - waits for all traces to reach ``write_status == 'written'`` in SQLite
    - validates on-disk trace timestamps fall within the expected recording
      window for every frame of every recording
    - validates every video trace holds the artefacts the case's codec implies,
      structurally sound and carrying the content ``video_detail`` declares
    - decodes the archive and checks every frame carries the code the producer
      painted, in order
    - asserts daemon and producer processes exit cleanly after stop
    - asserts no residual processes, files, sockets, or DB artefacts remain
      (isolation post-condition)
    """
    if case.preserve_artifacts_per_test:
        setup_per_test_artifact_dirs(case_id(case))

    results: list[ContextResult] = []
    dataset_name = create_testing_dataset_name(case)
    specs = build_context_specs(case, dataset_name=dataset_name)
    cloud_names = cloud_resource_names(specs)
    with (
        cloud_resource_deleter(*cloud_names),
        scoped_test_dir_state(case),
    ):
        try:
            with offline_daemon_running():
                assert_exactly_one_daemon_pid()
                results = run_case_contexts(case, specs=specs)
                wait_for_all_traces_written(results=results)
                assert_disk_recording_properties(results)
                if case.has_video:
                    assert_video_artifacts(results, case)
                    assert_disk_frame_codes(results, case)

        finally:
            set_case_analysis_report(
                request=request,
                case=case,
                results=results,
                test_wall_s=test_wall_timer(),
            )
