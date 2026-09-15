"""Synchronisation memory sweep against the platform.

Records a workload, waits for the dataset to finish synchronising, and emits a
``SYNC_BENCH_CASE`` line describing the case and the wall-clock window it
occupied. The memory itself is measured server-side by
``neuracore_shared.synchronization._memory_diag`` and read back out of Cloud
Logging afterwards::

    gcloud logging read \\
      'resource.type="cloud_run_revision"
       AND resource.labels.service_name=~"^staging-neuracore-backend"
       AND textPayload:"SYNC_MEM_DIAG"' \\
      --project=neuracore-staging --freshness=1d --format="value(textPayload)" \\
      > diag.txt

    python scripts/analyse_sync_diag.py diag.txt --cases bench-cases.jsonl

The ``SYNC_BENCH_CASE`` lines are what let the analyser attribute a server-side
sample to the case and independent variable that produced it; 
without them the samples are
just an undifferentiated cloud of points.

This suite deliberately does not assert on memory. It produces the workload and
the correlation data; the judgement lives in the analyser, where the whole
sweep is visible at once rather than one case at a time.
"""

from __future__ import annotations

import json
import logging
import time

import pytest

from tests.integration.platform.data_daemon.daemon_test_cases import SYNC_MEMORY_CASES
from tests.integration.platform.data_daemon.shared.db_helpers import (
    wait_for_dataset_ready,
)
from tests.integration.platform.data_daemon.shared.process_control import Timer
from tests.integration.platform.data_daemon.shared.runners import online_daemon_running
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestBatch,
    DataDaemonTestCase,
    case_ids,
    case_timeout_seconds,
    has_configured_org,
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
    scoped_storage_state,
)

logger = logging.getLogger(__name__)

BENCH_MARKER = "SYNC_BENCH_CASE"

# The control case every group varies from, mirrored from
# ``_SYNC_MEMORY_CONTROL`` in daemon_test_cases.
_CONTROL_DURATION_SEC = 40
_CONTROL_JOINT_COUNT = 7
_CONTROL_VIDEO_COUNT = 0
_CONTROL_RECORDING_COUNT = 1

CASES = DataDaemonTestBatch(
    cases=SYNC_MEMORY_CASES,
    storage_state_action=STORAGE_STATE_DELETE,
    stop_method=STOP_METHOD_CLI,
).as_cases()


def _independent_variable(case: DataDaemonTestCase) -> str:
    """Name the one variable this case moves off the control case.

    Args:
        case: The case being run.

    Returns:
        ``frames_per_trace``, ``joints_per_frame``, ``traces_per_recording``,
        ``recordings_per_dataset``, ``control`` when nothing moved, or
        ``confounded`` when more than one moved — in which case the sample
        cannot be attributed to a single cause and the analyser drops it.
    """
    moved = []
    if case.duration_sec != _CONTROL_DURATION_SEC:
        moved.append("frames_per_trace")
    if case.joint_count != _CONTROL_JOINT_COUNT:
        moved.append("joints_per_frame")
    if case.video_count != _CONTROL_VIDEO_COUNT:
        moved.append("traces_per_recording")
    if case.recording_count != _CONTROL_RECORDING_COUNT:
        moved.append("recordings_per_dataset")
    if not moved:
        return "control"
    return moved[0] if len(moved) == 1 else "confounded"


def _emit_case_record(
    case: DataDaemonTestCase,
    dataset_name: str,
    started_at: float,
    finished_at: float,
) -> None:
    """Log the case parameters and the window its synchronisations fell in."""
    record = {
        "marker": BENCH_MARKER,
        "independent_variable": _independent_variable(case),
        "dataset_name": dataset_name,
        "started_at": round(started_at, 3),
        "finished_at": round(finished_at, 3),
        "duration_sec": case.duration_sec,
        "joint_count": case.joint_count,
        "joint_fps": case.joint_fps,
        "video_count": case.video_count,
        "video_fps": case.video_fps,
        "depth_count": case.depth_count,
        "recording_count": case.recording_count,
        "frames_per_trace": case.joint_fps * case.duration_sec,
        "joints_per_frame": case.joint_count,
        "traces_per_recording": case.video_count + case.depth_count + 1,
        "recordings_per_dataset": case.recording_count,
    }
    logger.info("%s %s", BENCH_MARKER, json.dumps(record, sort_keys=True))


@pytest.mark.parametrize("case", CASES, ids=case_ids(CASES))
def test_sync_memory_sweep(
    case: DataDaemonTestCase,
    clear_daemon_timer_stats,
    performance_report,
) -> None:
    """Record one sweep point and wait for the platform to synchronise it.

    Asserts only that the workload completed and that joints remain the densest
    stream. Memory is read from the server-side diagnostics afterwards.
    """
    if not has_configured_org():
        pytest.skip(
            "Sync memory sweep requires NEURACORE_ORG_ID"
            " or a saved current organization."
        )

    # The densest-stream branch of `_synchronize_nc_data` picks whichever trace
    # has the most entries as the reference timeline. If a camera ever overtook
    # the joints, the traces_per_recording group would be moving the row count too.
    if case.video_count:
        assert case.joint_fps > case.video_fps, (
            f"joint_fps={case.joint_fps} must exceed video_fps={case.video_fps} "
            "or the camera becomes the reference timeline, which would change "
            "frames_per_trace while we are varying traces_per_recording"
        )

    dataset_name = create_testing_dataset_name(case)
    specs = build_context_specs(case, dataset_name=dataset_name, assert_deadline=True)

    with performance_report(case, dataset_name=dataset_name) as report:
        results: list[ContextResult] = []
        started_at = time.time()
        with scoped_storage_state(case, specs):
            with online_daemon_running():
                with report.step("Record workload and stop recordings"):
                    with Timer(
                        case_timeout_seconds(case),
                        label="sync_memory.recording_contexts",
                        always_log=True,
                    ):
                        results = report.capture_results(
                            run_case_contexts(case, specs=specs)
                        )
                with report.step("Wait for cloud dataset readiness"):
                    with Timer(
                        case_timeout_seconds(case),
                        label="sync_memory.dataset_ready_wait",
                        always_log=True,
                    ):
                        wait_for_dataset_ready(
                            results[0].dataset_name,
                            expected_recording_count=case.recording_count,
                            timeout_s=case_timeout_seconds(case),
                        )
        finished_at = time.time()

    _emit_case_record(case, results[0].dataset_name, started_at, finished_at)
