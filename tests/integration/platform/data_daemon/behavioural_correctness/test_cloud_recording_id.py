"""Behavioural correctness tests for resolving a recording's cloud id.

The cloud id is minted asynchronously and read from the daemon's live state, so
it is resolvable only while the recording is open. These pin both halves of that:
what the narrowed window costs, and the one caller that must not pay it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import pytest

import neuracore as nc
from neuracore.api.core import _await_cloud_recording_id
from neuracore.core.utils import backend_utils
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
)
from tests.integration.platform.data_daemon.shared.process_control import Timer
from tests.integration.platform.data_daemon.shared.runners import online_daemon_running
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestBatch,
    DataDaemonTestCase,
    Synchronous,
    case_ids,
    has_configured_org,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    MAX_TIME_TO_START_S,
)
from tests.integration.platform.data_daemon.shared.test_case.context_spec import (
    ContextResult,
    build_context_specs,
)
from tests.integration.platform.data_daemon.shared.test_case.context_worker import (
    create_testing_dataset_name,
    log_frames,
)
from tests.integration.platform.data_daemon.shared.test_case.recording_control import (
    await_gate,
)
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    scoped_storage_state,
    set_case_analysis_report,
)

logger = logging.getLogger(__name__)

_UPLOAD_WAIT_TIMEOUT_S = 180.0

_CASES = DataDaemonTestBatch(
    cases=(
        Synchronous(
            duration_sec=2,
            joint_count=4,
        ),
    ),
).as_cases()


def _await_window_open(robot: object) -> None:
    """Wait for the daemon to report the window open.

    A local start returns before the daemon has applied it, so the window is
    not open the instant the call comes back.
    """
    await_gate(
        robot,
        open_gate=True,
        deadline=time.time() + MAX_TIME_TO_START_S,
        label="cloud_id.open_gate_wait",
        overdue="the recording never opened",
        assert_deadline=False,
    )


@pytest.mark.parametrize("case", _CASES, ids=case_ids(_CASES))
def test_a_stopped_recordings_cloud_id_is_unresolvable(
    case: DataDaemonTestCase,
    clear_daemon_timer_stats,
    request: pytest.FixtureRequest,
    test_wall_timer: Callable[[], float],
) -> None:
    """A cloud id is readable while the recording is open, and not after it stops.

    The daemon answers from the state it holds for the source and drops that
    entry on stop, so there is no id to read afterwards. This is deliberate, not
    an oversight: anything needing the id must ask for it before stopping, which
    is what ``nc.stop_recording(wait=True)`` does. The test exists so the
    narrowed window is a decision on the record rather than a surprise.
    """
    if not has_configured_org():
        pytest.skip(
            "Cloud-recording-id behavioural tests require NEURACORE_ORG_ID"
            " or a saved current organization."
        )

    dataset_name = create_testing_dataset_name(case)
    specs = build_context_specs(case, dataset_name=dataset_name)
    spec = specs[0]
    results: list[ContextResult] = []

    try:
        with scoped_storage_state(case, specs):
            with online_daemon_running():
                assert_exactly_one_daemon_pid()

                with Timer(
                    MAX_TIME_TO_START_S, label="nc.create_dataset", always_log=True
                ):
                    nc.create_dataset(
                        dataset_name,
                        description="Cloud recording id resolution test",
                    )
                with Timer(
                    MAX_TIME_TO_START_S, label="nc.connect_robot", always_log=True
                ):
                    robot = nc.connect_robot(spec.robot_name, overwrite=False)

                nc.start_recording(robot_name=spec.robot_name)
                _await_window_open(robot)

                open_recording_id = _await_cloud_recording_id(robot)
                assert open_recording_id is not None, (
                    "the daemon never minted a cloud id for an open recording, "
                    "so nothing downstream could wait on its upload"
                )

                log_frames(
                    spec, robot=robot, recording_index=0, marker_name="marker_cloud_id"
                )

                nc.stop_recording(robot_name=spec.robot_name)

                assert robot.get_current_recording_id() is None, (
                    "a stopped recording's cloud id must read as absent — the "
                    "daemon no longer holds the source's entry"
                )
    finally:
        set_case_analysis_report(
            request=request,
            case=case,
            results=results,
            test_wall_s=test_wall_timer(),
        )


@pytest.mark.parametrize("case", _CASES, ids=case_ids(_CASES))
def test_a_short_recording_still_waits_for_upload(
    case: DataDaemonTestCase,
    clear_daemon_timer_stats,
    request: pytest.FixtureRequest,
    test_wall_timer: Callable[[], float],
) -> None:
    """``wait=True`` must not quietly degrade into ``wait=False``.

    ``nc.stop_recording(wait=True)`` resolves the cloud id before publishing the
    stop, and returns early when it has none. A recording stopped moments after
    it started is where that early return would bite, so this drives exactly that
    shape and asserts the upload really did finish by the time the call returned.
    """
    if not has_configured_org():
        pytest.skip(
            "Cloud-recording-id behavioural tests require NEURACORE_ORG_ID"
            " or a saved current organization."
        )

    dataset_name = create_testing_dataset_name(case)
    specs = build_context_specs(case, dataset_name=dataset_name)
    spec = specs[0]
    results: list[ContextResult] = []

    try:
        with scoped_storage_state(case, specs):
            with online_daemon_running():
                assert_exactly_one_daemon_pid()

                with Timer(
                    MAX_TIME_TO_START_S, label="nc.create_dataset", always_log=True
                ):
                    nc.create_dataset(
                        dataset_name,
                        description="Short-recording upload wait test",
                    )
                with Timer(
                    MAX_TIME_TO_START_S, label="nc.connect_robot", always_log=True
                ):
                    robot = nc.connect_robot(spec.robot_name, overwrite=False)

                nc.start_recording(robot_name=spec.robot_name)
                _await_window_open(robot)

                # Captured here because the stop is what makes it unresolvable;
                # the assertion below needs a handle that outlives the window.
                recording_id = _await_cloud_recording_id(robot)
                assert recording_id is not None

                log_frames(
                    spec, robot=robot, recording_index=0, marker_name="marker_short"
                )

                nc.stop_recording(
                    robot_name=spec.robot_name,
                    wait=True,
                    wait_timeout_s=_UPLOAD_WAIT_TIMEOUT_S,
                )

                assert backend_utils.is_recording_upload_complete(recording_id), (
                    "stop_recording(wait=True) returned before the upload "
                    "completed, so it resolved no cloud id and skipped the wait"
                )
    finally:
        set_case_analysis_report(
            request=request,
            case=case,
            results=results,
            test_wall_s=test_wall_timer(),
        )
