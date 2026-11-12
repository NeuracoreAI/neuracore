"""Behavioural correctness test for tags given when a recording stops.

The tags travel from ``nc.stop_recording`` through the daemon to the backend.
This pins that a stop with tags saves them as user tags of the recording, and
that a stop without tags saves none.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pytest
from neuracore_types import RecordingTagSource

import neuracore as nc
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    wait_for_dataset_ready,
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
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    scoped_storage_state,
    set_case_analysis_report,
)

logger = logging.getLogger(__name__)

_UPLOAD_WAIT_TIMEOUT_S = 180.0

_CASES = DataDaemonTestBatch(
    cases=(Synchronous(duration_sec=1, joint_count=1),),
).as_cases()


@pytest.mark.parametrize("case", _CASES, ids=case_ids(_CASES))
@pytest.mark.parametrize(
    ("stop_tags", "expected_tag_names"),
    [(None, set()), (["Stop Tag", "stop tag"], {"stop-tag"})],
    ids=["without-tags", "with-tags"],
)
def test_stop_tags_appear_on_the_saved_recording(
    case: DataDaemonTestCase,
    stop_tags: list[str] | None,
    expected_tag_names: set[str],
    clear_daemon_timer_stats,
    request: pytest.FixtureRequest,
    test_wall_timer: Callable[[], float],
) -> None:
    """Tags given to ``stop_recording`` are the user tags of the saved recording."""
    if not has_configured_org():
        pytest.skip(
            "Stop-tag behavioural tests require NEURACORE_ORG_ID"
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
                        dataset_name, description="Stop tags behavioural test"
                    )
                with Timer(
                    MAX_TIME_TO_START_S, label="nc.connect_robot", always_log=True
                ):
                    robot = nc.connect_robot(spec.robot_name, overwrite=False)

                nc.start_recording(robot_name=spec.robot_name)
                log_frames(
                    spec, robot=robot, recording_index=0, marker_name="marker_tags"
                )
                nc.stop_recording(
                    robot_name=spec.robot_name,
                    wait=True,
                    wait_timeout_s=_UPLOAD_WAIT_TIMEOUT_S,
                    tags=stop_tags,
                )

                wait_for_dataset_ready(dataset_name, timeout_s=_UPLOAD_WAIT_TIMEOUT_S)
                recording = next(iter(nc.get_dataset(dataset_name)))
                user_tag_names = {
                    tag.name
                    for tag in recording.tags
                    if tag.source == RecordingTagSource.USER
                }
                assert user_tag_names == expected_tag_names, (
                    f"stop_recording(tags={stop_tags!r}) saved the user tags "
                    f"{sorted(user_tag_names)}, expected {sorted(expected_tag_names)}"
                )
    finally:
        set_case_analysis_report(
            request=request,
            case=case,
            results=results,
            test_wall_s=test_wall_timer(),
        )
