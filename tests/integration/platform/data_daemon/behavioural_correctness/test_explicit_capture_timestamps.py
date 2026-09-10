"""Behavioural tests for the capture timestamps passed to start/stop recording."""

from __future__ import annotations

import time
from typing import Any

import pytest

import neuracore as nc
from neuracore.data_daemon.helpers import get_daemon_recordings_root_path
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
)
from tests.integration.platform.data_daemon.shared.auth import ensure_login
from tests.integration.platform.data_daemon.shared.db_constants import (
    COLUMN_START_TIMESTAMP_NS,
    COLUMN_STOP_TIMESTAMP_NS,
    TRACE_WRITE_WRITTEN,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    fetch_all_traces,
    fetch_recordings_for_source,
    wait_for_recording_index_for_source,
)
from tests.integration.platform.data_daemon.shared.disk_helpers import (
    collect_trace_timestamps_per_file,
)
from tests.integration.platform.data_daemon.shared.process_control import Timer
from tests.integration.platform.data_daemon.shared.runners import offline_daemon_running
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestCase,
    has_configured_org,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case_context import (  # noqa: E501
    ContextSpec,
    build_context_specs,
    create_testing_dataset_name,
    log_frames,
    precompute_timestamps,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    MAX_TIME_TO_START_S,
    STOP_RECORDING_NO_WAIT_SLA_S,
    STORAGE_STATE_DELETE,
)
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    scoped_storage_state,
)

_CASE = DataDaemonTestCase(
    duration_sec=3,
    joint_count=4,
    video_count=1,
    image_width=64,
    image_height=64,
    # Equal rates, so every trace shares one expected timestamp list.
    joint_fps=10,
    video_fps=10,
    storage_state_action=STORAGE_STATE_DELETE,
)

_MARKER_NAME = "marker_capture_timestamps"

_SYNTHETIC_CAPTURE_START_S = 1_000_000.0
"""A capture clock decades away from any publish clock."""

_TRACE_WRITE_TIMEOUT_S = 30.0


def _wait_for_written_traces(recording_index: int) -> list[dict[str, Any]]:
    """Wait until every trace of *recording_index* is sealed to disk."""
    deadline = time.monotonic() + _TRACE_WRITE_TIMEOUT_S
    traces: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        traces = fetch_all_traces(
            recording_index, columns=["data_type_name", "write_status"]
        )
        if traces and all(
            trace.get("write_status") == TRACE_WRITE_WRITTEN for trace in traces
        ):
            return traces
        time.sleep(0.1)

    raise AssertionError(
        "Daemon did not finalize every trace for recording "
        f"{recording_index}; traces={traces}"
    )


def _record_one(
    robot: Any,
    spec: ContextSpec,
    *,
    capture_start_s: float | None,
    capture_stop_s: float | None,
) -> tuple[int, tuple[float, float], tuple[float, float]]:
    """Record one recording end to end, with or without capture timestamps.

    Returns:
        The daemon's ``recording_index`` and the wall-clock brackets around the
        control calls.
    """
    start_called_at = time.time()
    with Timer(MAX_TIME_TO_START_S, label="nc.start_recording", always_log=True):
        nc.start_recording(robot_name=spec.robot_name, timestamp=capture_start_s)
    start_returned_at = time.time()

    recording_index = wait_for_recording_index_for_source(
        str(robot.id), int(robot.instance), timeout_s=MAX_TIME_TO_START_S
    )
    log_frames(spec, robot=robot, recording_index=0, marker_name=_MARKER_NAME)

    stop_called_at = time.time()
    with Timer(
        STOP_RECORDING_NO_WAIT_SLA_S,
        label="nc.stop_recording",
        always_log=True,
        assert_deadline=False,
    ):
        nc.stop_recording(
            robot_name=spec.robot_name, wait=False, timestamp=capture_stop_s
        )
    stop_returned_at = time.time()

    _wait_for_written_traces(recording_index)
    return (
        recording_index,
        (start_called_at, start_returned_at),
        (stop_called_at, stop_returned_at),
    )


def _fetch_only_recording(robot: Any) -> dict[str, Any]:
    """Return the single recording row this source produced."""
    rows = fetch_recordings_for_source(str(robot.id), int(robot.instance))
    assert len(rows) == 1, f"Expected exactly one recording for the source; got {rows}"
    return rows[0]


def test_explicit_capture_timestamps_are_stored_and_leave_the_window_alone() -> None:
    """Capture timestamps reach the row verbatim and do not move the window."""
    if not has_configured_org():
        pytest.skip(
            "Capture-timestamp behavioural tests require NEURACORE_ORG_ID"
            " or a saved current organization."
        )

    ensure_login()
    dataset_name = create_testing_dataset_name(_CASE)
    spec = build_context_specs(_CASE)[0]
    capture_start_s = _SYNTHETIC_CAPTURE_START_S
    capture_stop_s = capture_start_s + _CASE.duration_sec

    with scoped_storage_state(_CASE):
        with offline_daemon_running():
            assert_exactly_one_daemon_pid()
            with Timer(MAX_TIME_TO_START_S, label="nc.create_dataset", always_log=True):
                nc.create_dataset(dataset_name)
            with Timer(MAX_TIME_TO_START_S, label="nc.connect_robot", always_log=True):
                robot = nc.connect_robot(spec.robot_name, overwrite=False)

            recording_index, _, _ = _record_one(
                robot,
                spec,
                capture_start_s=capture_start_s,
                capture_stop_s=capture_stop_s,
            )

            row = _fetch_only_recording(robot)
            assert row[COLUMN_START_TIMESTAMP_NS] == int(capture_start_s * 1e9), (
                "Recording row did not store the capture start time passed to"
                f" start_recording; row={row}"
            )
            assert row[COLUMN_STOP_TIMESTAMP_NS] == int(capture_stop_s * 1e9), (
                "Recording row did not store the capture stop time passed to"
                f" stop_recording; row={row}"
            )

            expected_timestamps = precompute_timestamps(
                spec.timestamp_start_s,
                spec.expected_joint_frames,
                spec.case.joint_fps,
            )
            recording_dir = get_daemon_recordings_root_path() / str(recording_index)
            on_disk = collect_trace_timestamps_per_file(recording_dir)
            assert on_disk, (
                "No traces on disk: the recording window did not hold the data"
                f" logged inside it. recording_dir={recording_dir}"
            )
            for trace_key, timestamps in sorted(on_disk.items()):
                assert timestamps == expected_timestamps, (
                    f"trace {trace_key} does not hold exactly the logged frames:"
                    f" expected {len(expected_timestamps)}, got {len(timestamps)}"
                )
