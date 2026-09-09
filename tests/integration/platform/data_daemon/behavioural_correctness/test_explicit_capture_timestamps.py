"""Behavioural tests for the capture timestamps passed to start/stop recording."""

from __future__ import annotations

import dataclasses
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
    COLUMN_RECORDING_INDEX,
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
from tests.integration.platform.data_daemon.shared.runners import (
    offline_daemon_running,
    online_daemon_running,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    Synchronous,
    has_configured_org,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    MAX_TIME_TO_START_S,
    STOP_RECORDING_NO_WAIT_SLA_S,
    STORAGE_STATE_DELETE,
)
from tests.integration.platform.data_daemon.shared.test_case.context_spec import (
    ContextSpec,
    build_context_specs,
    precompute_timestamps,
)
from tests.integration.platform.data_daemon.shared.test_case.context_worker import (
    create_testing_dataset_name,
    log_frames,
)
from tests.integration.platform.data_daemon.shared.test_case.recording_control import (
    LocalRecordingController,
    RecordingController,
    RemoteRecordingController,
)
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    scoped_storage_state,
)

_CASE = Synchronous(
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

_EARLIER_CAPTURE_START_S = _SYNTHETIC_CAPTURE_START_S - 86_400.0
"""A day before the first recording's, for the newest-first case below."""

_TRACE_WRITE_TIMEOUT_S = 30.0

_EPOCH_VISIBLE_TIMEOUT_S = 5.0
"""How long the producer's cache may take to see a recording it did not open.

A recording bracketed here is known the moment `start_recording` returns. One
opened from the web reaches this process only through the bridge's refresh, so
it is waited for — and that wait is the assertion that the refresh works.
"""


def _await_recording_epoch(robot: Any) -> int:
    """Block until this process can see the recording its source has open."""
    deadline = time.monotonic() + _EPOCH_VISIBLE_TIMEOUT_S
    while time.monotonic() < deadline:
        epoch = robot._recording_epoch()
        if epoch is not None:
            return epoch
        time.sleep(0.02)
    raise AssertionError(
        "The producer never saw the open recording: the monotonic-timestamp"
        f" check stayed off for {_EPOCH_VISIBLE_TIMEOUT_S}s after the window"
        " opened."
    )


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
    after_index: int = 0,
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
        str(robot.id),
        int(robot.instance),
        after_index=after_index,
        timeout_s=MAX_TIME_TO_START_S,
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


def test_a_recording_may_start_below_where_the_last_one_ended() -> None:
    """A source's second recording is free to carry an earlier capture clock.

    An importer replaying episodes newest-first does exactly this: each episode
    is its own recording, and the one it uploads second is stamped a day before
    the one it uploaded first. The monotonic-timestamp check is scoped to a
    recording precisely so this is not a violation — and it is scoped by asking
    the daemon, which is the only party that knows where one recording ends and
    the next begins.

    Both recordings run on one source, in one process, so nothing but the
    recording boundary separates the two timelines.
    """
    if not has_configured_org():
        pytest.skip(
            "Capture-timestamp behavioural tests require NEURACORE_ORG_ID"
            " or a saved current organization."
        )

    ensure_login()
    dataset_name = create_testing_dataset_name(_CASE)
    spec = build_context_specs(_CASE)[0]
    # The second recording's whole timeline — its bracket and its frames — sits
    # a day below the first's.
    earlier_spec = dataclasses.replace(
        spec,
        timestamp_start_s=_EARLIER_CAPTURE_START_S,
        timestamp_end_s=_EARLIER_CAPTURE_START_S + _CASE.duration_sec,
    )

    with scoped_storage_state(_CASE):
        with offline_daemon_running():
            assert_exactly_one_daemon_pid()
            with Timer(MAX_TIME_TO_START_S, label="nc.create_dataset", always_log=True):
                nc.create_dataset(dataset_name)
            with Timer(MAX_TIME_TO_START_S, label="nc.connect_robot", always_log=True):
                robot = nc.connect_robot(spec.robot_name, overwrite=False)

            later_index, _, _ = _record_one(
                robot,
                spec,
                capture_start_s=_SYNTHETIC_CAPTURE_START_S,
                capture_stop_s=_SYNTHETIC_CAPTURE_START_S + _CASE.duration_sec,
            )
            # A ValueError out of any log_* call here is the regression: before
            # the check was scoped to a recording, the first frame of this one
            # was rejected for trailing the last frame of the one above.
            earlier_index, _, _ = _record_one(
                robot,
                earlier_spec,
                capture_start_s=_EARLIER_CAPTURE_START_S,
                capture_stop_s=_EARLIER_CAPTURE_START_S + _CASE.duration_sec,
                after_index=later_index,
            )

            rows = {
                int(row[COLUMN_RECORDING_INDEX]): row
                for row in fetch_recordings_for_source(
                    str(robot.id), int(robot.instance)
                )
            }
            assert rows.keys() >= {later_index, earlier_index}, (
                "Both recordings must survive as their own rows; the second is"
                f" not a continuation of the first. rows={sorted(rows)}"
            )
            assert rows[earlier_index][COLUMN_START_TIMESTAMP_NS] == int(
                _EARLIER_CAPTURE_START_S * 1e9
            ), (
                "The later recording did not store its own, earlier capture"
                f" start; row={rows[earlier_index]}"
            )

            # Each recording holds exactly its own grid: the daemon partitions
            # by publish clock, so the earlier-stamped data belongs to the
            # recording that was open when it was published, not to the one
            # whose capture clock brackets it.
            for recording_index, timeline_start_s in (
                (later_index, spec.timestamp_start_s),
                (earlier_index, earlier_spec.timestamp_start_s),
            ):
                _wait_for_written_traces(recording_index)
                expected_timestamps = precompute_timestamps(
                    timeline_start_s,
                    spec.expected_joint_frames,
                    spec.case.joint_fps,
                )
                recording_dir = get_daemon_recordings_root_path() / str(recording_index)
                on_disk = collect_trace_timestamps_per_file(recording_dir)
                assert on_disk, (
                    "No traces on disk for recording"
                    f" {recording_index}; recording_dir={recording_dir}"
                )
                for trace_key, timestamps in sorted(on_disk.items()):
                    assert timestamps == expected_timestamps, (
                        f"trace {trace_key} of recording {recording_index} does"
                        " not hold exactly its own recording's frames:"
                        f" expected {len(expected_timestamps)},"
                        f" got {len(timestamps)}"
                    )


@pytest.mark.parametrize(
    "controller_type",
    [LocalRecordingController, RemoteRecordingController],
    ids=["local", "remote"],
)
def test_a_backwards_timestamp_inside_one_recording_is_rejected(
    controller_type: type[RecordingController],
) -> None:
    """The check is live, and scoped to a recording rather than switched off.

    Pairs with the newest-first case above, which asserts only that nothing
    raises — on its own that passes just as well if the check never runs. This
    asserts the other half: inside a recording a timestamp that fails to advance
    is refused at the `log_*` call, and outside one it is not, because nothing
    logged there reaches a trace.

    Run under both controls because they reach this process by different
    routes. A local start seeds the producer's cache itself; a remote one is
    minted by the backend and never touches this process, so the check can only
    arm through the bridge's refresh — which is exactly the path that stayed
    dark while arming came from a local recording handle.

    The controller is named here rather than set as the case's
    ``recording_control``. That field is how the *matrix* picks one, and the
    matrix rightly refuses remote control for a synchronous producer, whose
    leading frames would land before the window opens. This test builds no
    producer and logs nothing until the window is open, so the case stays
    honest about what runs: one thread, no frame in flight at a boundary.
    """
    if not has_configured_org():
        pytest.skip(
            "Capture-timestamp behavioural tests require NEURACORE_ORG_ID"
            " or a saved current organization."
        )

    ensure_login()
    dataset_name = create_testing_dataset_name(_CASE)
    spec = dataclasses.replace(build_context_specs(_CASE)[0], dataset_name=dataset_name)
    joint_name = "joint_0"
    # A remote start is the backend's to mint, so the daemon has to be online
    # to be told about it.
    remote = controller_type is RemoteRecordingController
    daemon = online_daemon_running if remote else offline_daemon_running

    with scoped_storage_state(_CASE):
        with daemon():
            assert_exactly_one_daemon_pid()
            with Timer(MAX_TIME_TO_START_S, label="nc.create_dataset", always_log=True):
                nc.create_dataset(dataset_name)
            with Timer(MAX_TIME_TO_START_S, label="nc.connect_robot", always_log=True):
                robot = nc.connect_robot(spec.robot_name, overwrite=False)

            # No recording open: there is no timeline to keep, so a producer
            # free-running between recordings is not failed for its clock.
            idle_s = time.time()
            nc.log_joint_positions(
                positions={joint_name: 0.0},
                robot_name=spec.robot_name,
                timestamp=idle_s,
            )
            nc.log_joint_positions(
                positions={joint_name: 0.1},
                robot_name=spec.robot_name,
                timestamp=idle_s - 10.0,
            )

            controller = controller_type(spec, robot)
            # Stamped here, not earlier: a remote start is held to an
            # announcement SLA measured from the start time the cloud records,
            # so any setup between the two would be charged to the round trip.
            capture_start_s = time.time()
            controller.open(capture_start_s)
            try:
                _await_recording_epoch(robot)
                nc.log_joint_positions(
                    positions={joint_name: 0.2},
                    robot_name=spec.robot_name,
                    timestamp=capture_start_s + 1.0,
                )
                with pytest.raises(ValueError, match="Non-monotonic timestamp"):
                    nc.log_joint_positions(
                        positions={joint_name: 0.3},
                        robot_name=spec.robot_name,
                        timestamp=capture_start_s + 1.0,
                    )
                with pytest.raises(ValueError, match="Non-monotonic timestamp"):
                    nc.log_joint_positions(
                        positions={joint_name: 0.4},
                        robot_name=spec.robot_name,
                        timestamp=capture_start_s + 0.5,
                    )
            finally:
                controller.close(capture_start_s + _CASE.duration_sec)
                controller.shutdown()
