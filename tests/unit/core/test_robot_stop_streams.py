from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from neuracore_types import DataType

from neuracore.api.logging import start_stream
from neuracore.core.robot import Robot
from neuracore.core.streaming.data_stream import RGBDataStream
from neuracore.data_daemon.bridge import (
    LiveRecording,
    RecordingStateUnavailableError,
)


class _ActiveStream:
    data_type = DataType.JOINT_POSITIONS

    def __init__(self) -> None:
        self.stop_calls = 0

    def stop_recording(self) -> None:
        self.stop_calls += 1


class _FailingStopStream(_ActiveStream):
    def stop_recording(self) -> None:
        super().stop_recording()
        raise RuntimeError("stop failed")


def test_stop_all_streams_logs_stop_failure_and_continues() -> None:
    """One stream failing to stop must not prevent the others from stopping."""
    robot = Robot("robot", instance=0, org_id="org-1")
    failing = _FailingStopStream()
    active = _ActiveStream()

    robot.add_data_stream("JOINT_POSITIONS:failing", failing)  # type: ignore[arg-type]
    robot.add_data_stream("JOINT_POSITIONS:active", active)  # type: ignore[arg-type]

    robot._stop_all_streams()

    assert failing.stop_calls == 1
    assert active.stop_calls == 1


def test_a_stream_built_mid_recording_arms_itself() -> None:
    """A recording opens before its cameras have streams.

    `nc.start_recording()` then `log_rgb()` is the ordinary flow, so at start
    time there is usually nothing to arm — every stream is built during the
    recording and has to arm as it is created. Arming resets the stream's
    monotonic-timestamp timeline; it carries no recording identity, which the
    daemon owns.
    """
    robot = Robot("robot", instance=0, org_id="org-1")
    robot.id = "robot-id-1"
    robot._daemon_recording_context = MagicMock()

    stream = RGBDataStream("front", width=4, height=3)
    start_stream(robot, stream)
    assert stream.is_recording() is False

    robot.start_recording("dataset-1")
    start_stream(robot, stream)
    assert stream.is_recording() is True

    robot.id = None  # keep __del__ off the real daemon


def test_arming_never_asks_the_daemon() -> None:
    """The log path reads a process-local flag, never the daemon.

    `start_stream` runs per logged frame, and outside a recording the armed
    short-circuit does not apply — so it runs its full body on every frame. A
    daemon round trip there would be per-frame IPC.
    """
    robot = Robot("robot", instance=0, org_id="org-1")
    robot.id = "robot-id-1"
    robot._daemon_recording_context = MagicMock()

    with patch(
        "neuracore.core.robot.recording_context.query_recording_state",
        side_effect=AssertionError("the log path must not ask the daemon"),
    ):
        start_stream(robot, RGBDataStream("front", width=4, height=3))
        robot.start_recording("dataset-1")
        start_stream(robot, RGBDataStream("wrist", width=4, height=3))

    robot.id = None


def test_is_recording_answers_from_the_daemon() -> None:
    """The daemon owns recording state, so it answers even for a recording
    this process never started — which is what makes web-driven and
    cross-process recordings visible here."""
    robot = Robot("robot", instance=0, org_id="org-1")
    robot.id = "robot-id-1"

    live = LiveRecording(
        recording_index=7, recording_id="cloud-1", start_timestamp_ns=1_000
    )
    with patch(
        "neuracore.core.robot.recording_context.query_recording_state",
        return_value=live,
    ):
        assert robot.is_recording() is True
        assert robot.get_current_recording_id() == "cloud-1"

    with patch(
        "neuracore.core.robot.recording_context.query_recording_state",
        return_value=None,
    ):
        assert robot.is_recording() is False
        assert robot.get_current_recording_id() is None

    robot.id = None


def test_an_unanswered_query_raises_rather_than_reading_as_not_recording() -> None:
    """A silent daemon must not read as "not recording".

    `nc.stop_recording` gives up when `is_recording` is False, so answering
    False on silence would leave a recording running with its stop never
    published. The caller is told the state is unknown instead.
    """
    robot = Robot("robot", instance=0, org_id="org-1")
    robot.id = "robot-id-1"
    robot._daemon_recording_context = MagicMock()

    with patch(
        "neuracore.core.robot.recording_context.query_recording_state",
        side_effect=RecordingStateUnavailableError("no answer"),
    ):
        with pytest.raises(RecordingStateUnavailableError):
            robot.is_recording()
        with pytest.raises(RecordingStateUnavailableError):
            robot.get_current_recording_id()

    robot.id = None
