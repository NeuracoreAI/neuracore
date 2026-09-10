from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from neuracore.core.robot import Robot
from neuracore.data_daemon.bridge import LiveRecording, RecordingStateUnavailableError


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
