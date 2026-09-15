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
        recording_index=7, cloud_recording_id="cloud-1", start_timestamp_ns=1_000
    )
    with patch(
        "neuracore.core.robot.recording_context.query_recording_state",
        return_value=live,
    ):
        assert robot.is_recording() is True
        assert robot.get_cloud_recording_id(timeout_s=0.0) == "cloud-1"

    with patch(
        "neuracore.core.robot.recording_context.query_recording_state",
        return_value=None,
    ):
        assert robot.is_recording() is False
        assert robot.get_cloud_recording_id(timeout_s=0.0) is None

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
            robot.get_cloud_recording_id(timeout_s=0.0)

    robot.id = None


def test_the_cloud_id_wait_is_handed_to_the_bridge() -> None:
    """The daemon mints the cloud id after the recording opens, so a caller has
    to wait for it. The wait belongs in the bridge, which drops the GIL while it
    polls, rather than in a Python loop across the boundary."""
    robot = Robot("robot", instance=0, org_id="org-1")
    robot.id = "robot-id-1"
    minted = LiveRecording(
        recording_index=7, cloud_recording_id="cloud-1", start_timestamp_ns=1_000
    )

    with patch(
        "neuracore.core.robot.recording_context.query_recording_state",
        return_value=minted,
    ) as query_recording_state:
        assert robot.get_cloud_recording_id(timeout_s=1.0) == "cloud-1"

    query_recording_state.assert_called_once_with("robot-id-1", 0, timeout_s=1.0)
    robot.id = None


def test_a_cloud_id_that_never_arrives_gives_up() -> None:
    """An offline recording never gets an id, so this must end, not hang."""
    robot = Robot("robot", instance=0, org_id="org-1")
    robot.id = "robot-id-1"
    pending = LiveRecording(
        recording_index=7, cloud_recording_id=None, start_timestamp_ns=1_000
    )

    with patch(
        "neuracore.core.robot.recording_context.query_recording_state",
        return_value=pending,
    ):
        assert robot.get_cloud_recording_id(timeout_s=1.0) is None

    robot.id = None
