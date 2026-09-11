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


def test_a_timeout_waits_for_the_cloud_id_to_be_minted(monkeypatch) -> None:
    """The daemon mints the cloud id after the recording opens, so this polls.

    A caller that gave up on the first `None` would skip the upload wait for
    every recording stopped promptly after starting.
    """
    robot = Robot("robot", instance=0, org_id="org-1")
    robot.id = "robot-id-1"
    pending = LiveRecording(
        recording_index=7, cloud_recording_id=None, start_timestamp_ns=1_000
    )
    minted = pending._replace(cloud_recording_id="cloud-1")
    monkeypatch.setattr("neuracore.core.robot.time.sleep", lambda _seconds: None)

    with patch(
        "neuracore.core.robot.recording_context.query_recording_state",
        side_effect=(pending, pending, minted),
    ):
        assert robot.get_cloud_recording_id(timeout_s=1.0) == "cloud-1"

    robot.id = None


def test_a_timeout_gives_up_on_a_cloud_id_that_never_arrives(monkeypatch) -> None:
    """An offline recording never gets an id, so this must end, not hang."""
    robot = Robot("robot", instance=0, org_id="org-1")
    robot.id = "robot-id-1"
    pending = LiveRecording(
        recording_index=7, cloud_recording_id=None, start_timestamp_ns=1_000
    )
    # First call sets the deadline, second passes it.
    clock = iter((0.0, 2.0))
    monkeypatch.setattr("neuracore.core.robot.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("neuracore.core.robot.time.sleep", lambda _seconds: None)

    with patch(
        "neuracore.core.robot.recording_context.query_recording_state",
        return_value=pending,
    ):
        assert robot.get_cloud_recording_id(timeout_s=1.0) is None

    robot.id = None
