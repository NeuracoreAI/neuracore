"""Focused tests for recording notification ordering."""

import threading
from unittest.mock import MagicMock

from neuracore_types import RecordingStartPayload, RobotInstanceIdentifier

from neuracore.core.streaming.recording_state_manager import (
    RecordingStateManager,
    TrackedRecording,
)


def _manager() -> RecordingStateManager:
    """Build the state-only portion of a manager without opening an SSE stream."""
    manager = RecordingStateManager.__new__(RecordingStateManager)
    manager._connected_robot_id = "robot-1"
    manager._state_lock = threading.RLock()
    manager.recording_robot_instances = {}
    manager._completed_start_time_watermarks = {}
    manager._expired_recording_ids = set()
    manager._recording_timers = {}
    manager.active_dataset_ids = {}
    manager._drain_callbacks = {}
    manager._start_callbacks = {}
    manager._cancel_recording_timers = MagicMock()
    manager._schedule_recording_timers = MagicMock()
    manager._ensure_daemon_for_recording = MagicMock()
    return manager


def _start_payload(*, recording_id: str, start_time: float) -> RecordingStartPayload:
    return RecordingStartPayload(
        recording_id=recording_id,
        robot_id="robot-1",
        instance=0,
        created_by="test",
        dataset_ids=["dataset-1"],
        data_types=set(),
        start_time=start_time,
    )


def test_delayed_start_after_local_stop_does_not_restart_daemon() -> None:
    manager = _manager()
    source = RobotInstanceIdentifier(robot_id="robot-1", robot_instance=0)
    manager.recording_robot_instances[source] = TrackedRecording(
        recording_id="local-handle",
        start_time=100.0,
        opened_locally=True,
    )

    manager.recording_stopped("robot-1", 0, "local-handle")
    manager.updated_recording_state(
        True,
        _start_payload(recording_id="cloud-id", start_time=100.0),
    )

    assert source not in manager.recording_robot_instances
    assert source not in manager.active_dataset_ids
    manager._ensure_daemon_for_recording.assert_not_called()


def test_newer_start_after_local_stop_is_applied_and_starts_daemon() -> None:
    manager = _manager()
    source = RobotInstanceIdentifier(robot_id="robot-1", robot_instance=0)
    manager.recording_robot_instances[source] = TrackedRecording(
        recording_id="first-recording",
        start_time=100.0,
        opened_locally=True,
    )
    manager.recording_stopped("robot-1", 0, "first-recording")

    manager.updated_recording_state(
        True,
        _start_payload(recording_id="second-recording", start_time=101.0),
    )

    assert manager.recording_robot_instances[source] == TrackedRecording(
        recording_id="second-recording",
        start_time=101.0,
        opened_locally=True,
    )
    assert manager.active_dataset_ids[source] == "dataset-1"
    manager._ensure_daemon_for_recording.assert_called_once_with()
