"""Focused tests for recording notification ordering."""

import asyncio
import threading
from unittest.mock import MagicMock

from neuracore_types import (
    BaseRecodingUpdatePayload,
    RecordingNotification,
    RecordingNotificationType,
    RecordingStartPayload,
    RobotInstanceIdentifier,
)

from neuracore.core.streaming.recording_state_manager import (
    RecordingStateManager,
    TrackedRecording,
)
from neuracore.data_daemon import bridge


def _join_relay_threads() -> None:
    """Wait for the discard relay's thread — it publishes off the event loop."""
    for thread in threading.enumerate():
        if thread.name.startswith("discard-"):
            thread.join(timeout=5)


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


def _discard_notification(*, recording_id: str, robot_id: str = "robot-1") -> str:
    return RecordingNotification(
        type=RecordingNotificationType.DISCARDED,
        payload=BaseRecodingUpdatePayload(
            recording_id=recording_id,
            robot_id=robot_id,
            instance=0,
        ),
    ).model_dump_json()


def test_discard_notification_stops_the_daemons_uploads(monkeypatch) -> None:
    """A DISCARDED must reach the daemon, which owns the uploads."""
    native = MagicMock()
    monkeypatch.setattr(bridge, "_DATA_BRIDGE_MODULE", native)
    manager = _manager()
    source = RobotInstanceIdentifier(robot_id="robot-1", robot_instance=0)
    manager.recording_robot_instances[source] = TrackedRecording(
        recording_id="cloud-id",
        start_time=100.0,
        opened_locally=True,
    )

    asyncio.run(manager.on_message(_discard_notification(recording_id="cloud-id")))
    _join_relay_threads()

    native.discard_recording.assert_called_once_with("cloud-id")
    assert source not in manager.recording_robot_instances


def test_discard_of_another_users_recording_still_stops_uploads(monkeypatch) -> None:
    """A web-initiated cancel names a recording this process never tracked,
    while this process is the one still uploading it, so the relay is
    unconditional.
    """
    native = MagicMock()
    monkeypatch.setattr(bridge, "_DATA_BRIDGE_MODULE", native)
    manager = _manager()

    # Nothing tracked locally, and a robot this client never connected to.
    asyncio.run(
        manager.on_message(
            _discard_notification(recording_id="someone-elses", robot_id="robot-9")
        )
    )
    _join_relay_threads()

    native.discard_recording.assert_called_once_with("someone-elses")


def test_stop_notification_does_not_touch_the_upload_queue(monkeypatch) -> None:
    """Only a discard drops uploads — a normal stop must still upload its data."""
    native = MagicMock()
    monkeypatch.setattr(bridge, "_DATA_BRIDGE_MODULE", native)
    manager = _manager()

    notification = RecordingNotification(
        type=RecordingNotificationType.STOP,
        payload=BaseRecodingUpdatePayload(
            recording_id="cloud-id", robot_id="robot-1", instance=0
        ),
    ).model_dump_json()
    asyncio.run(manager.on_message(notification))
    _join_relay_threads()

    native.discard_recording.assert_not_called()


def test_discard_is_not_relayed_from_a_process_that_never_recorded(
    monkeypatch,
) -> None:
    """A bystander has no uploads to stop, and publishing would needlessly
    initialise its producer IPC state.
    """
    monkeypatch.setattr(bridge, "_DATA_BRIDGE_MODULE", None)
    manager = _manager()

    asyncio.run(manager.on_message(_discard_notification(recording_id="cloud-id")))
    _join_relay_threads()

    # The point is that loading the bridge was never attempted.
    assert bridge._DATA_BRIDGE_MODULE is None
