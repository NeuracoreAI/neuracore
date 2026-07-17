import asyncio
import threading

from neuracore_types import (
    BaseRecodingUpdatePayload,
    RecordingNotification,
    RecordingNotificationType,
)

from neuracore.core.streaming.recording_state_manager import RecordingStateManager


def _make_manager() -> RecordingStateManager:
    manager = object.__new__(RecordingStateManager)
    manager._finalized_condition = threading.Condition()
    manager._watched_recording_ids = set()
    manager._terminal_notifications = {}
    return manager


def _notification_json(
    notification_type: RecordingNotificationType, recording_id: str
) -> str:
    return RecordingNotification(
        type=notification_type,
        payload=BaseRecodingUpdatePayload(
            recording_id=recording_id, robot_id="robot-1", instance=0
        ),
    ).model_dump_json()


def test_wait_for_terminal_notification_returns_saved_when_already_marked() -> None:
    manager = _make_manager()
    manager.start_tracking_recording("rec-1")
    manager._record_terminal_notification("rec-1", RecordingNotificationType.SAVED)

    notification = manager.wait_for_terminal_notification("rec-1", timeout_s=0.0)

    assert notification == RecordingNotificationType.SAVED


def test_wait_for_terminal_notification_returns_discarded_when_already_marked() -> None:
    manager = _make_manager()
    manager.start_tracking_recording("rec-1")
    manager._record_terminal_notification("rec-1", RecordingNotificationType.DISCARDED)

    notification = manager.wait_for_terminal_notification("rec-1", timeout_s=0.0)

    assert notification == RecordingNotificationType.DISCARDED


def test_wait_for_terminal_notification_times_out_without_notification() -> None:
    manager = _make_manager()
    manager.start_tracking_recording("rec-1")

    notification = manager.wait_for_terminal_notification("rec-1", timeout_s=0.05)

    assert notification is None


def test_wait_for_terminal_notification_wakes_on_mark_from_another_thread() -> None:
    manager = _make_manager()
    manager.start_tracking_recording("rec-1")

    def mark() -> None:
        manager._record_terminal_notification("rec-1", RecordingNotificationType.SAVED)

    marker = threading.Timer(0.05, mark)
    marker.start()
    try:
        notification = manager.wait_for_terminal_notification("rec-1", timeout_s=2.0)
    finally:
        marker.join(timeout=2.0)

    assert notification == RecordingNotificationType.SAVED


def test_unwatched_recording_notification_is_ignored() -> None:
    manager = _make_manager()

    manager._record_terminal_notification("rec-1", RecordingNotificationType.SAVED)

    assert manager.wait_for_terminal_notification("rec-1", timeout_s=0.0) is None


def test_stop_tracking_recording_drops_stored_notification() -> None:
    manager = _make_manager()
    manager.start_tracking_recording("rec-1")
    manager._record_terminal_notification("rec-1", RecordingNotificationType.SAVED)

    manager.stop_tracking_recording("rec-1")

    assert manager._terminal_notifications == {}
    assert manager._watched_recording_ids == set()


def test_on_message_saved_marks_watched_recording() -> None:
    manager = _make_manager()
    manager.start_tracking_recording("rec-1")

    asyncio.run(
        manager.on_message(_notification_json(RecordingNotificationType.SAVED, "rec-1"))
    )

    assert manager.wait_for_terminal_notification("rec-1", timeout_s=0.0) == (
        RecordingNotificationType.SAVED
    )


def test_on_message_discarded_marks_watched_recording() -> None:
    manager = _make_manager()
    manager.start_tracking_recording("rec-1")
    manager.updated_recording_state = lambda is_recording, details: None

    asyncio.run(
        manager.on_message(
            _notification_json(RecordingNotificationType.DISCARDED, "rec-1")
        )
    )

    assert manager.wait_for_terminal_notification("rec-1", timeout_s=0.0) == (
        RecordingNotificationType.DISCARDED
    )
