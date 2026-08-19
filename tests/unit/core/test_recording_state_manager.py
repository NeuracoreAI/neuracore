"""Tests for recording-state manager teardown."""

import asyncio
import threading
from concurrent.futures import Future
from unittest.mock import Mock

from neuracore_types import RecordingStartPayload, RobotInstanceIdentifier

from neuracore.core.streaming import recording_state_manager as manager_module
from neuracore.core.streaming.p2p.enabled_manager import EnabledManager
from neuracore.core.streaming.recording_state_manager import (
    RecordingStateManager,
    TrackedRecording,
    shutdown_recording_state_manager,
)


def test_shutdown_cancels_timers_clears_state_and_closes_session():
    """No recording callbacks or authenticated SSE resources survive logout."""
    loop = asyncio.new_event_loop()
    loop_started = threading.Event()

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        loop_started.set()
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()
    loop_started.wait(timeout=1.0)

    class ClientSessionStub:
        closed = False

        async def close(self) -> None:
            self.closed = True

    manager = object.__new__(RecordingStateManager)
    manager.loop = loop
    manager.enabled_manager = EnabledManager(True, loop=loop)
    manager.enabled_manager.add_listener(EnabledManager.DISABLED, manager._on_close)
    manager.background_tracker = Mock()
    manager.signalling_stream_future = Mock()
    manager.client_session = ClientSessionStub()
    manager._state_lock = threading.RLock()

    key = RobotInstanceIdentifier(robot_id="robot-id", robot_instance=0)
    timer_handle = Mock()
    manager.recording_robot_instances = {
        key: TrackedRecording(recording_id="recording-id", start_time=1.0)
    }
    manager._expired_recording_ids = {"expired-id"}
    manager._recording_timers = {"recording-id": [timer_handle]}
    manager._latest_stopped_times = {key: 2.0}
    manager.active_dataset_ids = {key: "dataset-id"}
    manager._drain_callbacks = {key: Mock()}
    manager._connected_robot_id = "robot-id"

    try:
        manager.shutdown()

        manager.background_tracker.stop_background_coroutines.assert_called_once_with()
        manager.signalling_stream_future.cancel.assert_called_once_with()
        timer_handle.cancel.assert_called_once_with()
        assert manager.client_session.closed is True
        assert manager.recording_robot_instances == {}
        assert manager._expired_recording_ids == set()
        assert manager._recording_timers == {}
        assert manager._latest_stopped_times == {}
        assert manager.active_dataset_ids == {}
        assert manager._drain_callbacks == {}
        assert manager._connected_robot_id is None
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=1.0)
        loop.close()


def test_queued_timer_registration_is_rejected_after_shutdown(monkeypatch):
    """A timer callback queued before logout cannot recreate expiry state."""
    queued_callbacks = []
    loop = Mock()
    loop.call_soon_threadsafe.side_effect = queued_callbacks.append
    monkeypatch.setattr(manager_module, "get_running_loop", lambda: loop)

    manager = object.__new__(RecordingStateManager)
    manager.loop = loop
    manager.enabled_manager = Mock()
    manager._state_lock = threading.RLock()
    manager._stopped = False
    manager.recording_robot_instances = {}
    manager._expired_recording_ids = set()
    manager._recording_timers = {}
    manager._latest_stopped_times = {}
    manager.active_dataset_ids = {}
    manager._drain_callbacks = {}
    manager._connected_robot_id = None

    manager._schedule_recording_timers("robot-id", 0, "recording-id")
    manager._close_recording_resources()

    for callback in queued_callbacks:
        callback()

    assert manager._recording_timers == {}
    loop.call_later.assert_not_called()


def test_shutdown_global_forgets_manager_so_it_can_be_recreated(monkeypatch):
    """The next login/connect must not reuse a closed recording manager."""
    manager = Mock()
    manager_future: Future[RecordingStateManager] = Future()
    manager_future.set_result(manager)
    monkeypatch.setattr(manager_module, "_recording_manager", manager_future)

    shutdown_recording_state_manager()

    manager.shutdown.assert_called_once_with()
    assert manager_module._recording_manager is None


def test_late_start_after_local_stop_does_not_restart_daemon(monkeypatch):
    """A delayed START for a finished recording must remain stopped."""
    key = RobotInstanceIdentifier(robot_id="robot-id", robot_instance=0)
    loop = Mock()
    loop.call_soon_threadsafe.side_effect = lambda callback: callback()
    monkeypatch.setattr(manager_module, "get_running_loop", lambda: loop)

    manager = object.__new__(RecordingStateManager)
    manager.loop = loop
    manager._state_lock = threading.RLock()
    manager._stopped = False
    manager._connected_robot_id = "robot-id"
    manager.recording_robot_instances = {
        key: TrackedRecording(recording_id="local-handle", start_time=100.0)
    }
    manager._expired_recording_ids = set()
    manager._recording_timers = {}
    manager._latest_stopped_times = {}
    manager.active_dataset_ids = {}
    manager._drain_callbacks = {}
    manager._ensure_daemon_for_recording = Mock()

    manager.recording_stopped(
        robot_id="robot-id",
        instance=0,
        recording_id="local-handle",
        stop_time=105.0,
    )
    manager.updated_recording_state(
        is_recording=True,
        details=RecordingStartPayload(
            recording_id="delayed-cloud-id",
            robot_id="robot-id",
            instance=0,
            created_by="test",
            dataset_ids=["dataset-id"],
            start_time=100.0,
        ),
    )

    assert manager.recording_robot_instances == {}
    assert manager.active_dataset_ids == {}
    manager._ensure_daemon_for_recording.assert_not_called()
