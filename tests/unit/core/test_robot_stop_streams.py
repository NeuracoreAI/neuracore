from __future__ import annotations

from unittest.mock import MagicMock, patch

from neuracore_types import DataType

from neuracore.core.robot import Robot


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


def _remote_stop_callback(robot: Robot) -> object:
    """Register the connect-time handler and hand back the callback it stored."""
    captured: dict[str, object] = {}

    class _FakeManager:
        def register_remote_stop_handler(
            self, robot_id: str, instance: int, callback: object
        ) -> None:
            captured["callback"] = callback

        def deregister_remote_stop_handler(self, robot_id: str, instance: int) -> None:
            pass

    with patch(
        "neuracore.core.robot.get_recording_state_manager", return_value=_FakeManager()
    ):
        robot._register_remote_stop_handler()
        robot.id = None  # prevent __del__ from hitting the real manager

    callback = captured["callback"]
    assert callable(callback)
    return callback


def test_web_stop_drains_streams_and_notifies_daemon() -> None:
    """Callback registered at connect time must drain streams and notify the daemon."""
    robot = Robot("robot", instance=0, org_id="org-1")
    robot.id = "robot-id-1"
    # This process opened the window, so it is the one that closes it.
    robot._owns_daemon_recording = True

    active = _ActiveStream()
    robot.add_data_stream("JOINT_POSITIONS:joint", active)  # type: ignore[arg-type]

    fake_daemon = MagicMock()
    robot._daemon_recording_context = fake_daemon

    _remote_stop_callback(robot)("rec-abc")

    assert active.stop_calls == 1
    fake_daemon.stop_recording.assert_called_once_with(timestamp=None)
    fake_daemon.flush_source.assert_called_once_with()


def test_web_stop_in_a_non_owning_process_flushes_without_publishing_a_stop() -> None:
    """A process that never started the recording must not publish its stop."""
    robot = Robot("robot", instance=0, org_id="org-1")
    robot.id = "robot-id-1"

    active = _ActiveStream()
    robot.add_data_stream("JOINT_POSITIONS:joint", active)  # type: ignore[arg-type]

    fake_daemon = MagicMock()
    robot._daemon_recording_context = fake_daemon

    _remote_stop_callback(robot)("rec-abc")

    assert active.stop_calls == 1
    fake_daemon.stop_recording.assert_not_called()
    fake_daemon.flush_source.assert_called_once_with()


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
