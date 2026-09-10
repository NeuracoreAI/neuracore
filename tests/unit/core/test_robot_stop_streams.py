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


def test_web_stop_drains_streams_without_publishing_a_stop() -> None:
    """Callback registered at connect time drains locally and publishes no stop.

    A stop this process only heard about is not its to publish: the envelope
    names no recording, so a publish arriving a network hop late would close
    whichever window is live by then — the next recording, back-to-back. The
    daemon reads the same stream and closes the named recording itself. What
    still has to happen here is everything only this process can do: stop its
    streams and seal its writer's tail chunks.
    """
    robot = Robot("robot", instance=0, org_id="org-1")
    robot.id = "robot-id-1"

    active = _ActiveStream()
    robot.add_data_stream("JOINT_POSITIONS:joint", active)  # type: ignore[arg-type]

    fake_daemon = MagicMock()
    robot._daemon_recording_context = fake_daemon

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
    callback("rec-abc")

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


def test_close_uses_registered_manager_without_creating_another() -> None:
    """Post-logout cleanup must not recreate the global recording manager."""
    robot = Robot("robot", instance=0, org_id="org-1")
    robot.id = "robot-id-1"
    manager = MagicMock()

    with patch(
        "neuracore.core.robot.get_recording_state_manager", return_value=manager
    ):
        robot._register_remote_stop_handler()

    with patch(
        "neuracore.core.robot.get_recording_state_manager",
        side_effect=AssertionError("must not create a manager during close"),
    ):
        robot.close()

    manager.deregister_remote_stop_handler.assert_called_once_with("robot-id-1", 0)
    assert robot._recording_state_manager is None
