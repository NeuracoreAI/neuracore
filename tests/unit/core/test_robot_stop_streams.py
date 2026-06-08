from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

from neuracore_types import DataType

from neuracore.core.robot import Robot


class _FakeCoordinator:
    """Stands in for the robot-scoped producer coordinator."""

    def __init__(self, producer_id: str = "robot:robot-id-1:0", cutoff: int = 42):
        self.producer_id = producer_id
        self.cutoff = cutoff
        self.stop_calls: list[bool] = []
        self.closed = False

    def stop_recording(self, *, wait_for_drain: bool = True) -> int:
        self.stop_calls.append(wait_for_drain)
        return self.cutoff

    def close(self) -> None:
        self.closed = True


class _FakeStream:
    def __init__(self, data_type: DataType = DataType.JOINT_POSITIONS) -> None:
        self.data_type = data_type
        self.stopped = False

    def mark_recording_stopped(self) -> None:
        self.stopped = True


def test_stop_all_streams_returns_coordinator_cutoff() -> None:
    robot = Robot("robot", instance=0, org_id="org-1")
    coordinator = _FakeCoordinator()
    robot._producer_coordinator = coordinator  # type: ignore[assignment]
    stream = _FakeStream()
    robot.add_data_stream("JOINT_POSITIONS:joint", stream)  # type: ignore[arg-type]

    sequence_numbers = robot._stop_all_streams(wait_for_producer_drain=False)

    assert sequence_numbers == {coordinator.producer_id: 42}
    assert coordinator.stop_calls == [False]
    assert stream.stopped is True


def test_stop_all_streams_without_coordinator_returns_empty() -> None:
    robot = Robot("robot", instance=0, org_id="org-1")
    stream = _FakeStream()
    robot.add_data_stream("JOINT_POSITIONS:joint", stream)  # type: ignore[arg-type]

    assert robot._stop_all_streams() == {}


def test_web_stop_drains_streams_and_notifies_daemon() -> None:
    """Callback registered at connect time must drain streams and notify the daemon."""
    robot = Robot("robot", instance=0, org_id="org-1")
    robot.id = "robot-id-1"
    coordinator = _FakeCoordinator()
    robot._producer_coordinator = coordinator  # type: ignore[assignment]

    stream = _FakeStream()
    robot.add_data_stream("JOINT_POSITIONS:joint", stream)  # type: ignore[arg-type]

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

    assert coordinator.stop_calls == [False]
    assert stream.stopped is True
    fake_daemon.stop_recording.assert_called_once_with(
        recording_id="rec-abc",
        producer_stop_sequence_numbers={coordinator.producer_id: 42},
    )


def test_get_producer_coordinator_is_thread_safe() -> None:
    robot = Robot("robot", instance=0, org_id="org-1")
    robot.id = "robot-id-1"
    created: list[_FakeCoordinator] = []
    results: list[_FakeCoordinator] = []
    thread_count = 8
    barrier = threading.Barrier(thread_count)

    def build_coordinator(producer_id: str) -> _FakeCoordinator:
        time.sleep(0.01)
        coordinator = _FakeCoordinator(producer_id=producer_id)
        created.append(coordinator)
        return coordinator

    def worker() -> None:
        barrier.wait()
        results.append(robot.get_producer_coordinator())  # type: ignore[arg-type]

    with patch(
        "neuracore.core.robot.RobotProducerCoordinator",
        side_effect=build_coordinator,
    ):
        threads = [threading.Thread(target=worker) for _ in range(thread_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert len(created) == 1
    assert len({id(result) for result in results}) == 1
    assert results[0] is created[0]
