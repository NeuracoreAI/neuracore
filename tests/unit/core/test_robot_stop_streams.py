from __future__ import annotations

from unittest.mock import MagicMock, patch

from neuracore_types import DataType

from neuracore.core.robot import Robot
from neuracore.core.streaming.data_stream import MissingProducerChannelError


class _FakeProducerChannel:
    channel_id = "active-channel"

    def __init__(self) -> None:
        self.stop_requested = False

    def mark_recording_stop_requested(self) -> int:
        self.stop_requested = True
        return 42


class _StaleStream:
    data_type = DataType.CUSTOM_1D

    def prepare_recording_stopped(self) -> tuple[_FakeProducerChannel, int]:
        raise MissingProducerChannelError("Stream stale has no ProducerChannel")


class _ActiveStream:
    data_type = DataType.JOINT_POSITIONS

    def __init__(self) -> None:
        self.channel = _FakeProducerChannel()
        self.stop_calls: list[dict[str, object]] = []

    def prepare_recording_stopped(self) -> tuple[_FakeProducerChannel, int]:
        return self.channel, self.channel.mark_recording_stop_requested()

    def stop_recording(
        self,
        *,
        stop_cutoff_sequence_number: int,
        wait_for_producer_drain: bool = True,
    ) -> None:
        self.stop_calls.append({
            "stop_cutoff_sequence_number": stop_cutoff_sequence_number,
            "wait_for_producer_drain": wait_for_producer_drain,
        })


def test_stop_all_streams_removes_stale_stream_and_continues() -> None:
    robot = Robot("robot", instance=0, org_id="org-1")
    stale = _StaleStream()
    active = _ActiveStream()

    robot.add_data_stream("CUSTOM_1D:stale", stale)  # type: ignore[arg-type]
    robot.add_data_stream("JOINT_POSITIONS:joint", active)  # type: ignore[arg-type]

    sequence_numbers = robot._stop_all_streams(wait_for_producer_drain=False)

    assert sequence_numbers == {"active-channel": 42}
    assert active.stop_calls == [{
        "stop_cutoff_sequence_number": 42,
        "wait_for_producer_drain": False,
    }]
    assert "CUSTOM_1D:stale" not in robot.list_all_streams()
    assert "JOINT_POSITIONS:joint" in robot.list_all_streams()
    assert robot._data_stream_counts[DataType.CUSTOM_1D] == 0
    assert robot._data_stream_counts[DataType.JOINT_POSITIONS] == 1


def test_web_stop_drains_streams_and_notifies_daemon() -> None:
    """Callback registered at connect time must drain streams and notify the daemon."""
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

    assert active.stop_calls == [{
        "stop_cutoff_sequence_number": 42,
        "wait_for_producer_drain": False,
    }]
    fake_daemon.stop_recording.assert_called_once_with(
        recording_id="rec-abc",
        producer_stop_sequence_numbers={"active-channel": 42},
    )
