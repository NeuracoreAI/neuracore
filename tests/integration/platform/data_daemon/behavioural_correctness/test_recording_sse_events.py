"""Integration-style coverage for recording lifecycle SSE notifications."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from neuracore_types import DataType, RobotInstanceIdentifier

from neuracore.api import logging as api_logging
from neuracore.core.robot import Robot
from neuracore.core.streaming import recording_state_manager as rsm
from neuracore.core.streaming.data_stream import DataRecordingContext
from neuracore.core.streaming.recording_state_manager import RecordingStateManager


class _StartableStream:
    data_type = DataType.JOINT_POSITIONS

    def __init__(self) -> None:
        self._recording = False
        self.start_contexts: list[DataRecordingContext] = []

    def is_recording(self) -> bool:
        return self._recording

    def start_recording(
        self, context: DataRecordingContext, coordinator: object
    ) -> None:
        del coordinator
        self._recording = True
        self.start_contexts.append(context)


class _FakeCoordinator:
    producer_id = "robot:robot-1:0"

    def __init__(self) -> None:
        self.stop_calls: list[bool] = []

    def stop_recording(self, *, wait_for_drain: bool = True) -> int:
        self.stop_calls.append(wait_for_drain)
        return 12

    def close(self) -> None:
        pass


class _ActiveStream:
    data_type = DataType.JOINT_POSITIONS

    def __init__(self) -> None:
        self.stopped = False

    def mark_recording_stopped(self) -> None:
        self.stopped = True


class _ImmediateThread:
    def __init__(
        self,
        *,
        target: object,
        args: tuple[object, ...],
        daemon: bool,
        name: str,
    ) -> None:
        del daemon, name
        self._target = target
        self._args = args

    def start(self) -> None:
        assert callable(self._target)
        self._target(*self._args)


@pytest.fixture
def recording_manager(monkeypatch: pytest.MonkeyPatch) -> RecordingStateManager:
    """Build a manager without starting the background SSE socket loop."""
    monkeypatch.setattr(
        rsm.BaseSSEConsumer,
        "__init__",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(rsm, "ensure_daemon_running", MagicMock())

    manager = RecordingStateManager(org_id="org-1", auth=MagicMock())
    manager.register_connected_robot("robot-1")
    manager._schedule_recording_timers = MagicMock()  # type: ignore[method-assign]
    manager._cancel_recording_timers = MagicMock()  # type: ignore[method-assign]
    return manager


def _start_notification(
    *,
    recording_id: str = "rec-1",
    robot_id: str = "robot-1",
    instance: int = 0,
    dataset_id: str = "dataset-1",
) -> str:
    return json.dumps({
        "type": "START",
        "payload": {
            "recording_id": recording_id,
            "robot_id": robot_id,
            "instance": instance,
            "created_by": "server-user",
            "dataset_ids": [dataset_id],
            "data_types": [],
            "start_time": 1000.0,
        },
    })


def _stop_notification(
    *,
    recording_id: str = "rec-1",
    robot_id: str = "robot-1",
    instance: int = 0,
) -> str:
    return json.dumps({
        "type": "STOP",
        "payload": {
            "recording_id": recording_id,
            "robot_id": robot_id,
            "instance": instance,
        },
    })


@pytest.mark.asyncio
async def test_mocked_sse_start_event_marks_robot_instance_recording(
    recording_manager: RecordingStateManager,
) -> None:
    await recording_manager.on_message(_start_notification())

    instance_key = RobotInstanceIdentifier(robot_id="robot-1", robot_instance=0)
    assert recording_manager.is_recording("robot-1", 0)
    assert recording_manager.get_current_recording_id("robot-1", 0) == "rec-1"
    assert recording_manager.active_dataset_ids[instance_key] == "dataset-1"
    recording_manager._schedule_recording_timers.assert_called_once_with(
        robot_id="robot-1",
        instance=0,
        recording_id="rec-1",
    )


@pytest.mark.asyncio
async def test_mocked_sse_start_event_starts_next_logged_stream(
    recording_manager: RecordingStateManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = Robot("robot", instance=0, org_id="org-1")
    robot.id = "robot-1"
    stream = _StartableStream()

    monkeypatch.setattr(
        "neuracore.core.robot.get_recording_state_manager",
        lambda: recording_manager,
    )
    monkeypatch.setattr(
        api_logging,
        "get_recording_state_manager",
        lambda: recording_manager,
    )

    try:
        await recording_manager.on_message(
            _start_notification(recording_id="rec-2", dataset_id="dataset-2")
        )
        api_logging.start_stream(robot, stream)  # type: ignore[arg-type]
    finally:
        robot.id = None

    assert stream.is_recording()
    assert len(stream.start_contexts) == 1
    context = stream.start_contexts[0]
    assert context.recording_id == "rec-2"
    assert context.robot_id == "robot-1"
    assert context.robot_name == "robot"
    assert context.robot_instance == 0
    assert context.dataset_id == "dataset-2"


@pytest.mark.asyncio
async def test_mocked_sse_stop_event_drains_robot_streams_and_notifies_daemon(
    recording_manager: RecordingStateManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = Robot("robot", instance=0, org_id="org-1")
    robot.id = "robot-1"
    stream = _ActiveStream()
    coordinator = _FakeCoordinator()
    fake_daemon = MagicMock()
    robot.add_data_stream("JOINT_POSITIONS:joint", stream)  # type: ignore[arg-type]
    robot._producer_coordinator = coordinator  # type: ignore[assignment]
    robot._daemon_recording_context = fake_daemon

    monkeypatch.setattr(
        "neuracore.core.robot.get_recording_state_manager",
        lambda: recording_manager,
    )
    monkeypatch.setattr(rsm.threading, "Thread", _ImmediateThread)

    try:
        robot._register_remote_stop_handler()
        await recording_manager.on_message(_start_notification(recording_id="rec-3"))
        await recording_manager.on_message(_stop_notification(recording_id="rec-3"))
    finally:
        robot.id = None

    assert not recording_manager.is_recording("robot-1", 0)
    assert stream.stopped is True
    assert coordinator.stop_calls == [False]
    fake_daemon.stop_recording.assert_called_once_with(
        recording_id="rec-3",
        producer_stop_sequence_numbers={coordinator.producer_id: 12},
    )
