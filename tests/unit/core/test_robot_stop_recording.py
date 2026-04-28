from __future__ import annotations

import queue
import time

import pytest

import neuracore.core.robot as robot_module
from neuracore.core.exceptions import RobotError


class _FakeRecordingStateManager:
    def recording_stopped(
        self, robot_id: str, instance: int, recording_id: str
    ) -> None:
        return None


class _FakeDaemonRecordingContext:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, int]]] = []
        self.stop_ack_endpoint = "ipc://fake-stop-ack"
        self._acks: queue.Queue[dict[str, object]] = queue.Queue()
        self._closed = False

    def stop_recording(
        self,
        recording_id: str | None = None,
        producer_stop_sequence_numbers: dict[str, int] | None = None,
    ) -> None:
        assert recording_id is not None
        self.calls.append((recording_id, producer_stop_sequence_numbers or {}))

    def emit_stop_ack(self, recording_id: str, stopped_at: float = 1234.5) -> None:
        self._acks.put({
            "event": "recording_stop_acked",
            "recording_id": recording_id,
            "stopped_at": stopped_at,
        })

    def receive_stop_ack(self, timeout_ms: int = 0) -> dict[str, object] | None:
        if self._closed:
            raise RuntimeError("closed")
        timeout_s = timeout_ms / 1000 if timeout_ms > 0 else 0
        try:
            return self._acks.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._closed = True


@pytest.fixture
def robot(monkeypatch) -> robot_module.Robot:
    monkeypatch.setattr(robot_module, "get_current_org", lambda: "org-1")
    monkeypatch.setattr(
        robot_module,
        "get_recording_state_manager",
        lambda: _FakeRecordingStateManager(),
    )
    fake_context = _FakeDaemonRecordingContext()
    monkeypatch.setattr(robot_module, "DaemonRecordingContext", lambda: fake_context)

    robot = robot_module.Robot("robot", 0)
    robot.id = "robot-id"

    def stop_all_streams_stub(wait_for_producer_drain: bool = True) -> dict[str, int]:
        return {"producer": 7}

    robot._stop_all_streams = stop_all_streams_stub  # type: ignore[method-assign]
    try:
        yield robot
    finally:
        robot.close()


def test_stop_recording_wait_false_skips_backend_wait(robot, monkeypatch) -> None:
    waited: list[str] = []

    monkeypatch.setattr(
        robot,
        "_wait_for_daemon_stop_ack",
        lambda recording_id, **kwargs: waited.append(recording_id),
    )
    monkeypatch.setattr(
        robot_module.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
    )

    robot.stop_recording("rec-1", wait_for_producer_drain=False)

    fake_context = robot._get_daemon_recording_context()
    assert fake_context.calls == [("rec-1", {"producer": 7})]
    assert waited == []


def test_stop_recording_wait_true_waits_for_daemon_stop_report(
    robot, monkeypatch
) -> None:
    waited: list[str] = []

    monkeypatch.setattr(
        robot,
        "_wait_for_daemon_stop_ack",
        lambda recording_id, **kwargs: waited.append(recording_id),
    )

    robot.stop_recording("rec-2", wait_for_producer_drain=True)

    fake_context = robot._get_daemon_recording_context()
    assert fake_context.calls == [("rec-2", {"producer": 7})]
    assert waited == ["rec-2"]


def test_wait_for_daemon_stop_ack_returns_when_daemon_acknowledges(
    robot, monkeypatch
) -> None:
    fake_context = robot._get_daemon_recording_context()
    original_stop_recording = fake_context.stop_recording

    def stop_recording_with_ack(
        recording_id: str | None = None,
        producer_stop_sequence_numbers: dict[str, int] | None = None,
    ) -> None:
        original_stop_recording(
            recording_id=recording_id,
            producer_stop_sequence_numbers=producer_stop_sequence_numbers,
        )
        assert recording_id is not None
        fake_context.emit_stop_ack(recording_id, stopped_at=1234.5)

    fake_context.stop_recording = stop_recording_with_ack  # type: ignore[method-assign]

    robot.stop_recording("rec-3", wait_for_producer_drain=True)

    assert fake_context.calls == [("rec-3", {"producer": 7})]
    assert robot._daemon_stop_acked_at["rec-3"] == 1234.5


def test_wait_for_daemon_stop_ack_times_out(robot, monkeypatch) -> None:
    fake_context = robot._get_daemon_recording_context()
    monkeypatch.setattr(robot_module, "_STOP_REPORT_WAIT_TIMEOUT_S", 0.01)

    robot._prepare_daemon_stop_ack_wait("rec-timeout")

    with pytest.raises(
        RobotError,
        match="Timed out waiting for the daemon to acknowledge recording stop",
    ):
        robot._wait_for_daemon_stop_ack("rec-timeout")

    assert fake_context.calls == []


def test_listener_records_ack_without_active_waiter(robot) -> None:
    fake_context = robot._get_daemon_recording_context()
    fake_context.emit_stop_ack("rec-late", stopped_at=222.0)

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if robot._daemon_stop_acked_at.get("rec-late") == 222.0:
            break
        time.sleep(0.01)

    assert robot._daemon_stop_acked_at["rec-late"] == 222.0
