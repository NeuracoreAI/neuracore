from __future__ import annotations

import sqlite3

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

    def stop_recording(
        self,
        recording_id: str | None = None,
        producer_stop_sequence_numbers: dict[str, int] | None = None,
    ) -> None:
        assert recording_id is not None
        self.calls.append((recording_id, producer_stop_sequence_numbers or {}))


@pytest.fixture
def robot(monkeypatch) -> robot_module.Robot:
    monkeypatch.setattr(robot_module, "get_current_org", lambda: "org-1")
    monkeypatch.setattr(
        robot_module,
        "get_recording_state_manager",
        lambda: _FakeRecordingStateManager(),
    )
    robot = robot_module.Robot("robot", 0)
    robot.id = "robot-id"

    def stop_all_streams_stub(wait_for_producer_drain: bool = True) -> dict[str, int]:
        return {"producer": 7}

    robot._stop_all_streams = stop_all_streams_stub  # type: ignore[method-assign]
    return robot


def test_stop_recording_wait_false_skips_backend_wait(robot, monkeypatch) -> None:
    fake_context = _FakeDaemonRecordingContext()
    waited: list[str] = []

    monkeypatch.setattr(robot, "_get_daemon_recording_context", lambda: fake_context)
    monkeypatch.setattr(
        robot,
        "_wait_for_daemon_stop_report",
        lambda recording_id: waited.append(recording_id),
    )
    monkeypatch.setattr(
        robot_module.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
    )

    robot.stop_recording("rec-1", wait_for_producer_drain=False)

    assert fake_context.calls == [("rec-1", {"producer": 7})]
    assert waited == []


def test_stop_recording_wait_true_waits_for_daemon_stop_report(
    robot, monkeypatch
) -> None:
    fake_context = _FakeDaemonRecordingContext()
    waited: list[str] = []

    monkeypatch.setattr(robot, "_get_daemon_recording_context", lambda: fake_context)
    monkeypatch.setattr(
        robot,
        "_wait_for_daemon_stop_report",
        lambda recording_id: waited.append(recording_id),
    )

    robot.stop_recording("rec-2", wait_for_producer_drain=True)

    assert fake_context.calls == [("rec-2", {"producer": 7})]
    assert waited == ["rec-2"]


def test_wait_for_daemon_stop_report_returns_when_reported(
    robot, monkeypatch, tmp_path
) -> None:
    db_path = tmp_path / "state.db"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE recordings (
                recording_id TEXT PRIMARY KEY,
                stop_report_status TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO recordings (recording_id, stop_report_status) VALUES (?, ?)",
            ("rec-3", "reported"),
        )
        conn.commit()

    monkeypatch.setattr(robot_module, "get_daemon_db_path", lambda: db_path)

    robot._wait_for_daemon_stop_report("rec-3")


def test_wait_for_daemon_stop_report_times_out(robot, monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "state.db"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE recordings (
                recording_id TEXT PRIMARY KEY,
                stop_report_status TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO recordings (recording_id, stop_report_status) VALUES (?, ?)",
            ("rec-timeout", "pending"),
        )
        conn.commit()

    monkeypatch.setattr(robot_module, "get_daemon_db_path", lambda: db_path)
    monkeypatch.setattr(robot_module, "_STOP_REPORT_WAIT_TIMEOUT_S", 0.01)
    monkeypatch.setattr(robot_module, "_STOP_REPORT_POLL_INTERVAL_S", 0.001)

    with pytest.raises(
        RobotError, match="Timed out waiting for the daemon to report recording stop"
    ):
        robot._wait_for_daemon_stop_report("rec-timeout")
