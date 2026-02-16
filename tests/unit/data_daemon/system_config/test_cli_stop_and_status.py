from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import neuracore.data_daemon.config_manager.args_handler as ah


def test_stop_prints_not_running_if_no_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "get_daemon_db_path", lambda: db_path)
    monkeypatch.setattr(ah, "read_pid_from_file", lambda p: None)

    ah.handle_stop(SimpleNamespace())
    assert capsys.readouterr().out.strip() == "Daemon is not running."


def test_stop_cleans_up_if_pid_file_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "get_daemon_db_path", lambda: db_path)
    monkeypatch.setattr(ah, "read_pid_from_file", lambda p: 999)
    monkeypatch.setattr(ah, "pid_is_running", lambda pid: False)

    called = {"shutdown": 0}

    def fake_shutdown(pid_path, socket_paths, db_path):
        called["shutdown"] += 1

    monkeypatch.setattr(ah, "shutdown", fake_shutdown)

    ah.handle_stop(SimpleNamespace())
    assert called["shutdown"] == 1
    assert capsys.readouterr().out.strip() == "Daemon stopped."


def test_stop_terminate_then_shutdown_happy_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "get_daemon_db_path", lambda: db_path)
    monkeypatch.setattr(ah, "read_pid_from_file", lambda p: 123)
    monkeypatch.setattr(ah, "pid_is_running", lambda pid: True)
    monkeypatch.setattr(ah, "terminate_pid", lambda pid: True)
    monkeypatch.setattr(ah, "wait_for_exit", lambda pid, timeout_s: True)

    called = {"shutdown": 0}

    def fake_shutdown(**kwargs):
        called["shutdown"] += 1

    monkeypatch.setattr(ah, "shutdown", fake_shutdown)

    ah.handle_stop(SimpleNamespace())
    assert called["shutdown"] == 1
    assert capsys.readouterr().out.strip() == "Daemon stopped."


def test_stop_force_kill_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    pid_path.write_text("1234", encoding="utf-8")

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "get_daemon_db_path", lambda: db_path)
    monkeypatch.setattr(ah, "read_pid_from_file", lambda p: 1234)
    monkeypatch.setattr(ah, "pid_is_running", lambda pid: True)

    monkeypatch.setattr(ah, "terminate_pid", lambda pid: True)

    calls = {"n": 0}

    def fake_wait_for_exit(pid: int, timeout_s: float) -> bool:
        calls["n"] += 1
        return calls["n"] == 2  # first wait fails, second succeeds

    monkeypatch.setattr(ah, "wait_for_exit", fake_wait_for_exit)
    monkeypatch.setattr(ah, "force_kill", lambda pid: True)

    shutdown_called = {"called": False}

    def fake_shutdown(*, pid_path, socket_paths, db_path):
        shutdown_called["called"] = True

    monkeypatch.setattr(ah, "shutdown", fake_shutdown)

    ah.handle_stop(SimpleNamespace())  # type: ignore[arg-type]

    out = capsys.readouterr().out
    assert "Daemon stopped (forced)." in out
    assert shutdown_called["called"] is True


def test_stop_permission_denied_on_sigterm_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "get_daemon_db_path", lambda: db_path)
    monkeypatch.setattr(ah, "read_pid_from_file", lambda p: 123)
    monkeypatch.setattr(ah, "pid_is_running", lambda pid: True)
    monkeypatch.setattr(ah, "terminate_pid", lambda pid: False)

    with pytest.raises(SystemExit) as e:
        ah.handle_stop(SimpleNamespace())
    assert e.value.code == 1
    assert "Permission denied sending SIGTERM to pid=123." in capsys.readouterr().out


def test_status_prints_not_running_if_no_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "get_daemon_db_path", lambda: db_path)
    monkeypatch.setattr(ah, "read_pid_from_file", lambda p: None)

    ah.handle_status(SimpleNamespace())
    assert capsys.readouterr().out.strip() == "Daemon not running."


def test_status_cleans_up_stale_client_state_when_pid_not_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "get_daemon_db_path", lambda: db_path)
    monkeypatch.setattr(ah, "read_pid_from_file", lambda p: 777)
    monkeypatch.setattr(ah, "pid_is_running", lambda pid: False)

    called = {"cleanup": 0}

    def fake_cleanup_stale_client_state(**kwargs):
        called["cleanup"] += 1

    monkeypatch.setattr(
        ah, "cleanup_stale_client_state", fake_cleanup_stale_client_state
    )

    ah.handle_status(SimpleNamespace())
    assert called["cleanup"] == 1
    assert capsys.readouterr().out.strip() == "Daemon not running."


def test_status_prints_running_when_pid_is_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "get_daemon_db_path", lambda: db_path)
    monkeypatch.setattr(ah, "read_pid_from_file", lambda p: 456)
    monkeypatch.setattr(ah, "pid_is_running", lambda pid: True)

    ah.handle_status(SimpleNamespace())
    assert capsys.readouterr().out.strip() == "Daemon running (pid=456)."
