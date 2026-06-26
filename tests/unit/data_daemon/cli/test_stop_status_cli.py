from __future__ import annotations

from pathlib import Path

import pytest
import typer

import neuracore.data_daemon.config_manager.args_handler as ah


def test_stop_prints_not_running_if_no_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "get_daemon_db_path", lambda: db_path)
    monkeypatch.setattr(ah, "read_pid_from_file", lambda p: None)

    ah.run_stop()
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

    def fake_shutdown(*, pid_path, socket_paths, db_path):
        called["shutdown"] += 1

    monkeypatch.setattr(ah, "shutdown", fake_shutdown)

    ah.run_stop()
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

    ah.run_stop()
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
        return calls["n"] == 2

    monkeypatch.setattr(ah, "wait_for_exit", fake_wait_for_exit)
    monkeypatch.setattr(ah, "force_kill", lambda pid: True)

    shutdown_called = {"called": False}

    def fake_shutdown(*, pid_path, socket_paths, db_path):
        shutdown_called["called"] = True

    monkeypatch.setattr(ah, "shutdown", fake_shutdown)

    ah.run_stop()

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

    with pytest.raises(typer.Exit) as e:
        ah.run_stop()
    assert e.value.exit_code == 1
    assert "Permission denied sending SIGTERM to pid=123." in capsys.readouterr().err


def test_status_prints_not_running_if_no_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "get_daemon_db_path", lambda: db_path)
    monkeypatch.setattr(ah, "read_pid_from_file", lambda p: None)

    ah.run_status()
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

    ah.run_status()
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

    ah.run_status()
    assert capsys.readouterr().out.strip() == "Daemon running (pid=456)."


def test_reset_delegates_to_binary_and_passes_resolved_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "get_daemon_db_path", lambda: db_path)

    seen: dict[str, object] = {}

    def fake_reset_daemon_state(
        *, pid_path: Path, db_path: Path, assume_yes: bool
    ) -> int:
        seen["pid_path"] = pid_path
        seen["db_path"] = db_path
        seen["assume_yes"] = assume_yes
        return 0

    monkeypatch.setattr(ah, "reset_daemon_state", fake_reset_daemon_state)

    ah.run_reset(yes=True)
    assert seen == {"pid_path": pid_path, "db_path": db_path, "assume_yes": True}


def test_reset_defaults_to_prompting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: tmp_path / "daemon.pid")
    monkeypatch.setattr(ah, "get_daemon_db_path", lambda: tmp_path / "state.db")

    seen: dict[str, object] = {}

    def fake_reset_daemon_state(*, assume_yes: bool, **kwargs: object) -> int:
        seen["assume_yes"] = assume_yes
        return 0

    monkeypatch.setattr(ah, "reset_daemon_state", fake_reset_daemon_state)

    ah.run_reset()
    assert seen == {"assume_yes": False}


def test_reset_propagates_nonzero_exit_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: tmp_path / "daemon.pid")
    monkeypatch.setattr(ah, "get_daemon_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(ah, "reset_daemon_state", lambda **kwargs: 3)

    with pytest.raises(typer.Exit) as exit_info:
        ah.run_reset()
    assert exit_info.value.exit_code == 3


def test_reset_reports_missing_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: tmp_path / "daemon.pid")
    monkeypatch.setattr(ah, "get_daemon_db_path", lambda: tmp_path / "state.db")

    def fake_reset_daemon_state(**kwargs):
        raise ah.DaemonLifecycleError("no binary")

    monkeypatch.setattr(ah, "reset_daemon_state", fake_reset_daemon_state)

    with pytest.raises(typer.Exit) as exit_info:
        ah.run_reset()
    assert exit_info.value.exit_code == 1
    assert capsys.readouterr().err.strip() == "no binary"
