from __future__ import annotations

from pathlib import Path

import pytest
import typer

import neuracore.data_daemon.config_manager.args_handler as ah


class FakePopen:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.sent_sigint = False

    def wait(self) -> int:
        return 0

    def send_signal(self, _sig: int) -> None:
        self.sent_sigint = True


def test_launch_background_exits_if_pid_is_running(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "get_daemon_db_path", lambda: db_path)

    def fake_launch(**_: object) -> FakePopen:
        raise ah.DaemonLifecycleError("Daemon already running (pid=12345).")

    monkeypatch.setattr(ah, "launch_new_daemon_subprocess", fake_launch)

    with pytest.raises(typer.Exit) as e:
        ah.run_launch(profile=None, background=True)

    assert e.value.exit_code == 1
    assert "Daemon already running (pid=12345)." in capsys.readouterr().err


def test_launch_background_propagates_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "get_daemon_db_path", lambda: db_path)

    def fake_launch(**_: object) -> FakePopen:
        raise RuntimeError("Daemon failed to start.")

    monkeypatch.setattr(ah, "launch_new_daemon_subprocess", fake_launch)

    with pytest.raises(typer.Exit) as e:
        ah.run_launch(profile=None, background=True)

    assert e.value.exit_code == 1
    assert "Daemon failed to start." in capsys.readouterr().err


def test_launch_background_passes_expected_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "get_daemon_db_path", lambda: db_path)

    captured_kwargs: dict[str, object] = {}

    def fake_launch(**kwargs: object) -> FakePopen:
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        return FakePopen(pid=44444)

    monkeypatch.setattr(ah, "launch_new_daemon_subprocess", fake_launch)

    ah.run_launch(profile=None, background=True)

    assert captured_kwargs["pid_path"] == pid_path
    assert captured_kwargs["db_path"] == db_path
    assert captured_kwargs["background"] is True
    assert captured_kwargs["env_overrides"] is None


def test_launch_foreground_passes_background_false(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "get_daemon_db_path", lambda: db_path)

    captured_kwargs: dict[str, object] = {}

    def fake_launch(**kwargs: object) -> FakePopen:
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        return FakePopen(pid=55555)

    monkeypatch.setattr(ah, "launch_new_daemon_subprocess", fake_launch)

    ah.run_launch(profile=None, background=False)

    assert captured_kwargs["background"] is False


def test_launch_foreground_waits_and_sends_sigint_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "get_daemon_db_path", lambda: db_path)

    popen_obj = FakePopen(pid=66666)

    def fake_launch(**_: object) -> FakePopen:
        return popen_obj

    monkeypatch.setattr(ah, "launch_new_daemon_subprocess", fake_launch)

    calls = {"n": 0}

    def fake_wait() -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt
        return 0

    popen_obj.wait = fake_wait  # type: ignore[method-assign]

    ah.run_launch(profile=None, background=False)

    assert popen_obj.sent_sigint is True


def test_launch_with_profile_sets_env_and_validates_profile_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "get_daemon_db_path", lambda: db_path)
    monkeypatch.setattr(ah.profile_manager, "get_profile", lambda name: object())

    captured_kwargs: dict[str, object] = {}

    def fake_launch(**kwargs: object) -> FakePopen:
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        return FakePopen(pid=77777)

    monkeypatch.setattr(ah, "launch_new_daemon_subprocess", fake_launch)

    ah.run_launch(profile="demo", background=True)

    assert captured_kwargs["env_overrides"] == {"NEURACORE_DAEMON_PROFILE": "demo"}


def test_launch_with_missing_profile_exits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "get_daemon_db_path", lambda: db_path)

    def fake_get_profile(name: str) -> object:
        raise ah.ProfileNotFound(f"Profile {name!r} not found.")

    monkeypatch.setattr(ah.profile_manager, "get_profile", fake_get_profile)

    with pytest.raises(typer.Exit) as e:
        ah.run_launch(profile="missing-prof", background=True)

    assert e.value.exit_code == 1
    assert "Profile 'missing-prof' not found." in capsys.readouterr().err
