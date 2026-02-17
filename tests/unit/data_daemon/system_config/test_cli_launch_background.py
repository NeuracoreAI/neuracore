from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import neuracore.data_daemon.config_manager.args_handler as ah


class FakePopen:
    def __init__(self, pid: int, poll_sequence: list[int | None]) -> None:
        self.pid = pid
        self._poll_sequence = poll_sequence[:]
        self.wait_called = False
        self.sent_sigint = False

    def poll(self) -> int | None:
        if self._poll_sequence:
            return self._poll_sequence.pop(0)
        return None

    def wait(self) -> int:
        self.wait_called = True
        return 0

    def send_signal(self, _sig: int) -> None:
        self.sent_sigint = True


def test_launch_background_exits_if_pid_is_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "read_pid_from_file", lambda p: 12345)
    monkeypatch.setattr(ah, "pid_is_running", lambda pid: True)

    args = SimpleNamespace(profile=None, background=True)
    with pytest.raises(SystemExit) as e:
        ah.handle_launch(args)  # type: ignore[arg-type]
    assert e.value.code == 1
    assert "Daemon already running (pid=12345)." in capsys.readouterr().out


def test_launch_unlinks_stale_pid_then_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text("999", encoding="utf-8")

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "read_pid_from_file", lambda p: 999)
    monkeypatch.setattr(ah, "pid_is_running", lambda pid: False)

    popen_obj = FakePopen(pid=22222, poll_sequence=[None, None])
    monkeypatch.setattr(ah.subprocess, "Popen", lambda *a, **k: popen_obj)

    args = SimpleNamespace(profile=None, background=True)
    ah.handle_launch(args)  # type: ignore[arg-type]

    assert pid_path.read_text(encoding="utf-8").strip() == "22222"


def test_launch_background_does_not_write_pid_if_child_exits_fast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "read_pid_from_file", lambda p: None)

    popen_obj = FakePopen(pid=33333, poll_sequence=[1])
    monkeypatch.setattr(ah.subprocess, "Popen", lambda *a, **k: popen_obj)

    args = SimpleNamespace(profile=None, background=True)
    with pytest.raises(SystemExit) as e:
        ah.handle_launch(args)  # type: ignore[arg-type]
    assert e.value.code == 1

    out = capsys.readouterr().out
    assert "Daemon failed to start." in out
    assert not pid_path.exists()


def test_launch_background_sets_expected_env_vars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "read_pid_from_file", lambda p: None)

    captured_env = {}

    def fake_popen(cmd, **kwargs):
        nonlocal captured_env
        captured_env = kwargs["env"]
        return FakePopen(pid=44444, poll_sequence=[None, None])

    monkeypatch.setattr(ah.subprocess, "Popen", fake_popen)

    args = SimpleNamespace(profile=None, background=True)
    ah.handle_launch(args)  # type: ignore[arg-type]

    assert captured_env["NEURACORE_DAEMON_PID_PATH"] == str(pid_path)
    assert captured_env["NEURACORE_DAEMON_MANAGE_PID"] == "0"


def test_launch_background_redirects_output_when_background_true(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "read_pid_from_file", lambda p: None)

    captured_kwargs = {}

    def fake_popen(cmd, **kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        return FakePopen(pid=55555, poll_sequence=[None, None])

    monkeypatch.setattr(ah.subprocess, "Popen", fake_popen)

    args = SimpleNamespace(profile=None, background=True)
    ah.handle_launch(args)  # type: ignore[arg-type]

    assert captured_kwargs["stdin"] is ah.subprocess.DEVNULL
    assert captured_kwargs["stdout"] is ah.subprocess.DEVNULL
    assert captured_kwargs["stderr"] is ah.subprocess.DEVNULL
    assert captured_kwargs["start_new_session"] is True


def test_launch_foreground_waits_and_sends_sigint_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "read_pid_from_file", lambda p: None)

    popen_obj = FakePopen(pid=66666, poll_sequence=[None, None])
    monkeypatch.setattr(ah.subprocess, "Popen", lambda *a, **k: popen_obj)

    calls = {"n": 0}

    def fake_wait() -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt
        return 0

    popen_obj.wait = fake_wait  # type: ignore[method-assign]

    args = SimpleNamespace(profile=None, background=False)
    ah.handle_launch(args)  # type: ignore[arg-type]

    assert popen_obj.sent_sigint is True


def test_launch_with_profile_sets_env_and_validates_profile_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ah, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(ah, "read_pid_from_file", lambda p: None)

    monkeypatch.setattr(ah.profile_manager, "get_profile", lambda name: object())

    captured_env = {}

    def fake_popen(cmd, **kwargs):
        nonlocal captured_env
        captured_env = kwargs["env"]
        return FakePopen(pid=77777, poll_sequence=[None, None])

    monkeypatch.setattr(ah.subprocess, "Popen", fake_popen)

    args = SimpleNamespace(profile="demo", background=True)
    ah.handle_launch(args)  # type: ignore[arg-type]

    assert captured_env["NEURACORE_DAEMON_PROFILE"] == "demo"
