from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest

import neuracore.data_daemon.lifecycle.daemon_os_control as daemon_os_control
from neuracore.data_daemon.lifecycle.daemon_os_control import (
    DaemonLifecycleError,
    acquire_pid_file,
    install_signal_handlers,
    launch_daemon_subprocess,
    remove_pid_file,
)


class _FakePopen:
    def __init__(self, pid: int = 12345, poll_value: int | None = None) -> None:
        self.pid = pid
        self._poll_value = poll_value

    def poll(self) -> int | None:
        return self._poll_value


def test_acquire_pid_file_rejects_running_pid(tmp_path: Path) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    with pytest.raises(DaemonLifecycleError):
        acquire_pid_file(pid_path)


def test_acquire_pid_file_clears_stale_pid(tmp_path: Path) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text("999999", encoding="utf-8")

    assert acquire_pid_file(pid_path) is True
    assert pid_path.read_text(encoding="utf-8").strip() == str(os.getpid())


def test_remove_pid_file_removes(tmp_path: Path) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text("123", encoding="utf-8")

    remove_pid_file(pid_path)

    assert not pid_path.exists()


def test_launch_daemon_subprocess_redirects_stdio_in_background(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    fake_socket_path = tmp_path / "management.sock"
    fake_socket_path.touch()
    captured: dict[str, object] = {}

    def fake_popen(command: list[str], **kwargs: object) -> _FakePopen:
        captured["command"] = command
        captured.update(kwargs)
        return _FakePopen(pid=54321, poll_value=None)

    monkeypatch.setattr(daemon_os_control.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon_os_control.time, "sleep", lambda _: None)
    monkeypatch.setattr(daemon_os_control, "SOCKET_PATH", fake_socket_path)

    proc = launch_daemon_subprocess(
        pid_path=pid_path,
        db_path=db_path,
        background=True,
    )

    assert proc.pid == 54321
    assert captured["start_new_session"] is True
    assert captured["stdin"] is daemon_os_control.subprocess.DEVNULL
    assert captured["stdout"] is daemon_os_control.subprocess.DEVNULL
    assert captured["stderr"] is daemon_os_control.subprocess.DEVNULL
    assert captured["close_fds"] is True
    assert captured["cwd"] == str(Path.cwd())

    env = captured["env"]
    assert isinstance(env, dict)
    assert env["NEURACORE_DAEMON_PID_PATH"] == str(pid_path)
    assert env["NEURACORE_DAEMON_DB_PATH"] == str(db_path)
    assert env["NEURACORE_DAEMON_MANAGE_PID"] == "0"
    assert pid_path.read_text(encoding="utf-8").strip() == "54321"


def test_launch_daemon_subprocess_keeps_foreground_stdio_attached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    fake_socket_path = tmp_path / "management.sock"
    fake_socket_path.touch()
    captured: dict[str, object] = {}

    def fake_popen(command: list[str], **kwargs: object) -> _FakePopen:
        captured["command"] = command
        captured.update(kwargs)
        return _FakePopen(pid=65432, poll_value=None)

    monkeypatch.setattr(daemon_os_control.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon_os_control.time, "sleep", lambda _: None)
    monkeypatch.setattr(daemon_os_control, "SOCKET_PATH", fake_socket_path)

    proc = launch_daemon_subprocess(
        pid_path=pid_path,
        db_path=db_path,
        background=False,
        env_overrides={"NEURACORE_DAEMON_PROFILE": "demo"},
    )

    assert proc.pid == 65432
    assert captured["start_new_session"] is False
    assert not captured.get("stdin")
    assert not captured.get("stdout")
    assert not captured.get("stderr")

    env = captured["env"]
    assert isinstance(env, dict)
    assert env["NEURACORE_DAEMON_PROFILE"] == "demo"
    assert pid_path.read_text(encoding="utf-8").strip() == "65432"


def test_install_signal_handlers_invokes_shutdown() -> None:
    called: list[int] = []

    def on_shutdown(signum: int) -> None:
        called.append(signum)

    orig_term = signal.getsignal(signal.SIGTERM)
    orig_int = signal.getsignal(signal.SIGINT)
    try:
        install_signal_handlers(on_shutdown=on_shutdown)
        handler = signal.getsignal(signal.SIGTERM)
        assert handler is not None
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGTERM, None)
        assert called == [signal.SIGTERM]
    finally:
        signal.signal(signal.SIGTERM, orig_term)
        signal.signal(signal.SIGINT, orig_int)
