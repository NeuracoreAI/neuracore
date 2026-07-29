"""OS control coverage for the legacy Python daemon backend.

Every test here runs with the daemon pinned to the Python runner, including the
mode-agnostic ones (PID file, signal handlers, premature exit) that
[test_daemon_os_control.py](test_daemon_os_control.py) also runs against the
default Rust backend — so neither backend can regress them alone. What is unique
to this file is the Python runner's own launch contract: unix-socket readiness,
a parent-owned PID file, and its own argv.
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from typing import IO, cast

import pytest

import neuracore.data_daemon.lifecycle.daemon_os_control as daemon_os_control
from neuracore.data_daemon.lifecycle.daemon_os_control import (
    DaemonLifecycleError,
    acquire_pid_file,
    install_signal_handlers,
    launch_daemon_subprocess,
    remove_pid_file,
)

DEAD_PID = 999999


class _FakePopen:
    def __init__(
        self,
        pid: int = 12345,
        poll_value: int | None = None,
    ) -> None:
        self.pid = pid
        self._poll_value = poll_value
        self.returncode = poll_value
        self.stderr = None

    def poll(self) -> int | None:
        return self._poll_value

    def terminate(self) -> None:
        self.returncode = -15
        self._poll_value = -15


@pytest.fixture(autouse=True)
def python_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Pin the module to the Python daemon and return its readiness socket path.

    The socket is not created — a test signals readiness by touching it — and the
    flag is pinned so these tests stay on the legacy path regardless of whether a
    bundled ``data-daemon`` binary is installed on the machine running the suite.
    """
    socket_path = tmp_path / "management.sock"
    monkeypatch.setattr(daemon_os_control, "is_rust_daemon_enabled", lambda: False)
    monkeypatch.setattr(daemon_os_control, "SOCKET_PATH", socket_path)
    return socket_path


def test_acquire_pid_file_rejects_running_pid(tmp_path: Path) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    with pytest.raises(DaemonLifecycleError):
        acquire_pid_file(pid_path)


def test_acquire_pid_file_clears_stale_pid(tmp_path: Path) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text(str(DEAD_PID), encoding="utf-8")

    assert acquire_pid_file(pid_path) is True
    assert pid_path.read_text(encoding="utf-8").strip() == str(os.getpid())


def test_remove_pid_file_removes(tmp_path: Path) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text("123", encoding="utf-8")

    remove_pid_file(pid_path)

    assert not pid_path.exists()


def test_launch_daemon_subprocess_runs_python_runner_and_redirects_stdio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, python_mode: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    python_mode.touch()
    captured: dict[str, object] = {}

    def fake_popen(command: list[str], **kwargs: object) -> _FakePopen:
        captured["command"] = command
        captured.update(kwargs)
        return _FakePopen(pid=54321, poll_value=None)

    monkeypatch.setattr(daemon_os_control.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon_os_control.time, "sleep", lambda _: None)

    proc = launch_daemon_subprocess(
        pid_path=pid_path,
        db_path=db_path,
        background=True,
    )

    assert proc.pid == 54321
    assert captured["command"] == [
        sys.executable,
        "-m",
        "neuracore.data_daemon.runner_entry",
    ]
    assert captured["start_new_session"] is True
    assert captured["stdin"] is daemon_os_control.subprocess.DEVNULL
    assert captured["stdout"] is daemon_os_control.subprocess.DEVNULL
    # Background stderr is routed to a sibling log file (not an undrained PIPE
    # that would deadlock the daemon, nor DEVNULL that would hide failures).
    stderr_target = captured["stderr"]
    assert stderr_target is not daemon_os_control.subprocess.PIPE
    assert stderr_target is not daemon_os_control.subprocess.DEVNULL
    assert Path(stderr_target.name) == db_path.parent / "daemon.log"
    assert (db_path.parent / "daemon.log").exists()
    assert captured["close_fds"] is True
    assert captured["cwd"] == str(Path.cwd())

    env = captured["env"]
    assert isinstance(env, dict)
    assert env["NEURACORE_DAEMON_PID_PATH"] == str(pid_path)
    assert env["NEURACORE_DAEMON_DB_PATH"] == str(db_path)
    # The Python runner relies on the parent for the PID file, both for the
    # hand-off flag and for the write itself.
    assert env["NEURACORE_DAEMON_MANAGE_PID"] == "0"
    assert pid_path.read_text(encoding="utf-8").strip() == "54321"


def test_launch_daemon_subprocess_keeps_foreground_stdio_attached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, python_mode: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    python_mode.touch()
    captured: dict[str, object] = {}

    def fake_popen(command: list[str], **kwargs: object) -> _FakePopen:
        captured["command"] = command
        captured.update(kwargs)
        return _FakePopen(pid=65432, poll_value=None)

    monkeypatch.setattr(daemon_os_control.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon_os_control.time, "sleep", lambda _: None)

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
    assert env["NEURACORE_DAEMON_MANAGE_PID"] == "0"
    assert pid_path.read_text(encoding="utf-8").strip() == "65432"


def test_launch_daemon_subprocess_times_out_when_socket_never_appears(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, python_mode: Path
) -> None:
    """The Python counterpart of the Rust IPC-probe timeout names the socket."""
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"

    def fake_popen(command: list[str], **kwargs: object) -> _FakePopen:
        return _FakePopen(pid=54321, poll_value=None)

    monkeypatch.setattr(daemon_os_control.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon_os_control.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError) as exc_info:
        launch_daemon_subprocess(
            pid_path=pid_path,
            db_path=db_path,
            background=True,
            timeout_s=0.0,
        )

    assert f"socket {python_mode}" in str(exc_info.value)
    assert not pid_path.exists()


def test_launch_daemon_subprocess_premature_exit_includes_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"

    def fake_popen(command: list[str], **kwargs: object) -> _FakePopen:
        # The real daemon writes its failure to the stderr target before it
        # exits; emulate that so the parent can read it back from the log file.
        stderr_target = cast(IO[bytes], kwargs["stderr"])
        stderr_target.write(b"ImportError: No module named 'foo'")
        return _FakePopen(pid=99999, poll_value=1)

    monkeypatch.setattr(daemon_os_control.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon_os_control.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError) as exc_info:
        launch_daemon_subprocess(pid_path=pid_path, db_path=db_path, background=True)

    message = str(exc_info.value)
    assert "exit code 1" in message
    assert "ImportError: No module named 'foo'" in message


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
