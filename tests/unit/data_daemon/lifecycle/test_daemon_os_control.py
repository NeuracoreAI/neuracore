"""OS control coverage for the default (Rust) daemon backend.

Behaviour the legacy Python runner does differently — unix-socket readiness,
parent-owned PID file, its own argv — lives in
[test_python_daemon_os_control.py](test_python_daemon_os_control.py).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import IO, cast

import pytest

import neuracore.data_daemon.lifecycle.daemon_os_control as daemon_os_control
from neuracore.data_daemon.lifecycle.daemon_os_control import (
    DaemonLifecycleError,
    acquire_pid_file,
    install_signal_handlers,
    launch_daemon_subprocess,
    remove_pid_file,
    reset_daemon_state,
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
def rust_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Pin the module to the Rust daemon and return the fake binary path.

    Both the flag and the binary lookup are pinned so these tests do not depend
    on whether a bundled ``data-daemon`` binary happens to be installed on the
    machine running the suite.
    """
    binary_path = tmp_path / "data-daemon"
    monkeypatch.setattr(daemon_os_control, "is_rust_daemon_enabled", lambda: True)
    monkeypatch.setattr(
        daemon_os_control, "rust_daemon_binary_path", lambda: binary_path
    )
    return binary_path


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


def test_rust_launch_waits_for_ipc_health_not_pid_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    fake_pid = os.getpid()
    readiness_checks = iter([False, True])
    captured: dict[str, object] = {}

    def fake_popen(command: list[str], **kwargs: object) -> _FakePopen:
        captured["command"] = command
        captured.update(kwargs)
        pid_path.write_text(str(fake_pid), encoding="utf-8")
        return _FakePopen(pid=fake_pid, poll_value=None)

    monkeypatch.setattr(daemon_os_control.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon_os_control.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        daemon_os_control,
        "_rust_daemon_ready",
        lambda _pid_path, *, timeout_s=0.0: next(readiness_checks),
    )

    proc = launch_daemon_subprocess(
        pid_path=pid_path,
        db_path=db_path,
        background=True,
    )

    assert proc.pid == fake_pid
    env = captured["env"]
    assert isinstance(env, dict)
    assert "NEURACORE_DAEMON_MANAGE_PID" not in env


def test_rust_launch_times_out_when_only_pid_appears(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    fake_pid = os.getpid()

    def fake_popen(command: list[str], **kwargs: object) -> _FakePopen:
        pid_path.write_text(str(fake_pid), encoding="utf-8")
        return _FakePopen(pid=fake_pid, poll_value=None)

    monkeypatch.setattr(daemon_os_control.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        daemon_os_control, "_rust_daemon_ready", lambda *_args, **_kwargs: False
    )

    with pytest.raises(RuntimeError) as exc_info:
        launch_daemon_subprocess(
            pid_path=pid_path,
            db_path=db_path,
            background=True,
            timeout_s=0.0,
        )

    assert "Rust daemon IPC health probe" in str(exc_info.value)


def test_ensure_daemon_running_existing_rust_pid_waits_for_health(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    calls: list[float] = []

    def fake_ready(_pid_path: Path, *, timeout_s: float = 0.0) -> bool:
        calls.append(timeout_s)
        return True

    monkeypatch.setattr(daemon_os_control, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(daemon_os_control, "get_daemon_db_path", lambda: db_path)
    monkeypatch.setattr(daemon_os_control, "_rust_daemon_ready", fake_ready)

    assert daemon_os_control.ensure_daemon_running(timeout_s=2.5) == os.getpid()
    assert calls == [2.5]


def test_ensure_daemon_running_rejects_running_but_never_ready_daemon(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A live PID with a dead IPC listener must fail rather than look healthy."""
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    monkeypatch.setattr(daemon_os_control, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(daemon_os_control, "get_daemon_db_path", lambda: db_path)
    monkeypatch.setattr(
        daemon_os_control, "_rust_daemon_ready", lambda *_args, **_kwargs: False
    )

    with pytest.raises(DaemonLifecycleError) as exc_info:
        daemon_os_control.ensure_daemon_running(timeout_s=0.0)

    assert "did not become ready" in str(exc_info.value)


def test_launch_daemon_subprocess_runs_bundled_binary_and_redirects_stdio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rust_mode: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    captured: dict[str, object] = {}

    def fake_popen(command: list[str], **kwargs: object) -> _FakePopen:
        captured["command"] = command
        captured.update(kwargs)
        return _FakePopen(pid=54321, poll_value=None)

    monkeypatch.setattr(daemon_os_control.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon_os_control.time, "sleep", lambda _: None)
    monkeypatch.setattr(daemon_os_control, "_rust_daemon_ready", lambda *_, **__: True)

    proc = launch_daemon_subprocess(
        pid_path=pid_path,
        db_path=db_path,
        background=True,
    )

    assert proc.pid == 54321
    assert captured["command"] == [str(rust_mode), "launch"]
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
    # The Rust binary owns its PID file, so the parent must neither claim it via
    # NEURACORE_DAEMON_MANAGE_PID=0 nor write it itself.
    assert "NEURACORE_DAEMON_MANAGE_PID" not in env
    assert not pid_path.exists()


def test_launch_daemon_subprocess_keeps_foreground_stdio_attached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    captured: dict[str, object] = {}

    def fake_popen(command: list[str], **kwargs: object) -> _FakePopen:
        captured["command"] = command
        captured.update(kwargs)
        return _FakePopen(pid=65432, poll_value=None)

    monkeypatch.setattr(daemon_os_control.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon_os_control.time, "sleep", lambda _: None)
    monkeypatch.setattr(daemon_os_control, "_rust_daemon_ready", lambda *_, **__: True)

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
    assert "NEURACORE_DAEMON_MANAGE_PID" not in env
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
        stderr_target.write(b"failed to open iceoryx2 service")
        return _FakePopen(pid=99999, poll_value=1)

    monkeypatch.setattr(daemon_os_control.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon_os_control.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError) as exc_info:
        launch_daemon_subprocess(pid_path=pid_path, db_path=db_path, background=True)

    message = str(exc_info.value)
    assert "exit code 1" in message
    assert "failed to open iceoryx2 service" in message


def _install_fake_bridge(
    monkeypatch: pytest.MonkeyPatch, wait_until_ready: object
) -> None:
    """Expose a stand-in ``_data_bridge`` module to the health probe.

    ``importlib.import_module`` returns an existing ``sys.modules`` entry, so
    seeding one lets these tests drive the probe without a built extension.
    """
    monkeypatch.setitem(
        sys.modules,
        "neuracore.data_daemon._data_bridge",
        SimpleNamespace(wait_until_ready=wait_until_ready),
    )


def test_rust_daemon_ready_returns_false_without_pid_file(tmp_path: Path) -> None:
    assert daemon_os_control._rust_daemon_ready(tmp_path / "missing.pid") is False


def test_rust_daemon_ready_returns_false_for_dead_pid(tmp_path: Path) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text(str(DEAD_PID), encoding="utf-8")

    assert daemon_os_control._rust_daemon_ready(pid_path) is False


def test_rust_daemon_ready_returns_false_when_bridge_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A source install without the built extension must not report readiness."""
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    def raise_import_error(_name: str) -> object:
        raise ImportError("no module named _data_bridge")

    monkeypatch.setattr(
        daemon_os_control,
        "importlib",
        SimpleNamespace(import_module=raise_import_error),
    )

    assert daemon_os_control._rust_daemon_ready(pid_path) is False


@pytest.mark.parametrize(
    "error", [RuntimeError("probe failed"), AttributeError("gone")]
)
def test_rust_daemon_ready_returns_false_when_probe_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: Exception
) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    def raise_error(_timeout_s: float) -> int:
        raise error

    _install_fake_bridge(monkeypatch, raise_error)

    assert daemon_os_control._rust_daemon_ready(pid_path) is False


def test_rust_daemon_ready_returns_false_when_probe_answers_other_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale PID file must not be validated by a different live daemon."""
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    _install_fake_bridge(monkeypatch, lambda _timeout_s: os.getpid() + 1)

    assert daemon_os_control._rust_daemon_ready(pid_path) is False


def test_rust_daemon_ready_returns_true_when_probe_answers_same_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    timeouts: list[float] = []

    def fake_wait_until_ready(timeout_s: float) -> int:
        timeouts.append(timeout_s)
        return os.getpid()

    _install_fake_bridge(monkeypatch, fake_wait_until_ready)

    assert daemon_os_control._rust_daemon_ready(pid_path, timeout_s=1.5) is True
    assert timeouts == [1.5]


def test_reset_daemon_state_delegates_to_bundled_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rust_mode: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        captured["command"] = command
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 3)

    monkeypatch.setattr(daemon_os_control.subprocess, "run", fake_run)

    exit_code = reset_daemon_state(pid_path=pid_path, db_path=db_path, assume_yes=True)

    assert exit_code == 3
    assert captured["command"] == [str(rust_mode), "reset", "--yes"]
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["NEURACORE_DAEMON_PID_PATH"] == str(pid_path)
    assert env["NEURACORE_DAEMON_DB_PATH"] == str(db_path)


def test_reset_daemon_state_without_assume_yes_keeps_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rust_mode: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(daemon_os_control.subprocess, "run", fake_run)

    assert (
        reset_daemon_state(
            pid_path=tmp_path / "daemon.pid",
            db_path=tmp_path / "state.db",
            assume_yes=False,
        )
        == 0
    )
    assert captured["command"] == [str(rust_mode), "reset"]


def test_reset_daemon_state_requires_bundled_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(daemon_os_control, "rust_daemon_binary_path", lambda: None)

    with pytest.raises(DaemonLifecycleError):
        reset_daemon_state(
            pid_path=tmp_path / "daemon.pid",
            db_path=tmp_path / "state.db",
            assume_yes=True,
        )


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
