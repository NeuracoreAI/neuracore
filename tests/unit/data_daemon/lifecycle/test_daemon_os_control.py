from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import IO, cast

import pytest

import neuracore.data_daemon.lifecycle.daemon_os_control as daemon_os_control
from neuracore.data_daemon.helpers import bridge_sdk_org_id_env
from neuracore.data_daemon.lifecycle.daemon_os_control import (
    DaemonLifecycleError,
    acquire_pid_file,
    install_signal_handlers,
    launch_daemon_subprocess,
    remove_pid_file,
)


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
    assert env["NEURACORE_DAEMON_MANAGE_PID"] == "0"
    assert pid_path.read_text(encoding="utf-8").strip() == "54321"


def test_bridge_sdk_org_id_env_bridges_to_daemon_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEURACORE_ORG_ID", "org-from-sdk-env")
    monkeypatch.delenv("NCD_CURRENT_ORG_ID", raising=False)

    bridge_sdk_org_id_env()

    assert os.environ["NCD_CURRENT_ORG_ID"] == "org-from-sdk-env"


def test_bridge_sdk_org_id_env_keeps_explicit_daemon_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEURACORE_ORG_ID", "org-from-sdk-env")
    monkeypatch.setenv("NCD_CURRENT_ORG_ID", "org-explicit")

    bridge_sdk_org_id_env()

    assert os.environ["NCD_CURRENT_ORG_ID"] == "org-explicit"


def test_bridge_sdk_org_id_env_noop_without_sdk_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NEURACORE_ORG_ID", raising=False)
    monkeypatch.delenv("NCD_CURRENT_ORG_ID", raising=False)

    bridge_sdk_org_id_env()

    assert "NCD_CURRENT_ORG_ID" not in os.environ


def _stub_daemon_launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Stub out process/auth side effects so entry points run hermetically."""
    monkeypatch.setattr(
        daemon_os_control, "get_daemon_pid_path", lambda: tmp_path / "daemon.pid"
    )
    monkeypatch.setattr(
        daemon_os_control, "get_daemon_db_path", lambda: tmp_path / "state.db"
    )
    monkeypatch.setattr(
        daemon_os_control, "cleanup_stale_client_state", lambda **_: None
    )
    monkeypatch.setattr(daemon_os_control, "ensure_daemon_auth_ready", lambda *_: None)
    monkeypatch.setattr(
        daemon_os_control,
        "launch_daemon_subprocess",
        lambda **_: _FakePopen(pid=54321),
    )


def test_ensure_daemon_running_bridges_sdk_org_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NEURACORE_ORG_ID", "org-from-sdk-env")
    monkeypatch.delenv("NCD_CURRENT_ORG_ID", raising=False)
    _stub_daemon_launch(monkeypatch, tmp_path)

    daemon_os_control.ensure_daemon_running()

    assert os.environ["NCD_CURRENT_ORG_ID"] == "org-from-sdk-env"


def test_launch_new_daemon_subprocess_bridges_sdk_org_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NEURACORE_ORG_ID", "org-from-sdk-env")
    monkeypatch.delenv("NCD_CURRENT_ORG_ID", raising=False)
    _stub_daemon_launch(monkeypatch, tmp_path)

    daemon_os_control.launch_new_daemon_subprocess(
        pid_path=tmp_path / "daemon.pid",
        db_path=tmp_path / "state.db",
        background=True,
    )

    assert os.environ["NCD_CURRENT_ORG_ID"] == "org-from-sdk-env"


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


def test_launch_daemon_subprocess_premature_exit_includes_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    fake_socket_path = tmp_path / "management.sock"

    def fake_popen(command: list[str], **kwargs: object) -> _FakePopen:
        # The real daemon writes its failure to the stderr target before it
        # exits; emulate that so the parent can read it back from the log file.
        stderr_target = cast(IO[bytes], kwargs["stderr"])
        stderr_target.write(b"ImportError: No module named 'foo'")
        return _FakePopen(pid=99999, poll_value=1)

    monkeypatch.setattr(daemon_os_control.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon_os_control.time, "sleep", lambda _: None)
    monkeypatch.setattr(daemon_os_control, "SOCKET_PATH", fake_socket_path)

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
