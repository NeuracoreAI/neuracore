"""OS control coverage for launching and probing the data daemon."""

from __future__ import annotations

import importlib.metadata
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import IO, cast

import pytest

import neuracore.data_daemon.daemon_control as daemon_control
from neuracore.data_daemon.daemon_control import (
    DaemonLifecycleError,
    DaemonVersionMismatchError,
    launch_daemon_subprocess,
)

DEAD_PID = 999999
# Not the repo's real version, so a test that stops using the _sdk_version
# seam fails instead of matching the genuine installed metadata.
INSTALLED_SDK_VERSION = "99.9.9"
STALE_DAEMON_VERSION = "13.0.1"


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
        self.wait_calls = 0

    def poll(self) -> int | None:
        return self._poll_value

    def terminate(self) -> None:
        self.returncode = -15
        self._poll_value = -15

    def wait(self, timeout: float | None = None) -> int | None:
        self.wait_calls += 1
        return self.returncode


@pytest.fixture(autouse=True)
def bundled_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Pin the binary lookup so tests do not depend on a built install."""
    binary_path = tmp_path / "data-daemon"
    monkeypatch.setattr(
        daemon_control, "require_data_daemon_binary", lambda: binary_path
    )
    monkeypatch.setattr(daemon_control, "data_daemon_binary_path", lambda: binary_path)
    return binary_path


def test_launch_waits_for_ipc_health_not_pid_only(
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

    monkeypatch.setattr(daemon_control.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon_control.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        daemon_control,
        "_daemon_ready",
        lambda _pid_path, *, timeout_s=0.0: next(readiness_checks),
    )

    proc = launch_daemon_subprocess(
        pid_path=pid_path,
        db_path=db_path,
        background=True,
    )

    assert proc.pid == fake_pid


def test_launch_times_out_when_only_pid_appears(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    fake_pid = os.getpid()

    def fake_popen(command: list[str], **kwargs: object) -> _FakePopen:
        pid_path.write_text(str(fake_pid), encoding="utf-8")
        return _FakePopen(pid=fake_pid, poll_value=None)

    monkeypatch.setattr(daemon_control.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        daemon_control, "_daemon_ready", lambda *_args, **_kwargs: False
    )

    with pytest.raises(RuntimeError) as exc_info:
        launch_daemon_subprocess(
            pid_path=pid_path,
            db_path=db_path,
            background=True,
            timeout_s=0.0,
        )

    assert "IPC health probe" in str(exc_info.value)


def test_ensure_daemon_running_existing_pid_waits_for_health(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    calls: list[float] = []

    def fake_ready(_pid_path: Path, *, timeout_s: float = 0.0) -> bool:
        calls.append(timeout_s)
        return True

    monkeypatch.setattr(daemon_control, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(daemon_control, "get_daemon_db_path", lambda: db_path)
    monkeypatch.setattr(daemon_control, "_daemon_ready", fake_ready)
    _install_matching_version_daemon(monkeypatch)

    assert daemon_control.ensure_daemon_running(timeout_s=2.5) == os.getpid()
    assert calls == [2.5]


def test_ensure_daemon_running_rejects_running_but_never_ready_daemon(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A live PID with a dead IPC listener must fail rather than look healthy."""
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    monkeypatch.setattr(daemon_control, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(daemon_control, "get_daemon_db_path", lambda: db_path)
    monkeypatch.setattr(
        daemon_control, "_daemon_ready", lambda *_args, **_kwargs: False
    )

    with pytest.raises(DaemonLifecycleError) as exc_info:
        daemon_control.ensure_daemon_running(timeout_s=0.0)

    assert "did not become ready" in str(exc_info.value)


def test_launch_runs_bundled_binary_and_redirects_stdio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bundled_binary: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    captured: dict[str, object] = {}

    def fake_popen(command: list[str], **kwargs: object) -> _FakePopen:
        captured["command"] = command
        captured.update(kwargs)
        return _FakePopen(pid=54321, poll_value=None)

    monkeypatch.setattr(daemon_control.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon_control.time, "sleep", lambda _: None)
    monkeypatch.setattr(daemon_control, "_daemon_ready", lambda *_, **__: True)

    proc = launch_daemon_subprocess(
        pid_path=pid_path,
        db_path=db_path,
        background=True,
    )

    assert proc.pid == 54321
    assert captured["command"] == [str(bundled_binary), "launch"]
    assert captured["start_new_session"] is True
    assert captured["stdin"] is daemon_control.subprocess.DEVNULL
    assert captured["stdout"] is daemon_control.subprocess.DEVNULL
    # Background stderr is routed to a sibling log file (not an undrained PIPE
    # that would deadlock the daemon, nor DEVNULL that would hide failures).
    stderr_target = captured["stderr"]
    assert stderr_target is not daemon_control.subprocess.PIPE
    assert stderr_target is not daemon_control.subprocess.DEVNULL
    assert Path(stderr_target.name) == db_path.parent / "daemon.log"
    assert (db_path.parent / "daemon.log").exists()
    assert captured["close_fds"] is True
    assert captured["cwd"] == str(Path.cwd())

    env = captured["env"]
    assert isinstance(env, dict)
    assert env["NEURACORE_DAEMON_PID_PATH"] == str(pid_path)
    assert env["NEURACORE_DAEMON_DB_PATH"] == str(db_path)
    # The daemon owns its PID file, so the parent must not write it itself.
    assert not pid_path.exists()


def test_launch_keeps_foreground_stdio_attached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    captured: dict[str, object] = {}

    def fake_popen(command: list[str], **kwargs: object) -> _FakePopen:
        captured["command"] = command
        captured.update(kwargs)
        return _FakePopen(pid=65432, poll_value=None)

    monkeypatch.setattr(daemon_control.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon_control.time, "sleep", lambda _: None)
    monkeypatch.setattr(daemon_control, "_daemon_ready", lambda *_, **__: True)

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
    assert not pid_path.exists()


def test_launch_premature_exit_includes_stderr(
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

    monkeypatch.setattr(daemon_control.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon_control.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError) as exc_info:
        launch_daemon_subprocess(pid_path=pid_path, db_path=db_path, background=True)

    message = str(exc_info.value)
    assert "exit code 1" in message
    assert "failed to open iceoryx2 service" in message


def _install_fake_bridge(
    monkeypatch: pytest.MonkeyPatch,
    wait_until_ready: object,
    daemon_version: object | None = None,
) -> None:
    """Expose a stand-in ``_data_bridge`` module to the health probe.

    ``importlib.import_module`` returns an existing ``sys.modules`` entry, so
    seeding one lets these tests drive the probe without a built extension.
    A ``daemon_version`` of ``None`` leaves that name off the module, which is
    what an extension built before the version query looks like.
    """
    fake_bridge = SimpleNamespace(wait_until_ready=wait_until_ready)
    if daemon_version is not None:
        fake_bridge.daemon_version = daemon_version
    monkeypatch.setitem(sys.modules, "neuracore.data_daemon._data_bridge", fake_bridge)


def _install_sdk_version(monkeypatch: pytest.MonkeyPatch, sdk_version: str) -> None:
    """Pin the version the installed neuracore package reports."""
    monkeypatch.setattr(daemon_control, "_sdk_version", lambda: sdk_version)


def _install_matching_version_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the fake daemon report exactly the installed neuracore version."""
    _install_sdk_version(monkeypatch, INSTALLED_SDK_VERSION)
    _install_fake_bridge(
        monkeypatch,
        lambda _timeout_s: os.getpid(),
        daemon_version=lambda _timeout_s: INSTALLED_SDK_VERSION,
    )


def _use_live_daemon_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the lifecycle helpers at a PID file holding a live PID."""
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    monkeypatch.setattr(daemon_control, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(
        daemon_control, "get_daemon_db_path", lambda: tmp_path / "state.db"
    )


def test_daemon_ready_returns_false_without_pid_file(tmp_path: Path) -> None:
    assert daemon_control._daemon_ready(tmp_path / "missing.pid") is False


def test_daemon_ready_returns_false_for_dead_pid(tmp_path: Path) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text(str(DEAD_PID), encoding="utf-8")

    assert daemon_control._daemon_ready(pid_path) is False


def test_daemon_ready_returns_false_when_bridge_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A source install without the built extension must not report readiness."""
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    def raise_import_error(_name: str) -> object:
        raise ImportError("no module named _data_bridge")

    monkeypatch.setattr(
        daemon_control,
        "importlib",
        SimpleNamespace(import_module=raise_import_error),
    )

    assert daemon_control._daemon_ready(pid_path) is False


@pytest.mark.parametrize(
    "error", [RuntimeError("probe failed"), AttributeError("gone")]
)
def test_daemon_ready_returns_false_when_probe_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: Exception
) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    def raise_error(_timeout_s: float) -> int:
        raise error

    _install_fake_bridge(monkeypatch, raise_error)

    assert daemon_control._daemon_ready(pid_path) is False


def test_daemon_ready_returns_false_when_probe_answers_other_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale PID file must not be validated by a different live daemon."""
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    _install_fake_bridge(monkeypatch, lambda _timeout_s: os.getpid() + 1)

    assert daemon_control._daemon_ready(pid_path) is False


def test_daemon_ready_returns_true_when_probe_answers_same_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    timeouts: list[float] = []

    def fake_wait_until_ready(timeout_s: float) -> int:
        timeouts.append(timeout_s)
        return os.getpid()

    _install_fake_bridge(monkeypatch, fake_wait_until_ready)

    assert daemon_control._daemon_ready(pid_path, timeout_s=1.5) is True
    assert timeouts == [1.5]


def test_ensure_daemon_running_adopts_daemon_built_from_the_installed_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A daemon reporting the installed neuracore version is used as it is."""
    _use_live_daemon_paths(monkeypatch, tmp_path)
    _install_sdk_version(monkeypatch, INSTALLED_SDK_VERSION)
    version_timeouts: list[float] = []

    def fake_daemon_version(timeout_s: float) -> str:
        version_timeouts.append(timeout_s)
        return INSTALLED_SDK_VERSION

    _install_fake_bridge(
        monkeypatch,
        lambda _timeout_s: os.getpid(),
        daemon_version=fake_daemon_version,
    )

    assert daemon_control.ensure_daemon_running(timeout_s=2.5) == os.getpid()
    # The query gets the same time budget as the health probe, so a daemon
    # that is slow to answer is not misread as one without a version service.
    assert version_timeouts == [2.5]


def test_ensure_daemon_running_accepts_an_alpha_daemon(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An alpha wheel ships the two spellings of one version, not two versions."""
    _use_live_daemon_paths(monkeypatch, tmp_path)
    _install_sdk_version(monkeypatch, "99.9.9a7")
    _install_fake_bridge(
        monkeypatch,
        lambda _timeout_s: os.getpid(),
        daemon_version=lambda _timeout_s: "99.9.9-alpha.7",
    )

    assert daemon_control.ensure_daemon_running(timeout_s=0.0) == os.getpid()


def test_ensure_daemon_running_rejects_a_daemon_from_another_alpha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two alphas of the same release are still two different builds."""
    _use_live_daemon_paths(monkeypatch, tmp_path)
    _install_sdk_version(monkeypatch, "99.9.9a7")
    _install_fake_bridge(
        monkeypatch,
        lambda _timeout_s: os.getpid(),
        daemon_version=lambda _timeout_s: "99.9.9-alpha.6",
    )

    with pytest.raises(DaemonVersionMismatchError):
        daemon_control.ensure_daemon_running(timeout_s=0.0)


def test_ensure_daemon_running_rejects_daemon_built_from_another_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A daemon left over from an earlier install must be named, not adopted."""
    _use_live_daemon_paths(monkeypatch, tmp_path)
    _install_sdk_version(monkeypatch, INSTALLED_SDK_VERSION)
    _install_fake_bridge(
        monkeypatch,
        lambda _timeout_s: os.getpid(),
        daemon_version=lambda _timeout_s: STALE_DAEMON_VERSION,
    )

    with pytest.raises(DaemonVersionMismatchError) as exc_info:
        daemon_control.ensure_daemon_running(timeout_s=0.0)

    assert isinstance(exc_info.value, DaemonLifecycleError)
    message = str(exc_info.value)
    assert STALE_DAEMON_VERSION in message
    assert INSTALLED_SDK_VERSION in message
    assert "neuracore data-daemon stop" in message
    assert "build_wheel_artefacts.sh" in message


def test_ensure_daemon_running_rejects_daemon_that_answers_no_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A healthy daemon that never answers the version query is too old."""
    _use_live_daemon_paths(monkeypatch, tmp_path)
    _install_sdk_version(monkeypatch, INSTALLED_SDK_VERSION)
    _install_fake_bridge(
        monkeypatch,
        lambda _timeout_s: os.getpid(),
        daemon_version=lambda _timeout_s: None,
    )

    with pytest.raises(DaemonVersionMismatchError) as exc_info:
        daemon_control.ensure_daemon_running(timeout_s=0.0)

    message = str(exc_info.value)
    assert daemon_control.UNKNOWN_DAEMON_VERSION in message
    assert INSTALLED_SDK_VERSION in message
    assert "neuracore data-daemon stop" in message


def test_ensure_daemon_running_rejects_extension_without_the_version_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An extension without the query names itself as stale, not the daemon."""
    _use_live_daemon_paths(monkeypatch, tmp_path)
    _install_sdk_version(monkeypatch, INSTALLED_SDK_VERSION)
    _install_fake_bridge(monkeypatch, lambda _timeout_s: os.getpid())

    with pytest.raises(DaemonLifecycleError) as exc_info:
        daemon_control.ensure_daemon_running(timeout_s=0.0)

    assert not isinstance(exc_info.value, DaemonVersionMismatchError)
    message = str(exc_info.value)
    assert "native extension" in message
    assert "build_wheel_artefacts.sh" in message


def test_version_check_is_skipped_when_the_query_fails_in_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An IPC failure is not evidence of a version mismatch."""
    _use_live_daemon_paths(monkeypatch, tmp_path)
    _install_sdk_version(monkeypatch, INSTALLED_SDK_VERSION)

    def raise_runtime_error(_timeout_s: float) -> str:
        raise RuntimeError("iceoryx2 node limit reached")

    _install_fake_bridge(
        monkeypatch,
        lambda _timeout_s: os.getpid(),
        daemon_version=raise_runtime_error,
    )

    with caplog.at_level("WARNING"):
        assert daemon_control.ensure_daemon_running(timeout_s=0.0) == os.getpid()

    assert "Skipped the daemon version check" in caplog.text


def test_ensure_daemon_running_checks_the_daemon_it_has_just_launched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale local build must be refused, not accepted because it is new.

    In an editable install the bundled binary can be older than the installed
    package, so the daemon this call starts can report an earlier version.
    """
    pid_path = tmp_path / "daemon.pid"
    launched_pid = 24680
    launched_processes: list[_FakePopen] = []

    def fake_popen(command: list[str], **kwargs: object) -> _FakePopen:
        process = _FakePopen(pid=launched_pid, poll_value=None)
        launched_processes.append(process)
        return process

    monkeypatch.setattr(daemon_control, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(
        daemon_control, "get_daemon_db_path", lambda: tmp_path / "state.db"
    )
    monkeypatch.setattr(daemon_control.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon_control, "_daemon_ready", lambda *_, **__: True)
    _install_sdk_version(monkeypatch, INSTALLED_SDK_VERSION)
    version_timeouts: list[float] = []

    def fake_daemon_version(timeout_s: float) -> str:
        version_timeouts.append(timeout_s)
        return STALE_DAEMON_VERSION

    _install_fake_bridge(
        monkeypatch,
        lambda _timeout_s: launched_pid,
        daemon_version=fake_daemon_version,
    )

    with pytest.raises(DaemonVersionMismatchError) as exc_info:
        daemon_control.ensure_daemon_running(timeout_s=1.0)

    message = str(exc_info.value)
    assert STALE_DAEMON_VERSION in message
    assert INSTALLED_SDK_VERSION in message
    assert "build_wheel_artefacts.sh" in message
    # The launch path must give the query the caller's budget as well.
    assert version_timeouts == [1.0]
    # The mismatched daemon this call started must not be left running, and
    # must be reaped so no zombie remains.
    assert [process.returncode for process in launched_processes] == [-15]
    assert [process.wait_calls for process in launched_processes] == [1]


def test_version_mismatch_message_asks_for_an_install_without_a_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An install with no daemon binary cannot start a correct daemon."""
    _use_live_daemon_paths(monkeypatch, tmp_path)
    _install_sdk_version(monkeypatch, INSTALLED_SDK_VERSION)
    monkeypatch.setattr(daemon_control, "data_daemon_binary_path", lambda: None)
    _install_fake_bridge(
        monkeypatch,
        lambda _timeout_s: os.getpid(),
        daemon_version=lambda _timeout_s: STALE_DAEMON_VERSION,
    )

    with pytest.raises(DaemonVersionMismatchError) as exc_info:
        daemon_control.ensure_daemon_running(timeout_s=0.0)

    message = str(exc_info.value)
    # The install must come first: the stop command itself needs the binary.
    assert message.index("pip install --force-reinstall neuracore") < message.index(
        "data-daemon stop"
    )


def test_version_check_is_skipped_without_package_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A bare source tree has no version to compare, so the check must not run.

    Raising here would reject every daemon forever: no stop, reinstall, or
    rebuild can give the package metadata it never had.
    """
    _use_live_daemon_paths(monkeypatch, tmp_path)

    def raise_package_not_found(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError("neuracore")

    monkeypatch.setattr(importlib.metadata, "version", raise_package_not_found)
    _install_fake_bridge(
        monkeypatch,
        lambda _timeout_s: os.getpid(),
        daemon_version=lambda _timeout_s: STALE_DAEMON_VERSION,
    )

    with caplog.at_level("WARNING"):
        assert daemon_control.ensure_daemon_running(timeout_s=0.0) == os.getpid()

    assert daemon_control._sdk_version() is None
    assert "no installed metadata" in caplog.text


def test_version_check_is_skipped_when_the_extension_import_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The check cannot run without the native extension, so it must not raise."""
    _install_sdk_version(monkeypatch, INSTALLED_SDK_VERSION)

    def raise_import_error(_name: str) -> object:
        raise ImportError("no module named _data_bridge")

    monkeypatch.setattr(
        daemon_control,
        "importlib",
        SimpleNamespace(import_module=raise_import_error),
    )

    with caplog.at_level("WARNING"):
        daemon_control._check_daemon_version(0.0)

    assert "not importable" in caplog.text
