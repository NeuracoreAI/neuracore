"""Tests for nc-data-daemon CLI handlers."""

import argparse
import os
from pathlib import Path
from typing import Any

import pytest

from neuracore.data_daemon.config_manager import args_handler
from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig


def _ns(**kwargs: Any) -> argparse.Namespace:
    """Build a simple argparse.Namespace for handler tests."""
    return argparse.Namespace(**kwargs)


def test_handle_profile_create_calls_manager_and_prints_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """handle_profile_create should call ProfileManager.create_profile."""
    called_with: list[str] = []

    def fake_create_profile(name: str) -> None:
        called_with.append(name)

    monkeypatch.setattr(
        args_handler.profile_manager,
        "create_profile",
        fake_create_profile,
    )

    args = _ns(name="favour")
    args_handler.handle_profile_create(args)

    out = capsys.readouterr().out.strip()

    assert called_with == ["favour"]
    assert out == "Created profile 'favour'."


def test_handle_profile_create_prints_error_when_profile_exists(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """handle_profile_create should print ProfileAlreadyExist message."""

    def fake_create_profile(name: str) -> None:
        raise args_handler.ProfileAlreadyExist(f"Profile {name!r} already exists.")

    monkeypatch.setattr(
        args_handler.profile_manager,
        "create_profile",
        fake_create_profile,
    )

    args = _ns(name="favour")
    args_handler.handle_profile_create(args)

    out = capsys.readouterr().out.strip()
    assert out == "Profile 'favour' already exists."


def test_handle_profile_update_builds_updates_and_calls_manager(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """handle_profile_update should build updates dict and call update_profile."""
    captured: dict[str, Any] = {}

    def fake_update_profile(name: str, updates: dict[str, Any]) -> DaemonConfig:
        captured["name"] = name
        captured["updates"] = updates
        return DaemonConfig(num_threads=2)

    monkeypatch.setattr(
        args_handler.profile_manager,
        "update_profile",
        fake_update_profile,
    )

    args = _ns(
        name="recording",
        storage_limit=None,
        bandwidth_limit=None,
        path_to_store_record=None,
        num_threads=2,
        keep_wakelock_while_upload=None,
        offline=None,
        api_key=None,
        current_org_id=None,
    )

    args_handler.handle_profile_update(args)
    out = capsys.readouterr().out.strip()

    assert captured["name"] == "recording"
    assert captured["updates"] == {"num_threads": 2}
    assert out == "Updated profile 'recording'."


def test_handle_profile_update_prints_error_when_profile_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """handle_profile_update should print error when profile does not exist."""

    def fake_update_profile(name: str, updates: dict[str, Any]) -> DaemonConfig:
        raise args_handler.ProfileNotFound(f"Profile {name!r} not found.")

    monkeypatch.setattr(
        args_handler.profile_manager,
        "update_profile",
        fake_update_profile,
    )

    args = _ns(
        name="missing",
        storage_limit=None,
        bandwidth_limit=None,
        storage_path=None,
        num_threads=None,
        keep_wakelock_while_upload=False,
        offline=False,
    )

    args_handler.handle_profile_update(args)
    out = capsys.readouterr().out.strip()

    assert out == "Profile 'missing' not found."


def test_handle_profile_show_prints_profile_config(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """handle_profile_show should print the profile configuration JSON."""
    config = DaemonConfig(storage_limit=1_234, offline=True)

    def fake_get_profile(name: str) -> DaemonConfig:
        assert name == "recording"
        return config

    monkeypatch.setattr(
        args_handler.profile_manager,
        "get_profile",
        fake_get_profile,
    )

    args = _ns(name="recording")
    args_handler.handle_profile_show(args)

    out = capsys.readouterr().out.strip()

    assert '"storage_limit": 1234' in out
    assert '"offline": true' in out


def test_handle_profile_show_prints_error_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """handle_profile_show should print an error when profile is missing."""

    def fake_get_profile(name: str) -> DaemonConfig:
        raise args_handler.ProfileNotFound(f"Profile {name!r} not found.")

    monkeypatch.setattr(
        args_handler.profile_manager,
        "get_profile",
        fake_get_profile,
    )

    args = _ns(name="missing")
    args_handler.handle_profile_show(args)

    out = capsys.readouterr().out.strip()
    assert out == "Profile 'missing' not found."


def test_handle_list_profile_prints_message_when_no_profiles(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """handle_list_profile should print a message when no profiles exist."""
    monkeypatch.setattr(
        args_handler.profile_manager,
        "list_profiles",
        lambda: [],
    )

    args_handler.handle_list_profile(_ns())
    out = capsys.readouterr().out.strip()

    assert out == "No profiles found."


def test_handle_list_profile_prints_profile_names(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """handle_list_profile should print one profile name per line."""
    monkeypatch.setattr(
        args_handler.profile_manager,
        "list_profiles",
        lambda: ["alpha", "beta"],
    )

    args_handler.handle_list_profile(_ns())
    out_lines = capsys.readouterr().out.strip().splitlines()

    assert out_lines == ["alpha", "beta"]


class _FakePopen:
    def __init__(self, pid: int, poll_value: int | None) -> None:
        self.pid = pid
        self._poll_value = poll_value

    def poll(self) -> int | None:
        return self._poll_value


def test_handle_launch_writes_pid_file_and_prints_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    test_state_dir = tmp_path / ".neuracore"
    test_pid_file = test_state_dir / "daemon.pid"

    monkeypatch.setattr(args_handler, "daemon_state_dir_path", test_state_dir)
    monkeypatch.setattr(args_handler, "pid_file_path", test_pid_file)
    monkeypatch.setattr(args_handler.time, "sleep", lambda _: None)

    monkeypatch.setattr(
        args_handler.subprocess,
        "Popen",
        lambda *a, **k: _FakePopen(pid=12345, poll_value=None),
    )

    args_handler.handle_launch(_ns())
    out = capsys.readouterr().out.strip()

    assert test_pid_file.read_text(encoding="utf-8").strip() == "12345"
    assert out == "Daemon launched (pid=12345)."


def test_handle_launch_rejects_running_pid(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    test_state_dir = tmp_path / ".neuracore"
    test_pid_file = test_state_dir / "daemon.pid"
    test_state_dir.mkdir(parents=True, exist_ok=True)
    test_pid_file.write_text(str(os.getpid()), encoding="utf-8")

    monkeypatch.setattr(args_handler, "daemon_state_dir_path", test_state_dir)
    monkeypatch.setattr(args_handler, "pid_file_path", test_pid_file)

    with pytest.raises(SystemExit) as excinfo:
        args_handler.handle_launch(_ns())

    out = capsys.readouterr().out.strip()
    assert excinfo.value.code == 1
    assert out == f"Daemon already running (pid={os.getpid()})."


def test_handle_launch_clears_stale_pid_and_starts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    test_state_dir = tmp_path / ".neuracore"
    test_pid_file = test_state_dir / "daemon.pid"
    test_state_dir.mkdir(parents=True, exist_ok=True)
    test_pid_file.write_text("999999", encoding="utf-8")

    monkeypatch.setattr(args_handler, "daemon_state_dir_path", test_state_dir)
    monkeypatch.setattr(args_handler, "pid_file_path", test_pid_file)
    monkeypatch.setattr(args_handler.time, "sleep", lambda _: None)

    monkeypatch.setattr(
        args_handler.subprocess,
        "Popen",
        lambda *a, **k: _FakePopen(pid=12345, poll_value=None),
    )

    args_handler.handle_launch(_ns())
    out = capsys.readouterr().out.strip()

    assert test_pid_file.read_text(encoding="utf-8").strip() == "12345"
    assert out == "Daemon launched (pid=12345)."


def test_handle_launch_exits_and_does_not_write_pid_file_when_runner_exits_immediately(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    test_state_dir = tmp_path / ".neuracore"
    test_pid_file = test_state_dir / "daemon.pid"

    monkeypatch.setattr(args_handler, "daemon_state_dir_path", test_state_dir)
    monkeypatch.setattr(args_handler, "pid_file_path", test_pid_file)
    monkeypatch.setattr(args_handler.time, "sleep", lambda _: None)

    monkeypatch.setattr(
        args_handler.subprocess,
        "Popen",
        lambda *a, **k: _FakePopen(pid=99999, poll_value=1),
    )

    with pytest.raises(SystemExit) as excinfo:
        args_handler.handle_launch(_ns())

    out = capsys.readouterr().out.strip()
    assert excinfo.value.code == 1
    assert out == "Daemon failed to start."
    assert not test_pid_file.exists()


def test_handle_stop_removes_pid_file_when_process_is_not_running(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    test_state_dir = tmp_path / ".neuracore"
    test_pid_file = test_state_dir / "daemon.pid"
    test_state_dir.mkdir(parents=True, exist_ok=True)
    test_pid_file.write_text("4242", encoding="utf-8")

    monkeypatch.setattr(args_handler, "daemon_state_dir_path", test_state_dir)
    monkeypatch.setattr(args_handler, "pid_file_path", test_pid_file)
    monkeypatch.setattr(args_handler.time, "sleep", lambda _: None)
    monkeypatch.setattr(args_handler, "_terminate_pid", lambda _: True)
    monkeypatch.setattr(args_handler, "_pid_is_running", lambda _: False)

    args_handler.handle_stop(_ns())
    out = capsys.readouterr().out.strip()

    assert out == "Daemon stopped."
    assert not test_pid_file.exists()


def test_handle_status_removes_stale_pid_file_and_prints_not_running(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    test_state_dir = tmp_path / ".neuracore"
    test_pid_file = test_state_dir / "daemon.pid"
    test_state_dir.mkdir(parents=True, exist_ok=True)
    test_pid_file.write_text("7777", encoding="utf-8")

    monkeypatch.setattr(args_handler, "daemon_state_dir_path", test_state_dir)
    monkeypatch.setattr(args_handler, "pid_file_path", test_pid_file)
    monkeypatch.setattr(args_handler, "_pid_is_running", lambda _: False)

    args_handler.handle_status(_ns())
    out = capsys.readouterr().out.strip()

    assert out == "Daemon not running."
    assert not test_pid_file.exists()


def test_handle_update_calls_resolve_effective_config_and_prints_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """handle_update should build updates dict and call resolve_effective_config."""
    captured: dict[str, Any] = {}

    def fake_resolve_effective_config(
        cli_config: dict[str, Any] | None = None,
    ) -> DaemonConfig:
        captured["cli_config"] = cli_config
        return DaemonConfig(num_threads=2, offline=False)

    monkeypatch.setattr(
        args_handler.config_manager,
        "resolve_effective_config",
        fake_resolve_effective_config,
    )

    args = _ns(
        storage_limit=None,
        bandwidth_limit=None,
        path_to_store_record=None,
        num_threads=2,
        keep_wakelock_while_upload=None,
        offline=None,
    )

    args_handler.handle_update(args)

    assert captured["cli_config"] == {"num_threads": 2}
