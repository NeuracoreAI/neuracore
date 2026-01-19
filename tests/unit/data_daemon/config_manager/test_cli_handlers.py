"""Tests for nc-data-daemon CLI handlers."""

import argparse
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


def test_handle_launch_prints_not_implemented_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """handle_launch should print a scaffolding message."""
    args_handler.handle_launch(_ns())
    out = capsys.readouterr().out.strip()
    assert out == "launch command is not implemented yet."


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
