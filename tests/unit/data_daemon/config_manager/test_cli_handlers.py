"""Tests for data-daemon Typer CLI handlers."""

from typing import Any

import pytest
import typer

from neuracore.data_daemon.config_manager import args_handler
from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig


def test_run_profile_create_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    called_with: list[str] = []

    def fake_create_profile(name: str) -> None:
        called_with.append(name)

    monkeypatch.setattr(
        args_handler.profile_manager, "create_profile", fake_create_profile
    )

    args_handler.run_profile_create(name="favour")
    out = capsys.readouterr().out.strip()

    assert called_with == ["favour"]
    assert out == "Created profile 'favour'."


def test_run_profile_update_validates_and_updates(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}

    def fake_update_profile(name: str, updates: dict[str, Any]) -> DaemonConfig:
        captured["name"] = name
        captured["updates"] = updates
        return DaemonConfig(num_threads=2)

    monkeypatch.setattr(
        args_handler.profile_manager, "update_profile", fake_update_profile
    )

    args_handler.run_profile_update(
        name="recording",
        storage_limit=None,
        bandwidth_limit=None,
        path_to_store_record=None,
        num_threads=2,
        max_concurrent_uploads=None,
        keep_wakelock_while_upload=None,
        offline=None,
        api_key=None,
        current_org_id=None,
    )
    out = capsys.readouterr().out.strip()

    assert captured["name"] == "recording"
    assert captured["updates"] == {"num_threads": 2}
    assert out == "Updated profile 'recording'."


def test_run_profile_get_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = DaemonConfig(storage_limit=1234, offline=True)

    def fake_get_profile(name: str) -> DaemonConfig:
        assert name == "recording"
        return config

    monkeypatch.setattr(args_handler.profile_manager, "get_profile", fake_get_profile)

    args_handler.run_profile_get(name="recording")
    out = capsys.readouterr().out.strip()

    assert '"storage_limit": 1234' in out
    assert '"offline": true' in out


def test_run_list_profiles_no_profiles(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(args_handler.profile_manager, "list_profiles", lambda: [])

    args_handler.run_list_profiles()
    out = capsys.readouterr().out.strip()

    assert out == "No profiles found."


def test_run_list_profiles_prints_each_profile(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        args_handler.profile_manager, "list_profiles", lambda: ["alpha", "beta"]
    )

    args_handler.run_list_profiles()
    out_lines = capsys.readouterr().out.strip().splitlines()

    assert out_lines == ["alpha", "beta"]


def test_run_profile_delete_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    called_with: list[str] = []

    def fake_delete_profile(name: str) -> None:
        called_with.append(name)

    monkeypatch.setattr(
        args_handler.profile_manager, "delete_profile", fake_delete_profile
    )

    args_handler.run_profile_delete(name="recording")
    out = capsys.readouterr().out.strip()

    assert called_with == ["recording"]
    assert out == "Deleted profile 'recording'."


def test_run_profile_delete_missing_profile_exits(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_delete_profile(name: str) -> None:
        raise args_handler.ProfileNotFound(f"Profile {name!r} not found.")

    monkeypatch.setattr(
        args_handler.profile_manager, "delete_profile", fake_delete_profile
    )

    with pytest.raises(typer.Exit) as exc_info:
        args_handler.run_profile_delete(name="missing-prof")

    captured = capsys.readouterr()
    assert exc_info.value.exit_code == 1
    assert "Profile 'missing-prof' not found." in captured.err


def test_run_profile_delete_blocks_default_profile(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(typer.Exit) as exc_info:
        args_handler.run_profile_delete(name=args_handler.DEFAULT_PROFILE_NAME)

    captured = capsys.readouterr()
    assert exc_info.value.exit_code == 1
    assert (
        f"Cannot delete default profile {args_handler.DEFAULT_PROFILE_NAME!r}."
        in captured.err
    )
