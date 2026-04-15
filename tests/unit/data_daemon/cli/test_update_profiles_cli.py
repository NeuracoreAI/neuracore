from __future__ import annotations

from typing import Any

import pytest
import typer

import neuracore.data_daemon.config_manager.args_handler as ah


def test_run_profile_create_happy_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    called: dict[str, Any] = {}

    def fake_create_profile(name: str) -> None:
        called["name"] = name

    monkeypatch.setattr(ah.profile_manager, "create_profile", fake_create_profile)

    ah.run_profile_create(name="demo")

    assert called["name"] == "demo"
    assert "Created profile 'demo'." in capsys.readouterr().out


def test_run_profile_update_validates_and_updates(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    called: dict[str, Any] = {}

    def fake_update_profile(name: str, updates: dict[str, Any]) -> None:
        called["name"] = name
        called["updates"] = updates

    monkeypatch.setattr(ah.profile_manager, "update_profile", fake_update_profile)

    ah.run_profile_update(
        name="demo",
        storage_limit=2048,
        bandwidth_limit=4096,
        path_to_store_record=None,
        num_threads=None,
        keep_wakelock_while_upload=None,
        offline=None,
        api_key=None,
        current_org_id=None,
    )

    assert called["name"] == "demo"
    assert called["updates"] == {"storage_limit": 2048, "bandwidth_limit": 4096}
    assert "Updated profile 'demo'." in capsys.readouterr().out


def test_run_profile_get_prints_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakeConfig:
        def model_dump_json(self, indent: int = 2) -> str:
            return '{\n  "storage_limit": 123\n}'

    monkeypatch.setattr(ah.profile_manager, "get_profile", lambda name: FakeConfig())

    ah.run_profile_get(name="demo")

    assert '"storage_limit": 123' in capsys.readouterr().out


def test_run_list_profiles_no_profiles(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ah.profile_manager, "list_profiles", lambda: [])
    ah.run_list_profiles()
    assert capsys.readouterr().out.strip() == "No profiles found."


def test_run_list_profiles_prints_each_profile(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ah.profile_manager, "list_profiles", lambda: ["a", "b"])
    ah.run_list_profiles()
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["a", "b"]


def test_run_profile_delete_happy_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    called: dict[str, Any] = {}

    def fake_delete_profile(name: str) -> None:
        called["name"] = name

    monkeypatch.setattr(ah.profile_manager, "delete_profile", fake_delete_profile)

    ah.run_profile_delete(name="demo")

    assert called["name"] == "demo"
    assert "Deleted profile 'demo'." in capsys.readouterr().out


def test_run_profile_delete_missing_profile_exits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_delete_profile(name: str) -> None:
        raise ah.ProfileNotFound(f"Profile {name!r} not found.")

    monkeypatch.setattr(ah.profile_manager, "delete_profile", fake_delete_profile)

    with pytest.raises(typer.Exit) as e:
        ah.run_profile_delete(name="missing-prof")

    assert e.value.exit_code == 1
    assert "Profile 'missing-prof' not found." in capsys.readouterr().err


def test_run_profile_delete_blocks_default_profile(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(typer.Exit) as e:
        ah.run_profile_delete(name=ah.DEFAULT_PROFILE_NAME)

    assert e.value.exit_code == 1
    assert (
        f"Cannot delete default profile {ah.DEFAULT_PROFILE_NAME!r}."
        in capsys.readouterr().err
    )
