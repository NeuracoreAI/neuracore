from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Any

import pytest

import neuracore.data_daemon.config_manager.args_handler as ah


def _patch_fake_daemon_config(
    monkeypatch: pytest.MonkeyPatch, allowed_fields: set[str]
) -> None:
    class FakeDaemonConfig:
        model_fields = {k: None for k in allowed_fields}

        @staticmethod
        def model_validate(d: dict[str, Any]) -> FakeDaemonConfig:
            obj = FakeDaemonConfig()
            obj._d = d
            return obj

        def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
            if exclude_none:
                return {k: v for k, v in self._d.items() if v is not None}
            return dict(self._d)

    monkeypatch.setattr(ah, "DaemonConfig", FakeDaemonConfig)


def test_add_common_config_args_argparse_wiring() -> None:
    parser = argparse.ArgumentParser()
    ah.add_common_config_args(parser)

    args = parser.parse_args([
        "--storage-limit",
        "2gb",
        "--bandwidth-limit",
        "50mb",
        "--storage-path",
        "/tmp/ncd_records",
        "--num-threads",
        "7",
        "--offline",
        "--api-key",
        "nrc_x",
        "--current-org-id",
        "org_123",
    ])
    assert args.storage_limit is not None
    assert args.bandwidth_limit is not None
    assert args.path_to_store_record == "/tmp/ncd_records"
    assert args.num_threads == 7
    assert args.offline is True
    assert args.api_key == "nrc_x"
    assert args.current_org_id == "org_123"


def test_extract_config_updates_filters_only_daemon_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_daemon_config(monkeypatch, {"storage_limit", "offline", "api_key"})

    args = SimpleNamespace(
        storage_limit=123,
        offline=True,
        api_key="k",
        not_a_field="ignore",
        another=None,
    )
    updates = ah._extract_config_updates(args)  # type: ignore[arg-type]
    assert updates == {"storage_limit": 123, "offline": True, "api_key": "k"}


def test_handle_update_prints_resolved_effective_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_daemon_config(monkeypatch, {"storage_limit", "offline", "api_key"})

    def fake_resolve_effective_config(validated_updates: dict[str, Any]) -> str:
        return f"resolved({validated_updates})"

    monkeypatch.setattr(
        ah.config_manager, "resolve_effective_config", fake_resolve_effective_config
    )

    args = SimpleNamespace(storage_limit=1024, offline=None, api_key=None)
    ah.handle_update(args)  # type: ignore[arg-type]

    out = capsys.readouterr().out.strip()
    assert out == "resolved({'storage_limit': 1024})"


def test_handle_update_excludes_none_fields(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_daemon_config(monkeypatch, {"storage_limit", "offline", "api_key"})

    monkeypatch.setattr(
        ah.config_manager,
        "resolve_effective_config",
        lambda validated_updates: f"resolved({validated_updates})",
    )

    args = SimpleNamespace(storage_limit=None, offline=True, api_key=None)
    ah.handle_update(args)  # type: ignore[arg-type]

    out = capsys.readouterr().out.strip()
    assert out == "resolved({'offline': True})"


def test_handle_profile_create_happy_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    called: dict[str, Any] = {}

    def fake_create_profile(name: str) -> None:
        called["name"] = name

    monkeypatch.setattr(ah.profile_manager, "create_profile", fake_create_profile)

    args = SimpleNamespace(name="demo")
    ah.handle_profile_create(args)  # type: ignore[arg-type]

    assert called["name"] == "demo"
    assert "Created profile 'demo'." in capsys.readouterr().out


def test_handle_profile_update_validates_and_updates(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_daemon_config(
        monkeypatch, {"storage_limit", "bandwidth_limit", "offline"}
    )

    called: dict[str, Any] = {}

    def fake_update_profile(name: str, updates: dict[str, Any]) -> None:
        called["name"] = name
        called["updates"] = updates

    monkeypatch.setattr(ah.profile_manager, "update_profile", fake_update_profile)

    args = SimpleNamespace(
        name="demo", storage_limit=2048, bandwidth_limit=4096, offline=None
    )
    ah.handle_profile_update(args)  # type: ignore[arg-type]

    assert called["name"] == "demo"
    assert called["updates"] == {"storage_limit": 2048, "bandwidth_limit": 4096}
    assert "Updated profile 'demo'." in capsys.readouterr().out


def test_handle_profile_show_prints_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakeConfig:
        def model_dump_json(self, indent: int = 2) -> str:
            return '{\n  "storage_limit": 123\n}'

    monkeypatch.setattr(ah.profile_manager, "get_profile", lambda name: FakeConfig())

    args = SimpleNamespace(name="demo")
    ah.handle_profile_show(args)  # type: ignore[arg-type]

    assert '"storage_limit": 123' in capsys.readouterr().out


def test_handle_list_profile_no_profiles(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ah.profile_manager, "list_profiles", lambda: [])
    ah.handle_list_profile(SimpleNamespace())
    assert capsys.readouterr().out.strip() == "No profiles found."


def test_handle_list_profile_prints_each_profile(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ah.profile_manager, "list_profiles", lambda: ["a", "b"])
    ah.handle_list_profile(SimpleNamespace())
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["a", "b"]
