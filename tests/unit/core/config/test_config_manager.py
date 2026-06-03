"""Tests for neuracore.core.config.config_manager."""

import json
from pathlib import Path

import pytest

from neuracore.core.config.config_manager import CONFIG_FILE, Config, ConfigManager
from neuracore.core.exceptions import ConfigError


def test_save_config_skips_when_unchanged(temp_config_dir: Path) -> None:
    config_file = temp_config_dir / CONFIG_FILE
    config_file.write_text(
        '{"api_key":"nrc_test","current_org_id":"org-1"}',
        encoding="utf-8",
    )
    mtime_before = config_file.stat().st_mtime

    manager = ConfigManager()
    manager.config = Config(api_key="nrc_test", current_org_id="org-1")
    manager.save_config()

    assert config_file.stat().st_mtime == mtime_before


def test_save_config_overwrites_when_changed(temp_config_dir: Path) -> None:
    config_file = temp_config_dir / CONFIG_FILE
    config_file.write_text(
        '{"api_key":"nrc_old","current_org_id":"org-old"}',
        encoding="utf-8",
    )

    manager = ConfigManager()
    manager.config = Config(api_key="nrc_new", current_org_id="org-new")
    manager.save_config()

    loaded = json.loads(config_file.read_text(encoding="utf-8"))
    assert loaded == {"api_key": "nrc_new", "current_org_id": "org-new"}


def test_save_config_overwrites_when_empty(temp_config_dir: Path) -> None:
    config_file = temp_config_dir / CONFIG_FILE
    config_file.write_text("", encoding="utf-8")

    manager = ConfigManager()
    manager.config = Config(api_key="nrc_test", current_org_id="org-1")
    manager.save_config()

    loaded = json.loads(config_file.read_text(encoding="utf-8"))
    assert loaded == {"api_key": "nrc_test", "current_org_id": "org-1"}


def test_save_config_writes_json(temp_config_dir: Path) -> None:
    config_file = temp_config_dir / CONFIG_FILE
    manager = ConfigManager()
    manager.config = Config(api_key="nrc_test", current_org_id="org-1")
    manager.save_config()

    assert config_file.exists()
    loaded = json.loads(config_file.read_text(encoding="utf-8"))
    assert loaded == {"api_key": "nrc_test", "current_org_id": "org-1"}


def test_load_config_raises_on_invalid_json(temp_config_dir: Path) -> None:
    config_file = temp_config_dir / CONFIG_FILE
    config_file.write_text("{not json")

    manager = ConfigManager()
    with pytest.raises(ConfigError, match="invalid structure"):
        _ = manager.config
