from pathlib import Path
from typing import Any

import pytest
import yaml

from neuracore.data_daemon.config_manager.config import ConfigManager
from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.config_manager.helpers import parse_bytes
from neuracore.data_daemon.config_manager.profiles import (
    ProfileAlreadyExist,
    ProfileManager,
    ProfileNotFound,
)


@pytest.fixture
def temporary_home(tmp_path: Path) -> Path:
    """Provide an isolated home directory for profile tests."""
    home_directory = tmp_path / "home"
    home_directory.mkdir()
    return home_directory


@pytest.fixture
def profile_manager(temporary_home: Path) -> ProfileManager:
    """Provide a ProfileManager rooted at the temporary home directory."""
    return ProfileManager(home_path=temporary_home)


@pytest.fixture
def profiles_directory(temporary_home: Path) -> Path:
    """Return the profiles directory under the temporary home."""
    return temporary_home / ".neuracore" / "data_daemon" / "profiles"


def test_create_profile_creates_yaml_with_default_config(
    profile_manager: ProfileManager,
    profiles_directory: Path,
) -> None:
    """create_profile should create a YAML file with default DaemonConfig values."""
    profile_name = "recording"

    profile_manager.create_profile(profile_name)

    profile_path = profiles_directory / f"{profile_name}.yaml"
    assert profile_path.is_file()

    with profile_path.open("r") as profile_file:
        stored_data = yaml.safe_load(profile_file)

    expected_config = DaemonConfig()
    assert stored_data == expected_config.model_dump()


def test_create_profile_raises_when_profile_exists(
    profile_manager: ProfileManager,
) -> None:
    """create_profile should raise when the profile already exists."""
    profile_name = "existing"
    profile_manager.create_profile(profile_name)

    with pytest.raises(ProfileAlreadyExist):
        profile_manager.create_profile(profile_name)


def test_get_profile_returns_config_from_yaml(
    profile_manager: ProfileManager,
    profiles_directory: Path,
) -> None:
    """get_profile should load configuration from the profile YAML."""
    profile_name = "custom"
    profiles_directory.mkdir(parents=True, exist_ok=True)
    profile_path = profiles_directory / f"{profile_name}.yaml"

    stored_content: dict[str, Any] = {
        "storage_limit": 10_000,
        "bandwidth_limit": 1_000,
        "path_to_store_record": "/data/records",
        "num_threads": 4,
        "keep_wakelock_while_upload": True,
        "offline": False,
    }

    with profile_path.open("w") as profile_file:
        yaml.safe_dump(stored_content, profile_file)

    loaded_config = profile_manager.get_profile(profile_name)

    assert loaded_config.storage_limit == 10_000
    assert loaded_config.bandwidth_limit == 1_000
    assert loaded_config.path_to_store_record == "/data/records"
    assert loaded_config.num_threads == 4
    assert loaded_config.keep_wakelock_while_upload is True
    assert loaded_config.offline is False


def test_get_profile_parses_unit_suffixed_byte_fields(
    profile_manager: ProfileManager,
    profiles_directory: Path,
) -> None:
    """get_profile should parse unit-suffixed byte values from YAML."""
    profile_name = "units"
    profiles_directory.mkdir(parents=True, exist_ok=True)
    profile_path = profiles_directory / f"{profile_name}.yaml"

    stored_content: dict[str, Any] = {
        "storage_limit": "2gb",
        "bandwidth_limit": "300m",
    }

    with profile_path.open("w") as profile_file:
        yaml.safe_dump(stored_content, profile_file)

    loaded_config = profile_manager.get_profile(profile_name)

    assert loaded_config.storage_limit == 2 * 1024**3
    assert loaded_config.bandwidth_limit == 300 * 1024**2


def test_get_profile_raises_when_missing(
    profile_manager: ProfileManager,
) -> None:
    """get_profile should raise when the profile YAML file does not exist."""
    with pytest.raises(ProfileNotFound):
        profile_manager.get_profile("does_not_exist")


def test_list_profiles_returns_sorted_names(
    profile_manager: ProfileManager,
    profiles_directory: Path,
) -> None:
    """list_profiles should return sorted profile names without extensions."""
    profiles_directory.mkdir(parents=True, exist_ok=True)

    profile_names = ["beta", "alpha", "gamma"]
    for profile_name in profile_names:
        profile_path = profiles_directory / f"{profile_name}.yaml"
        profile_path.write_text("{}", encoding="utf-8")

    listed_profiles = profile_manager.list_profiles()

    assert listed_profiles == ["alpha", "beta", "gamma"]


def test_update_profile_updates_only_specified_fields(
    profile_manager: ProfileManager,
) -> None:
    """update_profile should modify only the provided fields and keep others."""
    profile_name = "recording"
    profile_manager.create_profile(profile_name)

    initial_updates: dict[str, Any] = {
        "storage_limit": 5_000,
        "bandwidth_limit": 500,
        "offline": False,
    }
    profile_manager.update_profile(profile_name, initial_updates)

    second_updates: dict[str, Any] = {
        "bandwidth_limit": 750,
    }
    updated_config = profile_manager.update_profile(profile_name, second_updates)

    assert updated_config.storage_limit == 5_000
    assert updated_config.bandwidth_limit == 750
    assert updated_config.offline is False


def test_update_profile_raises_when_profile_missing(
    profile_manager: ProfileManager,
) -> None:
    """update_profile should raise when the profile does not exist."""
    profile_name = "missing"

    with pytest.raises(ProfileNotFound):
        profile_manager.update_profile(profile_name, {"storage_limit": 1_000})


def test_resolve_effective_config_env_overrides_profile(
    profile_manager: ProfileManager,
    profiles_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment variables should override values from the profile YAML."""
    profile_name = "recording"
    profiles_directory.mkdir(parents=True, exist_ok=True)
    profile_path = profiles_directory / f"{profile_name}.yaml"

    base_content: dict[str, Any] = {
        "storage_limit": 1_000,
        "bandwidth_limit": 2_000,
        "offline": False,
    }

    with profile_path.open("w") as profile_file:
        yaml.safe_dump(base_content, profile_file)

    monkeypatch.setenv("NCD_STORAGE_LIMIT", "5000")
    monkeypatch.setenv("NCD_OFFLINE", "true")

    config_manager = ConfigManager(
        profile_manager=profile_manager, profile=profile_name
    )
    effective_config = config_manager.resolve_effective_config(cli_config={})

    assert effective_config.storage_limit == 5_000
    assert effective_config.bandwidth_limit == 2_000
    assert effective_config.offline is True


def test_resolve_effective_config_env_supports_unit_suffixed_values(
    profile_manager: ProfileManager,
    profiles_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment variables support unit-suffixed byte values."""
    profile_name = "recording"
    profiles_directory.mkdir(parents=True, exist_ok=True)
    profile_path = profiles_directory / f"{profile_name}.yaml"

    base_content: dict[str, Any] = {
        "storage_limit": 1_000,
        "bandwidth_limit": 2_000,
    }

    with profile_path.open("w") as profile_file:
        yaml.safe_dump(base_content, profile_file)

    monkeypatch.setenv("NCD_STORAGE_LIMIT", "2gb")
    monkeypatch.setenv("NCD_BANDWIDTH_LIMIT", "300m")

    config_manager = ConfigManager(
        profile_manager=profile_manager, profile=profile_name
    )
    effective_config = config_manager.resolve_effective_config(cli_config={})

    assert effective_config.storage_limit == 2 * 1024**3
    assert effective_config.bandwidth_limit == 300 * 1024**2


def test_resolve_effective_config_env_boolean_values_are_case_insensitive(
    profile_manager: ProfileManager,
    profiles_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boolean environment values should be case-insensitive."""
    profile_name = "recording"
    profiles_directory.mkdir(parents=True, exist_ok=True)
    profile_path = profiles_directory / f"{profile_name}.yaml"

    base_content: dict[str, Any] = {
        "offline": False,
    }

    with profile_path.open("w") as profile_file:
        yaml.safe_dump(base_content, profile_file)

    monkeypatch.setenv("NCD_OFFLINE", "YeS")

    config_manager = ConfigManager(
        profile_manager=profile_manager, profile=profile_name
    )
    effective_config = config_manager.resolve_effective_config(cli_config={})

    assert effective_config.offline is True


def test_resolve_effective_config_cli_overrides_env_and_profile(
    profile_manager: ProfileManager,
    profiles_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI overrides should take precedence over both env and profile values."""
    profile_name = "recording"
    profiles_directory.mkdir(parents=True, exist_ok=True)
    profile_path = profiles_directory / f"{profile_name}.yaml"

    base_content: dict[str, Any] = {
        "storage_limit": 1_000,
        "bandwidth_limit": 2_000,
        "num_threads": 1,
    }

    with profile_path.open("w") as profile_file:
        yaml.safe_dump(base_content, profile_file)

    monkeypatch.setenv("NCD_STORAGE_LIMIT", "3000")
    monkeypatch.setenv("NCD_NUM_THREADS", "2")

    cli_overrides: dict[str, Any] = {
        "storage_limit": 7_000,
    }

    config_manager = ConfigManager(
        profile_manager=profile_manager, profile=profile_name
    )
    effective_config = config_manager.resolve_effective_config(cli_config=cli_overrides)

    assert effective_config.storage_limit == 7_000
    assert effective_config.bandwidth_limit == 2_000
    assert effective_config.num_threads == 2


def test_resolve_effective_config_ignores_invalid_integer_env_values(
    profile_manager: ProfileManager,
    profiles_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid integer/unit values in environment variables should be ignored."""
    profile_name = "recording"
    profiles_directory.mkdir(parents=True, exist_ok=True)
    profile_path = profiles_directory / f"{profile_name}.yaml"

    base_content: dict[str, Any] = {
        "storage_limit": 4_000,
    }

    with profile_path.open("w") as profile_file:
        yaml.safe_dump(base_content, profile_file)

    monkeypatch.setenv("NCD_STORAGE_LIMIT", "not_an_integer")

    config_manager = ConfigManager(
        profile_manager=profile_manager, profile=profile_name
    )
    effective_config = config_manager.resolve_effective_config(cli_config={})

    assert effective_config.storage_limit == 4_000


def test_parse_bytes_accepts_int_and_numeric_string() -> None:
    """parse_bytes should handle ints and plain numeric strings."""
    assert parse_bytes(1024) == 1024
    assert parse_bytes("2048") == 2048


def test_parse_bytes_parses_unit_suffixes() -> None:
    """parse_bytes should handle common unit suffixes."""
    assert parse_bytes("1kb") == 1024
    assert parse_bytes("1m") == 1024**2
    assert parse_bytes("2gb") == 2 * 1024**3
