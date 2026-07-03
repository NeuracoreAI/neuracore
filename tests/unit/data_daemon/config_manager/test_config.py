from __future__ import annotations

import pytest

from neuracore.data_daemon.config_manager.config import ConfigManager
from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig


class _StubProfileManager:
    def __init__(self, profile_config: DaemonConfig) -> None:
        self._profile_config = profile_config

    def get_profile(self, profile: str | None) -> DaemonConfig:
        return self._profile_config


def test_env_org_id_read_from_neuracore_org_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEURACORE_ORG_ID", "org-from-env")

    manager = ConfigManager(_StubProfileManager(DaemonConfig()))
    config = manager.resolve_effective_config()

    assert config.current_org_id == "org-from-env"


def test_env_org_id_overrides_profile_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEURACORE_ORG_ID", "org-from-env")

    manager = ConfigManager(
        _StubProfileManager(DaemonConfig(current_org_id="org-from-profile"))
    )
    config = manager.resolve_effective_config()

    assert config.current_org_id == "org-from-env"


def test_org_id_unset_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEURACORE_ORG_ID", raising=False)

    manager = ConfigManager(_StubProfileManager(DaemonConfig()))
    config = manager.resolve_effective_config()

    assert config.current_org_id is None
