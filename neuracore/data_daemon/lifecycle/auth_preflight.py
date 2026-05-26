"""Ensure authentication completes before daemon socket readiness polling starts."""

import os

from neuracore.core.auth import login
from neuracore.data_daemon.config_manager.config import ConfigManager
from neuracore.data_daemon.config_manager.profiles import (
    ProfileAlreadyExist,
    ProfileManager,
)
from neuracore.data_daemon.const import DEFAULT_PROFILE_NAME


def ensure_daemon_auth_ready(
    env_overrides: dict[str, str] | None = None,
) -> None:
    """Ensure authentication completes before daemon socket readiness polling starts."""
    profile_name = (
        (env_overrides or {}).get("NEURACORE_DAEMON_PROFILE")
        or os.environ.get("NEURACORE_DAEMON_PROFILE")
        or DEFAULT_PROFILE_NAME
    )

    profile_manager = ProfileManager()

    if profile_name == DEFAULT_PROFILE_NAME:
        try:
            profile_manager.create_profile(DEFAULT_PROFILE_NAME)
        except ProfileAlreadyExist:
            pass

    config_manager = ConfigManager(profile_manager, profile=profile_name)
    config = config_manager.resolve_effective_config()

    if config.offline:
        return

    login()
