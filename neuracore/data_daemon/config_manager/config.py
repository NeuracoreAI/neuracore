"""Resolve daemon configuration from profile, environment, and CLI overrides."""

from __future__ import annotations

import os
from typing import Any

from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.config_manager.helpers import parse_bytes
from neuracore.data_daemon.config_manager.profiles import ProfileManager

_ENV_MAP: dict[str, str] = {
    "storage_limit": "NCD_STORAGE_LIMIT",
    "bandwidth_limit": "NCD_BANDWIDTH_LIMIT",
    "path_to_store_record": "NCD_PATH_TO_STORE_RECORD",
    "num_threads": "NCD_NUM_THREADS",
    "keep_wakelock_while_upload": "NCD_KEEP_WAKELOCK_WHILE_UPLOAD",
    "offline": "NCD_OFFLINE",
    "api_key": "NCD_API_KEY",
    "current_org_id": "NCD_CURRENT_ORG_ID",
}

YES_CONFIRMATION = {"1", "true", "yes", "y"}


class ConfigManager:
    """Build effective daemon configuration from profile, env, and CLI overrides."""

    def __init__(
        self, profile_manager: ProfileManager, profile: str | None = None
    ) -> None:
        """Initialise ConfigManager.

        Args:
            profile_manager: ProfileManager instance
            profile: Name of the profile to load as the base configuration.
        """
        self.profile_manager = profile_manager
        self.profile = profile

    def _read_env_overrides(self) -> dict[str, Any]:
        """Read daemon configuration overrides from environment variables.

        Returns:
            A dictionary of configuration field names to override values.
        """
        overrides: dict[str, Any] = {}

        for field_name, env_var_name in _ENV_MAP.items():
            env_value = os.getenv(env_var_name)
            if env_value is None:
                continue

            if field_name in {"storage_limit", "bandwidth_limit"}:
                try:
                    overrides[field_name] = parse_bytes(env_value)
                except ValueError:
                    continue
            elif field_name == "num_threads":
                try:
                    overrides[field_name] = int(env_value)
                except ValueError:
                    continue
            elif field_name in {"keep_wakelock_while_upload", "offline"}:
                overrides[field_name] = env_value.lower() in YES_CONFIRMATION
            else:
                overrides[field_name] = env_value

        return overrides

    def resolve_effective_config(
        self, cli_config: dict[str, Any] | None = None
    ) -> DaemonConfig:
        """Resolve the effective daemon configuration for this run.

        Args:
            cli_config: Optional CLI-provided configuration overrides.

        Returns:
            The resolved ``DaemonConfig``.
        """
        base_config = self.profile_manager.get_profile(self.profile)

        env_overrides = self._read_env_overrides()
        merged_config = base_config.model_copy(update=env_overrides)

        if cli_config is not None:
            merged_config = merged_config.model_copy(update=cli_config)

        return merged_config
