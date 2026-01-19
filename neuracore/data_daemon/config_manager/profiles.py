"""API for handling data daemon profile information."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.config_manager.helpers import (
    build_default_daemon_config,
    parse_bytes,
)


class ProfileNotFound(Exception):
    """Raised when a requested profile cannot be found on disk."""


class ProfileAlreadyExist(Exception):
    """Raised when attempting to create a profile that already exists."""


class ProfileManager:
    """Manage daemon profiles stored on disk."""

    def __init__(
        self,
        home_path: Path | None = None,
    ) -> None:
        """Initialise ProfileManager."""
        self._home_path = home_path or Path.home()

    @property
    def home_path(self) -> Path:
        """Return the home path used for resolving configuration."""
        return self._home_path

    def _profiles_dir(self) -> Path:
        """Return the directory where daemon profiles are stored."""
        return self._home_path / ".neuracore" / "data_daemon" / "profiles"

    def _ensure_profiles_dir(self) -> Path:
        """Ensure that the profiles directory exists on disk.

        Returns:
            The path to the profiles directory.
        """
        profiles_dir = self._profiles_dir()
        profiles_dir.mkdir(parents=True, exist_ok=True)
        return profiles_dir

    def _get_profile_path(self, profile: str) -> Path:
        """Return the filesystem path for a given profile name.

        Args:
            profile: Name of the profile.

        Returns:
            Path to the profile YAML file.
        """
        profiles_dir = self._ensure_profiles_dir()
        return profiles_dir / f"{profile}.yaml"

    def list_profiles(self) -> list[str]:
        """List available profile names on the current node.

        Returns:
            List of profile names without the ``.yaml`` suffix.
        """
        profiles_dir = self._profiles_dir()
        if not profiles_dir.exists():
            return []

        names: list[str] = []
        for path in profiles_dir.iterdir():
            if path.is_file() and path.suffix == ".yaml":
                names.append(path.stem)
        return sorted(names)

    def get_profile(self, profile: str | None = None) -> DaemonConfig:
        """Load a profile configuration from disk.

        Args:
            profile: Name of the profile to load.

        Returns:
            Parsed daemon configuration for the profile.

        Raises:
            ProfileNotFound:
                If the profile YAML file does not exist.
        """
        if profile is None:
            return build_default_daemon_config()

        profile_path = self._get_profile_path(profile)

        try:
            with profile_path.open("r") as profile_file:
                profile_data = yaml.safe_load(profile_file) or {}
        except FileNotFoundError as exc:
            raise ProfileNotFound(f"Profile {profile!r} not found.") from exc

        for field_name in ("storage_limit", "bandwidth_limit"):
            raw_value = profile_data.get(field_name)
            if raw_value is not None:
                try:
                    profile_data[field_name] = parse_bytes(raw_value)
                except ValueError:
                    continue

        return DaemonConfig(**profile_data)

    def create_profile(self, profile: str) -> None:
        """Create a new profile with default configuration values.

        Args:
            profile: Name of the profile to create.

        Returns:
            None

        Raises:
            ProfileAlreadyExist:
                If a profile with the same name already exists.
        """
        profile_path = self._get_profile_path(profile)
        daemon_config = DaemonConfig()

        try:
            with profile_path.open("x") as profile_file:
                yaml.safe_dump(daemon_config.model_dump(), profile_file)
        except FileExistsError as exc:
            raise ProfileAlreadyExist(f"Profile {profile!r} already exists.") from exc

    def update_profile(self, profile: str, updates: dict[str, Any]) -> DaemonConfig:
        """Update an existing profile with the provided field values.

        Args:
            profile: Name of the profile to update.
            updates: Mapping of field names to new values. Fields with a value of
                ``None`` are ignored and do not overwrite existing values.

        Returns:
            The updated daemon configuration.

        Raises:
            ProfileNotFound:
                If the profile YAML file does not exist.
        """
        profile_path = self._get_profile_path(profile)

        current = self.get_profile(profile)
        filtered_updates = {
            name: value for name, value in updates.items() if value is not None
        }
        new_config = current.model_copy(update=filtered_updates)

        with profile_path.open("w") as profile_file:
            yaml.safe_dump(new_config.model_dump(), profile_file)

        return new_config
