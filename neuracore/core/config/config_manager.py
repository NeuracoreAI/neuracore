"""Config store management.

This module provides a class to manage the the state of the configuration stored
to disk.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ValidationError

from neuracore.core.exceptions import ConfigError

CONFIG_DIR = Path.home() / ".neuracore"
CONFIG_FILE = "config.json"
CONFIG_ENCODING = "utf-8"


class Config(BaseModel):
    """Pydantic Schema for the data stored in the config file."""

    api_key: Optional[str] = None
    current_org_id: Optional[str] = None


class ConfigManager:
    """Manager for retrieving and storing the configuration state."""

    def __init__(self) -> None:
        """Initialise the config manager."""
        self._config: Optional[Config] = None

    @property
    def config(self) -> Config:
        """Load authentication configuration from persistent storage.

        Attempts to load previously saved API key from the user's home
        directory configuration file. Provides a default config if not found.

        Raises:
            ConfigError: If there is an error trying to access the saved config.
        """
        if self._config:
            return self._config

        config_file = CONFIG_DIR / CONFIG_FILE
        if not config_file.exists():
            self._config = Config()
            return self._config

        try:
            with open(config_file, mode="r", encoding=CONFIG_ENCODING) as f:
                self._config = Config.model_validate_json(f.read())
                return self._config
        except ValidationError:
            raise ConfigError("Error loading config: invalid structure")
        except PermissionError:
            raise ConfigError("Error loading config: insufficient permissions")
        except UnicodeDecodeError:
            raise ConfigError("Error loading config: invalid encoding")
        except OSError:
            raise ConfigError("Error loading config: cannot open file")

    @config.setter
    def config(self, config: Config) -> None:
        """Setter method for updating the state of the config in memory.

        This does not save the config to disk use `save_config` to persist any
        changes

        Args:
            config (Config): the new config.
        """
        self._config = config

    def save_config(self) -> None:
        """Save current authentication configuration to persistent storage.

        Creates the configuration directory if it doesn't exist and saves
        the current configuration to a JSON configuration file in the user's
        home directory.

        Raises:
            ConfigError: If saving the config fails.
        """
        if not self._config:
            # Nothing to save
            return

        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config_file = CONFIG_DIR / CONFIG_FILE

        try:
            with open(config_file, "w", encoding=CONFIG_ENCODING) as f:
                f.write(self._config.model_dump_json())
        except PermissionError:
            raise ConfigError("Error saving config: insufficient permissions")
        except OSError:
            raise ConfigError("Error saving config: cannot write to file")

    def remove_config(self) -> None:
        """Remove the current authentication configuration from persistent storage.

        Deletes the configuration file from the user's home directory if it exists.

        Raises:
            ConfigError: If removing the config fails due to permission or file
                system errors.
        """
        config_file = CONFIG_DIR / CONFIG_FILE

        try:
            if config_file.exists():
                config_file.unlink()
        except PermissionError:
            raise ConfigError("Error removing config: insufficient permissions")
        except OSError:
            raise ConfigError("Error removing config: cannot delete file")


_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Gets the singleton config manager instance.

    Returns:
        ConfigManager: The config manager for updating the config state
    """
    global _config_manager
    if not _config_manager:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> Config:
    """A utility function to make simplify getting the current configuration.

    Returns:
        The current configuration
    Raises:
        ConfigError: If there is an error trying to access the saved config.
    """
    return get_config_manager().config
