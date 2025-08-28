"""This module provides a method to get the currently select organization id."""

import os

from neuracore.core.cli.input_lock import user_input_lock
from neuracore.core.cli.select_current_org import select_current_org
from neuracore.core.config.config_manager import get_config_manager
from neuracore.core.exceptions import (
    AuthenticationError,
    ConfigError,
    InputError,
    OrganizationError,
)


def get_current_org() -> str:
    """Get the current organization from storage or interactively select it.

    If the organization is selected interactively it is persisted to the config.

    overridden by the `NEURACORE_ORG_ID` environment variable.

    Returns:
        The selected organization's id

    Raises:
        ConfigError: If there is an error trying to get the config
    """
    if "NEURACORE_ORG_ID" in os.environ:
        return os.environ["NEURACORE_ORG_ID"]

    with user_input_lock:
        config_manager = get_config_manager()
        org_id = config_manager.config.current_org_id
        if org_id:
            return org_id
        try:
            organization = select_current_org()
            config_manager.config.current_org_id = organization.id
            config_manager.save_config()
            return organization.id
        except (AuthenticationError, OrganizationError, InputError) as e:
            raise ConfigError(f"Failed to select organization: {e}")
