"""This module provides a method to get the currently select organization id."""

import os

from neuracore.core.auth import get_auth
from neuracore.core.cli.input_lock import user_input_lock
from neuracore.core.cli.select_current_org import select_current_org
from neuracore.core.config.config_manager import get_config_manager
from neuracore.core.exceptions import (
    AuthenticationError,
    ConfigError,
    InputError,
    OrganizationError,
)


def get_saved_org() -> str | None:
    """Get the current organization without asking the user or the server.

    Returns:
        The `NEURACORE_ORG_ID` environment variable if set, otherwise the
        saved current organization's id, or None if neither is set.

    Raises:
        ConfigError: If there is an error trying to get the config
    """
    if "NEURACORE_ORG_ID" in os.environ:
        return os.environ["NEURACORE_ORG_ID"]
    return get_config_manager().config.current_org_id or None


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


OFFLINE_WITHOUT_ORG_MESSAGE = (
    "You are offline and no organization is saved. Connect to the internet "
    "and run `neuracore select-org` once, then you can record offline."
)


def require_saved_org() -> str:
    """Get the saved organization, which working offline needs.

    Offline, an organization cannot be selected, as that asks the server. A
    robot or dataset set up offline is registered in the saved organization
    once the server can be reached, so one must be saved.

    Returns:
        The saved organization's id, as :func:`get_saved_org` gives it.

    Raises:
        OrganizationError: If no organization is saved, or `NEURACORE_ORG_ID`
            is set but empty.
        ConfigError: If there is an error trying to get the config
    """
    org_id = get_saved_org()
    if not org_id:
        raise OrganizationError(OFFLINE_WITHOUT_ORG_MESSAGE)
    return org_id


def get_registry_org() -> str:
    """Get the organization to connect a robot or select a dataset in.

    Online this is :func:`get_current_org`, which may ask the user to pick
    one. Offline it is the saved organization, as nothing can be picked.

    Returns:
        The organization's id.

    Raises:
        OrganizationError: If offline and no organization is saved.
        ConfigError: If there is an error trying to get the config
    """
    if get_auth().is_offline:
        return require_saved_org()
    return get_current_org()
