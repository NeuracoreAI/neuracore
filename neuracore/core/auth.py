"""Authentication management for Neuracore API access.

This module provides authentication functionality including API key management,
access token handling, configuration persistence, and version validation.
It implements a singleton pattern to maintain authentication state across
the application.
"""

import os
from typing import Optional

import requests
from neuracore_types import __version__ as nc_types_version

from neuracore.api.orgs_fetch import fetch_org_ids
from neuracore.core.config.config_manager import get_config_manager
from neuracore.core.config.get_api_key import get_api_key
from neuracore.core.utils.singleton_metaclass import SingletonMetaclass

from .const import API_URL
from .exceptions import AuthenticationError


class Auth(metaclass=SingletonMetaclass):
    """Singleton class for managing Neuracore authentication state.

    This class handles API key management, access token retrieval, configuration
    persistence, and provides authenticated request headers. It maintains
    authentication state throughout the application lifecycle and automatically
    loads saved configuration on initialization.
    """

    def __init__(self) -> None:
        """Initialize the Auth instance and load saved configuration."""
        self._access_token = None

    def login(self, api_key: Optional[str] = None) -> None:
        """Authenticate with the Neuracore server using an API key.

        Attempts authentication using the provided API key, environment variable,
        or saved configuration. If no API key is available, initiates the API
        key generation process. Upon successful verification, saves the
        configuration for future use.

        Args:
            api_key: API key for authentication. If not provided, will attempt
                to use the NEURACORE_API_KEY environment variable or previously
                saved configuration. If none are available, will prompt for
                interactive API key generation.

        Raises:
            AuthenticationError: If API key verification fails due to invalid
                credentials, network issues, or server errors.
            InputError: If there is an issue with the user's input when gathering
                credentials
        """
        api_key = api_key or os.environ.get("NEURACORE_API_KEY") or get_api_key()

        # Verify API key with server and get access token
        try:
            response = requests.post(
                f"{API_URL}/auth/verify-api-key",
                json={"api_key": api_key},
            )
            if response.status_code != 200:
                raise AuthenticationError(
                    "Could not verify API key. Please check your key and try again."
                )
            token_data = response.json()
            self._access_token = token_data["access_token"]

            if self._access_token is None:
                raise AuthenticationError(
                    "Login succeeded but no access token returned."
                )

            config_manager = get_config_manager()
            prev_org_id = config_manager.config.current_org_id

            org_ids_optional = fetch_org_ids(self._access_token)
            fetched_orgs_success = org_ids_optional is not None
            user_org_ids = org_ids_optional or set()

            has_saved_org = prev_org_id is not None
            saved_org_no_longer_accessible = (
                fetched_orgs_success and prev_org_id not in user_org_ids
            )

            clear_saved_org = has_saved_org and saved_org_no_longer_accessible

            if clear_saved_org:
                # only clear if the saved org isn’t in the new account’s org list.
                config_manager.config.current_org_id = None

            # always persist the API key.
            config_manager.config.api_key = api_key
            config_manager.save_config()

        except requests.exceptions.ConnectionError:
            raise AuthenticationError((
                "Failed to connect to neuracore server, "
                "please check your internet connection and try again."
            ))
        except requests.exceptions.RequestException:
            raise AuthenticationError(
                "Could not verify API key. Please check your key and try again."
            )

    def logout(self) -> None:
        """Clear authentication state and remove saved configuration.

        Resets all authentication data including API key and access token,
        and removes the saved current org.

        Raises:
            ConfigError: If saving the config fails.
        """
        self._access_token = None
        config_manager = get_config_manager()
        config_manager.config.api_key = None
        config_manager.config.current_org_id = None
        config_manager.save_config()

    def validate_version(self) -> None:
        """Validate client version compatibility with the Neuracore server.

        Checks that the current Neuracore client version is compatible with
        the server API version. This helps ensure that API calls will work
        correctly and prevents issues from version mismatches.

        Raises:
            AuthenticationError: If version validation fails due to
                incompatible versions or server communication issues.
        """
        # Placeholder for version validation logic

        response = requests.get(
            f"{API_URL}/auth/verify-version",
            params={"version": nc_types_version},
        )
        if response.status_code != 200:
            raise AuthenticationError(
                f"Version validation failed: {response.json().get('detail')}"
            )

    @property
    def access_token(self) -> Optional[str]:
        """Get the current access token.

        Returns:
            The current access token received from the server, or None
            if not authenticated.
        """
        return self._access_token

    @property
    def is_authenticated(self) -> bool:
        """Check if currently authenticated with valid credentials.

        Returns:
            True if an access token is available, indicating
            successful authentication.
        """
        return self._access_token is not None

    def get_headers(self) -> dict:
        """Get HTTP headers for authenticated API requests.

        Provides the authorization header required for making authenticated
        requests to the Neuracore API.

        Returns:
            Dictionary containing the Authorization header with the bearer token.

        Raises:
            AuthenticationError: If not currently authenticated.
        """
        if not self.is_authenticated:
            raise AuthenticationError("Not authenticated. Please call login() first.")
        return {
            "Authorization": f"Bearer {self._access_token}",
        }


# Global instance
_auth = Auth()


def login(api_key: Optional[str] = None) -> None:
    """Global convenience function for authentication.

    Args:
        api_key: Optional API key for authentication.
    """
    _auth.login(api_key)


def logout() -> None:
    """Global convenience function for clearing authentication state.

    Raises:
        ConfigError: If saving the updated config fails
    """
    _auth.logout()


def get_auth() -> Auth:
    """Get the global Auth singleton instance.

    Returns:
        The global Auth instance used throughout the application.
    """
    return _auth
