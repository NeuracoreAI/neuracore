import json
import os
from pathlib import Path
from typing import Optional

import requests

from .const import API_URL
from .exceptions import AuthenticationError
from .generate_api_key import generate_api_key

CONFIG_DIR = Path.home() / ".neuracore"
CONFIG_FILE = "config.json"


class Auth:
    _instance = None
    _api_key: Optional[str] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._load_config()
        self._access_token = None

    def _load_config(self) -> None:
        """Load configuration from disk if it exists."""
        config_file = CONFIG_DIR / CONFIG_FILE
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
                self._api_key = config.get("api_key")

    def _save_config(self) -> None:
        """Save current configuration to disk."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config_file = CONFIG_DIR / CONFIG_FILE
        with open(config_file, "w") as f:
            json.dump({"api_key": self._api_key}, f)

    def login(self, api_key: Optional[str] = None) -> None:
        """
        Authenticate with the NeuraCore server using an API key.

        Args:
            api_key: Optional API key. If not provided, will try to use environment
                    variable NEURACORE_API_KEY or previously saved config.
        """
        self._api_key = api_key or os.environ.get("NEURACORE_API_KEY") or self._api_key

        if not self._api_key:
            print("No API key provided. Attempting to log you in...")
            self._api_key = generate_api_key()

        # Verify API key with server and get access token
        try:
            response = requests.post(
                f"{API_URL}/auth/verify-api-key",
                json={"api_key": self._api_key},
            )
            if response.status_code != 200:
                raise AuthenticationError(
                    "Could not verify API key. Please check your key and try again."
                )
            token_data = response.json()
            self._access_token = token_data["access_token"]
        except requests.exceptions.RequestException:
            raise AuthenticationError(
                "Could not verify API key. Please check your key and try again."
            )

        # Save configuration if verification successful
        self._save_config()

    def logout(self) -> None:
        """Clear authentication state."""
        self._api_key = None
        self._access_token = None
        config_file = CONFIG_DIR / CONFIG_FILE
        if config_file.exists():
            config_file.unlink()

    def validate_version(self) -> None:
        """Validate the NeuraCore version."""
        # Placeholder for version validation logic
        import neuracore as nc

        response = requests.get(
            f"{API_URL}/auth/verify-version",
            params={"version": nc.__version__},
        )
        if response.status_code != 200:
            raise AuthenticationError(
                f"Version validation failed: {response.json().get('detail')}"
            )

    @property
    def api_key(self) -> Optional[str]:
        """Get the current API key."""
        return self._api_key

    @property
    def access_token(self) -> Optional[str]:
        """Get the current access token."""
        return self._access_token

    @property
    def is_authenticated(self) -> bool:
        """Check if currently authenticated."""
        return self._api_key is not None and self._access_token is not None

    def get_headers(self) -> dict:
        """Get headers for authenticated requests."""
        if not self.is_authenticated:
            raise AuthenticationError("Not authenticated. Please call login() first.")
        return {
            "Authorization": f"Bearer {self._access_token}",
        }


# Global instance
_auth = Auth()


def login(api_key: Optional[str] = None) -> None:
    """Global login function."""
    _auth.login(api_key)


def logout() -> None:
    """Global logout function."""
    _auth.logout()


def get_auth() -> Auth:
    """Get the global Auth instance."""
    return _auth
