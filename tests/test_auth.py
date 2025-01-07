import json
import pathlib

import requests_mock

from neuracore.auth import get_auth, login, logout
from neuracore.const import API_URL


def test_login_with_api_key(temp_config_dir, monkeypatch):
    """Test login functionality."""
    # Create mock requests
    with requests_mock.Mocker() as m:
        # Mock the authentication endpoint to match the API_URL
        m.post(
            f"{API_URL}/auth/verify-api-key",
            json={"access_token": "test_token"},
            status_code=200,
        )

        # Perform login
        login("test_api_key")

        # Check config file was created
        config_file = pathlib.Path(temp_config_dir) / ".neuracore" / "config.json"
        assert config_file.exists()

        # Verify config contents
        with open(config_file) as f:
            config = json.load(f)
            assert config["api_key"] == "test_api_key"

    # Verify authentication state
    auth = get_auth()
    assert auth.is_authenticated
    assert auth.api_key == "test_api_key"


def test_logout(temp_config_dir, monkeypatch):
    """Test logout functionality."""
    # Create a dummy config directory
    config_dir = pathlib.Path(temp_config_dir) / ".neuracore"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.json"

    # Write initial config
    with open(config_file, "w") as f:
        json.dump({"api_key": "test_key"}, f)

    # Perform logout
    logout()

    # Check config file is deleted
    assert not config_file.exists()


def test_auth_instance_singleton():
    """Test that Auth is a singleton."""
    auth1 = get_auth()
    auth2 = get_auth()

    assert auth1 is auth2, "Auth should be a singleton"


def test_auth_headers(monkeypatch):
    """Test generation of authentication headers."""
    # Create mock authentication
    with requests_mock.Mocker() as m:
        # Mock the authentication endpoint to match the API_URL
        m.post(
            f"{API_URL}/auth/verify-api-key",
            json={"access_token": "test_token"},
            status_code=200,
        )

        # Perform login
        login("test_api_key")

    # Get auth instance
    auth = get_auth()

    # Get headers
    headers = auth.get_headers()

    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test_token"
