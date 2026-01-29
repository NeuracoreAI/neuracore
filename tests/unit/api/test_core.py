import json

import requests_mock

import neuracore as nc
from neuracore.api import core as api_core
from neuracore.core.auth import get_auth
from neuracore.core.const import API_URL


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
        nc.login("test_api_key")

        # Check config file was created
        config_file = temp_config_dir / "config.json"
        assert config_file.exists()

        # Verify config contents
        with open(config_file) as f:
            config = json.load(f)
            assert config["api_key"] == "test_api_key"

    # Verify authentication state
    auth = get_auth()
    assert auth.is_authenticated
    assert auth.access_token == "test_token"


def test_logout(temp_config_dir, monkeypatch):
    """Test logout functionality."""
    # Create a dummy config directory
    config_file = temp_config_dir / "config.json"

    # Write initial config
    with open(config_file, "w") as f:
        json.dump({"api_key": "test_key", "current_org_id": "test-org-id"}, f)

    # Perform logout
    nc.logout()

    # Verify config contents
    with open(config_file) as f:
        config = json.load(f)
        assert config["api_key"] is None
        assert config["current_org_id"] is None


def test_auth_instance_singleton():
    """Test that Auth is a singleton."""
    auth1 = get_auth()
    auth2 = get_auth()

    assert auth1 is auth2, "Auth should be a singleton"


def test_auth_headers(temp_config_dir, monkeypatch):
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
        nc.login("test_api_key")

    # Get auth instance
    auth = get_auth()

    # Get headers
    headers = auth.get_headers()

    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test_token"


def test_login_logout(temp_config_dir, mock_auth_requests, reset_neuracore):
    """Test login and logout functionality."""
    # Perform login
    nc.login("test_api_key")

    # Check authentication state
    auth = get_auth()
    assert auth.is_authenticated

    # Logout
    nc.logout()
    assert not auth.is_authenticated


def test_connect_robot(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf, mocked_org_id
):
    """Test robot connection."""
    # Ensure login first
    nc.login("test_api_key")

    # Mock robot creation endpoint with a full response
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )

    # Connect robot
    robot = nc.connect_robot("test_robot", urdf_path=mock_urdf)

    # Verify robot connection
    assert robot is not None
    assert robot.name == "test_robot"


def test_update_robot_name_calls_underlying_and_returns_robot_id(monkeypatch):
    calls: list[tuple] = []

    def fake_update_robot_name(
        robot_name: str,
        new_robot_name: str,
        instance: int = 0,
        shared: bool = False,
    ) -> str:
        calls.append((robot_name, new_robot_name, instance, shared))
        return "robot_id_123"

    monkeypatch.setattr(api_core, "_update_robot_name", fake_update_robot_name)

    robot_id = nc.update_robot_name(
        "old_name_or_id", "new_name", instance=2, shared=True
    )

    assert robot_id == "robot_id_123"
    assert calls == [("old_name_or_id", "new_name", 2, True)]


def test_update_robot_name_forwards_arguments(monkeypatch):
    def fake_update_robot_name(
        robot_name: str,
        new_robot_name: str,
        instance: int = 0,
        shared: bool = False,
    ) -> str:
        return "robot_id_123"

    monkeypatch.setattr(api_core, "_update_robot_name", fake_update_robot_name)

    robot_id = nc.update_robot_name("old", "new")

    assert robot_id == "robot_id_123"
