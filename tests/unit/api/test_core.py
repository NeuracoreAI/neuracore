import json
import time
from unittest import mock

import pytest
import requests
import requests_mock
from neuracore_types import RecordingNotificationType

import neuracore as nc
from neuracore.api import core as api_core
from neuracore.core.auth import get_auth
from neuracore.core.const import API_URL
from neuracore.core.exceptions import AuthenticationError, RecordingError


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
        m.get(f"{API_URL}/auth/verify-version", status_code=200)

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
        m.get(f"{API_URL}/auth/verify-version", status_code=200)

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


def test_login_version_mismatch_surfaces_installed_version(
    temp_config_dir, reset_neuracore
):
    """Test version validation surfaces the installed version and mitigation steps."""
    with requests_mock.Mocker() as m:
        m.get(
            f"{API_URL}/auth/verify-version",
            json={"detail": {"error": "Neuracore client version mismatch"}},
            status_code=400,
        )

        with pytest.raises(AuthenticationError) as exc_info:
            nc.login("test_api_key")

    message = str(exc_info.value)
    assert "Neuracore client version mismatch" in message
    assert f"Installed version: {nc.__version__}" in message
    assert "pip install --upgrade neuracore" in message


def test_login_version_check_connection_error_surfaces_cleanly(
    monkeypatch, reset_neuracore
):
    def raise_connection_error(*args, **kwargs):
        raise requests.exceptions.ConnectionError(
            "Connection reset by peer during verify-version"
        )

    monkeypatch.setattr(
        "neuracore.core.auth.thread_local_session",
        lambda: type("_Session", (), {"get": raise_connection_error})(),
    )

    with pytest.raises(AuthenticationError) as exc_info:
        nc.login("test_api_key")

    assert "Connection reset by peer during verify-version" in str(exc_info.value)


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


def test_stop_recording_no_wait_forwards_flag_and_skips_tracking(monkeypatch) -> None:
    robot = mock.Mock()
    robot.is_recording.return_value = True
    robot.get_current_recording_id.return_value = "rec-123"
    manager = mock.Mock()
    monkeypatch.setattr(api_core, "_get_robot", lambda robot_name, instance: robot)
    monkeypatch.setattr(api_core, "get_recording_state_manager", lambda: manager)
    monkeypatch.setattr(api_core, "is_rust_daemon_enabled", lambda: False)

    nc.stop_recording(wait=False)

    robot.stop_recording.assert_called_once_with(
        "rec-123", wait_for_producer_drain=False, timestamp=None
    )
    manager.start_tracking_recording.assert_not_called()
    manager.wait_for_terminal_notification.assert_not_called()


def test_stop_recording_wait_returns_on_saved_notification(monkeypatch) -> None:
    robot = mock.Mock()
    robot.is_recording.return_value = True
    robot.get_current_recording_id.return_value = "rec-123"
    manager = mock.Mock()
    manager.wait_for_terminal_notification.return_value = (
        RecordingNotificationType.SAVED
    )
    monkeypatch.setattr(api_core, "_get_robot", lambda robot_name, instance: robot)
    monkeypatch.setattr(api_core, "get_recording_state_manager", lambda: manager)
    monkeypatch.setattr(api_core, "is_rust_daemon_enabled", lambda: False)

    def fail_if_called(recording_id: str) -> object:
        raise AssertionError("backend lookup should be skipped on SAVED")

    monkeypatch.setattr(api_core.backend_utils, "get_recording", fail_if_called)

    nc.stop_recording(wait=True, timeout_s=17.0)

    robot.stop_recording.assert_called_once_with(
        "rec-123", wait_for_producer_drain=True, timestamp=None
    )
    manager.start_tracking_recording.assert_called_once_with("rec-123")
    manager.stop_tracking_recording.assert_called_once_with("rec-123")
    manager.wait_for_terminal_notification.assert_called_once_with(
        "rec-123", timeout_s=api_core.RECORDING_SAVE_POLL_INTERVAL_S
    )


def test_stop_recording_wait_raises_and_stops_tracking_on_discarded(
    monkeypatch,
) -> None:
    robot = mock.Mock()
    robot.is_recording.return_value = True
    robot.get_current_recording_id.return_value = "rec-123"
    manager = mock.Mock()
    manager.wait_for_terminal_notification.return_value = (
        RecordingNotificationType.DISCARDED
    )
    monkeypatch.setattr(api_core, "_get_robot", lambda robot_name, instance: robot)
    monkeypatch.setattr(api_core, "get_recording_state_manager", lambda: manager)
    monkeypatch.setattr(api_core, "is_rust_daemon_enabled", lambda: False)

    with pytest.raises(RecordingError, match="discarded"):
        nc.stop_recording(wait=True)

    manager.stop_tracking_recording.assert_called_once_with("rec-123")


def test_stop_recording_wait_falls_back_to_backend_lookup(monkeypatch) -> None:
    robot = mock.Mock()
    robot.is_recording.return_value = True
    robot.get_current_recording_id.return_value = "rec-123"
    manager = mock.Mock()
    manager.wait_for_terminal_notification.return_value = None
    monkeypatch.setattr(api_core, "_get_robot", lambda robot_name, instance: robot)
    monkeypatch.setattr(api_core, "get_recording_state_manager", lambda: manager)
    monkeypatch.setattr(api_core, "is_rust_daemon_enabled", lambda: False)
    monkeypatch.setattr(
        api_core.backend_utils, "get_recording", lambda recording_id: object()
    )

    nc.stop_recording(wait=True, timeout_s=0.05)


def test_stop_recording_wait_raises_on_timeout(monkeypatch) -> None:
    robot = mock.Mock()
    robot.is_recording.return_value = True
    robot.get_current_recording_id.return_value = "rec-123"
    manager = mock.Mock()

    def wait_out_interval(recording_id: str, timeout_s: float | None = None) -> None:
        if timeout_s:
            time.sleep(timeout_s)
        return None

    manager.wait_for_terminal_notification.side_effect = wait_out_interval
    monkeypatch.setattr(api_core, "_get_robot", lambda robot_name, instance: robot)
    monkeypatch.setattr(api_core, "get_recording_state_manager", lambda: manager)
    monkeypatch.setattr(api_core, "is_rust_daemon_enabled", lambda: False)
    monkeypatch.setattr(
        api_core.backend_utils, "get_recording", lambda recording_id: None
    )

    with pytest.raises(RecordingError, match="not saved within"):
        nc.stop_recording(wait=True, timeout_s=0.05)
