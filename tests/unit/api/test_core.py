import json
from unittest.mock import Mock

import pytest
import requests
import requests_mock

import neuracore as nc
from neuracore.api import core as api_core
from neuracore.core import robot as core_robot
from neuracore.core.auth import get_auth
from neuracore.core.const import API_URL
from neuracore.core.exceptions import AuthenticationError


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
    temp_config_dir,
    mock_auth_requests,
    reset_neuracore,
    mock_urdf,
    mocked_org_id,
    monkeypatch,
):
    """Test robot connection."""
    session_factory = Mock(wraps=core_robot.thread_local_session)
    monkeypatch.setattr(core_robot, "thread_local_session", session_factory)
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
    session_factory.assert_called_once_with(retry_transient=True)


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


def test_get_active_data_traces_uses_request_timeout(monkeypatch) -> None:
    response = Mock(status_code=200)
    response.json.return_value = []

    session = Mock()
    session.get.return_value = response

    auth = Mock()
    auth.get_headers.return_value = {"Authorization": "Bearer test-token"}

    monkeypatch.setattr(
        api_core.backend_utils,
        "thread_local_session",
        lambda: session,
    )
    monkeypatch.setattr(
        api_core.backend_utils,
        "get_current_org",
        lambda: "org-123",
    )
    monkeypatch.setattr(
        api_core.backend_utils,
        "get_auth",
        lambda: auth,
    )

    result = api_core.backend_utils.get_active_data_traces("recording-123")

    assert result == []
    session.get.assert_called_once()

    request_kwargs = session.get.call_args.kwargs
    assert request_kwargs.get("timeout") == 10


@pytest.mark.parametrize("complete", [True, False])
def test_is_recording_upload_complete_returns_backend_state(
    monkeypatch,
    complete: bool,
) -> None:
    response = Mock()
    response.json.return_value = complete

    session = Mock()
    session.get.return_value = response

    auth = Mock()
    auth.get_headers.return_value = {"Authorization": "Bearer test-token"}

    session_factory = Mock(return_value=session)

    monkeypatch.setattr(
        api_core.backend_utils,
        "thread_local_session",
        session_factory,
    )
    monkeypatch.setattr(
        api_core.backend_utils,
        "get_current_org",
        lambda: "org-123",
    )
    monkeypatch.setattr(
        api_core.backend_utils,
        "get_auth",
        lambda: auth,
    )

    result = api_core.backend_utils.is_recording_upload_complete("recording-123")

    session_factory.assert_called_once_with(retry_read_timeout=True)

    assert result is complete
    session.get.assert_called_once_with(
        (f"{API_URL}/org/org-123/recording/" "recording-123/traces/complete"),
        headers={"Authorization": "Bearer test-token"},
        timeout=3,
    )
    response.raise_for_status.assert_called_once_with()


def test_is_recording_upload_complete_rejects_invalid_response(
    monkeypatch,
) -> None:
    response = Mock()
    response.json.return_value = {"complete": True}

    session = Mock()
    session.get.return_value = response

    auth = Mock()
    auth.get_headers.return_value = {}

    session_factory = Mock(return_value=session)

    monkeypatch.setattr(
        api_core.backend_utils,
        "thread_local_session",
        session_factory,
    )
    monkeypatch.setattr(
        api_core.backend_utils,
        "get_current_org",
        lambda: "org-123",
    )
    monkeypatch.setattr(
        api_core.backend_utils,
        "get_auth",
        lambda: auth,
    )

    with pytest.raises(
        TypeError,
        match="Expected a boolean recording upload-completion response",
    ):
        api_core.backend_utils.is_recording_upload_complete("recording-123")

    session_factory.assert_called_once_with(retry_read_timeout=True)


def test_stop_recording_forwards_wait_flag_to_robot(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []
    completion_checks: list[str] = []

    class _FakeRobot:
        def is_recording(self) -> bool:
            return True

        def get_current_recording_id(self) -> str:
            return "rec-123"

        def stop_recording(
            self,
            recording_id: str,
            *,
            wait_for_producer_drain: bool = True,
            timestamp: float | None = None,
        ) -> None:
            calls.append((recording_id, wait_for_producer_drain))

    def is_upload_complete(recording_id: str) -> bool:
        completion_checks.append(recording_id)
        return True

    monkeypatch.setattr(
        api_core,
        "_get_robot",
        lambda robot_name, instance: _FakeRobot(),
    )
    monkeypatch.setattr(
        api_core,
        "is_rust_daemon_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        api_core.backend_utils,
        "is_recording_upload_complete",
        is_upload_complete,
    )

    nc.stop_recording(wait=False)
    nc.stop_recording(wait=True)

    assert calls == [
        ("rec-123", False),
        ("rec-123", True),
    ]
    assert completion_checks == ["rec-123"]


def test_stop_recording_wait_times_out_when_upload_never_completes(
    monkeypatch,
) -> None:
    poll_count = 0

    class _FakeRobot:
        def is_recording(self) -> bool:
            return True

        def get_current_recording_id(self) -> str:
            return "rec-123"

        def stop_recording(
            self,
            recording_id: str,
            *,
            wait_for_producer_drain: bool = True,
            timestamp: float | None = None,
        ) -> None:
            pass

    def upload_never_completes(recording_id: str) -> bool:
        nonlocal poll_count
        poll_count += 1
        return False

    # First call creates the deadline:
    #     0.0 + 1.0 = 1.0
    # Second call allows one poll:
    #     0.0 < 1.0
    # Third call passes the deadline:
    #     2.0 > 1.0
    clock = iter((0.0, 0.0, 2.0))

    monkeypatch.setattr(
        api_core,
        "_get_robot",
        lambda robot_name, instance: _FakeRobot(),
    )
    monkeypatch.setattr(
        api_core,
        "is_rust_daemon_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        api_core.backend_utils,
        "is_recording_upload_complete",
        upload_never_completes,
    )
    monkeypatch.setattr(
        api_core.time,
        "monotonic",
        lambda: next(clock),
    )
    monkeypatch.setattr(
        api_core.time,
        "sleep",
        lambda _seconds: None,
    )

    with pytest.raises(
        TimeoutError,
        match="Timed out waiting for recording uploads to complete",
    ):
        nc.stop_recording(wait=True, wait_timeout_s=1.0)

    assert poll_count == 1
