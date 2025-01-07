import numpy as np
import pytest

import neuracore as nc
from neuracore.const import API_URL


def test_login_logout(temp_config_dir, mock_auth_requests, reset_neuracore):
    """Test login and logout functionality."""
    # Perform login
    nc.login("test_api_key")

    # Check authentication state
    auth = nc.auth.get_auth()
    assert auth.is_authenticated
    assert auth.api_key == "test_api_key"

    # Logout
    nc.logout()
    assert not auth.is_authenticated


def test_connect_robot(temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf):
    """Test robot connection."""
    # Ensure login first
    nc.login("test_api_key")

    # Mock robot creation endpoint with a full response
    mock_auth_requests.post(
        f"{API_URL}/api/robots", json="mock_robot_id", status_code=200
    )

    # Connect robot
    nc.connect_robot("test_robot", mock_urdf)

    # Verify robot connection
    assert nc.core._active_robot is not None
    assert nc.core._active_robot.name == "test_robot"


def test_log_actions(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf, monkeypatch
):
    """Test logging actions and sensor data."""
    # Ensure login and robot connection
    nc.login("test_api_key")

    # Mock robot creation
    mock_auth_requests.post(
        f"{API_URL}/api/robots", json="mock_robot_id", status_code=200
    )

    # Mock WebSocket-related behaviors
    def mock_websockets_connect(*args, **kwargs):
        class MockWebSocket:
            async def send(self, message):
                pass

            async def close(self):
                pass

        return MockWebSocket()

    monkeypatch.setattr("websockets.connect", mock_websockets_connect)

    # Connect robot
    nc.connect_robot("test_robot", mock_urdf)

    # Test logging functions
    try:
        # Logging with mocked websocket endpoints
        nc.log_joints({"vx300s_left/waist": 0.5, "vx300s_right/waist": -0.3})
        nc.log_action({"action1": 0.1, "action2": 0.2})

        # Test RGB logging with various input types
        # Normalized float image
        rgb_float = np.random.random((100, 100, 3)).astype(np.float32)
        nc.log_rgb("top_camera", rgb_float)

        # Uint8 image
        rgb_uint8 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        nc.log_rgb("front_camera", rgb_uint8)

        # Test depth logging
        depth = np.random.random((100, 100)).astype(np.float32) * 10  # meters
        nc.log_depth("depth_camera", depth)

    except Exception as e:
        pytest.fail(f"Logging functions raised unexpected exception: {e}")


def test_create_dataset(temp_config_dir, mock_auth_requests, reset_neuracore):
    """Test dataset creation."""
    # Ensure login
    nc.login("test_api_key")

    # Mock dataset creation endpoint
    mock_auth_requests.post(
        f"{API_URL}/api/datasets", json={"id": "test_dataset_id"}, status_code=200
    )

    # Create dataset
    nc.create_dataset("Test Dataset")

    # Verify dataset was created
    assert nc.core._active_dataset_id is not None


def test_connect_endpoint(temp_config_dir, mock_auth_requests, reset_neuracore):
    """Test connecting to an endpoint."""
    # Ensure login
    nc.login("test_api_key")

    # Mock endpoint list
    mock_auth_requests.get(
        f"{API_URL}/api/models/endpoints",
        json=[{"id": "test_endpoint_id", "name": "test_endpoint", "status": "active"}],
        status_code=200,
    )

    # Mock endpoint prediction
    mock_auth_requests.post(
        f"{API_URL}/api/models/endpoints/test_endpoint_id/predict",
        json={"predictions": [0.1, 0.2, 0.3]},
        status_code=200,
    )

    endpoint = nc.connect_endpoint("test_endpoint")

    # Test prediction
    pred = endpoint.predict(
        joint_positions=[0, 0, 0],
        images={"top": np.zeros((100, 100, 3), dtype=np.uint8)},
    )
    assert isinstance(pred, np.ndarray)


def test_connect_local_endpoint(
    temp_config_dir, mock_model_mar, reset_neuracore, monkeypatch
):
    """Test connecting to a local endpoint."""

    # Mock torchserve subprocess
    def mock_subprocess_popen(*args, **kwargs):
        class MockProcess:
            def __init__(self):
                self.stdout = None
                self.stderr = None

            def terminate(self):
                pass

            def wait(self):
                pass

        return MockProcess()

    # Mock requests to simulate TorchServe endpoints
    def mock_requests_get(url, *args, **kwargs):
        class MockResponse:
            status_code = 200

            def raise_for_status(self):
                pass

        return MockResponse()

    def mock_requests_post(url, *args, **kwargs):
        class MockResponse:
            status_code = 200

            def json(self):
                return {"predictions": [0.1, 0.2, 0.3]}

            def raise_for_status(self):
                pass

        return MockResponse()

    # Apply mocks
    monkeypatch.setattr("subprocess.Popen", mock_subprocess_popen)
    monkeypatch.setattr("requests.get", mock_requests_get)
    monkeypatch.setattr("requests.post", mock_requests_post)

    local_endpoint = nc.connect_local_endpoint(mock_model_mar)

    # Test prediction
    pred = local_endpoint.predict(
        joint_positions=[0, 0, 0],
        images={"top": np.zeros((100, 100, 3), dtype=np.uint8)},
    )
    assert isinstance(pred, np.ndarray)


def test_stop_functions(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf, monkeypatch
):
    """Test stop and stop_all functions."""
    # Ensure login and robot connection
    nc.login("test_api_key")

    # Mock robot creation
    mock_auth_requests.post(
        f"{API_URL}/api/robots", json="mock_robot_id", status_code=200
    )

    # Mock WebSocket-related behaviors
    def mock_websockets_connect(*args, **kwargs):
        class MockWebSocket:
            async def send(self, message):
                pass

            async def close(self):
                pass

        return MockWebSocket()

    monkeypatch.setattr("websockets.connect", mock_websockets_connect)

    # Connect robot
    nc.connect_robot("test_robot", mock_urdf)

    # Test stop functions
    try:
        nc.stop("test_robot")
        nc.stop_all()
    except Exception as e:
        pytest.fail(f"Stop functions raised unexpected exception: {e}")

    # Verify global state reset
    assert nc.core._active_robot is None
