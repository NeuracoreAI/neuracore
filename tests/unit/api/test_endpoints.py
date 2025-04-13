import numpy as np
import pytest

import neuracore as nc
from neuracore.core.const import API_URL
from neuracore.core.nc_types import DataType, ModelPrediction


def test_connect_endpoint(temp_config_dir, mock_auth_requests, reset_neuracore):
    """Test connecting to an endpoint."""
    # Ensure login
    nc.login("test_api_key")
    mock_auth_requests.post(f"{API_URL}/robots", json="mock_robot_id", status_code=200)
    nc.connect_robot("test_robot")

    # Mock endpoint list
    mock_auth_requests.get(
        f"{API_URL}/models/endpoints",
        json=[{"id": "test_endpoint_id", "name": "test_endpoint", "status": "active"}],
        status_code=200,
    )

    # Mock endpoint prediction
    mock_auth_requests.post(
        f"{API_URL}/models/endpoints/test_endpoint_id/predict",
        json={
            "predictions": ModelPrediction(
                outputs={DataType.JOINT_TARGET_POSITIONS: [0.1, 0.2, 0.3]}
            ).model_dump()
        },
        status_code=200,
    )

    endpoint = nc.connect_endpoint("test_endpoint")

    nc.log_joint_positions({"joint1": 0.5, "joint2": 0.5, "joint3": 0.5})
    nc.log_rgb("top", np.zeros((100, 100, 3), dtype=np.uint8))

    # Test prediction
    pred = endpoint.predict()
    assert isinstance(pred, ModelPrediction)
    assert pred.outputs[DataType.JOINT_TARGET_POSITIONS].shape == (3,)
    assert list(pred.outputs[DataType.JOINT_TARGET_POSITIONS]) == [0.1, 0.2, 0.3]


def test_connect_nonexistent_endpoint(
    temp_config_dir, mock_auth_requests, reset_neuracore
):
    """Test connecting to a non-existent endpoint."""
    # Ensure login
    nc.login("test_api_key")

    # Mock empty endpoint list
    mock_auth_requests.get(
        f"{API_URL}/models/endpoints",
        json=[],
        status_code=200,
    )

    # Attempt to connect to non-existent endpoint should raise an error
    with pytest.raises(Exception, match="No endpoint found with name or ID"):
        nc.connect_endpoint("non_existent_endpoint")


def test_connect_inactive_endpoint(
    temp_config_dir, mock_auth_requests, reset_neuracore
):
    """Test connecting to an inactive endpoint."""
    # Ensure login
    nc.login("test_api_key")

    # Mock endpoint list with inactive endpoint
    mock_auth_requests.get(
        f"{API_URL}/models/endpoints",
        json=[
            {"id": "test_endpoint_id", "name": "test_endpoint", "status": "deploying"}
        ],
        status_code=200,
    )

    # Attempt to connect to inactive endpoint should raise an error
    with pytest.raises(Exception, match="Endpoint test_endpoint is not active"):
        nc.connect_endpoint("test_endpoint")


def test_connect_local_endpoint(
    temp_config_dir, mock_model_mar, reset_neuracore, monkeypatch, mock_auth_requests
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

    mock_auth_requests.get(
        "http://localhost:8080/ping",
        status_code=200,
    )

    mock_auth_requests.post(
        "http://localhost:8080/predictions/robot_model",
        json={
            "predictions": ModelPrediction(
                outputs={DataType.JOINT_TARGET_POSITIONS: [0.1, 0.2, 0.3]}
            ).model_dump()
        },
        status_code=200,
    )

    monkeypatch.setattr("subprocess.Popen", mock_subprocess_popen)

    nc.login("test_api_key")
    mock_auth_requests.post(f"{API_URL}/robots", json="mock_robot_id", status_code=200)
    nc.connect_robot("test_robot")

    local_endpoint = nc.connect_local_endpoint(mock_model_mar)

    nc.log_joint_positions({"joint1": 0.5, "joint2": 0.5, "joint3": 0.5})
    nc.log_rgb("top", np.zeros((100, 100, 3), dtype=np.uint8))

    # Test prediction
    pred = local_endpoint.predict()
    assert isinstance(pred, ModelPrediction)


def test_deploy_model(temp_config_dir, mock_auth_requests, reset_neuracore):
    """Test model deployment."""
    # Ensure login
    nc.login("test_api_key")

    # Mock deployment endpoint
    mock_auth_requests.post(
        f"{API_URL}/models/deploy",
        json={"id": "endpoint_123", "name": "test_endpoint", "status": "deploying"},
        status_code=200,
    )

    # Deploy model
    result = nc.deploy_model("job_123", "test_endpoint")

    # Verify result
    assert result is not None
    assert result["id"] == "endpoint_123"
    assert result["name"] == "test_endpoint"
    assert result["status"] == "deploying"


def test_get_endpoint_status(temp_config_dir, mock_auth_requests, reset_neuracore):
    """Test getting endpoint status."""
    # Ensure login
    nc.login("test_api_key")

    # Mock endpoint status
    mock_auth_requests.get(
        f"{API_URL}/models/endpoints/endpoint_123",
        json={"id": "endpoint_123", "name": "test_endpoint", "status": "active"},
        status_code=200,
    )

    # Get status
    status = nc.get_endpoint_status("endpoint_123")

    # Verify status
    assert status == "active"


def test_delete_endpoint(temp_config_dir, mock_auth_requests, reset_neuracore):
    """Test deleting an endpoint."""
    # Ensure login
    nc.login("test_api_key")

    # Mock delete endpoint
    mock_auth_requests.delete(
        f"{API_URL}/models/endpoints/endpoint_123",
        status_code=200,
    )

    # Delete endpoint (should not raise exception)
    nc.delete_endpoint("endpoint_123")

    # Verify the delete request was made
    assert mock_auth_requests.called
    assert mock_auth_requests.request_history[-1].method == "DELETE"
    assert (
        mock_auth_requests.request_history[-1].url
        == f"{API_URL}/models/endpoints/endpoint_123"
    )


def test_deploy_model_failure(temp_config_dir, mock_auth_requests, reset_neuracore):
    """Test handling of deployment failures."""
    # Ensure login
    nc.login("test_api_key")

    # Mock deployment endpoint to return an error
    mock_auth_requests.post(
        f"{API_URL}/models/deploy",
        status_code=500,
        text="Internal Server Error",
    )

    # Attempt to deploy should raise an exception
    with pytest.raises(ValueError, match="Error deploying model"):
        nc.deploy_model("job_123", "test_endpoint")


def test_connect_local_endpoint_with_train_run(
    temp_config_dir, mock_auth_requests, reset_neuracore, monkeypatch
):
    """Test connecting to a local endpoint using a training run name."""
    # Ensure login
    nc.login("test_api_key")
    mock_auth_requests.post(f"{API_URL}/robots", json="mock_robot_id", status_code=200)
    nc.connect_robot("test_robot")

    # Mock training jobs endpoint
    mock_auth_requests.get(
        f"{API_URL}/training/jobs",
        json=[{
            "id": "job_123",
            "name": "test_run",
            "status": "completed",
        }],
        status_code=200,
    )

    # Mock model download
    mock_auth_requests.get(
        f"{API_URL}/training/jobs/job_123/model",
        content=b"dummy model content",
        status_code=200,
    )

    mock_auth_requests.get(
        "http://localhost:8080/ping",
        status_code=200,
    )

    mock_auth_requests.post(
        "http://localhost:8080/predictions/robot_model",
        json={
            "predictions": ModelPrediction(
                outputs={DataType.JOINT_TARGET_POSITIONS: [0.1, 0.2, 0.3]}
            ).model_dump()
        },
        status_code=200,
    )

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

    # Apply mocks
    monkeypatch.setattr("subprocess.Popen", mock_subprocess_popen)

    # Connect using train run name
    local_endpoint = nc.connect_local_endpoint(train_run_name="test_run")
    nc.log_joint_positions({"joint1": 0.5, "joint2": 0.5, "joint3": 0.5})
    nc.log_rgb("top", np.zeros((100, 100, 3), dtype=np.uint8))

    # Test prediction
    pred = local_endpoint.predict()
    assert isinstance(pred, ModelPrediction)


def test_connect_local_endpoint_invalid_args(
    temp_config_dir, mock_auth_requests, reset_neuracore
):
    """Test connecting to a local endpoint with invalid arguments."""
    # Ensure login
    nc.login("test_api_key")

    # Both arguments provided should raise an error
    with pytest.raises(
        ValueError, match="Cannot provide both path_to_model and train_run_name"
    ):
        nc.connect_local_endpoint(path_to_model="model.mar", train_run_name="test_run")

    # Neither argument provided should raise an error
    with pytest.raises(
        ValueError, match="Must provide either path_to_model or train_run_name"
    ):
        nc.connect_local_endpoint()
