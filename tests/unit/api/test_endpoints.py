import numpy as np
import pytest
from neuracore_types import JointData, SyncPoint

import neuracore as nc
from neuracore.core.const import API_URL


def test_connect_endpoint(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Test connecting to an endpoint."""
    # Ensure login
    nc.login("test_api_key")
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )
    nc.connect_robot("test_robot")

    # Mock endpoint list
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/models/endpoints",
        json=[{"id": "test_endpoint_id", "name": "test_endpoint", "status": "active"}],
        status_code=200,
    )

    # Mock endpoint prediction
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/models/endpoints/test_endpoint_id/predict",
        json=[
            SyncPoint(
                joint_target_positions=JointData(
                    values={"joint_1": 0.1, "joint_2": 0.2, "joint_3": 0.3},
                )
            ).model_dump()
            for _ in range(3)
        ],
        status_code=200,
    )

    endpoint = nc.policy_remote_server("test_endpoint")

    nc.log_joint_positions({"joint1": 0.5, "joint2": 0.5, "joint3": 0.5})
    nc.log_rgb("top", np.zeros((100, 100, 3), dtype=np.uint8))

    # Test prediction
    preds = endpoint.predict()
    assert isinstance(preds, list)
    pred = preds[0]
    assert pred.joint_target_positions is not None
    assert pred.joint_target_positions.values == {
        "joint_1": 0.1,
        "joint_2": 0.2,
        "joint_3": 0.3,
    }


def test_connect_nonexistent_endpoint(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Test connecting to a non-existent endpoint."""
    # Ensure login
    nc.login("test_api_key")

    # Mock empty endpoint list
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/models/endpoints",
        json=[],
        status_code=200,
    )

    # Attempt to connect to non-existent endpoint should raise an error
    with pytest.raises(
        Exception, match="No endpoint found with name: non_existent_endpoint"
    ):
        nc.policy_remote_server("non_existent_endpoint")


def test_connect_inactive_endpoint(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Test connecting to an inactive endpoint."""
    # Ensure login
    nc.login("test_api_key")

    # Mock endpoint list with inactive endpoint
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/models/endpoints",
        json=[
            {"id": "test_endpoint_id", "name": "test_endpoint", "status": "deploying"}
        ],
        status_code=200,
    )

    # Attempt to connect to inactive endpoint should raise an error
    with pytest.raises(Exception, match="Endpoint test_endpoint is not active"):
        nc.policy_remote_server("test_endpoint")


def test_connect_local_endpoint(
    temp_config_dir,
    mock_model_mar,
    reset_neuracore,
    monkeypatch,
    mock_auth_requests,
    mocked_org_id,
):
    """Test connecting to a local endpoint."""

    port = np.random.randint(8000, 9000)
    localhost = f"http://127.0.0.1:{port}"

    # Mock torchserve subprocess
    def mock_subprocess_popen(*args, **kwargs):
        class MockProcess:
            def __init__(self):
                self.stdout = None
                self.stderr = None
                self.pid = -1

            def terminate(self):
                pass

            def wait(self):
                pass

        return MockProcess()

    mock_auth_requests.get(
        f"{localhost}/ping",
        status_code=200,
    )

    mock_auth_requests.post(
        f"{localhost}/predict",
        json=[
            SyncPoint(
                joint_positions=JointData(
                    values={"joint_1": 0.5, "joint_2": 0.6, "joint_3": 0.7},
                ),
            ).model_dump()
            for _ in range(3)
        ],
        status_code=200,
    )

    monkeypatch.setattr("subprocess.Popen", mock_subprocess_popen)
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)

    nc.login("test_api_key")
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )
    nc.connect_robot("test_robot")

    local_endpoint = nc.policy_local_server(model_file=mock_model_mar, port=port)

    nc.log_joint_positions({"joint1": 0.5, "joint2": 0.5, "joint3": 0.5})
    nc.log_rgb("top", np.zeros((100, 100, 3), dtype=np.uint8))

    # Test prediction
    pred = local_endpoint.predict()
    assert isinstance(pred, list)
    assert len(pred) == 3
    for p in pred:
        assert isinstance(p, SyncPoint)
        assert p.joint_positions is not None
        assert p.joint_positions.values == {
            "joint_1": 0.5,
            "joint_2": 0.6,
            "joint_3": 0.7,
        }
    local_endpoint.disconnect()


def test_deploy_model(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Test model deployment."""
    # Ensure login
    nc.login("test_api_key")

    # Mock deployment endpoint
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/models/deploy",
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


def test_get_endpoint_status(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Test getting endpoint status."""
    # Ensure login
    nc.login("test_api_key")

    # Mock endpoint status
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/models/endpoints/endpoint_123",
        json={"id": "endpoint_123", "name": "test_endpoint", "status": "active"},
        status_code=200,
    )

    # Get status
    status = nc.get_endpoint_status("endpoint_123")

    # Verify status
    assert status == "active"


def test_delete_endpoint(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Test deleting an endpoint."""
    # Ensure login
    nc.login("test_api_key")

    # Mock delete endpoint
    mock_auth_requests.delete(
        f"{API_URL}/org/{mocked_org_id}/models/endpoints/endpoint_123",
        status_code=200,
    )

    # Delete endpoint (should not raise exception)
    nc.delete_endpoint("endpoint_123")

    # Verify the delete request was made
    assert mock_auth_requests.called
    assert mock_auth_requests.request_history[-1].method == "DELETE"
    assert (
        mock_auth_requests.request_history[-1].url
        == f"{API_URL}/org/{mocked_org_id}/models/endpoints/endpoint_123"
    )


def test_deploy_model_failure(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Test handling of deployment failures."""
    # Ensure login
    nc.login("test_api_key")

    # Mock deployment endpoint to return an error
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/models/deploy",
        status_code=500,
        text="Internal Server Error",
    )

    # Attempt to deploy should raise an exception
    with pytest.raises(ValueError, match="Error deploying model"):
        nc.deploy_model("job_123", "test_endpoint")


def test_connect_local_endpoint_with_train_run(
    temp_config_dir, mock_auth_requests, reset_neuracore, monkeypatch, mocked_org_id
):
    """Test connecting to a local endpoint using a training run name."""
    # Ensure login
    nc.login("test_api_key")
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )
    nc.connect_robot("test_robot")
    port = np.random.randint(8000, 9000)

    # Mock training jobs endpoint
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/training/jobs",
        json=[{
            "id": "job_123",
            "name": "test_run",
            "status": "completed",
        }],
        status_code=200,
    )

    localhost = f"http://127.0.0.1:{port}"

    # Mock model download
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/training/jobs/job_123/model_url",
        json={
            "url": f"{localhost}/model.nc.zip",
        },
        status_code=200,
    )
    mock_auth_requests.get(
        f"{localhost}/model.nc.zip",
        content=b"dummy model content",
        status_code=200,
    )

    mock_auth_requests.get(
        f"{localhost}/ping",
        status_code=200,
    )

    mock_auth_requests.post(
        f"{localhost}/predict",
        json=[
            SyncPoint(
                joint_positions=JointData(
                    values={"joint_1": 0.5, "joint_2": 0.6, "joint_3": 0.7},
                )
            ).model_dump()
            for _ in range(3)
        ],
        status_code=200,
    )

    # Mock torchserve subprocess
    def mock_subprocess_popen(*args, **kwargs):
        class MockProcess:
            def __init__(self):
                self.stdout = None
                self.stderr = None
                self.pid = -1

            def terminate(self):
                pass

            def wait(self):
                pass

        return MockProcess()

    # Apply mocks
    monkeypatch.setattr("subprocess.Popen", mock_subprocess_popen)
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)

    # Connect using train run name
    local_endpoint = nc.policy_local_server(train_run_name="test_run", port=port)
    nc.log_joint_positions({"joint1": 0.5, "joint2": 0.5, "joint3": 0.5})
    nc.log_rgb("top", np.zeros((100, 100, 3), dtype=np.uint8))

    # Test prediction
    pred = local_endpoint.predict()
    assert isinstance(pred, list)
    assert len(pred) == 3
    for p in pred:
        assert isinstance(p, SyncPoint)
        assert p.joint_positions is not None
        assert p.joint_positions.values == {
            "joint_1": 0.5,
            "joint_2": 0.6,
            "joint_3": 0.7,
        }
    local_endpoint.disconnect()


def test_connect_local_endpoint_invalid_args(
    temp_config_dir, mock_auth_requests, reset_neuracore
):
    """Test connecting to a local endpoint with invalid arguments."""
    # Ensure login
    nc.login("test_api_key")

    # Both arguments provided should raise an error
    with pytest.raises(
        ValueError, match="Cannot specify both train_run_name and model_file"
    ):
        nc.policy_local_server(model_file="model.nc.zip", train_run_name="test_run")

    # Neither argument provided should raise an error
    with pytest.raises(
        ValueError, match="Must specify either train_run_name or model_file"
    ):
        nc.policy_local_server()
