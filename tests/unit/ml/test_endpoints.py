from typing import cast

import numpy as np
import pytest
import torch
from neuracore_types import BatchedJointData, BatchedNCData, DataType
from neuracore_types.endpoints.endpoint_requests import DeploymentRequest
from neuracore_types.training.training import GPUType

import neuracore as nc
from neuracore.core.const import API_URL

B = 1
PREDICTION_HORIZON = 3
FAKE_PREDICTED_DATA: dict[DataType, dict[str, BatchedNCData]] = {
    DataType.JOINT_TARGET_POSITIONS: {
        "joint1": BatchedJointData(value=torch.full((B, PREDICTION_HORIZON, 1), 0.1)),
        "joint2": BatchedJointData(value=torch.full((B, PREDICTION_HORIZON, 1), 0.2)),
        "joint3": BatchedJointData(value=torch.full((B, PREDICTION_HORIZON, 1), 0.3)),
    }
}
FAKE_PREDICTED_DATA_JSON = {
    k: {name: data.model_dump(mode="json") for name, data in v.items()}
    for k, v in FAKE_PREDICTED_DATA.items()
}


def _indexed_names(names: list[str] | tuple[str, ...]) -> dict[int, str]:
    return {index: name for index, name in enumerate(names)}


INPUT_EMBODIMENT_DESCRIPTION = {
    DataType.JOINT_POSITIONS: _indexed_names(["joint1", "joint2", "joint3"]),
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: _indexed_names(["left_arm", "right_arm"]),
    DataType.RGB_IMAGES: _indexed_names(["top_camera"]),
}

OUTPUT_EMBODIMENT_DESCRIPTION = {
    DataType.JOINT_TARGET_POSITIONS: _indexed_names(["joint1", "joint2", "joint3"]),
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: _indexed_names(["left_arm", "right_arm"]),
}


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

        def poll(self):
            pass

    return MockProcess()


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

    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/models/endpoints/test_endpoint_id/predict",
        json=FAKE_PREDICTED_DATA_JSON,
        status_code=200,
    )

    endpoint = nc.policy_remote_server("test_endpoint")

    nc.log_joint_positions(positions={"joint1": 0.5, "joint2": 0.5, "joint3": 0.5})
    nc.log_rgb("top_camera", np.zeros((100, 100, 3), dtype=np.uint8))

    # Test prediction
    preds = endpoint.predict()
    assert isinstance(preds, dict)
    assert isinstance(preds, dict)
    assert DataType.JOINT_TARGET_POSITIONS in preds
    assert (
        preds[DataType.JOINT_TARGET_POSITIONS].keys()
        == FAKE_PREDICTED_DATA[DataType.JOINT_TARGET_POSITIONS].keys()
    )
    pred_values = [
        cast(BatchedJointData, bjp).value.numpy()
        for bjp in preds[DataType.JOINT_TARGET_POSITIONS].values()
    ]
    expected_values = [
        cast(BatchedJointData, bjp).value.numpy()
        for bjp in FAKE_PREDICTED_DATA[DataType.JOINT_TARGET_POSITIONS].values()
    ]
    assert np.array_equal(pred_values, expected_values)


def test_remote_endpoint_filters_sync_point_from_endpoint_input_description(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Remote endpoints should only receive data types declared in metadata."""
    nc.login("test_api_key")
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )
    nc.connect_robot("test_robot")

    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/models/endpoints",
        json=[{
            "id": "test_endpoint_id",
            "name": "test_endpoint",
            "status": "active",
            "input_embodiment_description": {
                "JOINT_POSITIONS": {
                    "0": "joint1",
                    "1": "joint2",
                    "2": "joint3",
                }
            },
        }],
        status_code=200,
    )
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/models/endpoints/test_endpoint_id/predict",
        json=FAKE_PREDICTED_DATA_JSON,
        status_code=200,
    )

    endpoint = nc.policy_remote_server("test_endpoint")

    nc.log_joint_positions(positions={"joint1": 0.5, "joint2": 0.5, "joint3": 0.5})
    nc.log_rgb("top_camera", np.zeros((100, 100, 3), dtype=np.uint8))

    endpoint.predict()

    request_body = mock_auth_requests.request_history[-1].json()
    assert set(request_body["data"]) == {"JOINT_POSITIONS"}
    assert set(request_body["data"]["JOINT_POSITIONS"]) == {
        "joint1",
        "joint2",
        "joint3",
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


def test_connect_active_endpoint_with_duplicate_name(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Test connecting to active endpoint when duplicate names exist."""
    nc.login("test_api_key")
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )
    nc.connect_robot("test_robot")

    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/models/endpoints",
        json=[
            {"id": "inactive_endpoint_id", "name": "test_endpoint", "status": "failed"},
            {"id": "active_endpoint_id", "name": "test_endpoint", "status": "active"},
        ],
        status_code=200,
    )
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/models/endpoints/active_endpoint_id/predict",
        json=FAKE_PREDICTED_DATA_JSON,
        status_code=200,
    )

    endpoint = nc.policy_remote_server("test_endpoint")

    nc.log_joint_positions(positions={"joint1": 0.5, "joint2": 0.5, "joint3": 0.5})
    nc.log_rgb("top_camera", np.zeros((100, 100, 3), dtype=np.uint8))

    preds = endpoint.predict()
    assert DataType.JOINT_TARGET_POSITIONS in preds
    assert (
        preds[DataType.JOINT_TARGET_POSITIONS].keys()
        == FAKE_PREDICTED_DATA[DataType.JOINT_TARGET_POSITIONS].keys()
    )


def test_connect_multiple_active_endpoints_with_duplicate_name(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Test error when multiple active endpoints have the same name."""
    nc.login("test_api_key")
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/models/endpoints",
        json=[
            {"id": "active_endpoint_1", "name": "test_endpoint", "status": "active"},
            {"id": "active_endpoint_2", "name": "test_endpoint", "status": "active"},
        ],
        status_code=200,
    )

    with pytest.raises(
        Exception, match="Multiple active endpoints found with name test_endpoint"
    ):
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

    mock_auth_requests.get(
        f"{localhost}/ping",
        status_code=200,
    )

    mock_auth_requests.post(
        f"{localhost}/predict",
        json=FAKE_PREDICTED_DATA_JSON,
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

    local_endpoint = nc.policy_local_server(
        input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
        output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
        model_file=mock_model_mar,
        port=port,
    )

    nc.log_joint_positions(positions={"joint1": 0.5, "joint2": 0.5, "joint3": 0.5})
    nc.log_rgb("top_camera", np.zeros((100, 100, 3), dtype=np.uint8))

    preds = local_endpoint.predict()
    assert isinstance(preds, dict)
    assert DataType.JOINT_TARGET_POSITIONS in preds
    assert (
        preds[DataType.JOINT_TARGET_POSITIONS].keys()
        == FAKE_PREDICTED_DATA[DataType.JOINT_TARGET_POSITIONS].keys()
    )
    pred_values = [
        cast(BatchedJointData, bjp).value.numpy()
        for bjp in preds[DataType.JOINT_TARGET_POSITIONS].values()
    ]
    expected_values = [
        cast(BatchedJointData, bjp).value.numpy()
        for bjp in FAKE_PREDICTED_DATA[DataType.JOINT_TARGET_POSITIONS].values()
    ]
    assert np.array_equal(pred_values, expected_values)

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
    result = nc.deploy_model(
        job_id="job_123",
        name="test_endpoint",
        input_embodiment_description={
            DataType.RGB_IMAGES: _indexed_names(["top_camera"]),
            DataType.JOINT_POSITIONS: _indexed_names(["joint1", "joint2", "joint3"]),
        },
        output_embodiment_description={
            DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                ["joint1", "joint2", "joint3"]
            ),
        },
    )

    # Verify result
    assert result is not None
    assert result["id"] == "endpoint_123"
    assert result["name"] == "test_endpoint"
    assert result["status"] == "deploying"
    request_body = mock_auth_requests.request_history[-1].json()
    assert request_body["input_embodiment_description"]["RGB_IMAGES"] == {
        "0": "top_camera"
    }
    assert request_body["input_embodiment_description"]["JOINT_POSITIONS"] == {
        "0": "joint1",
        "1": "joint2",
        "2": "joint3",
    }
    assert request_body["output_embodiment_description"]["JOINT_TARGET_POSITIONS"] == {
        "0": "joint1",
        "1": "joint2",
        "2": "joint3",
    }
    assert request_body == DeploymentRequest(
        training_id="job_123",
        name="test_endpoint",
        input_embodiment_description={
            DataType.RGB_IMAGES: _indexed_names(["top_camera"]),
            DataType.JOINT_POSITIONS: _indexed_names(["joint1", "joint2", "joint3"]),
        },
        output_embodiment_description={
            DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                ["joint1", "joint2", "joint3"]
            ),
        },
        config={"gpu_type": GPUType.NVIDIA_TESLA_V100},
    ).model_dump(mode="json")


def test_deploy_model_includes_ttl_and_default_config(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Test model deployment serializes ttl and the default config."""
    nc.login("test_api_key")
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/models/deploy",
        json={"id": "endpoint_123", "name": "test_endpoint", "status": "deploying"},
        status_code=200,
    )

    nc.deploy_model(
        job_id="job_123",
        name="test_endpoint",
        input_embodiment_description={
            DataType.RGB_IMAGES: _indexed_names(["top_camera"]),
        },
        output_embodiment_description={
            DataType.JOINT_TARGET_POSITIONS: _indexed_names(["joint1"]),
        },
        ttl=1800,
    )

    request_body = mock_auth_requests.request_history[-1].json()
    assert request_body["ttl"] == 1800
    assert (
        request_body["config"]
        == DeploymentRequest(
            training_id="job_123",
            name="test_endpoint",
            ttl=1800,
            input_embodiment_description={
                DataType.RGB_IMAGES: _indexed_names(["top_camera"]),
            },
            output_embodiment_description={
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(["joint1"]),
            },
            config={"gpu_type": GPUType.NVIDIA_TESLA_V100},
        ).model_dump(mode="json")["config"]
    )


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
        nc.deploy_model(
            job_id="job_123",
            name="test_endpoint",
            input_embodiment_description={
                DataType.RGB_IMAGES: _indexed_names(["top_camera"]),
                DataType.JOINT_POSITIONS: _indexed_names(
                    ["joint1", "joint2", "joint3"]
                ),
            },
            output_embodiment_description={
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    ["joint1", "joint2", "joint3"]
                ),
            },
        )


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
        json=FAKE_PREDICTED_DATA_JSON,
        status_code=200,
    )

    # Apply mocks
    monkeypatch.setattr("subprocess.Popen", mock_subprocess_popen)
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)

    # Connect using train run name
    local_endpoint = nc.policy_local_server(
        input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
        output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
        train_run_name="test_run",
        port=port,
    )
    nc.log_joint_positions(positions={"joint1": 0.5, "joint2": 0.5, "joint3": 0.5})
    nc.log_rgb("top_camera", np.zeros((100, 100, 3), dtype=np.uint8))

    preds = local_endpoint.predict()
    assert isinstance(preds, dict)
    assert DataType.JOINT_TARGET_POSITIONS in preds
    assert (
        preds[DataType.JOINT_TARGET_POSITIONS].keys()
        == FAKE_PREDICTED_DATA[DataType.JOINT_TARGET_POSITIONS].keys()
    )
    pred_values = [
        cast(BatchedJointData, bjp).value.numpy()
        for bjp in preds[DataType.JOINT_TARGET_POSITIONS].values()
    ]
    expected_values = [
        cast(BatchedJointData, bjp).value.numpy()
        for bjp in FAKE_PREDICTED_DATA[DataType.JOINT_TARGET_POSITIONS].values()
    ]
    assert np.array_equal(pred_values, expected_values)

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
        nc.policy_local_server(
            input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
            output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
            model_file="model.nc.zip",
            train_run_name="test_run",
        )

    # Neither argument provided should raise an error
    with pytest.raises(
        ValueError, match="Must specify either train_run_name or model_file"
    ):
        nc.policy_local_server(
            input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
            output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
        )
