"""Tests for server endpoint connection, deployment, and inference."""

import re
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import requests
import torch
from neuracore_types import BatchedJointData, BatchedNCData, DataType
from neuracore_types.endpoints.endpoint_requests import DeploymentRequest
from neuracore_types.training.training import GPUType

import neuracore as nc
from neuracore.core.const import API_URL
from neuracore.core.endpoint import EndpointError, Policy
from neuracore.core.exceptions import InsufficientSynchronizedPointError

B = 1
PREDICTION_HORIZON = 3
TEST_API_KEY = "test_api_key"
TEST_ROBOT_ID = "test_robot"
TEST_ROBOT_PAYLOAD = {"robot_id": "mock_robot_id", "has_urdf": True}
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


def _login_and_connect_robot(mock_auth_requests, mocked_org_id: str) -> None:
    nc.login(TEST_API_KEY)
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json=TEST_ROBOT_PAYLOAD,
        status_code=200,
    )
    nc.create_robot(TEST_ROBOT_ID)
    nc.connect_robot(TEST_ROBOT_ID)


def _log_default_inputs() -> None:
    nc.log_joint_positions(positions={"joint1": 0.5, "joint2": 0.5, "joint3": 0.5})
    nc.log_rgb("top_camera", np.zeros((100, 100, 3), dtype=np.uint8))


def _assert_joint_prediction_matches_expected(
    predictions: dict[DataType, dict[str, BatchedNCData]],
) -> None:
    assert isinstance(predictions, dict)
    assert DataType.JOINT_TARGET_POSITIONS in predictions
    assert (
        predictions[DataType.JOINT_TARGET_POSITIONS].keys()
        == FAKE_PREDICTED_DATA[DataType.JOINT_TARGET_POSITIONS].keys()
    )
    prediction_values = [
        cast(BatchedJointData, batched_joint_data).value.numpy()
        for batched_joint_data in predictions[DataType.JOINT_TARGET_POSITIONS].values()
    ]
    expected_values = [
        cast(BatchedJointData, batched_joint_data).value.numpy()
        for batched_joint_data in FAKE_PREDICTED_DATA[
            DataType.JOINT_TARGET_POSITIONS
        ].values()
    ]
    assert np.array_equal(prediction_values, expected_values)


def _connect_test_remote_endpoint(mock_auth_requests, mocked_org_id: str):
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/models/endpoints",
        json=[{"id": "test_endpoint_id", "name": "test_endpoint", "status": "active"}],
        status_code=200,
    )
    return nc.policy_remote_server("test_endpoint")


class _StubPolicy(Policy):
    def __init__(self, predict_impl):
        self._predict_impl = predict_impl

    def _predict(self, sync_point=None):
        return self._predict_impl(sync_point)


INPUT_EMBODIMENT_DESCRIPTION = {
    DataType.JOINT_POSITIONS: _indexed_names(["joint1", "joint2", "joint3"]),
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: _indexed_names(["left_arm", "right_arm"]),
    DataType.RGB_IMAGES: _indexed_names(["top_camera"]),
}

OUTPUT_EMBODIMENT_DESCRIPTION = {
    DataType.JOINT_TARGET_POSITIONS: _indexed_names(["joint1", "joint2", "joint3"]),
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: _indexed_names(["left_arm", "right_arm"]),
}


def mock_subprocess_popen(*args, **kwargs):
    """Mock subprocess.Popen for local endpoint tests."""

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


def test_connect_remote_endpoint(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Test connecting to a remote server endpoint."""
    _login_and_connect_robot(mock_auth_requests, mocked_org_id)

    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/models/endpoints/test_endpoint_id/predict",
        json=FAKE_PREDICTED_DATA_JSON,
        status_code=200,
    )

    endpoint = _connect_test_remote_endpoint(mock_auth_requests, mocked_org_id)

    _log_default_inputs()

    _assert_joint_prediction_matches_expected(endpoint.predict())


def test_remote_endpoint_filters_sync_point_from_endpoint_input_description(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Remote endpoints should only receive data types declared in metadata."""
    _login_and_connect_robot(mock_auth_requests, mocked_org_id)

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

    _log_default_inputs()

    endpoint.predict()

    request_body = mock_auth_requests.request_history[-1].json()
    assert set(request_body["data"]) == {"JOINT_POSITIONS"}
    assert set(request_body["data"]["JOINT_POSITIONS"]) == {
        "joint1",
        "joint2",
        "joint3",
    }


def test_policy_predict_raises_for_non_positive_timeout():
    """Policy should reject timeout values <= 0."""
    policy = _StubPolicy(lambda _sync_point: {})

    with pytest.raises(ValueError, match="Timeout must be a positive number."):
        policy.predict(timeout=0)


def test_policy_predict_retries_until_success_before_timeout(monkeypatch):
    """Policy should retry on insufficient sync point until success."""
    attempts = {"count": 0}
    expected_prediction = {
        DataType.JOINT_TARGET_POSITIONS: {
            "joint1": BatchedJointData(value=torch.full((1, 1, 1), 0.1))
        }
    }

    def predict_impl(_sync_point):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise InsufficientSynchronizedPointError("not enough data yet")
        return expected_prediction

    time_values = iter([100.0, 100.1, 100.2])
    monkeypatch.setattr("neuracore.core.endpoint.time.time", lambda: next(time_values))
    monkeypatch.setattr("neuracore.core.endpoint.time.sleep", lambda _seconds: None)
    policy = _StubPolicy(predict_impl)

    prediction = policy.predict(timeout=1)
    assert prediction == expected_prediction
    assert attempts["count"] == 3


def test_policy_predict_raises_after_timeout_when_insufficient_data(monkeypatch):
    """Policy.predict should re-raise insufficient data error after timeout."""
    policy = _StubPolicy(
        lambda _sync_point: (_ for _ in ()).throw(
            InsufficientSynchronizedPointError("not enough data")
        )
    )

    time_values = iter([100.0, 100.05, 100.2])
    monkeypatch.setattr("neuracore.core.endpoint.time.time", lambda: next(time_values))
    monkeypatch.setattr("neuracore.core.endpoint.time.sleep", lambda _seconds: None)

    with pytest.raises(InsufficientSynchronizedPointError, match="not enough data"):
        policy.predict(timeout=0.1)


def test_set_checkpoint_success(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Server policy should post epoch to set_checkpoint endpoint."""
    nc.login(TEST_API_KEY)
    endpoint = _connect_test_remote_endpoint(mock_auth_requests, mocked_org_id)
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/models/endpoints/test_endpoint_id/set_checkpoint",
        json={},
        status_code=200,
    )

    endpoint.set_checkpoint(epoch=3)

    request_body = mock_auth_requests.request_history[-1].json()
    assert request_body == {"epoch": 3}


@pytest.mark.parametrize(
    "kwargs,error_message",
    [
        (
            {"checkpoint_file": "checkpoint.pt"},
            "Setting checkpoint by file is not supported in server policies.",
        ),
        ({"epoch": None}, "Must specify epoch to set checkpoint."),
        (
            {"epoch": -2},
            re.escape("Epoch must be -1 (last) or a non-negative integer."),
        ),
    ],
)
def test_set_checkpoint_validation_errors(
    temp_config_dir,
    mock_auth_requests,
    reset_neuracore,
    mocked_org_id,
    kwargs,
    error_message,
):
    """set_checkpoint should validate epoch/checkpoint_file arguments."""
    nc.login(TEST_API_KEY)
    endpoint = _connect_test_remote_endpoint(mock_auth_requests, mocked_org_id)

    with pytest.raises(ValueError, match=error_message):
        endpoint.set_checkpoint(**kwargs)


def test_set_checkpoint_raises_endpoint_error_for_non_200_response(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """set_checkpoint should raise EndpointError when response is non-200."""
    nc.login(TEST_API_KEY)
    endpoint = _connect_test_remote_endpoint(mock_auth_requests, mocked_org_id)
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/models/endpoints/test_endpoint_id/set_checkpoint",
        status_code=500,
        text="Internal Server Error",
    )

    with pytest.raises(
        EndpointError,
        match="Failed to set checkpoint: 500 - Internal Server Error",
    ):
        endpoint.set_checkpoint(epoch=1)


def test_set_checkpoint_raises_endpoint_error_for_connection_error(
    temp_config_dir,
    mock_auth_requests,
    reset_neuracore,
    mocked_org_id,
    mock_session,
):
    """set_checkpoint should wrap requests ConnectionError in EndpointError."""
    nc.login(TEST_API_KEY)
    endpoint = _connect_test_remote_endpoint(mock_auth_requests, mocked_org_id)
    mock_session.post.side_effect = requests.exceptions.ConnectionError()
    with patch(
        "neuracore.core.endpoint.get_session", return_value=mock_session
    ), pytest.raises(
        EndpointError,
        match=(
            "Failed to connect to endpoint, please check your internet "
            "connection and try again."
        ),
    ):
        endpoint.set_checkpoint(epoch=1)


def test_set_checkpoint_raises_endpoint_error_for_request_exception(
    temp_config_dir,
    mock_auth_requests,
    reset_neuracore,
    mocked_org_id,
    mock_session,
):
    """set_checkpoint should wrap generic RequestException in EndpointError."""
    nc.login(TEST_API_KEY)
    endpoint = _connect_test_remote_endpoint(mock_auth_requests, mocked_org_id)
    mock_session.post.side_effect = requests.exceptions.Timeout("timeout")
    with patch(
        "neuracore.core.endpoint.get_session", return_value=mock_session
    ), pytest.raises(EndpointError, match="Failed to set checkpoint: timeout"):
        endpoint.set_checkpoint(epoch=1)


def test_connect_nonexistent_remote_endpoint(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Test connecting to a non-existent endpoint."""
    nc.login(TEST_API_KEY)
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


def test_connect_inactive_remote_endpoint(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Test connecting to an inactive endpoint."""
    nc.login(TEST_API_KEY)
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


def test_connect_active_remote_endpoint_with_duplicate_name(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Test connecting to active endpoint when duplicate names exist."""
    _login_and_connect_robot(mock_auth_requests, mocked_org_id)

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

    _log_default_inputs()

    _assert_joint_prediction_matches_expected(endpoint.predict())


def test_connect_multiple_active_remote_endpoints_with_duplicate_name(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Test error when multiple active endpoints have the same name."""
    nc.login(TEST_API_KEY)
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

    _login_and_connect_robot(mock_auth_requests, mocked_org_id)

    local_endpoint = nc.policy_local_server(
        input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
        output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
        model_file=mock_model_mar,
        port=port,
    )

    _log_default_inputs()

    _assert_joint_prediction_matches_expected(local_endpoint.predict())

    local_endpoint.disconnect()


def test_local_endpoint_filters_sync_point_from_input_description(
    temp_config_dir,
    mock_model_mar,
    reset_neuracore,
    monkeypatch,
    mock_auth_requests,
    mocked_org_id,
):
    """Local endpoints should only receive configured input data types."""
    port = np.random.randint(8000, 9000)
    localhost = f"http://127.0.0.1:{port}"

    mock_auth_requests.get(f"{localhost}/ping", status_code=200)
    mock_auth_requests.post(
        f"{localhost}/predict",
        json=FAKE_PREDICTED_DATA_JSON,
        status_code=200,
    )

    monkeypatch.setattr("subprocess.Popen", mock_subprocess_popen)
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)

    _login_and_connect_robot(mock_auth_requests, mocked_org_id)

    local_endpoint = nc.policy_local_server(
        input_embodiment_description={
            DataType.JOINT_POSITIONS: _indexed_names(["joint1", "joint2", "joint3"]),
        },
        output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
        model_file=mock_model_mar,
        port=port,
    )

    _log_default_inputs()
    local_endpoint.predict()

    request_body = mock_auth_requests.request_history[-1].json()
    assert set(request_body["data"]) == {"JOINT_POSITIONS"}
    assert set(request_body["data"]["JOINT_POSITIONS"]) == {
        "joint1",
        "joint2",
        "joint3",
    }

    local_endpoint.disconnect()


def test_deploy_model(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Test model deployment."""
    nc.login(TEST_API_KEY)

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
    nc.login(TEST_API_KEY)
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


def test_get_remote_endpoint_status(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Test getting endpoint status."""
    nc.login(TEST_API_KEY)

    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/models/endpoints/endpoint_123",
        json={"id": "endpoint_123", "name": "test_endpoint", "status": "active"},
        status_code=200,
    )

    # Get status
    status = nc.get_endpoint_status("endpoint_123")

    # Verify status
    assert status == "active"


def test_delete_remote_endpoint(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Test deleting an endpoint."""
    nc.login(TEST_API_KEY)

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
    nc.login(TEST_API_KEY)

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
    _login_and_connect_robot(mock_auth_requests, mocked_org_id)
    port = np.random.randint(8000, 9000)

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

    monkeypatch.setattr("subprocess.Popen", mock_subprocess_popen)
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)

    # Connect using train run name
    local_endpoint = nc.policy_local_server(
        input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
        output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
        train_run_name="test_run",
        port=port,
    )
    _log_default_inputs()

    _assert_joint_prediction_matches_expected(local_endpoint.predict())

    local_endpoint.disconnect()


def test_connect_local_endpoint_invalid_args(
    temp_config_dir, mock_auth_requests, reset_neuracore
):
    """Test connecting to a local endpoint with invalid arguments."""
    nc.login(TEST_API_KEY)

    with pytest.raises(
        ValueError, match="Cannot specify both train_run_name and model_file"
    ):
        nc.policy_local_server(
            input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
            output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
            model_file="model.nc.zip",
            train_run_name="test_run",
        )

    with pytest.raises(
        ValueError, match="Must specify either train_run_name or model_file"
    ):
        nc.policy_local_server(
            input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
            output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
        )
