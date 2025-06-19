import pytest

import neuracore as nc
from neuracore.core.const import API_URL
from neuracore.core.nc_types import DataType


@pytest.fixture
def training_job_response():
    """Create a mock training job response."""
    return {
        "id": "train_job_123",
        "name": "test_training_run",
        "dataset_id": "dataset_123",
        "algorithm_id": "algo_123",
        "status": "pending",
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-01-01T00:00:00Z",
    }


@pytest.fixture
def algorithm_list_response():
    """Create a mock algorithm list response."""
    return [
        {
            "id": "algo_123",
            "name": "cnnmlp",
            "description": "CNN + MLP",
            "is_shared": True,
            "supported_input_data_types": [
                DataType.RGB_IMAGE,
                DataType.JOINT_POSITIONS,
            ],
            "supported_output_data_types": [DataType.JOINT_TARGET_POSITIONS],
        },
        {
            "id": "algo_456",
            "name": "act",
            "description": "Action Chunking with Transformers",
            "is_shared": True,
            "supported_input_data_types": [
                DataType.RGB_IMAGE,
                DataType.JOINT_POSITIONS,
            ],
            "supported_output_data_types": [DataType.JOINT_TARGET_POSITIONS],
        },
    ]


def test_start_training_run(
    temp_config_dir,
    mock_auth_requests,
    reset_neuracore,
    training_job_response,
    algorithm_list_response,
    mocked_org_id,
):
    """Test starting a training run."""
    # Ensure login
    nc.login("test_api_key")

    dataset_response = {
        "id": "dataset_123",
        "name": "test_dataset",
        "size_bytes": 1024,
        "tags": ["test"],
        "is_shared": False,
        "num_demonstrations": 10,
    }

    # Mock datasets endpoint to return a dataset
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets",
        json=[dataset_response],
        status_code=200,
    )

    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/by-name/{dataset_response['name']}",
        json=dataset_response,
        status_code=200,
    )

    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_response['id']}/recordings",
        json={"recordings": []},
        status_code=200,
    )

    # Mock shared datasets endpoint
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/shared",
        json=[],
        status_code=200,
    )

    # Mock algorithms endpoint
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/algorithms",
        json=algorithm_list_response,
        status_code=200,
    )

    # Mock shared algorithms endpoint
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/algorithms?shared=true",
        json=[],
        status_code=200,
    )

    # Mock training job creation endpoint
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/training/jobs",
        json=training_job_response,
        status_code=200,
    )

    # Start training run
    algorithm_config = {
        "hidden_dim": 512,
        "num_layers": 3,
        "cnn_output_dim": 64,
    }

    job = nc.start_training_run(
        name="test_training_run",
        dataset_name="test_dataset",
        algorithm_name="cnnmlp",
        algorithm_config=algorithm_config,
        gpu_type="T4",
        num_gpus=1,
        frequency=10,
    )

    # Verify job was created with expected values
    assert job is not None
    assert job["id"] == "train_job_123"
    assert job["name"] == "test_training_run"
    assert job["dataset_id"] == "dataset_123"
    assert job["algorithm_id"] == "algo_123"
    assert job["status"] == "pending"


def test_get_training_job_data(
    temp_config_dir,
    mock_auth_requests,
    reset_neuracore,
    training_job_response,
    mocked_org_id,
):
    """Test getting training job data."""
    # Ensure login
    nc.login("test_api_key")

    # Mock training jobs endpoint
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/training/jobs",
        json=[training_job_response],
        status_code=200,
    )

    # Get job data
    job_data = nc.get_training_job_data("train_job_123")

    # Verify job data
    assert job_data is not None
    assert job_data["id"] == "train_job_123"
    assert job_data["name"] == "test_training_run"
    assert job_data["status"] == "pending"


def test_get_training_job_status(
    temp_config_dir,
    mock_auth_requests,
    reset_neuracore,
    training_job_response,
    mocked_org_id,
):
    """Test getting training job status."""
    # Ensure login
    nc.login("test_api_key")

    # Mock training jobs endpoint
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/training/jobs",
        json=[training_job_response],
        status_code=200,
    )

    # Get job status
    status = nc.get_training_job_status("train_job_123")

    # Verify status
    assert status == "pending"


def test_get_nonexistent_training_job(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Test getting a non-existent training job raises an error."""
    # Ensure login
    nc.login("test_api_key")

    # Mock training jobs endpoint with empty list
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/training/jobs",
        json=[],
        status_code=200,
    )

    # Attempt to get non-existent job
    with pytest.raises(ValueError, match="Job not found"):
        nc.get_training_job_data("nonexistent_job")


def test_failed_training_job_request(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Test handling of failed API requests."""
    # Ensure login
    nc.login("test_api_key")

    # Mock training jobs endpoint to return an error
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/training/jobs",
        status_code=500,
        text="Internal Server Error",
    )

    # Attempt to get job data should raise an exception
    with pytest.raises(ValueError, match="Error accessing job"):
        nc.get_training_job_data("train_job_123")
