import pytest
import requests
from neuracore_types import Dataset, DataType, GPUType

import neuracore as nc
from neuracore.core.const import API_URL


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
                DataType.RGB_IMAGES,
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
                DataType.RGB_IMAGES,
                DataType.JOINT_POSITIONS,
            ],
            "supported_output_data_types": [DataType.JOINT_TARGET_POSITIONS],
        },
    ]


def test_start_training_run(
    mock_auth_requests,
    training_job_response,
    algorithm_list_response,
    mocked_org_id,
):
    """Test starting a training run."""
    # Ensure login
    nc.login("test_api_key")
    dataset_id = "dataset123"
    dataset_response = Dataset(
        id=dataset_id,
        name="test_dataset",
        created_at=0.0,
        modified_at=0.0,
        description="A test dataset",
        size_bytes=2048,
        tags=["test"],
        is_shared=False,
        num_demonstrations=20,
        all_data_types={DataType.RGB_IMAGES: 1, DataType.JOINT_TARGET_POSITIONS: 1},
        common_data_types={DataType.RGB_IMAGES: 1, DataType.JOINT_TARGET_POSITIONS: 1},
    )

    # Mock datasets endpoint to return a dataset
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets",
        json=[dataset_response.model_dump(mode="json")],
        status_code=200,
    )

    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
        json=dataset_response.model_dump(mode="json"),
        status_code=200,
    )

    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_response.id}/recordings",
        json={"recordings": []},
        status_code=200,
    )
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/recording/by-dataset/dataset123",
        json={"data": [], "total": 10, "limit": 1, "start_after": None},
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

    robot_id = "fake_robot_id"
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_id}/robot_ids",
        json=[robot_id],
        status_code=200,
    )
    robot_name = "fake_robot_name"
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json=[{"id": robot_id, "name": robot_name}],
        status_code=200,
    )
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_id}/full-data-spec/{robot_id}",
        json={
            DataType.RGB_IMAGES: ["angle"],
            DataType.JOINT_TARGET_POSITIONS: ["joint1", "joint2", "joint3"],
        },
        status_code=200,
    )

    # Start training run
    algorithm_config = {
        "hidden_dim": 512,
        "num_layers": 3,
        "cnn_output_dim": 64,
    }
    input_robot_data_spec = {
        robot_name: {
            DataType.RGB_IMAGES: ["angle"],
        }
    }

    output_robot_data_spec = {
        robot_name: {
            DataType.JOINT_TARGET_POSITIONS: ["joint1", "joint2", "joint3"],
        }
    }

    job = nc.start_training_run(
        name="test_training_run",
        dataset_name="test_dataset",
        algorithm_name="cnnmlp",
        algorithm_config=algorithm_config,
        gpu_type=GPUType.NVIDIA_TESLA_T4,
        num_gpus=1,
        frequency=10,
        input_robot_data_spec=input_robot_data_spec,
        output_robot_data_spec=output_robot_data_spec,
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


def test_start_training_run_raises_on_duplicate_name(
    temp_config_dir,
    mock_auth_requests,
    reset_neuracore,
    algorithm_list_response,
    mocked_org_id,
):
    """Starting a cloud run with an existing name should raise an API error."""
    nc.login("test_api_key")
    dataset_id = "dataset123"
    dataset_response = Dataset(
        id=dataset_id,
        name="test_dataset",
        created_at=0.0,
        modified_at=0.0,
        description="A test dataset",
        size_bytes=2048,
        tags=["test"],
        is_shared=False,
        num_demonstrations=20,
        all_data_types={DataType.RGB_IMAGES: 1, DataType.JOINT_TARGET_POSITIONS: 1},
        common_data_types={DataType.RGB_IMAGES: 1, DataType.JOINT_TARGET_POSITIONS: 1},
    )

    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets",
        json=[dataset_response.model_dump(mode="json")],
        status_code=200,
    )
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
        json=dataset_response.model_dump(mode="json"),
        status_code=200,
    )
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_response.id}/recordings",
        json={"recordings": []},
        status_code=200,
    )
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/recording/by-dataset/dataset123",
        json={"data": [], "total": 10, "limit": 1, "start_after": None},
        status_code=200,
    )
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/shared",
        json=[],
        status_code=200,
    )
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/algorithms",
        json=algorithm_list_response,
        status_code=200,
    )
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/algorithms?shared=true",
        json=[],
        status_code=200,
    )

    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/training/jobs",
        status_code=409,
        json={"detail": "Training job with name already exists"},
    )

    robot_id = "fake_robot_id"
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_id}/robot_ids",
        json=[robot_id],
        status_code=200,
    )
    robot_name = "fake_robot_name"
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json=[{"id": robot_id, "name": robot_name}],
        status_code=200,
    )
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_id}/full-data-spec/{robot_id}",
        json={
            DataType.RGB_IMAGES: ["angle"],
            DataType.JOINT_TARGET_POSITIONS: ["joint1", "joint2", "joint3"],
        },
        status_code=200,
    )

    input_robot_data_spec = {robot_name: {DataType.RGB_IMAGES: ["angle"]}}
    output_robot_data_spec = {
        robot_name: {DataType.JOINT_TARGET_POSITIONS: ["joint1", "joint2", "joint3"]}
    }

    with pytest.raises(requests.exceptions.HTTPError):
        nc.start_training_run(
            name="test_training_run",
            dataset_name="test_dataset",
            algorithm_name="cnnmlp",
            algorithm_config={"hidden_dim": 512},
            gpu_type=GPUType.NVIDIA_TESLA_T4,
            num_gpus=1,
            frequency=10,
            input_robot_data_spec=input_robot_data_spec,
            output_robot_data_spec=output_robot_data_spec,
        )


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


def test_delete_training_job(
    temp_config_dir,
    mock_auth_requests,
    reset_neuracore,
    mocked_org_id,
):
    """Test deleting a training job."""
    # Ensure login
    nc.login("test_api_key")

    # Mock delete training job endpoint
    mock_auth_requests.delete(
        f"{API_URL}/org/{mocked_org_id}/training/jobs/train_job_123",
        status_code=204,
    )

    # Delete training job
    nc.delete_training_job("train_job_123")

    # Verify delete request was made
    assert mock_auth_requests.called
    last_request = mock_auth_requests.last_request
    assert last_request.method == "DELETE"
    assert (
        last_request.url == f"{API_URL}/org/{mocked_org_id}/training/jobs/train_job_123"
    )


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
