import pytest
import requests
from neuracore_types import Dataset, DataType, GPUType

import neuracore as nc
from neuracore.api.training import _resolve_next_name, _validate_gpu_to_algorithm
from neuracore.core.const import API_URL

TEST_ROBOT_ID = "20a621b7-2f9b-4699-a08e-7d080488a5a3"
TEST_DATASET_ID = "dataset123"

# What the dataset reports it contains, and the slice of it the tests train on.
TEST_EMBODIMENT_DESCRIPTION = {
    DataType.RGB_IMAGES: {0: "angle"},
    DataType.JOINT_TARGET_POSITIONS: {0: "joint1", 1: "joint2", 2: "joint3"},
}
TEST_INPUT_DESCRIPTION = {TEST_ROBOT_ID: {DataType.RGB_IMAGES: {0: "angle"}}}
TEST_OUTPUT_DESCRIPTION = {
    TEST_ROBOT_ID: {
        DataType.JOINT_TARGET_POSITIONS: {0: "joint1", 1: "joint2", 2: "joint3"}
    }
}


def test_resolve_next_name_uses_base_name_when_available():
    """When base name is not taken, return it as-is."""
    assert _resolve_next_name("my_run", set()) == "my_run"
    assert _resolve_next_name("my_run", {"other_run"}) == "my_run"


def test_resolve_next_name_increments_when_base_name_taken():
    """When base name is taken, return base_1, base_2, etc."""
    assert _resolve_next_name("my_run", {"my_run"}) == "my_run_1"
    assert _resolve_next_name("my_run", {"my_run", "my_run_1"}) == "my_run_2"
    assert (
        _resolve_next_name("my_run", {"my_run", "my_run_1", "my_run_2"}) == "my_run_3"
    )


def test_resolve_next_name_ignores_unrelated_names():
    """Only exact base_name and base_name_N are considered taken."""
    assert _resolve_next_name("foo", {"foo_abc", "foo_12"}) == "foo"
    assert _resolve_next_name("foo", {"foo", "foo_1"}) == "foo_2"


def test_validate_gpu_to_algorithm_accepts_supported_gpu():
    """A GPU declared by the algorithm is accepted."""
    _validate_gpu_to_algorithm(
        GPUType.NVIDIA_A100_80GB,
        "ACT",
        [{
            "name": "ACT",
            "supported_gpus": [
                "NVIDIA_A100_80GB",
                "NVIDIA_TESLA_V100",
            ],
        }],
    )


def test_validate_gpu_to_algorithm_rejects_unsupported_gpu():
    """An unsupported GPU is rejected with the valid choices."""
    with pytest.raises(
        ValueError,
        match=(
            "GPU NVIDIA_TESLA_T4 is not supported by algorithm Pi05.*"
            "NVIDIA_A100_80GB, NVIDIA_H100_80GB"
        ),
    ):
        _validate_gpu_to_algorithm(
            GPUType.NVIDIA_TESLA_T4,
            "Pi05",
            [{
                "name": "Pi05",
                "supported_gpus": [
                    "NVIDIA_H100_80GB",
                    "NVIDIA_A100_80GB",
                ],
            }],
        )


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
            "supported_gpus": [GPUType.NVIDIA_TESLA_T4],
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
            "supported_gpus": [GPUType.NVIDIA_TESLA_T4],
        },
    ]


@pytest.fixture
def mock_start_training_run_endpoints(
    mock_auth_requests,
    mocked_org_id,
    algorithm_list_response,
    training_job_response,
):
    """Register the endpoints start_training_run walks through.

    Returns a callable so each test only states what it cares about: the
    dataset's size, the response to the job-creation POST, and any jobs that
    already exist.
    """

    def register(
        dataset_size_bytes: int = 2048,
        job_response: dict | None = None,
        job_status_code: int = 200,
        existing_jobs: list[dict] | None = None,
    ) -> None:
        dataset = Dataset(
            id=TEST_DATASET_ID,
            name="test_dataset",
            created_at=0.0,
            modified_at=0.0,
            description="A test dataset",
            size_bytes=dataset_size_bytes,
            tags=[],
            is_shared=False,
            num_demonstrations=20,
            all_data_types={DataType.RGB_IMAGES: 1, DataType.JOINT_TARGET_POSITIONS: 1},
            common_data_types={
                DataType.RGB_IMAGES: 1,
                DataType.JOINT_TARGET_POSITIONS: 1,
            },
        )
        org_url = f"{API_URL}/org/{mocked_org_id}"

        mock_auth_requests.get(
            f"{org_url}/datasets",
            json=[dataset.model_dump(mode="json")],
            status_code=200,
        )
        mock_auth_requests.get(
            f"{org_url}/datasets/search/by-name",
            json=dataset.model_dump(mode="json"),
            status_code=200,
        )
        mock_auth_requests.get(
            f"{org_url}/datasets/shared",
            json=[],
            status_code=200,
        )
        mock_auth_requests.get(
            f"{org_url}/datasets/{TEST_DATASET_ID}/recordings",
            json={"recordings": []},
            status_code=200,
        )
        mock_auth_requests.post(
            f"{org_url}/recording/by-dataset/{TEST_DATASET_ID}",
            json={"data": [], "total": 10, "limit": 1, "start_after": None},
            status_code=200,
        )
        mock_auth_requests.get(
            f"{org_url}/datasets/{TEST_DATASET_ID}/robot_ids",
            json=[TEST_ROBOT_ID],
            status_code=200,
        )
        mock_auth_requests.get(
            f"{org_url}/robots",
            json=[{"id": TEST_ROBOT_ID, "name": "fake_robot_name"}],
            status_code=200,
        )
        mock_auth_requests.get(
            f"{org_url}/datasets/{TEST_DATASET_ID}"
            f"/full-embodiment-description/{TEST_ROBOT_ID}",
            json=TEST_EMBODIMENT_DESCRIPTION,
            status_code=200,
        )
        mock_auth_requests.get(
            f"{org_url}/algorithms",
            json=algorithm_list_response,
            status_code=200,
        )
        mock_auth_requests.get(
            f"{org_url}/algorithms?shared=true",
            json=[],
            status_code=200,
        )
        mock_auth_requests.get(
            f"{org_url}/training/jobs",
            json=existing_jobs or [],
            status_code=200,
        )
        mock_auth_requests.post(
            f"{org_url}/training/jobs",
            json=training_job_response if job_response is None else job_response,
            status_code=job_status_code,
        )

    return register


def _training_job_post_requests(mock_auth_requests) -> list:
    """Every job-creation request the client made."""
    return [
        request
        for request in mock_auth_requests.request_history
        if request.method == "POST" and request.url.endswith("/training/jobs")
    ]


def test_start_training_run(mock_start_training_run_endpoints):
    """Test starting a training run."""
    # Ensure login
    nc.login("test_api_key")
    mock_start_training_run_endpoints()

    job = nc.start_training_run(
        name="test_training_run",
        dataset_name="test_dataset",
        algorithm_name="cnnmlp",
        algorithm_config={
            "hidden_dim": 512,
            "num_layers": 3,
            "cnn_output_dim": 64,
        },
        gpu_type=GPUType.NVIDIA_TESLA_T4,
        num_gpus=1,
        frequency=10,
        input_cross_embodiment_description=TEST_INPUT_DESCRIPTION,
        output_cross_embodiment_description=TEST_OUTPUT_DESCRIPTION,
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

    # Mock training job endpoint
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/training/jobs/train_job_123",
        json=training_job_response,
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
    reset_neuracore,
    mock_start_training_run_endpoints,
):
    """Starting a cloud run with an existing name should raise an API error."""
    nc.login("test_api_key")
    mock_start_training_run_endpoints(
        job_response={"detail": "Training job with name already exists"},
        job_status_code=409,
    )

    with pytest.raises(requests.exceptions.HTTPError):
        nc.start_training_run(
            name="test_training_run",
            dataset_name="test_dataset",
            algorithm_name="cnnmlp",
            algorithm_config={"hidden_dim": 512},
            gpu_type=GPUType.NVIDIA_TESLA_T4,
            num_gpus=1,
            frequency=10,
            input_cross_embodiment_description=TEST_INPUT_DESCRIPTION,
            output_cross_embodiment_description=TEST_OUTPUT_DESCRIPTION,
        )


def test_start_training_run_name_auto_increment(
    temp_config_dir,
    mock_auth_requests,
    reset_neuracore,
    training_job_response,
    mock_start_training_run_endpoints,
):
    """With name_auto_increment=True, an existing name is resolved to name_1."""
    nc.login("test_api_key")
    mock_start_training_run_endpoints(
        # An existing job with the same name is what forces the increment.
        existing_jobs=[{"id": "existing_123", "name": "test_training_run"}],
        job_response={
            **training_job_response,
            "name": "test_training_run_1",
            "id": "train_job_456",
        },
    )

    job = nc.start_training_run(
        name="test_training_run",
        dataset_name="test_dataset",
        algorithm_name="cnnmlp",
        algorithm_config={"hidden_dim": 512},
        gpu_type=GPUType.NVIDIA_TESLA_T4,
        num_gpus=1,
        frequency=10,
        input_cross_embodiment_description=TEST_INPUT_DESCRIPTION,
        output_cross_embodiment_description=TEST_OUTPUT_DESCRIPTION,
        name_auto_increment=True,
    )

    assert job["name"] == "test_training_run_1"
    post_requests = _training_job_post_requests(mock_auth_requests)
    assert len(post_requests) == 1
    assert post_requests[0].json()["name"] == "test_training_run_1"


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
        f"{API_URL}/org/{mocked_org_id}/training/jobs/train_job_123",
        json=training_job_response,
        status_code=200,
    )

    # Get job status
    status = nc.get_training_job_status("train_job_123")

    # Verify status
    assert status == "pending"


def test_cancel_training_job(
    temp_config_dir,
    mock_auth_requests,
    reset_neuracore,
    mocked_org_id,
):
    """Test cancelling a training job."""
    nc.login("test_api_key")

    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/training/jobs/train_job_123",
        status_code=200,
    )

    nc.cancel_training_job("train_job_123")

    assert mock_auth_requests.called
    last_request = mock_auth_requests.last_request
    assert last_request.method == "POST"
    assert (
        last_request.url == f"{API_URL}/org/{mocked_org_id}/training/jobs/train_job_123"
    )


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

    # Mock training job endpoint with 404
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/training/jobs/nonexistent_job",
        status_code=404,
        json={"detail": "Not found"},
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

    # Mock training job endpoint to return an error
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/training/jobs/train_job_123",
        status_code=500,
        text="Internal Server Error",
    )

    # Attempt to get job data should raise the HTTP error
    with pytest.raises(requests.exceptions.HTTPError):
        nc.get_training_job_data("train_job_123")


def test_resume_training_run(
    temp_config_dir,
    mock_auth_requests,
    reset_neuracore,
    training_job_response,
    mocked_org_id,
):
    """Successful resume returns the updated job dict."""
    nc.login("test_api_key")

    resumed_response = {
        **training_job_response,
        "status": "PENDING",
        "resumed_at": 1704070800.0,
        "resume_points": [1],
        "previous_training_time": None,
    }
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/training/jobs/train_job_123/resume/2",
        json=resumed_response,
        status_code=200,
    )

    result = nc.resume_training_run("train_job_123", additional_epochs=2)

    assert result["id"] == "train_job_123"
    assert result["status"] == "PENDING"
    assert result["resumed_at"] == 1704070800.0
    assert result["resume_points"] == [1]

    post_requests = [
        r
        for r in mock_auth_requests.request_history
        if r.method == "POST" and "/resume/" in r.url
    ]
    assert len(post_requests) == 1
    assert post_requests[0].url == (
        f"{API_URL}/org/{mocked_org_id}/training/jobs/train_job_123/resume/2"
    )


def test_resume_training_run_not_found(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """A 404 from the backend is surfaced as a ValueError."""
    nc.login("test_api_key")

    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/training/jobs/nonexistent/resume/1",
        status_code=404,
        json={"detail": "Training job not found"},
    )

    with pytest.raises(ValueError, match="Error resuming job nonexistent"):
        nc.resume_training_run("nonexistent", additional_epochs=1)


def test_resume_training_run_server_error(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """A 500 from the backend is surfaced as a ValueError."""
    nc.login("test_api_key")

    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/training/jobs/train_job_123/resume/1",
        status_code=500,
        text="Internal Server Error",
    )

    with pytest.raises(ValueError, match="Error resuming job train_job_123"):
        nc.resume_training_run("train_job_123", additional_epochs=1)


def test_resume_training_run_propagates_error_detail(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """The backend error detail appears in the raised ValueError message."""
    nc.login("test_api_key")

    backend_error = (
        "Dataset 'my_dataset' with id dataset123 has been deleted"
        " or no longer exists."
    )
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/training/jobs/train_job_123/resume/1",
        status_code=404,
        json={"detail": {"error": backend_error, "status": 404}},
    )

    with pytest.raises(ValueError) as exc_info:
        nc.resume_training_run("train_job_123", additional_epochs=1)

    error_message = str(exc_info.value)
    assert "Error resuming job train_job_123" in error_message
    assert backend_error in error_message


# ---------------------------------------------------------------------------
# get_training_job_logs
# ---------------------------------------------------------------------------

MOCK_LOGS_RESPONSE = {
    "job_id": "train_job_123",
    "logs": [
        {"message": "Epoch 1 started", "severity": "INFO", "timestamp": "2024-01-01"},
        {"message": "Loss: 0.42", "severity": "INFO", "timestamp": "2024-01-01"},
    ],
    "total_entries": 2,
    "retrieved_at": "2024-01-01T01:00:00Z",
}


def test_get_training_job_logs(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Logs are returned with the expected structure."""
    nc.login("test_api_key")

    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/training/jobs/train_job_123/logs",
        json=MOCK_LOGS_RESPONSE,
        status_code=200,
    )

    logs = nc.get_training_job_logs("train_job_123", max_entries=50)

    assert logs["job_id"] == "train_job_123"
    assert isinstance(logs["logs"], list)
    assert len(logs["logs"]) == 2
    assert logs["total_entries"] == 2
    assert "retrieved_at" in logs

    get_requests = [
        r
        for r in mock_auth_requests.request_history
        if r.method == "GET" and "/logs" in r.url
    ]
    assert len(get_requests) == 1
    assert "max_entries=50" in get_requests[0].url


def test_get_training_job_logs_with_severity_filter(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """Severity filter is forwarded as a query parameter."""
    nc.login("test_api_key")

    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/training/jobs/train_job_123/logs",
        json={**MOCK_LOGS_RESPONSE, "logs": [], "total_entries": 0},
        status_code=200,
    )

    logs = nc.get_training_job_logs(
        "train_job_123", max_entries=10, severity_filter="ERROR"
    )

    assert isinstance(logs["logs"], list)
    get_requests = [
        r
        for r in mock_auth_requests.request_history
        if r.method == "GET" and "/logs" in r.url
    ]
    assert "severity_filter=ERROR" in get_requests[0].url


def test_get_training_job_logs_error(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    """A backend error is surfaced as a ValueError."""
    nc.login("test_api_key")

    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/training/jobs/train_job_123/logs",
        status_code=404,
        json={"detail": "Job not found"},
    )

    with pytest.raises(ValueError, match="Error getting training job logs"):
        nc.get_training_job_logs("train_job_123")


def test_start_training_run_sends_disk_size_gb(
    mock_auth_requests,
    mock_start_training_run_endpoints,
):
    """start_training_run sends the specified disk_size_gb in the POST body."""
    nc.login("test_api_key")
    mock_start_training_run_endpoints()

    nc.start_training_run(
        name="disk_test_run",
        dataset_name="test_dataset",
        algorithm_name="cnnmlp",
        algorithm_config={"hidden_dim": 64},
        gpu_type=GPUType.NVIDIA_TESLA_T4,
        num_gpus=1,
        frequency=10,
        disk_size_gb=1000,
        input_cross_embodiment_description=TEST_INPUT_DESCRIPTION,
        output_cross_embodiment_description=TEST_OUTPUT_DESCRIPTION,
    )

    post_request = _training_job_post_requests(mock_auth_requests)[0]
    assert post_request.json()["disk_size_gb"] == 1000


def test_start_training_run_raises_when_disk_size_too_small(
    mock_auth_requests,
    mock_start_training_run_endpoints,
):
    """start_training_run rejects disks that cannot fit the dataset."""
    nc.login("test_api_key")
    mock_start_training_run_endpoints(dataset_size_bytes=2_000_000_000)

    expected_message = (
        "Dataset test_dataset is 2.00 GB, but selected VM disk is 1 GB. "
        "Please increase disk_size_gb."
    )
    with pytest.raises(ValueError, match=expected_message):
        nc.start_training_run(
            name="small_disk_run",
            dataset_name="test_dataset",
            algorithm_name="cnnmlp",
            algorithm_config={"hidden_dim": 64},
            gpu_type=GPUType.NVIDIA_TESLA_T4,
            num_gpus=1,
            frequency=10,
            disk_size_gb=1,
            input_cross_embodiment_description=TEST_INPUT_DESCRIPTION,
            output_cross_embodiment_description=TEST_OUTPUT_DESCRIPTION,
        )

    assert _training_job_post_requests(mock_auth_requests) == []


def test_start_training_run_default_disk_size_gb(
    mock_auth_requests,
    mock_start_training_run_endpoints,
):
    """start_training_run sends disk_size_gb=500 by default."""
    nc.login("test_api_key")
    mock_start_training_run_endpoints()

    nc.start_training_run(
        name="default_disk_run",
        dataset_name="test_dataset",
        algorithm_name="cnnmlp",
        algorithm_config={"hidden_dim": 64},
        gpu_type=GPUType.NVIDIA_TESLA_T4,
        num_gpus=1,
        frequency=10,
        input_cross_embodiment_description=TEST_INPUT_DESCRIPTION,
        output_cross_embodiment_description=TEST_OUTPUT_DESCRIPTION,
    )

    post_request = _training_job_post_requests(mock_auth_requests)[0]
    assert post_request.json()["disk_size_gb"] == 500


def test_start_training_run_sends_resume_from_job_id(
    mock_auth_requests,
    mock_start_training_run_endpoints,
):
    """start_training_run forwards the job whose checkpoint the run continues."""
    nc.login("test_api_key")
    mock_start_training_run_endpoints()

    nc.start_training_run(
        name="continued_run",
        dataset_name="test_dataset",
        algorithm_name="cnnmlp",
        algorithm_config={"hidden_dim": 64},
        gpu_type=GPUType.NVIDIA_TESLA_T4,
        num_gpus=1,
        frequency=10,
        input_cross_embodiment_description=TEST_INPUT_DESCRIPTION,
        output_cross_embodiment_description=TEST_OUTPUT_DESCRIPTION,
        resume_from_job_id="previous_job_123",
    )

    post_request = _training_job_post_requests(mock_auth_requests)[0]
    assert post_request.json()["resume_from_job_id"] == "previous_job_123"


def test_start_training_run_defaults_to_no_resume(
    mock_auth_requests,
    mock_start_training_run_endpoints,
):
    """start_training_run trains from scratch unless a source job is given."""
    nc.login("test_api_key")
    mock_start_training_run_endpoints()

    nc.start_training_run(
        name="fresh_run",
        dataset_name="test_dataset",
        algorithm_name="cnnmlp",
        algorithm_config={"hidden_dim": 64},
        gpu_type=GPUType.NVIDIA_TESLA_T4,
        num_gpus=1,
        frequency=10,
        input_cross_embodiment_description=TEST_INPUT_DESCRIPTION,
        output_cross_embodiment_description=TEST_OUTPUT_DESCRIPTION,
    )

    post_request = _training_job_post_requests(mock_auth_requests)[0]
    assert post_request.json()["resume_from_job_id"] is None
