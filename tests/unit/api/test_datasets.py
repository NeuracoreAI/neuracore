import pytest

import neuracore as nc
from neuracore.core.const import API_URL
from neuracore.core.exceptions import DatasetError


@pytest.fixture
def dataset_response():
    """Create a mock dataset response."""
    return {
        "id": "dataset_123",
        "name": "test_dataset",
        "size_bytes": 1024,
        "tags": ["test", "robotics"],
        "is_shared": False,
    }


def test_create_dataset(
    temp_config_dir, mock_auth_requests, reset_neuracore, dataset_response
):
    """Test dataset creation."""
    # Ensure login
    nc.login("test_api_key")

    # Mock dataset creation endpoint
    mock_auth_requests.post(
        f"{API_URL}/datasets",
        json=dataset_response,
        status_code=200,
    )

    # Mock recordings endpoint
    mock_auth_requests.get(
        f"{API_URL}/datasets/{dataset_response['id']}/recordings",
        json={"recordings": []},
        status_code=200,
    )

    mock_auth_requests.get(
        f"{API_URL}/datasets/by-name/{dataset_response['name']}",
        json=dataset_response,
        status_code=200,
    )

    # Create dataset
    dataset = nc.create_dataset("test_dataset")

    # Verify dataset was created
    assert dataset is not None
    assert dataset.id == "dataset_123"
    assert dataset.name == "test_dataset"
    assert dataset.size_bytes == 1024
    assert dataset.tags == ["test", "robotics"]
    assert dataset.is_shared is False


def test_create_dataset_with_params(
    temp_config_dir, mock_auth_requests, reset_neuracore, dataset_response
):
    """Test dataset creation with additional parameters."""
    # Ensure login
    nc.login("test_api_key")

    # Mock dataset creation endpoint
    mock_auth_requests.post(
        f"{API_URL}/datasets",
        json=dataset_response,
        status_code=200,
    )

    # Mock recordings endpoint
    mock_auth_requests.get(
        f"{API_URL}/datasets/{dataset_response['id']}/recordings",
        json={"recordings": []},
        status_code=200,
    )
    mock_auth_requests.get(
        f"{API_URL}/datasets/by-name/{dataset_response['name']}",
        json=dataset_response,
        status_code=200,
    )

    # Create dataset with additional parameters
    dataset = nc.create_dataset(
        name="test_dataset",
        description="Test dataset description",
        tags=["test", "robotics"],
        shared=True,
    )

    # Verify dataset was created
    assert dataset is not None
    assert dataset.id == "dataset_123"
    assert dataset.name == "test_dataset"


def test_get_dataset(
    temp_config_dir, mock_auth_requests, reset_neuracore, dataset_response
):
    """Test getting an existing dataset."""
    # Ensure login
    nc.login("test_api_key")

    # Mock datasets endpoint
    mock_auth_requests.get(
        f"{API_URL}/datasets",
        json=[dataset_response],
        status_code=200,
    )

    # Mock shared datasets endpoint
    mock_auth_requests.get(
        f"{API_URL}/datasets/shared",
        json=[],
        status_code=200,
    )
    mock_auth_requests.get(
        f"{API_URL}/datasets/by-name/{dataset_response['name']}",
        json=dataset_response,
        status_code=200,
    )

    # Mock recordings endpoint
    mock_auth_requests.get(
        f"{API_URL}/datasets/{dataset_response['id']}/recordings",
        json={"recordings": []},
        status_code=200,
    )

    # Get dataset
    dataset = nc.get_dataset("test_dataset")

    # Verify dataset was retrieved
    assert dataset is not None
    assert dataset.id == "dataset_123"
    assert dataset.name == "test_dataset"


def test_get_nonexistent_dataset(temp_config_dir, mock_auth_requests, reset_neuracore):
    """Test getting a non-existent dataset raises an error."""
    # Ensure login
    nc.login("test_api_key")

    # Mock empty datasets endpoints
    mock_auth_requests.get(
        f"{API_URL}/datasets",
        json=[],
        status_code=200,
    )

    mock_auth_requests.get(
        f"{API_URL}/datasets/shared",
        json=[],
        status_code=200,
    )
    mock_auth_requests.get(
        f"{API_URL}/datasets/by-name/nonexistent", json={}, status_code=404
    )

    # Attempt to get non-existent dataset
    with pytest.raises(DatasetError, match="Dataset 'nonexistent' not found"):
        nc.get_dataset("nonexistent")


def test_dataset_shared_property(temp_config_dir, mock_auth_requests, reset_neuracore):
    """Test dataset shared property."""
    # Ensure login
    nc.login("test_api_key")

    # Mock dataset creation for a shared dataset
    shared_dataset_response = {
        "id": "dataset_456",
        "name": "shared_dataset",
        "size_bytes": 2048,
        "tags": ["shared", "robotics"],
        "is_shared": True,
        "num_demonstrations": 5,
    }

    mock_auth_requests.post(
        f"{API_URL}/datasets",
        json=shared_dataset_response,
        status_code=200,
    )

    # Mock recordings endpoint
    mock_auth_requests.get(
        f"{API_URL}/datasets/{shared_dataset_response['id']}/recordings",
        json={"recordings": []},
        status_code=200,
    )

    mock_auth_requests.get(
        f"{API_URL}/datasets/by-name/{shared_dataset_response['name']}",
        json=shared_dataset_response,
        status_code=200,
    )

    # Create shared dataset
    dataset = nc.create_dataset(
        name="shared_dataset",
        shared=True,
    )

    # Verify dataset was created with is_shared=True
    assert dataset is not None
    assert dataset.is_shared is True


def test_dataset_global_state(
    temp_config_dir, mock_auth_requests, reset_neuracore, dataset_response
):
    """Test that dataset ID is stored in global state."""
    # Ensure login
    nc.login("test_api_key")

    # Mock dataset creation endpoint
    mock_auth_requests.post(
        f"{API_URL}/datasets",
        json=dataset_response,
        status_code=200,
    )

    # Mock recordings endpoint
    mock_auth_requests.get(
        f"{API_URL}/datasets/{dataset_response['id']}/recordings",
        json={"recordings": []},
        status_code=200,
    )
    mock_auth_requests.get(
        f"{API_URL}/datasets/by-name/{dataset_response['name']}",
        json=dataset_response,
        status_code=200,
    )

    # Create dataset
    dataset = nc.create_dataset("test_dataset")

    # Verify global state has dataset ID
    from neuracore.api.globals import GlobalSingleton

    assert GlobalSingleton()._active_dataset_id == dataset.id
