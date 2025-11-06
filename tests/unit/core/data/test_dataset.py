"""Tests for Dataset class."""

import re

import pytest

from neuracore.core.const import API_URL
from neuracore.core.data.dataset import Dataset
from neuracore.core.data.recording import Recording
from neuracore.core.data.synced_dataset import SynchronizedDataset
from neuracore.core.exceptions import DatasetError


class TestDatasetInitialization:
    """Tests for Dataset initialization."""

    def test_init_with_dict(self, dataset_dict, mock_auth_requests):
        """Test initializing a Dataset with a dictionary."""
        dataset = Dataset(**dataset_dict)

        assert dataset.id == "dataset123"
        assert dataset.name == "test_dataset"
        assert dataset.size_bytes == 1024
        assert dataset.tags == ["test", "robotics"]
        assert dataset.is_shared is False
        assert dataset.org_id == dataset_dict["org_id"]

    def test_init_with_recordings(self, dataset_dict, recordings_list):
        """Test initializing a Dataset with provided recordings."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        assert dataset.id == "dataset123"
        assert dataset.name == "test_dataset"
        assert len(dataset) == 2
        assert len(dataset.recordings) == 2
        assert dataset.recordings[0]["id"] == "rec1"
        assert dataset.recordings[1]["id"] == "rec2"

    def test_init_without_recordings_fetches_from_api(
        self, mock_auth_requests, dataset_dict, recordings_list, mocked_org_id
    ):
        """Test that initializing without recordings fetches them from API."""

        # Mock recordings endpoint
        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_dict['id']}/recordings",
            json={"recordings": recordings_list},
            status_code=200,
        )

        dataset = Dataset(**dataset_dict)

        assert len(dataset.recordings) == 2
        assert dataset.recordings[0]["id"] == "rec1"


class TestDatasetRetrieval:
    """Tests for retrieving existing datasets."""

    def test_get_by_name(
        self,
        temp_config_dir,
        reset_neuracore,
        mock_auth_requests,
        dataset_dict,
        recordings_list,
        mocked_org_id,
    ):
        """Test getting an existing dataset by name."""

        # Mock datasets endpoint
        mock_auth_requests.get(
            re.compile(f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name"),
            json=dataset_dict,
            status_code=200,
        )

        # Mock recordings endpoint
        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_dict['id']}/recordings",
            json={"recordings": recordings_list},
            status_code=200,
        )

        dataset = Dataset.get_by_name("test_dataset")

        assert dataset.id == "dataset123"
        assert dataset.name == "test_dataset"

    def test_get_by_name_not_found(self, mock_auth_requests, mocked_org_id):
        """Test getting a non-existent dataset by name raises an error."""

        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
            json={},
            status_code=404,
        )

        with pytest.raises(DatasetError, match="Dataset 'nonexistent' not found"):
            Dataset.get_by_name("nonexistent")

    def test_get_by_name_non_exist_ok(self, mock_auth_requests, mocked_org_id):
        """Test get_by_name with non_exist_ok returns None instead of raising."""

        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
            json={},
            status_code=404,
        )

        result = Dataset.get_by_name("nonexistent", non_exist_ok=True)
        assert result is None

    def test_get_by_id(
        self, mock_auth_requests, dataset_dict, recordings_list, mocked_org_id
    ):
        """Test getting a dataset by ID."""

        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_dict['id']}",
            json=dataset_dict,
            status_code=200,
        )

        # Mock recordings endpoint
        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_dict['id']}/recordings",
            json={"recordings": recordings_list},
            status_code=200,
        )

        dataset = Dataset.get_by_id("dataset123")

        assert dataset.id == "dataset123"
        assert dataset.name == "test_dataset"

    def test_get_by_id_not_found(self, mock_auth_requests, mocked_org_id):
        """Test getting a non-existent dataset by ID raises an error."""

        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/nonexistent",
            json={},
            status_code=404,
        )

        with pytest.raises(
            DatasetError, match="Dataset with ID 'nonexistent' not found"
        ):
            Dataset.get_by_id("nonexistent")

    def test_get_by_id_non_exist_ok(self, mock_auth_requests, mocked_org_id):
        """Test get_by_id with non_exist_ok returns None instead of raising."""

        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/nonexistent",
            json={},
            status_code=404,
        )

        result = Dataset.get_by_id("nonexistent", non_exist_ok=True)
        assert result is None


class TestDatasetCreation:
    """Tests for creating new datasets."""

    def test_create_dataset(
        self,
        temp_config_dir,
        reset_neuracore,
        mock_auth_requests,
        dataset_dict,
        recordings_list,
        mocked_org_id,
    ):
        """Test creating a new dataset."""

        # Mock check if exists
        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
            json={},
            status_code=404,
        )

        # Mock creation endpoint
        mock_auth_requests.post(
            f"{API_URL}/org/{mocked_org_id}/datasets",
            json=dataset_dict,
            status_code=200,
        )

        # Mock recordings endpoint
        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_dict['id']}/recordings",
            json={"recordings": recordings_list},
            status_code=200,
        )

        dataset = Dataset.create(
            "test_dataset", description="Test description", tags=["test"], shared=False
        )

        assert dataset.id == "dataset123"
        assert dataset.name == "test_dataset"

    def test_create_dataset_already_exists(
        self, mock_auth_requests, dataset_dict, recordings_list, mocked_org_id
    ):
        """Test that creating a dataset that already exists returns the existing one."""

        # Mock that dataset already exists
        mock_auth_requests.get(
            re.compile(f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name"),
            json=dataset_dict,
            status_code=200,
        )

        # Mock recordings endpoint
        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_dict['id']}/recordings",
            json={"recordings": recordings_list},
            status_code=200,
        )

        dataset = Dataset.create("test_dataset")

        assert dataset.id == "dataset123"
        assert dataset.name == "test_dataset"

    def test_create_shared_dataset(
        self, mock_auth_requests, dataset_dict, recordings_list, mocked_org_id
    ):
        """Test creating a shared dataset."""

        # Mock check if exists
        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
            json={},
            status_code=404,
        )

        shared_dataset_dict = {**dataset_dict, "is_shared": True}

        # Mock creation endpoint
        mock_auth_requests.post(
            f"{API_URL}/org/{mocked_org_id}/datasets",
            json=shared_dataset_dict,
            status_code=200,
        )

        # Mock recordings endpoint
        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_dict['id']}/recordings",
            json={"recordings": recordings_list},
            status_code=200,
        )

        dataset = Dataset.create("test_dataset", shared=True)

        assert dataset.is_shared is True

    def test_create_with_special_characters_in_name(
        self, mock_auth_requests, mocked_org_id
    ):
        """Test creating a dataset with special characters in name."""

        special_name = "ghjdidnia-dd/X0551-Ker-Pieb87-846483-CNNMLPP"

        # Mock check if exists
        mock_auth_requests.get(
            re.compile(f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name"),
            json={},
            status_code=404,
        )

        dataset_dict = {
            "id": "special123",
            "org_id": mocked_org_id,
            "name": special_name,
            "size_bytes": 0,
            "tags": [],
            "is_shared": False,
        }

        # Mock creation endpoint
        mock_auth_requests.post(
            f"{API_URL}/org/{mocked_org_id}/datasets",
            json=dataset_dict,
            status_code=200,
        )

        # Mock recordings endpoint
        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/special123/recordings",
            json={"recordings": []},
            status_code=200,
        )

        dataset = Dataset.create(special_name)

        assert dataset.name == special_name


class TestDatasetIndexingAndSlicing:
    """Tests for dataset indexing and slicing operations."""

    def test_len(self, dataset_dict, recordings_list):
        """Test the __len__ method."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)
        assert len(dataset) == 2

    def test_getitem_single_index(self, dataset_dict, recordings_list):
        """Test accessing a single recording by index."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        recording = dataset[0]

        assert isinstance(recording, Recording)
        assert recording.id == "rec1"

    def test_getitem_negative_index(self, dataset_dict, recordings_list):
        """Test accessing recordings with negative indices."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        recording = dataset[-1]

        assert isinstance(recording, Recording)
        assert recording.id == "rec2"

    def test_getitem_out_of_range(self, dataset_dict, recordings_list):
        """Test that out of range index raises IndexError."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        with pytest.raises(IndexError, match="Dataset index out of range"):
            _ = dataset[10]

    def test_getitem_slice(self, dataset_dict, recordings_list):
        """Test getting a slice of the dataset."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        result = dataset[0:1]

        assert isinstance(result, Dataset)
        assert len(result) == 1
        assert result.recordings[0]["id"] == "rec1"

    def test_getitem_slice_full(self, dataset_dict, recordings_list):
        """Test slicing entire dataset."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        result = dataset[:]

        assert isinstance(result, Dataset)
        assert len(result) == 2

    def test_getitem_slice_with_step(self, dataset_dict):
        """Test slicing with step parameter."""
        # Create dataset with more recordings
        recordings = [
            {
                "id": f"rec{i}",
                "robot_id": f"robot{i}",
                "instance": 1,
                "total_bytes": 100,
                "created_at": "2023-01-01T00:00:00Z",
            }
            for i in range(5)
        ]

        dataset = Dataset(**dataset_dict, recordings=recordings)

        result = dataset[0:5:2]

        assert len(result) == 3
        assert result.recordings[0]["id"] == "rec0"
        assert result.recordings[1]["id"] == "rec2"
        assert result.recordings[2]["id"] == "rec4"

    def test_getitem_invalid_type(self, dataset_dict, recordings_list):
        """Test that non-integer/slice index raises TypeError."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        with pytest.raises(
            TypeError, match="Dataset indices must be integers or slices"
        ):
            _ = dataset["invalid"]


class TestDatasetIteration:
    """Tests for iterating through datasets."""

    def test_iteration(self, dataset_dict, recordings_list):
        """Test iterating through dataset recordings."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        recordings = list(dataset)

        assert len(recordings) == 2
        assert all(isinstance(r, Recording) for r in recordings)
        assert recordings[0].id == "rec1"
        assert recordings[1].id == "rec2"

    def test_iteration_multiple_times(self, dataset_dict, recordings_list):
        """Test that dataset can be iterated multiple times."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        recordings1 = list(dataset)
        recordings2 = list(dataset)

        assert len(recordings1) == len(recordings2)
        assert recordings1[0].id == recordings2[0].id

    def test_iter_reset(self, dataset_dict, recordings_list):
        """Test that __iter__ resets the iteration index."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        dataset._recording_idx = 5
        result = iter(dataset)

        assert result is dataset
        assert dataset._recording_idx == 0

    def test_next_stop_iteration(self, dataset_dict, recordings_list):
        """Test that __next__ raises StopIteration when exhausted."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        iter(dataset)

        # Exhaust the iterator
        for _ in range(len(recordings_list)):
            next(dataset)

        with pytest.raises(StopIteration):
            next(dataset)


class TestDatasetSynchronization:
    """Tests for dataset synchronization."""

    def test_synchronize_with_no_data_types(
        self, mock_auth_requests, dataset_dict, recordings_list, mocked_org_id
    ):
        """Test synchronizing a dataset."""

        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        synced = dataset.synchronize(frequency=30)

        assert isinstance(synced, SynchronizedDataset)
        assert synced.frequency == 30

    def test_synchronize_with_data_types(
        self, mock_auth_requests, dataset_dict, recordings_list, mocked_org_id
    ):
        """Test synchronizing with specific data types."""

        from neuracore.core.nc_types import DataType

        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        data_types = [DataType.RGB_IMAGE, DataType.DEPTH_IMAGE]
        synced = dataset.synchronize(frequency=30, data_types=data_types)

        assert synced.data_types == data_types


class TestDatasetMixedOperations:
    """Tests for mixed dataset operations."""

    def test_mixed_access_patterns(self, dataset_dict, recordings_list):
        """Test mixing iteration and indexing."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        # Access by index
        rec0 = dataset[0]

        # Then iterate
        recordings = list(dataset)

        # Access by negative index
        rec_last = dataset[-1]

        assert rec0.id == "rec1"
        assert recordings[0].id == "rec1"
        assert rec_last.id == "rec2"

    def test_slice_then_iterate(self, dataset_dict, recordings_list):
        """Test slicing dataset then iterating through slice."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        sliced = dataset[0:1]
        recordings = list(sliced)

        assert len(recordings) == 1
        assert recordings[0].id == "rec1"

    def test_cache_dir_default(self, dataset_dict, recordings_list):
        """Test that cache_dir is set to default location."""
        from neuracore.core.data.dataset import DEFAULT_CACHE_DIR

        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        assert dataset.cache_dir == DEFAULT_CACHE_DIR
