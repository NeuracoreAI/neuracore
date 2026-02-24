"""Tests for Dataset class."""

import copy
import re

import pytest
import requests_mock
from neuracore_types import Dataset as DatasetModel
from neuracore_types import DataType, SynchronizationProgress

import neuracore as nc
from neuracore.api.globals import GlobalSingleton
from neuracore.core.const import API_URL, DEFAULT_RECORDING_CACHE_DIR
from neuracore.core.data.dataset import Dataset
from neuracore.core.data.recording import Recording
from neuracore.core.data.synced_dataset import SynchronizedDataset
from neuracore.core.exceptions import DatasetError
from neuracore.core.utils.robot_mapping import RobotMapping


@pytest.fixture
def dataset_response() -> Dataset:
    """Create a mock dataset response."""
    return DatasetModel(
        id="dataset_123",
        name="test_dataset",
        created_at=0.0,
        modified_at=0.0,
        description="A test dataset",
        size_bytes=1024,
        tags=["test", "robotics"],
        is_shared=False,
        num_demonstrations=20,
    )


def test_nc_create_dataset_basic(
    temp_config_dir,
    mock_data_requests,
    reset_neuracore,
    dataset_response,
    mocked_org_id,
):
    """Test dataset creation via the public nc.create_dataset API."""
    # Ensure login
    nc.login("test_api_key")

    # Mock dataset creation endpoint
    mock_data_requests.post(
        f"{API_URL}/org/{mocked_org_id}/datasets",
        json=dataset_response.model_dump(mode="json"),
        status_code=200,
    )

    # Mock recordings endpoint (Dataset will often hit this to init num_recordings)
    mock_data_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_response.id}/recordings",
        json={"data": [], "total": 0, "limit": 1, "start_after": None},
        status_code=200,
    )

    # Mock search-by-name (if nc.create_dataset internally checks for existence)
    mock_data_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
        json={},
        status_code=404,
    )

    # Create dataset via nc
    dataset = nc.create_dataset("test_dataset")

    # Verify dataset was created and mapped correctly
    assert dataset is not None
    assert dataset.id == "dataset_123"
    assert dataset.name == "test_dataset"
    assert dataset.size_bytes == 1024
    assert dataset.tags == ["test", "robotics"]
    assert dataset.is_shared is False


def test_nc_create_dataset_with_params(
    temp_config_dir,
    mock_data_requests,
    reset_neuracore,
    dataset_response,
    mocked_org_id,
):
    """Test dataset creation with additional parameters via nc.create_dataset."""
    nc.login("test_api_key")

    # Assume nc.create_dataset first checks if the dataset exists
    mock_data_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
        json={},
        status_code=404,
    )

    # Mock dataset creation endpoint
    mock_data_requests.post(
        f"{API_URL}/org/{mocked_org_id}/datasets",
        json=dataset_response.model_dump(mode="json"),
        status_code=200,
    )

    # Mock recordings endpoint
    mock_data_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_response.id}/recordings",
        json={"data": [], "total": 0, "limit": 1, "start_after": None},
        status_code=200,
    )

    dataset = nc.create_dataset(
        name="test_dataset",
        description="Test dataset description",
        tags=["test", "robotics"],
        shared=True,
    )

    assert dataset is not None
    assert dataset.id == "dataset_123"
    assert dataset.name == "test_dataset"
    # We can't easily assert that description/tags/shared were sent,
    # but this ensures the path doesn’t break when parameters are passed.


def test_nc_get_dataset_existing(
    temp_config_dir,
    mock_data_requests,
    reset_neuracore,
    dataset_response,
    mocked_org_id,
):
    """Test getting an existing dataset via nc.get_dataset."""
    nc.login("test_api_key")

    # Mock datasets list endpoint (if nc.get_dataset uses it)
    mock_data_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets",
        json=[dataset_response.model_dump(mode="json")],
        status_code=200,
    )

    # Mock shared datasets endpoint (if used)
    mock_data_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/shared",
        json=[],
        status_code=200,
    )

    # Mock search-by-name endpoint (most likely used)
    mock_data_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
        json=dataset_response.model_dump(mode="json"),
        status_code=200,
    )

    # Mock recordings endpoint
    mock_data_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_response.id}/recordings",
        json={"data": [], "total": 0, "limit": 1, "start_after": None},
        status_code=200,
    )

    dataset = nc.get_dataset("test_dataset")

    assert dataset is not None
    assert dataset.id == "dataset_123"
    assert dataset.name == "test_dataset"


def test_nc_get_dataset_nonexistent_raises(
    temp_config_dir, mock_data_requests, reset_neuracore, mocked_org_id
):
    """Test getting a non-existent dataset via nc.get_dataset raises DatasetError."""
    nc.login("test_api_key")

    # Mock endpoints returning no datasets
    mock_data_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets",
        json=[],
        status_code=200,
    )
    mock_data_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/shared",
        json=[],
        status_code=200,
    )
    mock_data_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
        json={},
        status_code=404,
    )

    with pytest.raises(DatasetError, match="Dataset 'nonexistent' not found"):
        nc.get_dataset("nonexistent")


def test_nc_create_shared_dataset_sets_is_shared(
    temp_config_dir,
    mock_data_requests,
    reset_neuracore,
    mocked_org_id,
    dataset_response,
):
    """Test creating a shared dataset via nc.create_dataset sets is_shared=True."""
    nc.login("test_api_key")

    dataset_response.is_shared = True

    # Existence check
    mock_data_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
        json={},
        status_code=404,
    )

    # Creation endpoint
    mock_data_requests.post(
        f"{API_URL}/org/{mocked_org_id}/datasets",
        json=dataset_response.model_dump(mode="json"),
        status_code=200,
    )

    # Recordings endpoint
    mock_data_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_response.id}/recordings",
        json={"data": [], "total": 0, "limit": 1, "start_after": None},
        status_code=200,
    )

    dataset = nc.create_dataset(name="shared_dataset", shared=True)

    assert dataset is not None
    assert dataset.is_shared is True


def test_nc_create_dataset_sets_global_state(
    temp_config_dir,
    mock_data_requests,
    reset_neuracore,
    dataset_response,
    mocked_org_id,
):
    """Test that nc.create_dataset stores the dataset ID in global state."""
    nc.login("test_api_key")

    mock_data_requests.post(
        f"{API_URL}/org/{mocked_org_id}/datasets",
        json=dataset_response.model_dump(mode="json"),
        status_code=200,
    )

    mock_data_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_response.id}/recordings",
        json={"data": [], "total": 0, "limit": 1, "start_after": None},
        status_code=200,
    )

    mock_data_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
        json={},
        status_code=404,
    )

    dataset = nc.create_dataset("test_dataset")

    # Verify global state has dataset ID
    assert GlobalSingleton()._active_dataset_id == dataset.id


class TestDatasetInitialization:
    """Tests for Dataset initialization."""

    def test_init_with_dict(self, dataset_dict, mock_login, mock_data_requests):
        """Test initializing a Dataset with a dictionary."""
        # Mock the recordings endpoint for num_recordings initialization

        dataset = Dataset(**dataset_dict)

        assert dataset.id == dataset_dict["id"]
        assert dataset.name == dataset_dict["name"]
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
        assert dataset._num_recordings == 2
        assert dataset[0].id == "rec1"
        assert dataset[1].id == "rec2"

    def test_init_without_recordings_inits_num_recordings(
        self,
        mock_login,
        mock_data_requests,
        dataset_dict,
        recordings_list,
        mocked_org_id,
    ):
        """Test that initializing without recordings fetches them from API."""

        mock_data_requests.post(
            f"{API_URL}/org/{mocked_org_id}/recording/by-dataset/{dataset_dict['id']}?limit=1&is_shared={dataset_dict['is_shared']}",
            json={
                "data": recordings_list[0:2],
                "total": 2,
                "limit": 1,
                "start_after": None,
            },
            status_code=200,
        )

        dataset = Dataset(**dataset_dict)

        assert len(dataset) == 2


class TestDatasetRetrieval:
    """Tests for retrieving existing datasets."""

    def test_get_by_name(self, mock_data_requests):
        """Test getting an existing dataset by name."""
        dataset = Dataset.get_by_name("test_dataset")

        assert dataset.id == "dataset123"
        assert dataset.name == "test_dataset"

    def test_get_by_name_not_found(self, mock_data_requests, mocked_org_id):
        """Test getting a non-existent dataset by name raises an error."""

        mock_data_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
            json={},
            status_code=404,
        )

        with pytest.raises(DatasetError, match="Dataset 'nonexistent' not found"):
            Dataset.get_by_name("nonexistent")

    def test_get_by_name_non_exist_ok(self, mock_data_requests, mocked_org_id):
        """Test get_by_name with non_exist_ok returns None instead of raising."""

        mock_data_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
            json={},
            status_code=404,
        )

        result = Dataset.get_by_name("nonexistent", non_exist_ok=True)
        assert result is None

    def test_get_by_id(self, mock_data_requests, dataset_model, mocked_org_id):
        """Test getting a dataset by ID."""

        mock_data_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_model.id}",
            json=dataset_model.model_dump(mode="json"),
            status_code=200,
        )

        dataset = Dataset.get_by_id("dataset123")

        assert dataset.id == "dataset123"
        assert dataset.name == "test_dataset"

    def test_get_by_id_not_found(self, mock_data_requests, mocked_org_id):
        """Test getting a non-existent dataset by ID raises an error."""

        mock_data_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/nonexistent",
            json={},
            status_code=404,
        )

        with pytest.raises(
            DatasetError, match="Dataset with ID 'nonexistent' not found"
        ):
            Dataset.get_by_id("nonexistent")

    def test_get_by_id_non_exist_ok(self, mock_data_requests, mocked_org_id):
        """Test get_by_id with non_exist_ok returns None instead of raising."""

        mock_data_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/nonexistent",
            json={},
            status_code=404,
        )

        result = Dataset.get_by_id("nonexistent", non_exist_ok=True)
        assert result is None

    def test_get_full_data_spec(
        self,
        mock_data_requests,
        dataset_dict,
        recordings_list,
        mocked_org_id,
    ):
        """Test getting full data spec for a robot ID in the dataset."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)
        robot_id = "test-robot-id"

        # Mock the API response for get_full_data_spec
        expected_data_spec = {
            DataType.RGB_IMAGES.value: ["camera_left", "camera_right"],
            DataType.JOINT_POSITIONS.value: ["joint_pos_1", "joint_pos_2"],
        }

        mock_data_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/{dataset.id}/full-data-spec/{robot_id}",
            json=expected_data_spec,
            status_code=200,
        )

        # Execute
        data_spec = dataset.get_full_data_spec(robot_id)

        # Verify
        assert data_spec == expected_data_spec
        assert DataType.RGB_IMAGES.value in data_spec
        assert DataType.JOINT_POSITIONS.value in data_spec
        assert data_spec[DataType.RGB_IMAGES.value] == ["camera_left", "camera_right"]
        assert data_spec[DataType.JOINT_POSITIONS.value] == [
            "joint_pos_1",
            "joint_pos_2",
        ]


class TestDatasetCreation:
    """Tests for creating new datasets."""

    def test_create_dataset(
        self,
        mock_data_requests,
        dataset_model,
        mocked_org_id,
    ):
        """Test creating a new dataset."""

        # Mock check if exists
        mock_data_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
            json={},
            status_code=404,
        )

        # Mock creation endpoint
        mock_data_requests.post(
            f"{API_URL}/org/{mocked_org_id}/datasets",
            json=dataset_model.model_dump(mode="json"),
            status_code=200,
        )

        # Mock recordings endpoint for num_recordings initialization
        mock_data_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_model.id}/recordings",
            json={"data": [], "total": 0, "limit": 1, "start_after": None},
            status_code=200,
        )

        dataset = Dataset.create(
            "test_dataset", description="Test description", tags=["test"], shared=False
        )

        assert dataset.id == "dataset123"
        assert dataset.name == "test_dataset"

    def test_create_dataset_already_exists(
        self, mock_data_requests, dataset_model, recordings_list, mocked_org_id
    ):
        """Test that creating a dataset that already exists returns the existing one."""

        # Mock that dataset already exists
        mock_data_requests.get(
            re.compile(f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name"),
            json=dataset_model.model_dump(mode="json"),
            status_code=200,
        )

        # Mock recordings endpoint for num_recordings initialization
        mock_data_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_model.id}/recordings",
            json={
                "data": recordings_list[:1],
                "total": 2,
                "limit": 1,
                "start_after": None,
            },
            status_code=200,
        )

        dataset = Dataset.create("test_dataset")

        assert dataset.id == "dataset123"
        assert dataset.name == "test_dataset"

    def test_create_shared_dataset(
        self, mock_data_requests, dataset_model, recordings_list, mocked_org_id
    ):
        """Test creating a shared dataset."""

        # Mock check if exists
        mock_data_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
            json={},
            status_code=404,
        )

        dataset_model.is_shared = True

        # Mock creation endpoint
        mock_data_requests.post(
            f"{API_URL}/org/{mocked_org_id}/datasets",
            json=dataset_model.model_dump(mode="json"),
            status_code=200,
        )

        # Mock recordings endpoint for num_recordings initialization
        mock_data_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_model.id}/recordings",
            json={"data": [], "total": 0, "limit": 1, "start_after": None},
            status_code=200,
        )

        dataset = Dataset.create("test_dataset", shared=True)

        assert dataset.is_shared is True

    @pytest.mark.usefixtures("mock_login")
    def test_create_dataset_unauthorized_error_detail(self, dataset_model):
        """Test dataset creation errors include backend details."""
        mocked_org_id = "test-org-id"
        error_detail = (
            "User is not authorized to upload shared datasets. Uploading shared "
            "data is a privileged action. Please email contact@neuracore.com to "
            "request access."
        )

        with requests_mock.Mocker() as m:
            m.get(
                f"{API_URL}/org-management/my-orgs",
                json=[{"org": {"id": mocked_org_id, "name": "test organization"}}],
            )
            m.get(
                f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
                json={},
                status_code=404,
            )
            m.post(
                f"{API_URL}/org/{mocked_org_id}/datasets",
                json={"detail": {"error": error_detail}},
                status_code=403,
            )

            with pytest.raises(
                DatasetError, match=f"Failed to create dataset: {error_detail}"
            ):
                Dataset.create("unauthorized_shared_dataset", shared=True)

    @pytest.mark.usefixtures("mock_login")
    def test_create_with_special_characters_in_name(self, dataset_model):
        """Test creating a dataset with special characters in name."""

        special_name = "ghjdidnia-dd/X0551-Ker-Pieb87-846483-CNNMLPP"
        mocked_org_id = "test-org-id"

        dataset_model.name = special_name

        # Fully self-contained mocking
        with requests_mock.Mocker() as m:
            # Mock org list call (needed by get_current_org)
            m.get(
                f"{API_URL}/org-management/my-orgs",
                json=[{"org": {"id": mocked_org_id, "name": "test organization"}}],
            )

            # Mock GET dataset by name (called inside Dataset.create → get_by_name)
            m.get(
                f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
                json={},  # empty so Dataset.create proceeds to create
                status_code=404,
            )

            # Mock POST create dataset
            m.post(
                f"{API_URL}/org/{mocked_org_id}/datasets",
                json=dataset_model.model_dump(mode="json"),
                status_code=201,
            )

            # Mock recordings fetch endpoint
            m.post(
                f"{API_URL}/org/{mocked_org_id}/recording/by-dataset/special123",
                json={"data": [], "total": 0, "start_after": None},
            )

            # Now call Dataset.create
            dataset = Dataset.create(special_name)

        # Assert it returned exactly the special dataset
        assert dataset.name == special_name
        assert dataset.id == dataset_model.id


class TestDatasetIndexingAndSlicing:
    """Tests for dataset indexing and slicing operations."""

    def test_len(self, dataset_dict, recordings_list):
        """Test the __len__ method."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)
        assert len(dataset) == 2

    def test_getitem_single_index(
        self, dataset_dict, recordings_list, mock_data_requests, mocked_org_id
    ):
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
        assert isinstance(result[0], Recording)
        assert len(result) == 1
        assert result[0].id == "rec1"

    def test_getitem_slice_full(self, dataset_dict, recordings_list):
        """Test slicing entire dataset."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        result = dataset[:]

        assert isinstance(result, Dataset)
        assert isinstance(result[0], Recording)
        assert len(result) == 2

    def test_getitem_slice_with_step(self, dataset_dict, recordings_list):
        """Test slicing with step parameter."""
        # Modify recordings_list to have more entries
        new_recordings = [copy.deepcopy(recordings_list[0]) for i in range(5)]
        for i in range(5):
            new_recordings[i]["id"] = f"rec{i}"
        dataset = Dataset(**dataset_dict, recordings=new_recordings)

        result = dataset[0:5:2]

        assert len(result) == 3
        assert result[0].id == "rec0"
        assert result[1].id == "rec2"
        assert result[2].id == "rec4"

    def test_getitem_invalid_type(self, dataset_dict, recordings_list):
        """Test that non-integer/slice index raises TypeError."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        with pytest.raises(TypeError, match="Dataset indices must be int or slice"):
            _ = dataset["invalid"]

    def test_lazy_loading_on_index(
        self, dataset_dict, recordings_list, mock_data_requests, mocked_org_id
    ):
        """Test that indexing triggers lazy loading of recordings."""
        # Mock initial total count fetch
        mock_data_requests.post(
            f"{API_URL}/org/{mocked_org_id}/recording/by-dataset/{dataset_dict['id']}",
            [
                # First call: get total count
                {
                    "json": {
                        "data": recordings_list[0],
                        "total": 2,
                        "limit": 1,
                        "start_after": None,
                    },
                    "status_code": 200,
                },
                # Second call: get all recordings when accessing index
                {
                    "json": {
                        "data": recordings_list,
                        "total": 2,
                        "limit": 100,
                        "start_after": None,
                    },
                    "status_code": 200,
                },
            ],
        )

        dataset = Dataset(**dataset_dict)

        # Initially, recordings should not be loaded
        len(dataset)
        assert dataset._num_recordings == 2
        assert len(dataset._recordings_cache) == 0

        # Accessing an index should trigger loading
        recording = dataset[0]

        assert isinstance(recording, Recording)
        assert recording.id == "rec1"
        assert len(dataset) == 2  # Now loaded


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

    def test_iter_returns_fresh_iterator(self, dataset_dict, recordings_list):
        """Each call to iter(dataset) should start iteration from the beginning."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        # First iterator: consume only the first recording
        it1 = iter(dataset)
        first = next(it1)
        assert isinstance(first, Recording)
        assert first.id == "rec1"

        # Second iterator: should start from the beginning again
        it2 = iter(dataset)
        all_from_second = list(it2)

        assert len(all_from_second) == 2
        assert [r.id for r in all_from_second] == ["rec1", "rec2"]

        # And __iter__ should return a new iterator, not the dataset itself
        assert it1 is not dataset
        assert it2 is not dataset

    def test_next_stop_iteration(self, dataset_dict, recordings_list):
        """Test that __next__ raises StopIteration when exhausted."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        iterator = iter(dataset)

        # Exhaust the iterator
        for _ in range(len(recordings_list)):
            next(iterator)

        with pytest.raises(StopIteration):
            next(iterator)

    def test_lazy_loading_on_iteration(
        self, dataset_dict, recordings_list, mock_data_requests, mocked_org_id
    ):
        """Test that iteration triggers lazy loading of recordings."""
        # Mock API calls
        mock_data_requests.post(
            f"{API_URL}/org/{mocked_org_id}/recording/by-dataset/dataset123?limit=30&is_shared=False",
            [
                # First call: get total count
                {
                    "json": {
                        "data": recordings_list[:1],
                        "total": len(recordings_list),
                        "limit": 1,
                        "start_after": None,
                    },
                    "status_code": 200,
                },
                # Second call: get all recordings when iterating
                {
                    "json": {
                        "data": recordings_list,
                        "total": len(recordings_list),
                        "limit": 100,
                        "start_after": None,
                    },
                    "status_code": 200,
                },
            ],
        )

        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        # Initially, recordings should not be loaded
        assert dataset._num_recordings == len(recordings_list)
        assert len(dataset) == len(recordings_list)

        # Iterating should trigger loading
        recordings = list(dataset)

        assert len(recordings) == 2
        assert len(dataset) == 2  # Now loaded


class TestDatasetSynchronization:
    """Tests for dataset synchronization."""

    def test_synchronize_with_no_data_types(
        self, mock_data_requests, dataset_dict, recordings_list, mocked_org_id
    ):
        """Test synchronizing a dataset."""

        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        synced = dataset.synchronize(frequency=30)

        assert isinstance(synced, SynchronizedDataset)
        assert synced.frequency == 30

    def test_synchronize_with_data_types(
        self, mock_data_requests, dataset_dict, recordings_list
    ):
        """Test synchronizing with specific data types."""
        nc.login("test_api_key")
        dataset = Dataset(**dataset_dict, recordings=recordings_list)
        dataset._robot_ids = ["robot_id"]
        mapping = RobotMapping("test-org")
        mapping._id_to_name = {"robot_id": "robot_name"}
        mapping._name_to_ids = {"robot_name": ["robot_id"]}
        mapping._initialized = True
        dataset._robot_mapping = mapping

        robot_data_spec = {
            "robot_name": {
                DataType.RGB_IMAGES: [],
                DataType.DEPTH_IMAGES: [],
                DataType.JOINT_POSITIONS: [],
            }
        }
        synced = dataset.synchronize(frequency=30, robot_data_spec=robot_data_spec)

        assert synced.robot_data_spec == {"robot_id": robot_data_spec["robot_name"]}

    def test_synchronize_raises_when_backend_reports_failed_recordings(
        self, mock_data_requests, dataset_dict, recordings_list, monkeypatch
    ):
        """Test synchronization fails immediately when backend reports failures."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        def failed_progress(_: str) -> SynchronizationProgress:
            return SynchronizationProgress(
                synchronized_dataset_id="synced_dataset_123",
                num_synchronized_demonstrations=0,
                has_failures=True,
                num_failed_recordings=1,
                failed_recording_ids=["rec-1"],
            )

        monkeypatch.setattr(
            dataset,
            "_get_synchronization_progress",
            failed_progress,
        )

        with pytest.raises(
            DatasetError, match="Synchronization failed for recording\\(s\\): rec-1"
        ):
            dataset.synchronize(frequency=30)

    def test_get_synchronization_progress_raises_dataset_error_on_409(
        self, mock_data_requests, dataset_dict, recordings_list, mocked_org_id
    ):
        dataset = Dataset(**dataset_dict, recordings=recordings_list)
        mock_data_requests.get(
            f"{API_URL}/org/{mocked_org_id}/synchronize/synchronization-progress/synced_dataset_123",
            json={
                "detail": {
                    "error": "Synchronization failed for recording(s): rec-1",
                    "status": 409,
                }
            },
            status_code=409,
        )

        with pytest.raises(
            DatasetError, match="Synchronization failed for recording\\(s\\): rec-1"
        ):
            dataset._get_synchronization_progress("synced_dataset_123")


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
        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        assert dataset.cache_dir == DEFAULT_RECORDING_CACHE_DIR
