"""Tests for Dataset class."""

import copy
import logging
import pickle
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest
import requests_mock
from neuracore_types import Dataset as DatasetModel
from neuracore_types import DataType
from neuracore_types import Recording as RecordingModel
from neuracore_types import RecordingMetadata

import neuracore as nc
from neuracore.api.globals import GlobalSingleton
from neuracore.core.const import API_URL, DEFAULT_RECORDING_CACHE_DIR
from neuracore.core.data.dataset import PAGE_SIZE, Dataset
from neuracore.core.data.recording import Recording
from neuracore.core.data.synced_dataset import SynchronizedDataset
from neuracore.core.exceptions import DatasetError

TEST_ROBOT_ID = "20a621b7-2f9b-4699-a08e-7d080488a5a3"


def _indexed_names(*names: str) -> dict[int, str]:
    return dict(enumerate(names))


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

    # Mock search-by-name endpoint
    mock_data_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
        json=dataset_response.model_dump(mode="json"),
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
        f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
        json={},
        status_code=404,
    )

    dataset = nc.create_dataset("test_dataset")

    # Verify global state has dataset ID
    assert GlobalSingleton()._active_dataset_id == dataset.id


def test_nc_clone_dataset_from_dataset_object(
    temp_config_dir,
    mock_data_requests,
    reset_neuracore,
    dataset_response,
    mocked_org_id,
    caplog,
):
    """Test cloning from a Dataset object via the public API."""
    nc.login("test_api_key")
    source = Dataset(
        id="source_dataset",
        org_id=mocked_org_id,
        name="source",
        size_bytes=512,
        tags=[],
        data_types=[],
        is_shared=False,
    )
    dataset_response.id = "cloned_dataset"
    dataset_response.name = "clone"
    mock_data_requests.post(
        f"{API_URL}/org/{mocked_org_id}/datasets/clone",
        json=dataset_response.model_dump(mode="json"),
        status_code=200,
    )

    caplog.set_level(logging.WARNING, logger="neuracore.api.datasets")

    cloned = nc.clone_dataset("clone", source_dataset=source, wait=False)

    assert cloned.id == "cloned_dataset"
    assert cloned.name == "clone"
    assert GlobalSingleton()._active_dataset_id == cloned.id
    assert mock_data_requests.last_request.json() == {
        "name": "clone",
        "sourceDatasetId": "source_dataset",
    }
    assert "recordings may not be available immediately" in caplog.text


def test_nc_clone_dataset_from_dataset_id(
    temp_config_dir,
    mock_data_requests,
    reset_neuracore,
    dataset_response,
    mocked_org_id,
):
    """Test cloning when the source is provided by ID."""
    nc.login("test_api_key")
    dataset_response.id = "cloned_dataset"
    dataset_response.name = "clone"
    mock_data_requests.post(
        f"{API_URL}/org/{mocked_org_id}/datasets/clone",
        json=dataset_response.model_dump(mode="json"),
        status_code=200,
    )

    cloned = nc.clone_dataset("clone", dataset_id="source_dataset", wait=False)

    assert cloned.id == "cloned_dataset"
    assert mock_data_requests.last_request.json()["sourceDatasetId"] == "source_dataset"


def test_nc_clone_dataset_from_dataset_name(
    temp_config_dir,
    mock_data_requests,
    reset_neuracore,
    dataset_response,
    mocked_org_id,
):
    """Test cloning when the source is provided by name."""
    nc.login("test_api_key")
    mock_data_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
        json=dataset_response.model_dump(mode="json"),
        status_code=200,
    )
    dataset_response.id = "cloned_dataset"
    dataset_response.name = "clone"
    mock_data_requests.post(
        f"{API_URL}/org/{mocked_org_id}/datasets/clone",
        json=dataset_response.model_dump(mode="json"),
        status_code=200,
    )

    cloned = nc.clone_dataset(
        dataset_name="source_name", new_dataset_name="clone", wait=False
    )

    assert cloned.id == "cloned_dataset"
    assert mock_data_requests.last_request.json()["sourceDatasetId"] == "dataset_123"


def test_nc_clone_dataset_missing_source_raises(
    temp_config_dir,
    mock_data_requests,
    reset_neuracore,
    mocked_org_id,
):
    """Test that an unresolved source string raises DatasetError."""
    nc.login("test_api_key")
    mock_data_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/missing",
        status_code=404,
    )
    mock_data_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
        status_code=404,
    )

    with pytest.raises(DatasetError, match="Dataset 'missing' not found"):
        nc.clone_dataset("clone", dataset_name="missing")


def test_nc_clone_dataset_api_error(
    temp_config_dir,
    mock_data_requests,
    reset_neuracore,
    mocked_org_id,
):
    """Test that clone API errors include the backend detail."""
    nc.login("test_api_key")
    source = Dataset(
        id="source_dataset",
        org_id=mocked_org_id,
        name="source",
        size_bytes=512,
        tags=[],
        data_types=[],
        is_shared=False,
    )
    mock_data_requests.post(
        f"{API_URL}/org/{mocked_org_id}/datasets/clone",
        json={"detail": {"error": "Failed to clone dataset: Dataset already exists"}},
        status_code=409,
    )

    with pytest.raises(
        DatasetError,
        match=r"^Failed to clone dataset: Dataset already exists$",
    ):
        nc.clone_dataset("clone", source_dataset=source)


def test_nc_merge_dataset_api_error(
    temp_config_dir,
    mock_data_requests,
    reset_neuracore,
    dataset_response,
    mocked_org_id,
):
    """Test that merge API errors include the backend detail."""
    nc.login("test_api_key")
    mock_data_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
        json=dataset_response.model_dump(mode="json"),
        status_code=200,
    )
    mock_data_requests.post(
        f"{API_URL}/org/{mocked_org_id}/datasets/merge",
        json={"detail": {"error": "Failed to merge datasets: Dataset not found."}},
        status_code=404,
    )

    with pytest.raises(
        DatasetError,
        match=r"^Failed to merge datasets: Dataset not found\.$",
    ):
        nc.merge_datasets("merged", ["source_a", "source_b"])


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

    def test_get_full_embodiment_description(
        self,
        mock_data_requests,
        dataset_dict,
        recordings_list,
        mocked_org_id,
    ):
        """Test getting full embodiment description for a robot ID in the dataset."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)
        robot_id = "test-robot-id"

        # Mock the API response for get_full_embodiment_description
        expected_embodiment_description = {
            DataType.RGB_IMAGES.value: _indexed_names("camera_left", "camera_right"),
            DataType.JOINT_POSITIONS.value: _indexed_names(
                "joint_pos_1", "joint_pos_2"
            ),
        }

        mock_data_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/{dataset.id}/full-embodiment-description/{robot_id}",
            json=expected_embodiment_description,
            status_code=200,
        )

        # Execute
        embodiment_description = dataset.get_full_embodiment_description(robot_id)

        # Verify
        assert embodiment_description == {
            DataType.RGB_IMAGES: _indexed_names("camera_left", "camera_right"),
            DataType.JOINT_POSITIONS: _indexed_names("joint_pos_1", "joint_pos_2"),
        }
        assert DataType.RGB_IMAGES in embodiment_description
        assert DataType.JOINT_POSITIONS in embodiment_description
        assert embodiment_description[DataType.RGB_IMAGES] == _indexed_names(
            "camera_left", "camera_right"
        )
        assert embodiment_description[DataType.JOINT_POSITIONS] == _indexed_names(
            "joint_pos_1", "joint_pos_2"
        )


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

        dataset = Dataset.create("test_dataset", shared=True)

        assert dataset.is_shared is True

    @pytest.mark.usefixtures("mock_login")
    @pytest.mark.parametrize(
        "shared,existing_is_shared,found,expect_create",
        [
            pytest.param(False, False, True, False, id="private_reuses_private"),
            pytest.param(
                False, True, True, True, id="private_creates_when_shared_exists"
            ),
            pytest.param(False, False, False, True, id="private_creates_when_none"),
            pytest.param(
                True, False, True, True, id="shared_creates_when_private_exists"
            ),
            pytest.param(True, True, True, False, id="shared_reuses_shared"),
            pytest.param(True, False, False, True, id="shared_creates_when_none"),
        ],
    )
    def test_create_dataset_namespace(
        self,
        shared,
        existing_is_shared,
        found,
        expect_create,
        dataset_model,
        mocked_org_id,
    ):
        """Test create reuses same-namespace datasets and creates across namespaces."""
        existing = copy.deepcopy(dataset_model)
        existing.is_shared = existing_is_shared

        created = copy.deepcopy(dataset_model)
        created.is_shared = shared
        if expect_create and found:
            created.id = "dataset456"

        with requests_mock.Mocker() as m:
            m.get(
                f"{API_URL}/org-management/my-orgs",
                json=[{"org": {"id": mocked_org_id, "name": "test organization"}}],
            )
            if found:
                m.get(
                    f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
                    json=existing.model_dump(mode="json"),
                    status_code=200,
                )
            else:
                m.get(
                    f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
                    json={},
                    status_code=404,
                )
            m.post(
                f"{API_URL}/org/{mocked_org_id}/datasets",
                json=created.model_dump(mode="json"),
                status_code=200,
            )

            dataset = Dataset.create("test_dataset", shared=shared)

            post_calls = [
                call
                for call in m.request_history
                if call.method == "POST"
                and call.url == f"{API_URL}/org/{mocked_org_id}/datasets"
            ]
            if expect_create:
                assert len(post_calls) == 1
                assert dataset.id == created.id
                assert dataset.is_shared == shared
            else:
                assert len(post_calls) == 0
                assert dataset.id == existing.id
                assert dataset.is_shared == existing_is_shared

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
        # data_types populated when present in API response
        assert recordings[0].data_types == {
            DataType.JOINT_POSITIONS,
            DataType.RGB_IMAGES,
        }
        # data_types defaults to empty set when absent from API response
        assert recordings[1].data_types == set()

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

    def test_get_robot_names_returns_names_by_id(
        self, mock_data_requests, dataset_dict
    ):
        nc.login("test_api_key")
        dataset = Dataset(**dataset_dict)

        assert dataset.get_robot_names() == {
            "20a621b7-2f9b-4699-a08e-7d080488a5a3": "robot-1",
            "30b731c8-3f9c-5799-b19e-8d190599b6b4": "robot-2",
        }
        assert dataset.robot_ids == [
            "20a621b7-2f9b-4699-a08e-7d080488a5a3",
            "30b731c8-3f9c-5799-b19e-8d190599b6b4",
        ]

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
        dataset._robot_ids = [TEST_ROBOT_ID]

        cross_embodiment_union = {
            TEST_ROBOT_ID: {
                DataType.RGB_IMAGES: [],
                DataType.DEPTH_IMAGES: [],
                DataType.JOINT_POSITIONS: [],
            }
        }
        synced = dataset.synchronize(
            frequency=30, cross_embodiment_union=cross_embodiment_union
        )

        assert synced.cross_embodiment_union == {
            TEST_ROBOT_ID: cross_embodiment_union[TEST_ROBOT_ID]
        }

    @pytest.mark.usefixtures("mock_login")
    def test_synchronize_polls_until_complete(
        self,
        mock_data_requests,
        dataset_dict,
        recordings_list,
        mocked_org_id,
        monkeypatch,
    ):
        """synchronize() polls progress until processed reaches total.

        The conftest mock returns a fully-synced progress immediately (the
        already-synchronized path). Override it with a partial-then-complete
        sequence so the tqdm polling loop runs.
        """
        monkeypatch.setattr(
            "neuracore.core.data.dataset.SYNC_PROGRESS_POLL_INTERVAL_S", 0
        )
        dataset = Dataset(**dataset_dict, recordings=recordings_list)
        total = len(recordings_list)

        def progress_json(num_synchronized: int) -> dict:
            return {
                "synchronized_dataset_id": "synced_dataset_123",
                "num_synchronized_demonstrations": num_synchronized,
                "has_failures": False,
                "num_failed_recordings": 0,
                "failed_recording_ids": [],
            }

        progress_endpoint = (
            f"{API_URL}/org/{mocked_org_id}/synchronize/"
            f"synchronization-progress/synced_dataset_123"
        )
        # First poll: nothing done yet (enters the loop); then one more, then
        # complete. The last response repeats if polled again.
        progress_matcher = mock_data_requests.get(
            progress_endpoint,
            [
                {"json": progress_json(0), "status_code": 200},
                {"json": progress_json(1), "status_code": 200},
                {"json": progress_json(total), "status_code": 200},
            ],
        )

        synced = dataset.synchronize(frequency=30)

        assert isinstance(synced, SynchronizedDataset)
        # Initial read + at least two polls to advance 0 -> 1 -> total.
        assert progress_matcher.call_count >= 3

    @pytest.mark.usefixtures("mock_login")
    def test_synchronize_propagates_dataset_error_on_failure(
        self, mock_data_requests, dataset_dict, recordings_list, mocked_org_id
    ):
        """synchronize() propagates the DatasetError from progress polling."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)
        error = (
            "Synchronization failed for recording: rec1. "
            "Reason: No sensors found for data type DEPTH_IMAGES"
        )
        mock_data_requests.get(
            f"{API_URL}/org/{mocked_org_id}/synchronize/synchronization-progress/synced_dataset_123",
            json={"detail": {"error": error, "status": 422}},
            status_code=422,
        )

        with pytest.raises(DatasetError, match="No sensors found for data type"):
            dataset.synchronize(frequency=30)

    @pytest.mark.usefixtures("mock_login")
    def test_get_synchronization_progress_surfaces_detail_on_422(
        self, mock_data_requests, dataset_dict, recordings_list, mocked_org_id
    ):
        """A failure response surfaces the backend detail verbatim."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)
        error = (
            "Synchronization failed for recording: rec1. "
            "Reason: No sensors found for data type DEPTH_IMAGES"
        )
        mock_data_requests.get(
            f"{API_URL}/org/{mocked_org_id}/synchronize/synchronization-progress/synced_dataset_123",
            json={"detail": {"error": error, "status": 422}},
            status_code=422,
        )

        with pytest.raises(DatasetError, match=error):
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


class TestDatasetConcurrentPagination:
    """Regression tests for thread-safety of lazy page fetching.

    SynchronizedDataset prefetches recordings from a thread pool, so
    concurrent Dataset[idx] calls race on the shared pagination cursor and
    recordings cache. Without locking, the same page could be fetched and
    appended multiple times, duplicating recordings and dropping the tail.
    """

    def test_concurrent_getitem_no_duplicate_pages(self, dataset_dict):
        """Concurrent indexing must fetch each page exactly once."""
        import neuracore.core.data.dataset as ds_mod

        total = 496
        all_recordings = [
            RecordingModel(
                id=f"rec-{i:04d}",
                robot_id=TEST_ROBOT_ID,
                instance=1,
                org_id="test-org-id",
                start_time=float(i),
                end_time=float(i) + 1.0,
                total_bytes=512,
                metadata=RecordingMetadata(),
            ).model_dump(mode="json")
            for i in range(total)
        ]
        request_count = 0
        count_lock = threading.Lock()

        def fake_post(url, headers=None, params=None, json=None, timeout=None):
            nonlocal request_count
            with count_lock:
                request_count += 1
            start = 0 if json is None else int(json["id"].split("-")[1]) + 1
            batch = all_recordings[start : start + params.get("limit", PAGE_SIZE)]
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {"data": batch, "total": total}
            response.raise_for_status = MagicMock()
            return response

        session = MagicMock()
        session.post.side_effect = fake_post

        with (
            patch.object(ds_mod, "thread_local_session", return_value=session),
            patch.object(ds_mod, "get_auth", return_value=MagicMock(get_headers=dict)),
        ):
            for _ in range(10):
                dataset = Dataset(**dataset_dict)
                dataset._num_recordings = total
                with ThreadPoolExecutor(max_workers=8) as executor:
                    recordings = list(executor.map(lambda i: dataset[i], range(total)))

                ids = [r.id for r in recordings]
                assert len(set(ids)) == total, "duplicate recordings from page race"
                assert ids == sorted(ids), "recording order corrupted by page race"

        num_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
        assert request_count == 10 * num_pages, "pages fetched more than once"


class TestDatasetCopyAndPickle:
    """Regression tests for copy/pickle support around the page lock.

    Dataset holds a threading.Lock (_page_lock) for thread-safe page
    fetching. train.py deep-copies the dataset for batch-size validation and
    autotuning, and spawn subprocesses pickle it. A raw lock cannot be
    deep-copied or pickled, so it must be dropped and recreated on copy/restore.
    """

    def test_dataset_deepcopy_recreates_page_lock(self, dataset_dict):
        """copy.deepcopy must not choke on the page lock."""
        dataset = Dataset(**dataset_dict)

        clone = copy.deepcopy(dataset)

        # Each instance gets its own usable lock, never a shared/dead one.
        assert clone._page_lock is not dataset._page_lock
        with clone._page_lock:
            pass

    def test_dataset_pickle_recreates_page_lock(self, dataset_dict):
        """Pickling must work so the autotuner can spawn subprocesses."""
        dataset = Dataset(**dataset_dict)

        restored = pickle.loads(pickle.dumps(dataset))

        assert restored._page_lock is not None
        with restored._page_lock:
            pass

    def test_getstate_snapshots_cache_into_private_list(self, dataset_dict):
        """__getstate__ drops the lock and copies the cache into a private list.

        The private list is what makes a copy safe against a concurrent page
        fetch extending the live cache mid-copy.
        """
        dataset = Dataset(**dataset_dict)

        state = dataset.__getstate__()

        assert state["_page_lock"] is None
        assert state["_recordings_cache"] is not dataset._recordings_cache
        assert state["_recordings_cache"] == dataset._recordings_cache


def _register_dataset_delete(mock_data_requests, dataset_dict):
    mock_data_requests.delete(
        f"{API_URL}/org/{dataset_dict['org_id']}/datasets/{dataset_dict['id']}",
        status_code=200,
    )


@pytest.fixture
def deleted_dataset(dataset_dict, mock_data_requests, mock_login):
    """A Dataset that has already been deleted."""
    _register_dataset_delete(mock_data_requests, dataset_dict)
    dataset = Dataset(**dataset_dict)
    dataset.delete()
    return dataset


@pytest.mark.usefixtures("mock_login")
class TestDatasetDeletion:
    """Tests for delete invalidating the in-memory Dataset object."""

    def test_deleted_false_before_delete(self, dataset_dict):
        """A live dataset reports deleted as False."""
        dataset = Dataset(**dataset_dict)

        assert dataset.deleted is False

    def test_delete_calls_api(self, dataset_dict, mock_data_requests):
        """delete issues the DELETE request to the datasets endpoint."""
        _register_dataset_delete(mock_data_requests, dataset_dict)
        dataset = Dataset(**dataset_dict)

        dataset.delete()

        last = mock_data_requests.request_history[-1]
        assert last.method == "DELETE"
        assert last.url.endswith(f"/datasets/{dataset_dict['id']}")

    def test_deleted_true_after_delete(self, deleted_dataset):
        """After delete, deleted reports True."""
        assert deleted_dataset.deleted is True

    def test_deleted_dataset_is_still_a_dataset(self, deleted_dataset):
        """A deleted dataset keeps its Dataset type for isinstance checks."""
        assert isinstance(deleted_dataset, Dataset)

    def test_attribute_access_after_delete_raises(self, deleted_dataset):
        """Reading any attribute of a deleted dataset raises DatasetError."""
        with pytest.raises(DatasetError, match="deleted"):
            _ = deleted_dataset.name

    def test_method_call_after_delete_raises(self, deleted_dataset):
        """Calling any method of a deleted dataset raises DatasetError."""
        with pytest.raises(DatasetError, match="deleted"):
            deleted_dataset.get_robot_names()

    def test_delete_releases_recording_cache(
        self, dataset_dict, recordings_list, mock_data_requests
    ):
        """delete clears the recording cache so its memory is reclaimable."""
        _register_dataset_delete(mock_data_requests, dataset_dict)
        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        dataset.delete()

        with pytest.raises(DatasetError, match="deleted"):
            _ = dataset._recordings_cache
