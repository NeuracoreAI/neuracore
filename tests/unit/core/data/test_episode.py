"""Tests for Recording class."""

import pytest
import requests_mock
from neuracore_types import DataType
from neuracore_types import Recording as RecordingModel
from neuracore_types import RecordingMetadata, RecordingStatus

import neuracore as nc
from neuracore.core.const import API_URL
from neuracore.core.data.dataset import Dataset
from neuracore.core.data.recording import Recording
from neuracore.core.data.synced_recording import SynchronizedRecording
from neuracore.core.exceptions import SynchronizationError


class TestRecording:
    """Tests for the Recording class."""

    @pytest.fixture
    def dataset_mock(
        self, dataset_dict, recordings_list, mock_data_requests
    ) -> Dataset:
        """Create a mock dataset object."""
        nc.login("test_api_key")

        return Dataset(**dataset_dict, recordings=recordings_list)

    @pytest.fixture
    def recording_model(self, mocked_org_id) -> RecordingModel:
        """Create a mock recording metadata."""
        return RecordingModel(
            id="rec1",
            org_id=mocked_org_id,
            total_bytes=512,
            robot_id="robot1",
            instance=1,
            start_time=0,
            end_time=10,
            metadata=RecordingMetadata(
                name="recording1",
                org_id="test-org-id",
                created_at="2023-01-01T00:00:00Z",
            ),
        )

    @pytest.fixture
    def recording(self, dataset_mock: Dataset, recording_model) -> Recording:
        """Create a Recording instance for testing."""
        return dataset_mock._wrap_raw_recording(recording_model)

    def test_init(self, recording: Recording, dataset_mock: Dataset):
        """Test Recording initialization."""
        assert recording.dataset == dataset_mock
        assert recording.id == "rec1"
        assert recording.total_bytes == 512
        assert recording.robot_id == "robot1"
        assert recording.instance == 1

    def test_synchronize_with_valid_frequency(self, recording: Recording):
        """Test synchronizing a recording with valid frequency."""
        synced_rec = recording.synchronize(frequency=30)

        assert isinstance(synced_rec, SynchronizedRecording)
        assert synced_rec.frequency == 30
        assert synced_rec.id == "rec1"
        assert synced_rec.robot_id == "robot1"
        assert synced_rec.instance == 1

    def test_synchronize_with_zero_frequency(self, recording: Recording):
        """Test that synchronizing with frequency=0 raises an error."""
        synced_rec = recording.synchronize(frequency=0)
        assert isinstance(synced_rec, SynchronizedRecording)
        assert synced_rec.frequency == 0
        assert len(synced_rec) == 2

    def test_synchronize_with_negative_frequency(self, recording: Recording):
        """Test that synchronizing with negative frequency raises an error."""
        with pytest.raises(SynchronizationError, match="Frequency must be >= 0"):
            recording.synchronize(frequency=-10)

    def test_synchronize_with_valid_data_types(self, recording: Recording):
        """Test synchronizing with specific data types."""

        robot_data_spec = {
            recording.robot_id: {
                DataType.RGB_IMAGES: ["camera_front", "camera_rear"],
                DataType.JOINT_POSITIONS: ["arm", "gripper"],
            }
        }
        synced = recording.synchronize(frequency=30, robot_data_spec=robot_data_spec)

        assert isinstance(synced, SynchronizedRecording)
        assert synced.robot_data_spec == robot_data_spec

    def test_synchronize_with_invalid_data_types(self, recording: Recording):
        """Test synchronizing with invalid data types raises an error."""

        robot_data_spec = {
            recording.robot_id: {
                "INVALID_DATA_TYPE": ["some_item"],
            }
        }
        with pytest.raises(
            ValueError,
            match="Invalid data types provided. ",
        ):
            recording.synchronize(frequency=30, robot_data_spec=robot_data_spec)

    def test_synchronize_with_empty_data_types(self, recording: Recording):
        """Test synchronizing with empty data types list."""
        synced = recording.synchronize(frequency=30, robot_data_spec={})
        assert isinstance(synced, SynchronizedRecording)
        assert synced.robot_data_spec == {}

    def test_synchronize_with_none_data_types(self, recording: Recording):
        """Test synchronizing with None data types (should default to empty list)."""
        synced = recording.synchronize(frequency=30, robot_data_spec=None)

        assert isinstance(synced, SynchronizedRecording)
        assert synced.robot_data_spec is None

    def test_iter_raises_runtime_error(self, recording: Recording):
        """Test that iterating over unsynchronized recording raises RuntimeError."""
        with pytest.raises(
            RuntimeError, match="Only synchronized recordings can be iterated over"
        ):
            iter(recording)

    def test_iter_in_for_loop_raises_error(self, recording: Recording):
        """Test that using unsynchronized recording in for loop raises error."""
        with pytest.raises(
            RuntimeError, match="Only synchronized recordings can be iterated over"
        ):
            for _ in recording:
                pass

    def test_multiple_synchronize_calls(self, recording: Recording):
        """Test that multiple synchronize calls create independent instances."""
        synced1 = recording.synchronize(frequency=30)
        synced2 = recording.synchronize(frequency=60)

        assert synced1 is not synced2
        assert synced1.frequency == 30
        assert synced2.frequency == 60

    def test_set_status(
        self,
        recording: Recording,
        recording_model: RecordingModel,
        mock_data_requests: requests_mock.Mocker,
        mocked_org_id: str,
    ):
        mock_data_requests.get(
            f"{API_URL}/org/{mocked_org_id}/recording/{recording.id}",
            json=recording_model.model_dump(mode="json"),
            status_code=200,
        )

        recording_clone = recording_model.model_copy(deep=True)
        recording_clone.metadata.status = RecordingStatus.FLAGGED

        mock_data_requests.put(
            f"{API_URL}/org/{mocked_org_id}/recording/{recording.id}/metadata",
            json=recording_clone.model_dump(mode="json"),
            status_code=200,
        )

        assert recording.metadata.status == RecordingStatus.NORMAL
        recording.set_status(RecordingStatus.FLAGGED)
        assert mock_data_requests.request_history[-1].json().get("status") == "FLAGGED"
        assert recording.metadata.status == RecordingStatus.FLAGGED

    def test_set_notes(
        self,
        recording: Recording,
        recording_model: RecordingModel,
        mock_data_requests: requests_mock.Mocker,
        mocked_org_id: str,
    ):
        mock_data_requests.get(
            f"{API_URL}/org/{mocked_org_id}/recording/{recording.id}",
            json=recording_model.model_dump(mode="json"),
            status_code=200,
        )

        recording_clone = recording_model.model_copy(deep=True)
        recording_clone.metadata.notes = "Test notes"

        mock_data_requests.put(
            f"{API_URL}/org/{mocked_org_id}/recording/{recording.id}/metadata",
            json=recording_clone.model_dump(mode="json"),
            status_code=200,
        )

        assert recording.metadata.notes == ""
        recording.set_notes("Test notes")
        assert (
            mock_data_requests.request_history[-1].json().get("notes") == "Test notes"
        )
        assert recording.metadata.notes == "Test notes"
