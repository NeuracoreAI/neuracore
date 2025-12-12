"""Tests for Recording class."""

import pytest
from neuracore_types import DataType

import neuracore as nc
from neuracore.core.data.recording import Recording
from neuracore.core.data.synced_recording import SynchronizedRecording
from neuracore.core.exceptions import SynchronizationError


class TestRecording:
    """Tests for the Recording class."""

    @pytest.fixture
    def dataset_mock(self, dataset_dict, recordings_list, mock_auth_requests):
        """Create a mock dataset object."""
        nc.login("test_api_key")
        from neuracore.core.data.dataset import Dataset

        return Dataset(**dataset_dict, recordings=recordings_list)

    @pytest.fixture
    def recording(self, dataset_mock):
        """Create a Recording instance for testing."""
        return Recording(
            dataset=dataset_mock,
            recording_id="rec1",
            total_bytes=512,
            robot_id="robot1",
            instance=1,
        )

    def test_init(self, recording: Recording, dataset_mock):
        """Test Recording initialization."""
        assert recording.dataset == dataset_mock
        assert recording.id == "rec1"
        assert recording.total_bytes == 512
        assert recording.robot_id == "robot1"
        assert recording.instance == 1

    def test_synchronize_with_valid_frequency(self, recording):
        """Test synchronizing a recording with valid frequency."""
        synced_rec = recording.synchronize(frequency=30)

        assert isinstance(synced_rec, SynchronizedRecording)
        assert synced_rec.frequency == 30
        assert synced_rec.id == "rec1"
        assert synced_rec.robot_id == "robot1"
        assert synced_rec.instance == 1

    def test_synchronize_with_zero_frequency(self, recording):
        """Test that synchronizing with frequency=0 raises an error."""
        synced_rec = recording.synchronize(frequency=0)
        assert isinstance(synced_rec, SynchronizedRecording)
        assert synced_rec.frequency == 0
        assert len(synced_rec) == 2

    def test_synchronize_with_negative_frequency(self, recording):
        """Test that synchronizing with negative frequency raises an error."""
        with pytest.raises(SynchronizationError, match="Frequency must be >= 0"):
            recording.synchronize(frequency=-10)

    def test_synchronize_with_valid_data_types(self, recording):
        """Test synchronizing with specific data types."""

        data_types = [DataType.RGB_IMAGE, DataType.JOINT_POSITIONS]
        synced = recording.synchronize(frequency=30, data_types=data_types)

        assert isinstance(synced, SynchronizedRecording)
        assert synced.data_types == data_types

    def test_synchronize_with_invalid_data_types(self, recording):
        """Test synchronizing with invalid data types raises an error."""

        data_types = [DataType.RGB_IMAGE, DataType.DEPTH_IMAGE]
        with pytest.raises(
            SynchronizationError,
            match="Invalid data type requested for synchronization",
        ):
            recording.synchronize(frequency=30, data_types=data_types)

    def test_synchronize_with_empty_data_types(self, recording):
        """Test synchronizing with empty data types list."""
        synced = recording.synchronize(frequency=30, data_types=[])

        assert isinstance(synced, SynchronizedRecording)
        assert synced.data_types == []

    def test_synchronize_with_none_data_types(self, recording):
        """Test synchronizing with None data types (should default to empty list)."""
        synced = recording.synchronize(frequency=30, data_types=None)

        assert isinstance(synced, SynchronizedRecording)
        assert synced.data_types == []

    def test_iter_raises_runtime_error(self, recording):
        """Test that iterating over unsynchronized recording raises RuntimeError."""
        with pytest.raises(
            RuntimeError, match="Only synchronized recordings can be iterated over"
        ):
            iter(recording)

    def test_iter_in_for_loop_raises_error(self, recording):
        """Test that using unsynchronized recording in for loop raises error."""
        with pytest.raises(
            RuntimeError, match="Only synchronized recordings can be iterated over"
        ):
            for _ in recording:
                pass

    def test_multiple_synchronize_calls(self, recording):
        """Test that multiple synchronize calls create independent instances."""
        synced1 = recording.synchronize(frequency=30)
        synced2 = recording.synchronize(frequency=60)

        assert synced1 is not synced2
        assert synced1.frequency == 30
        assert synced2.frequency == 60
