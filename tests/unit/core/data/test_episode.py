"""Tests for Recording class."""

from urllib.parse import urlparse

import pytest
import requests_mock
from neuracore_types import DataType, QAFailureReason, QAFinding
from neuracore_types import Recording as RecordingModel
from neuracore_types import RecordingMetadata, RecordingTag, RecordingTagSource

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

        cross_embodiment_union = {
            recording.robot_id: {
                DataType.RGB_IMAGES: ["camera_front", "camera_rear"],
                DataType.JOINT_POSITIONS: ["arm", "gripper"],
            }
        }
        synced = recording.synchronize(
            frequency=30, cross_embodiment_union=cross_embodiment_union
        )

        assert isinstance(synced, SynchronizedRecording)
        assert synced.cross_embodiment_union == cross_embodiment_union

    def test_synchronize_with_invalid_data_types(self, recording: Recording):
        """Test synchronizing with invalid data types raises an error."""

        cross_embodiment_union = {
            recording.robot_id: {
                "INVALID_DATA_TYPE": ["some_item"],
            }
        }
        with pytest.raises(
            ValueError,
            match="Invalid data types provided. ",
        ):
            recording.synchronize(
                frequency=30, cross_embodiment_union=cross_embodiment_union
            )

    def test_synchronize_with_empty_data_types(self, recording: Recording):
        """Test synchronizing with empty data types list."""
        synced = recording.synchronize(frequency=30, cross_embodiment_union={})
        assert isinstance(synced, SynchronizedRecording)
        assert synced.cross_embodiment_union == {}

    def test_synchronize_with_none_data_types(self, recording: Recording):
        """Test synchronizing with None data types (should default to empty list)."""
        synced = recording.synchronize(frequency=30, cross_embodiment_union=None)

        assert isinstance(synced, SynchronizedRecording)
        assert synced.cross_embodiment_union is None

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

    @pytest.fixture
    def entry_url(self, recording: Recording, mocked_org_id: str) -> str:
        """URL of the recording's entry in its dataset."""
        return (
            f"{API_URL}/org/{mocked_org_id}/recording/{recording.dataset.id}"
            f"/{recording.id}"
        )

    @staticmethod
    def _entry_response(recording: Recording, metadata: RecordingMetadata) -> dict:
        return {
            "dataset_id": recording.dataset.id,
            "recording_id": recording.id,
            "metadata": metadata.model_dump(mode="json"),
        }

    @staticmethod
    def _metadata_with_tags(*names: str) -> RecordingMetadata:
        return RecordingMetadata(
            tags=[
                RecordingTag(name=name, source=RecordingTagSource.USER)
                for name in names
            ]
        )

    def test_tags_reads_the_metadata(self, recording: Recording):
        finding = QAFinding(
            reason=QAFailureReason.NEVER_MOVED, start_time=0, end_time=1
        )
        tags = [
            RecordingTag(
                name="NEVER_MOVED", source=RecordingTagSource.QA, findings=[finding]
            ),
            RecordingTag(name="flagged", source=RecordingTagSource.USER),
        ]
        recording.metadata = RecordingMetadata(tags=tags)

        assert recording.tags == tags
        recording.tags.clear()
        assert recording.metadata.tags == tags

    def test_add_tag_posts_the_normalized_name_and_reads_back_the_entry(
        self,
        recording: Recording,
        entry_url: str,
        mock_data_requests: requests_mock.Mocker,
    ):
        tagged = self._metadata_with_tags("bad-grip")
        tags_request = mock_data_requests.post(
            f"{entry_url}/tags", json=self._entry_response(recording, tagged)
        )

        recording.add_tag("  Bad   Grip ")

        assert tags_request.last_request.json() == {"name": "bad-grip"}
        assert recording.tags == tagged.tags

    def test_remove_tag_sends_the_name_as_a_query_parameter(
        self,
        recording: Recording,
        entry_url: str,
        mock_data_requests: requests_mock.Mocker,
    ):
        recording.metadata = self._metadata_with_tags("left/right?50%")
        untagged = self._metadata_with_tags()
        tags_request = mock_data_requests.delete(
            f"{entry_url}/tags", json=self._entry_response(recording, untagged)
        )

        recording.remove_tag("Left/Right?50%")

        query = urlparse(tags_request.last_request.url).query
        assert query == "name=left%2Fright%3F50%25"
        assert recording.tags == []

    def test_add_tag_rejects_an_invalid_name_without_a_request(
        self,
        recording: Recording,
        entry_url: str,
        mock_data_requests: requests_mock.Mocker,
    ):
        tags_request = mock_data_requests.post(f"{entry_url}/tags")

        with pytest.raises(ValueError, match="comma"):
            recording.add_tag("bad,grip")

        assert not tags_request.called

    def test_add_tag_raises_when_the_cloud_refuses_it(
        self,
        recording: Recording,
        entry_url: str,
        mock_data_requests: requests_mock.Mocker,
    ):
        mock_data_requests.post(f"{entry_url}/tags", status_code=403)

        with pytest.raises(RuntimeError, match="403"):
            recording.add_tag("bad-grip")

        assert recording.tags == []

    def test_set_notes_sends_only_the_name_and_notes(
        self,
        recording: Recording,
        recording_model: RecordingModel,
        entry_url: str,
        mock_data_requests: requests_mock.Mocker,
        mocked_org_id: str,
    ):
        recording.metadata = self._metadata_with_tags("flagged")
        in_dataset = recording_model.model_copy(deep=True)
        in_dataset.metadata.name = "Renamed in dataset"
        mock_data_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/{recording.dataset.id}"
            f"/recording/{recording.id}",
            json=in_dataset.model_dump(mode="json"),
        )
        updated = in_dataset.metadata.model_copy(update={"notes": "Test notes"})
        metadata_request = mock_data_requests.put(
            f"{entry_url}/metadata", json=self._entry_response(recording, updated)
        )

        recording.set_notes("Test notes")

        assert metadata_request.last_request.json() == {
            "name": "Renamed in dataset",
            "notes": "Test notes",
        }
        assert recording.name == "Renamed in dataset"
        assert recording.metadata.notes == "Test notes"
        assert recording.tags == []
