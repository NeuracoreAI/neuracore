"""Tests for SynchronizedRecording class."""

import re
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
from neuracore_types import CameraData, DataType, JointData, SynchronizedPoint
from PIL import Image

from neuracore.core.const import API_URL
from neuracore.core.data.synced_recording import SynchronizedRecording


class TestSynchronizedRecording:
    """Tests for the SynchronizedRecording class."""

    @pytest.fixture
    def dataset_mock(self, dataset_dict, recordings_list, tmp_path):
        """Create a mock dataset object."""
        from neuracore.core.data.dataset import Dataset

        dataset = Dataset(**dataset_dict, recordings=recordings_list)
        dataset.cache_dir = tmp_path / "cache"
        dataset.cache_dir.mkdir(parents=True, exist_ok=True)
        return dataset

    @pytest.fixture
    def mock_synced_api(self, mock_auth_requests, synced_data, mocked_org_id):
        """Set up mocks for synchronization API endpoints."""
        # Mock sync endpoint
        mock_auth_requests.post(
            re.compile(
                f"{API_URL}/org/{mocked_org_id}/synchronize/synchronize-recording"
            ),
            json=synced_data.model_dump(mode="json"),
            status_code=200,
        )
        yield mock_auth_requests

    @pytest.fixture
    def synced_recording(
        self, dataset_mock, mock_auth_requests
    ) -> SynchronizedRecording:
        """Create a SynchronizedRecording instance for testing."""
        return SynchronizedRecording(
            dataset=dataset_mock,
            recording_id="rec1",
            robot_id="robot1",
            instance=1,
            start_time=0.0,
            end_time=2.0,
            frequency=30,
            robot_data_spec=None,
        )

    def test_init(self, synced_recording: SynchronizedRecording, dataset_mock):
        """Test SynchronizedRecording initialization."""
        assert synced_recording.dataset == dataset_mock
        assert synced_recording.id == "rec1"
        assert synced_recording.frequency == 30
        assert synced_recording.robot_id == "robot1"
        assert synced_recording.instance == 1
        assert synced_recording.robot_data_spec is None
        assert synced_recording._iter_idx == 0

    def test_init_with_data_types(self, dataset_mock, mock_auth_requests):
        """Test initialization with specific data types."""
        from neuracore_types import DataType

        robot_data_spec = {
            "robot1": {
                DataType.RGB_IMAGES: [],
                DataType.DEPTH_IMAGES: [],
            }
        }
        synced = SynchronizedRecording(
            dataset=dataset_mock,
            recording_id="rec1",
            robot_id="robot1",
            instance=1,
            start_time=0.0,
            end_time=2.0,
            frequency=30,
            robot_data_spec=robot_data_spec,
        )

        assert synced.robot_data_spec == robot_data_spec

    def test_get_synced_data(
        self, synced_recording: SynchronizedRecording, synced_data
    ):
        """Test that _get_synced_data correctly retrieves synchronized data."""
        result = synced_recording._episode_synced

        assert result.robot_id == synced_data.robot_id
        assert len(result.observations) == len(synced_data.observations)
        assert result.start_time == synced_data.start_time
        assert result.end_time == synced_data.end_time

    def test_len(self, synced_recording):
        """Test __len__ returns correct number of frames."""
        assert len(synced_recording) == 2

    def test_iter_reset(self, synced_recording):
        """Test that __iter__ resets the iteration index."""
        synced_recording._iter_idx = 5
        result = iter(synced_recording)

        assert result is synced_recording
        assert synced_recording._iter_idx == 0

    def test_getitem_single_index(
        self, synced_recording: SynchronizedRecording, mock_wget_download
    ):
        """Test accessing a single frame by index."""
        sync_point = synced_recording[0]

        assert isinstance(sync_point, SynchronizedPoint)
        assert DataType.JOINT_POSITIONS in sync_point.data
        assert sync_point.timestamp == 0.0
        joint_data = cast(
            JointData, list(sync_point[DataType.JOINT_POSITIONS].values())[0]
        )
        assert joint_data.value == 0.5

    def test_getitem_negative_index(
        self, synced_recording: SynchronizedRecording, mock_wget_download
    ):
        """Test accessing frames with negative indices."""
        sync_point = synced_recording[-1]

        assert isinstance(sync_point, SynchronizedPoint)
        assert sync_point.timestamp == 1.0

    def test_getitem_out_of_range(self, synced_recording):
        """Test that out of range index raises IndexError."""
        with pytest.raises(IndexError, match="Index out of range"):
            _ = synced_recording[10]

    def test_getitem_negative_out_of_range(self, synced_recording):
        """Test that negative out of range index raises IndexError."""
        with pytest.raises(IndexError, match="Index out of range"):
            _ = synced_recording[-10]

    def test_getitem_slice(
        self, synced_recording: SynchronizedRecording, mock_wget_download
    ):
        """Test slicing synchronized recording."""
        frames = synced_recording[0:2]

        assert isinstance(frames, list)
        assert len(frames) == 2
        assert all(isinstance(f, SynchronizedPoint) for f in frames)

    def test_getitem_slice_with_step(
        self,
        dataset_mock,
        mock_auth_requests,
        mock_wget_download,
        synced_data_multiple_frames,
    ):
        """Test slicing with step parameter."""
        # Mock the API to return more frames
        mock_auth_requests.post(
            re.compile(
                f"{API_URL}/org/{dataset_mock.org_id}/synchronize/synchronize-recording"
            ),
            json=synced_data_multiple_frames.model_dump(mode="json"),
            status_code=200,
        )

        synced = SynchronizedRecording(
            dataset=dataset_mock,
            recording_id="rec1",
            robot_id="robot1",
            instance=1,
            start_time=0.0,
            end_time=2.0,
            frequency=30,
            robot_data_spec=None,
        )

        frames = synced[0:5:2]

        assert len(frames) == 3
        assert frames[0].timestamp == 0.0
        assert frames[1].timestamp == 2.0
        assert frames[2].timestamp == 4.0

    def test_iteration(
        self, synced_recording: SynchronizedRecording, mock_wget_download
    ):
        """Test iterating through synchronized recording."""
        frames = list(synced_recording)

        assert len(frames) == 2
        assert all(isinstance(f, SynchronizedPoint) for f in frames)
        assert frames[0].timestamp == 0.0
        assert frames[1].timestamp == 1.0

    def test_iteration_multiple_times(
        self, synced_recording: SynchronizedRecording, mock_wget_download
    ):
        """Test that the recording can be iterated multiple times."""
        frames1 = list(synced_recording)
        frames2 = list(synced_recording)

        assert len(frames1) == len(frames2)
        assert frames1[0].timestamp == frames2[0].timestamp

    def test_next_stop_iteration(self, synced_recording):
        """Test that __next__ raises StopIteration when exhausted."""
        iter(synced_recording)

        # Exhaust the iterator
        synced_recording._iter_idx = len(synced_recording._episode_synced.observations)

        with pytest.raises(StopIteration):
            next(synced_recording)

    def test_video_caching(
        self, synced_recording: SynchronizedRecording, mock_wget_download, tmp_path
    ):
        """Test that videos are cached correctly."""
        # First access should download and cache
        synced_recording[0]

        # Check that cache directory was created
        cache_path = synced_recording.cache_dir / f"{synced_recording.id}"
        assert cache_path.exists()

    def test_video_cache_reuse(
        self, dataset_mock, mock_auth_requests, mock_wget_download, tmp_path
    ):
        """Test that cached videos are reused on subsequent access."""
        # Create cache directory and add a fake cached frame
        cache_path = (
            dataset_mock.cache_dir
            / "rec1"
            / "30Hz"
            / DataType.RGB_IMAGES.value
            / "cam1"
        )
        cache_path.mkdir(parents=True, exist_ok=True)

        # Create a fake cached image
        fake_image = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 128)
        fake_image.save(cache_path / "0.png")

        synced = SynchronizedRecording(
            dataset=dataset_mock,
            recording_id="rec1",
            robot_id="robot1",
            instance=1,
            start_time=0.0,
            end_time=2.0,
            frequency=30,
            robot_data_spec=None,
        )

        sync_point = cast(SynchronizedPoint, synced[0])

        # Should have loaded from cache
        assert DataType.RGB_IMAGES in sync_point.data
        assert "cam1" in sync_point[DataType.RGB_IMAGES]

    def test_prefetch_videos_skip_if_cached(
        self, dataset_mock, mock_auth_requests, mock_wget_download
    ):
        """Test that prefetch_videos parameter triggers video download on init."""
        synced = SynchronizedRecording(
            dataset=dataset_mock,
            recording_id="rec1",
            robot_id="robot1",
            instance=1,
            start_time=0.0,
            end_time=2.0,
            frequency=30,
            robot_data_spec=None,
            prefetch_videos=True,
        )

        # Cache directory should exist after prefetch
        cache_path = synced.cache_dir / f"{synced.id}"
        assert cache_path.exists()

        # Mock wget to track if it's called
        with patch("wget.download") as mock_download:
            SynchronizedRecording(
                dataset=dataset_mock,
                recording_id="rec1",
                robot_id="robot1",
                instance=1,
                start_time=0.0,
                end_time=2.0,
                frequency=30,
                robot_data_spec=None,
                prefetch_videos=True,
            )

            # wget.download should not be called since cache exists
            mock_download.assert_not_called()

    def test_depth_image_processing(
        self, synced_recording: SynchronizedRecording, mock_wget_download
    ):
        """Test that depth images are processed correctly."""
        sync_point = cast(SynchronizedPoint, synced_recording[0])

        for cam_id, cam_data in sync_point[DataType.DEPTH_IMAGES].items():
            cam_data = cast(CameraData, cam_data)
            assert cam_data.frame is not None
            assert isinstance(cam_data.frame, Image.Image)

    def test_camera_data_copy_independence(
        self, synced_recording: SynchronizedRecording, mock_wget_download
    ):
        """Test that returned sync points are independent copies."""
        sync_point1 = cast(SynchronizedPoint, synced_recording[0])
        sync_point2 = cast(SynchronizedPoint, synced_recording[0])

        # Should be different objects
        assert sync_point1 is not sync_point2

        # Modifying one shouldn't affect the other
        jp1 = cast(JointData, list(sync_point1[DataType.JOINT_POSITIONS].values())[0])
        original_value = jp1.value
        jp1.value = 999.0

        jp2 = cast(JointData, list(sync_point2[DataType.JOINT_POSITIONS].values())[0])
        assert jp2.value == original_value

    def test_cache_manager_initialization(self, synced_recording):
        """Test that cache manager is initialized correctly."""
        assert synced_recording.cache_manager is not None
        assert hasattr(synced_recording.cache_manager, "ensure_space_available")

    def test_suppress_wget_progress(self, synced_recording):
        """Test that wget progress is suppressed by default."""
        assert synced_recording._suppress_wget_progress is True

    def test_different_frequencies_different_cache(
        self, dataset_mock, mock_auth_requests, mock_wget_download
    ):
        """Test that different frequencies use different cache directories."""
        synced_30 = SynchronizedRecording(
            dataset=dataset_mock,
            recording_id="rec1",
            robot_id="robot1",
            instance=1,
            start_time=0.0,
            end_time=2.0,
            frequency=30,
            robot_data_spec=None,
        )

        synced_60 = SynchronizedRecording(
            dataset=dataset_mock,
            recording_id="rec1",
            robot_id="robot1",
            instance=1,
            start_time=0.0,
            end_time=2.0,
            frequency=60,
            robot_data_spec=None,
        )

        # Trigger cache creation
        _ = synced_30[0]
        _ = synced_60[0]

        cache_30 = dataset_mock.cache_dir / "rec1"
        cache_60 = dataset_mock.cache_dir / "rec1"

        assert cache_30.exists()
        assert cache_60.exists()
