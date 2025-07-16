import io
import pathlib
import re
import tempfile
from fractions import Fraction

import av
import numpy as np
import pytest

import neuracore as nc
from neuracore.core.const import API_URL
from neuracore.core.data.dataset import Dataset
from neuracore.core.exceptions import DatasetError
from neuracore.core.nc_types import CameraData, JointData, SyncedData, SyncPoint
from neuracore.core.utils.video_url_streamer import VideoStreamer

# Constants for video creation
CODEC = "h264"
PIX_FMT = "yuv420p"
FREQ = 30
PTS_FRACT = 90000  # Common timebase for h264


@pytest.fixture
def temp_config_dir(monkeypatch):
    """Fixture to create a temporary config directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock home directory for testing
        monkeypatch.setattr("pathlib.Path.home", lambda: pathlib.Path(tmpdir))
        yield tmpdir


@pytest.fixture
def create_test_video():
    """Create a test video in memory and return the bytes."""

    def _create_video(num_frames=10, width=224, height=224):
        # Create in-memory file-like object
        output = io.BytesIO()

        container = av.open(output, mode="w", format="mp4")
        stream = container.add_stream(CODEC)
        stream.width = width
        stream.height = height
        stream.pix_fmt = PIX_FMT
        stream.codec_context.options = {"qp": "0", "preset": "ultrafast"}
        stream.time_base = Fraction(1, PTS_FRACT)

        relative_time = 0
        for i in range(num_frames):
            # Create a frame with frame number encoded in pixels
            img = np.ones((height, width, 3), dtype=np.uint8) * (i % 255)
            # Add frame number as a pattern in the center
            img[
                height // 2 - 20 : height // 2 + 20, width // 2 - 20 : width // 2 + 20
            ] = (i % 255)

            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame = frame.reformat(format=PIX_FMT)
            pts = int(relative_time * PTS_FRACT)
            frame.pts = pts

            for packet in stream.encode(frame):
                container.mux(packet)
            relative_time += 1.0 / FREQ

        # Flush the stream
        for packet in stream.encode(None):
            container.mux(packet)

        container.close()

        # Get the video data
        video_data = output.getvalue()
        output.close()

        return video_data

    return _create_video


@pytest.fixture
def mock_wget_download(monkeypatch, create_test_video):
    """Mock wget.download calls to return fake video file."""
    import wget

    def mock_download(url, out=None, bar=None):
        """Mock wget.download to create a fake video file."""
        # Create fake video data
        video_data = create_test_video(num_frames=10)

        # Determine output filename
        if out:
            filename = out
        else:
            # Extract filename from URL or use default
            filename = url.split("/")[-1] if "/" in url else "downloaded_video.mp4"

        # Write fake video data to file
        with open(filename, "wb") as f:
            f.write(video_data)

        return filename

    monkeypatch.setattr(wget, "download", mock_download)
    yield


@pytest.fixture
def dataset_dict(mocked_org_id):
    return {
        "id": "dataset123",
        "org_id": mocked_org_id,
        "name": "test_dataset",
        "size_bytes": 1024,
        "tags": ["test", "robotics"],
        "is_shared": False,
    }


@pytest.fixture
def recordings_list():
    return [
        {
            "id": "rec1",
            "name": "recording1",
            "total_bytes": 512,
            "created_at": "2023-01-01T00:00:00Z",
        },
        {
            "id": "rec2",
            "name": "recording2",
            "total_bytes": 512,
            "created_at": "2023-01-02T00:00:00Z",
        },
    ]


@pytest.fixture
def synced_data():
    """Create synced data fixture."""
    # Create camera data with frame indices
    camera1 = CameraData(
        timestamp=1000.0,
        frame_idx=0,
        extrinsics=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        intrinsics=[[500, 0, 112], [0, 500, 112], [0, 0, 1]],
    )

    camera2 = CameraData(
        timestamp=1000.0,
        frame_idx=0,
        extrinsics=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        intrinsics=[[500, 0, 112], [0, 500, 112], [0, 0, 1]],
    )

    # Create sync points
    frame1 = SyncPoint(
        timestamp=0.0,
        joint_positions=JointData(timestamp=0.0, values={"joint1": 0.5}),
        joint_target_positions=JointData(timestamp=1000.0, values={"joint1": 1.0}),
        rgb_images={"cam1": camera1},
        depth_images={"cam2": camera2},
    )

    frame2 = SyncPoint(
        timestamp=1.0,
        joint_positions=JointData(timestamp=1.0, values={"joint1": 0.7}),
        joint_target_positions=JointData(timestamp=1.0, values={"joint1": 1.2}),
        rgb_images={"cam1": CameraData(timestamp=1.0, frame_idx=1)},
        depth_images={"cam2": CameraData(timestamp=1.0, frame_idx=1)},
    )

    return SyncedData(frames=[frame1, frame2], start_time=0.0, end_time=1.0)


@pytest.fixture
def mock_dataset_api(
    temp_config_dir,
    mock_auth_requests,
    dataset_dict,
    recordings_list,
    synced_data,
    mocked_org_id,
):
    """Set up mocks for Dataset API endpoints."""
    nc.login("test_api_key")

    # Mock datasets endpoint
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets", json=[dataset_dict], status_code=200
    )

    # Mock shared datasets endpoint
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/shared", json=[], status_code=200
    )

    mock_auth_requests.get(
        re.compile(f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name"),
        json=dataset_dict,
        status_code=200,
    )

    # Mock dataset creation endpoint
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/datasets", json=dataset_dict, status_code=200
    )

    # Mock recordings endpoint
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_dict['id']}/recordings",
        json={"recordings": recordings_list},
        status_code=200,
    )

    # Mock sync endpoint
    mock_auth_requests.post(
        re.compile(f"{API_URL}/org/{mocked_org_id}/synchronize/synchronize-recording"),
        json=synced_data.model_dump(mode="json"),
        status_code=200,
    )

    yield mock_auth_requests


@pytest.fixture
def mock_video_api(mock_auth_requests, create_test_video, mocked_org_id):
    """Set up mocks for video streaming API endpoints."""
    # Create test video data
    video_data = create_test_video(num_frames=10)

    # Mock video URL endpoint
    mock_auth_requests.get(
        re.compile(f"{API_URL}/org/{mocked_org_id}/recording/.*/download_url"),
        json={"url": "https://example.com/test-video.mp4"},
        status_code=200,
    )

    # Define a custom response handler for the video content
    def video_content_callback(request, context):
        context.status_code = 200
        context.headers["Content-Type"] = "video/mp4"
        return video_data

    # Mock the actual video content endpoint
    mock_auth_requests.get(
        "https://example.com/test-video.mp4", content=video_content_callback
    )

    yield mock_auth_requests


class TestVideoStreamer:
    """Tests for the VideoStreamer class."""

    def test_video_streaming(self, mock_video_api, create_test_video):
        """Test streaming a video using mocked API response."""
        with VideoStreamer("https://example.com/test-video.mp4") as streamer:
            frames = list(streamer)

            # Should receive frames
            assert len(frames) > 0

            # Each frame should be a numpy array with correct shape
            for frame in frames:
                assert isinstance(frame, np.ndarray)
                assert frame.shape == (224, 224, 3)  # Default size from fixture


class TestDataset:

    def test_init_with_dict(self, mock_auth_requests, mock_dataset_api, dataset_dict):
        """Test initializing a Dataset with a dictionary."""
        dataset = Dataset(**dataset_dict)

        assert dataset.id == "dataset123"
        assert dataset.name == "test_dataset"
        assert dataset.size_bytes == 1024
        assert dataset.tags == ["test", "robotics"]
        assert dataset.is_shared is False
        assert len(dataset) == 2

    def test_init_with_recordings(self, dataset_dict, recordings_list):
        """Test initializing a Dataset with provided recordings."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        assert dataset.id == "dataset123"
        assert dataset.name == "test_dataset"
        assert len(dataset) == 2
        assert len(dataset.recordings) == 2
        assert dataset.recordings[0]["id"] == "rec1"
        assert dataset.recordings[1]["id"] == "rec2"

    def test_get_existing_dataset(self, mock_auth_requests, mock_dataset_api):
        """Test getting an existing dataset by name."""
        dataset = Dataset.get_by_name("test_dataset")

        assert dataset.id == "dataset123"
        assert dataset.name == "test_dataset"

    def test_get_nonexistent_dataset(
        self, mock_auth_requests, mock_dataset_api, mocked_org_id
    ):
        """Test getting a non-existent dataset raises an error."""
        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
            json={},
            status_code=404,
        )
        with pytest.raises(DatasetError, match="Dataset 'nonexistent' not found"):
            Dataset.get_by_name("nonexistent")

    def test_create_dataset(self, mock_auth_requests, mock_dataset_api):
        """Test creating a new dataset."""
        dataset = Dataset.create(
            "test_dataset", description="Test description", tags=["test"], shared=False
        )

        assert dataset.id == "dataset123"
        assert dataset.name == "test_dataset"

    def test_len(self, dataset_dict, recordings_list):
        """Test the __len__ method."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)
        assert len(dataset) == 2

    def test_getitem_slice(self, dataset_dict, recordings_list):
        """Test getting a slice of the dataset."""
        dataset = Dataset(**dataset_dict, recordings=recordings_list)

        result = dataset[0:1]

        assert isinstance(result, Dataset)
        assert len(result) == 1
        assert result.recordings[0]["id"] == "rec1"

    def test_dataset_iteration(
        self,
        mock_auth_requests,
        mock_dataset_api,
        mock_video_api,
        dataset_dict,
        recordings_list,
        mock_wget_download,
    ):
        """Test iterating through a dataset of episodes."""
        # Create a dataset
        dataset = Dataset(**dataset_dict)

        # Make sure we can iterate through the dataset
        episode_count = 0
        frame_counts = []

        for episode in dataset:
            episode_count += 1
            # Count frames in this episode
            frames = list(episode.synchronize(frequency=30))
            frame_counts.append(len(frames))

        # Should have processed both episodes
        assert episode_count == 2
        # Each episode should have 2 frames
        assert frame_counts == [2, 2]

    def test_dataset_indexing(
        self,
        mock_auth_requests,
        mock_dataset_api,
        mock_video_api,
        dataset_dict,
        recordings_list,
    ):
        """Test accessing dataset episodes by index."""
        # Create a dataset
        dataset = Dataset(**dataset_dict)

        # Access first episode
        episode = dataset[0]
        frames = list(episode.synchronize(frequency=100))
        assert len(frames) == 2

        # Access second episode
        episode = dataset[1]
        frames = list(episode.synchronize(frequency=100))
        assert len(frames) == 2

    def test_nested_iteration(
        self,
        mock_auth_requests,
        mock_dataset_api,
        mock_video_api,
        dataset_dict,
        recordings_list,
    ):
        """Test nested iteration through dataset and episodes."""
        # Create a dataset
        dataset = Dataset(**dataset_dict)

        # Test double nested iteration
        episode_count = 0
        total_frames = 0

        for episode in dataset:
            episode_count += 1
            synced_ep = episode.synchronize(frequency=100)
            for frame in synced_ep:
                total_frames += 1
                # Check frame has expected properties
                assert hasattr(frame, "timestamp")
                assert hasattr(frame, "joint_positions")

                # Check if we have rgb images
                if frame.rgb_images:
                    for camera_id, camera_data in frame.rgb_images.items():
                        assert hasattr(camera_data, "frame_idx")

        # Should have processed 2 episodes with 2 frames each
        assert episode_count == 2
        assert total_frames == 4

    def test_get_dataset_by_name(self, mock_dataset_api, mocked_org_id):
        """Test getting a dataset by name."""
        dataset_name = "ghjdidnia-dd/X0551-Ker-Pieb87-846483-CNNMLPP"

        Dataset.create(
            dataset_name, description="Test description", tags=["test"], shared=False
        )
        dataset = Dataset.get_by_name(dataset_name)
        assert dataset is not None
