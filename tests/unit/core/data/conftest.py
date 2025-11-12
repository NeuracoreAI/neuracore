"""Shared test fixtures and utilities for dataset tests."""

import io
import re
from fractions import Fraction

import av
import numpy as np
import pytest
from neuracore_types import CameraData, DataType, JointData, SyncedData, SyncPoint

import neuracore as nc
from neuracore.core.const import API_URL

# Constants for video creation
CODEC = "h264"
PIX_FMT = "yuv420p"
FREQ = 30
PTS_FRACT = 90000  # Common timebase for h264


@pytest.fixture
def create_test_video_fn():
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
def mock_wget_download(monkeypatch, create_test_video_fn):
    """Mock wget.download calls to return fake video file."""
    import wget

    def mock_download(url, out=None, bar=None):
        """Mock wget.download to create a fake video file."""
        # Create fake video data
        video_data = create_test_video_fn(num_frames=10)

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
    """Basic dataset dictionary for testing."""
    return {
        "id": "dataset123",
        "org_id": mocked_org_id,
        "name": "test_dataset",
        "size_bytes": 1024,
        "tags": ["test", "robotics"],
        "is_shared": False,
        "data_types": [DataType.RGB_IMAGE, DataType.JOINT_POSITIONS],
    }


@pytest.fixture
def recordings_list():
    """List of recording dictionaries for testing."""
    return [
        {
            "id": "rec1",
            "name": "recording1",
            "robot_id": "robot1",
            "instance": 1,
            "total_bytes": 512,
            "created_at": "2023-01-01T00:00:00Z",
        },
        {
            "id": "rec2",
            "robot_id": "robot2",
            "instance": 1,
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

    return SyncedData(
        frames=[frame1, frame2], start_time=0.0, end_time=1.0, robot_id="robot1"
    )


@pytest.fixture
def synced_data_multiple_frames():
    """Create synced data fixture with more frames for testing."""
    frames = []
    for i in range(5):
        camera = CameraData(
            timestamp=float(i),
            frame_idx=i,
            extrinsics=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            intrinsics=[[500, 0, 112], [0, 500, 112], [0, 0, 1]],
        )

        frame = SyncPoint(
            timestamp=float(i),
            joint_positions=JointData(
                timestamp=float(i), values={"joint1": 0.5 + i * 0.1}
            ),
            joint_target_positions=JointData(
                timestamp=float(i), values={"joint1": 1.0 + i * 0.1}
            ),
            rgb_images={"cam1": camera},
            depth_images=None,
        )
        frames.append(frame)

    return SyncedData(frames=frames, start_time=0.0, end_time=4.0, robot_id="robot1")


@pytest.fixture
def mock_auth_requests(
    temp_config_dir,
    mock_auth_requests,
    dataset_dict,
    recordings_list,
    synced_data,
    mocked_org_id,
    create_test_video_fn,
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

    # Mock sync dataset
    mock_auth_requests.post(
        re.compile(f"{API_URL}/org/{mocked_org_id}/synchronize/synchronize-dataset"),
        json={
            "id": "synced_dataset_123",
            "parent_id": dataset_dict["id"],
            "freq": 30,
            "name": "synced_test_dataset",
            "created_at": 0.0,
            "modified_at": 0.0,
            "num_demonstrations": len(recordings_list),
            "num_processed_demonstrations": len(recordings_list),
        },
        status_code=200,
    )

    video_data = create_test_video_fn(num_frames=10)

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
