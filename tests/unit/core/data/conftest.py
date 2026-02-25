"""Shared test fixtures and utilities for dataset tests."""

import copy
import io
import re
from collections.abc import Generator
from fractions import Fraction

import av
import numpy as np
import pytest
import requests_mock
from neuracore_types import (
    Dataset,
    DataType,
    JointData,
    Recording,
    RGBCameraData,
    SynchronizationProgress,
    SynchronizedDataset,
    SynchronizedEpisode,
    SynchronizedPoint,
)

from neuracore.core.const import API_URL

# Constants for video creation
CODEC = "h264"
PIX_FMT = "yuv420p"
FREQ = 30
PTS_FRACT = 90000  # Common timebase for h264

MOCKED_ORG_ID = "test-org-id"
TEST_ROBOT_ID_1 = "20a621b7-2f9b-4699-a08e-7d080488a5a3"
TEST_ROBOT_ID_2 = "30b731c8-3f9c-5799-b19e-8d190599b6b4"


@pytest.fixture
def mocked_org_id():
    return MOCKED_ORG_ID


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
def dataset_model(mocked_org_id):
    """Basic dataset dictionary for testing."""
    return Dataset(
        id="dataset123",
        name="test_dataset",
        created_at=0.0,
        modified_at=0.0,
        description="A test dataset",
        size_bytes=1024,
        tags=["test", "robotics"],
        is_shared=False,
        num_demonstrations=20,
    )


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
        "description": "A test dataset",
        "data_types": {DataType.RGB_IMAGES: 1, DataType.JOINT_POSITIONS: 1},
    }


@pytest.fixture
def recordings_list():
    """List of recording dictionaries for testing."""
    return [
        Recording(
            id="rec1",
            name="recording1",
            robot_id=TEST_ROBOT_ID_1,
            instance=1,
            org_id="test-org-id",
            start_time=0.0,
            end_time=10.0,
            total_bytes=512,
            created_at="2023-01-01T00:00:00Z",
        ).model_dump(mode="json"),
        Recording(
            id="rec2",
            name="recording2",
            robot_id=TEST_ROBOT_ID_2,
            instance=1,
            org_id="test-org-id",
            start_time=0.0,
            end_time=8.0,
            total_bytes=512,
            created_at="2023-01-02T00:00:00Z",
        ).model_dump(mode="json"),
    ]


@pytest.fixture
def synced_data():
    """Create synced data fixture."""
    # Create camera data with frame indices
    camera1 = RGBCameraData(
        timestamp=1000.0,
        frame_idx=0,
        extrinsics=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        intrinsics=np.array([[500, 0, 112], [0, 500, 112], [0, 0, 1]]),
    )

    camera2 = RGBCameraData(
        timestamp=1000.0,
        frame_idx=0,
        extrinsics=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        intrinsics=np.array([[500, 0, 112], [0, 500, 112], [0, 0, 1]]),
    )

    # Create sync points
    frame1 = SynchronizedPoint(
        timestamp=0.0,
        data={
            DataType.JOINT_POSITIONS: {"joint1": JointData(timestamp=0.0, value=0.5)},
            DataType.JOINT_TARGET_POSITIONS: {
                "joint1": JointData(timestamp=1000.0, value=1.0)
            },
            DataType.RGB_IMAGES: {"cam1": camera1},
            DataType.DEPTH_IMAGES: {"cam2": camera2},
        },
    )

    camera1 = copy.deepcopy(camera1)
    camera2 = copy.deepcopy(camera2)
    camera1.frame_idx = 1
    camera2.frame_idx = 1

    frame2 = SynchronizedPoint(
        timestamp=1.0,
        data={
            DataType.JOINT_POSITIONS: {"joint1": JointData(timestamp=0.0, value=0.5)},
            DataType.JOINT_TARGET_POSITIONS: {
                "joint1": JointData(timestamp=1000.0, value=1.0)
            },
            DataType.RGB_IMAGES: {"cam1": camera1},
            DataType.DEPTH_IMAGES: {"cam2": camera2},
        },
    )

    return SynchronizedEpisode(
        observations=[frame1, frame2], start_time=0.0, end_time=1.0, robot_id="robot1"
    )


@pytest.fixture
def synced_data_multiple_frames():
    """Create synced data fixture with more frames for testing."""
    frames = []
    for i in range(5):
        camera = RGBCameraData(
            timestamp=float(i),
            frame_idx=i,
            extrinsics=np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            ),
            intrinsics=np.array([[500, 0, 112], [0, 500, 112], [0, 0, 1]]),
        )

        frame = SynchronizedPoint(
            timestamp=float(i),
            data={
                DataType.JOINT_POSITIONS: {
                    "joint1": JointData(timestamp=0.0, value=0.5 + i * 0.1)
                },
                DataType.JOINT_TARGET_POSITIONS: {
                    "joint1": JointData(timestamp=1000.0, value=1.0 + i * 0.1)
                },
                DataType.RGB_IMAGES: {"cam1": camera},
            },
        )
        frames.append(frame)

    return SynchronizedEpisode(
        observations=frames, start_time=0.0, end_time=4.0, robot_id="robot1"
    )


@pytest.fixture
def mock_data_requests(
    mock_auth_requests: requests_mock.Mocker,
    dataset_model,
    recordings_list,
    synced_data,
    mocked_org_id,
    create_test_video_fn,
) -> Generator[requests_mock.Mocker, None, None]:
    """Set up mocks for Dataset API endpoints."""
    # Mock datasets endpoint
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets",
        json=[dataset_model.model_dump(mode="json")],
        status_code=200,
    )

    # Mock shared datasets endpoint
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/shared", json=[], status_code=200
    )

    mock_auth_requests.get(
        re.compile(f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name"),
        json=dataset_model.model_dump(mode="json"),
        status_code=200,
    )
    mock_auth_requests.get(
        re.compile(f"{API_URL}/org/{mocked_org_id}/datasets/dataset123$"),
        json=dataset_model.model_dump(mode="json"),
        status_code=200,
    )
    mock_auth_requests.get(
        f"{API_URL}/org/{mocked_org_id}/datasets/{dataset_model.id}/robot_ids",
        json=[recording["robot_id"] for recording in recordings_list],
        status_code=200,
    )

    # Mock dataset creation endpoint
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/datasets",
        json=dataset_model.model_dump(mode="json"),
        status_code=200,
    )

    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/recording/by-dataset/{dataset_model.id}",
        additional_matcher=lambda request: request.text in (None, ""),
        json={"data": recordings_list, "total": len(recordings_list)},
        status_code=200,
    )

    # Mock sync endpoint
    mock_auth_requests.post(
        re.compile(f"{API_URL}/org/{mocked_org_id}/synchronize/synchronize-recording"),
        json=synced_data.model_dump(mode="json"),
        status_code=200,
    )
    mock_auth_requests.get(
        re.compile(
            f"{API_URL}/org/{mocked_org_id}/synchronize/synchronization-progress/synced_dataset_123"
        ),
        json=SynchronizationProgress(
            synchronized_dataset_id="synced_dataset_123",
            num_synchronized_demonstrations=len(recordings_list),
        ).model_dump(mode="json"),
        status_code=200,
    )

    synced_dataset = SynchronizedDataset(
        id="synced_dataset_123",
        parent_id=dataset_model.id,
        name="synced_test_dataset",
        created_at=0.0,
        modified_at=0.0,
        description="",
        num_demonstrations=len(recordings_list),
        total_duration_seconds=0.0,
        is_shared=False,
        metadata={},
        all_data_types={},
        common_data_types={},
        frequency=30.0,
        max_delay_s=0.1,
        allow_duplicates=True,
        trim_start_end=True,
    )

    # Mock sync dataset
    mock_auth_requests.post(
        re.compile(f"{API_URL}/org/{mocked_org_id}/synchronize/synchronize-dataset"),
        json=synced_dataset.model_dump(),
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
