"""Tests for data stream out-of-order timestamp warning."""

from unittest.mock import patch

import numpy as np
import pytest
from neuracore_types import CameraData, DataType, JointData

from neuracore.core.streaming.data_stream import (
    DataRecordingContext,
    JsonDataStream,
    VideoDataStream,
)


@pytest.fixture
def recording_context():
    """Create a DataRecordingContext for testing."""
    return DataRecordingContext(
        recording_id="test-recording-id",
        robot_id="test-robot-id",
        robot_name="test-robot",
        robot_instance=0,
        dataset_id="test-dataset-id",
        dataset_name="test-dataset",
    )


@pytest.fixture
def json_stream():
    """Create a JsonDataStream for testing."""
    return JsonDataStream(
        data_type=DataType.JOINT_POSITIONS,
        data_type_name="test_joints",
    )


@pytest.fixture
def video_stream():
    """Create a VideoDataStream for testing."""
    return VideoDataStream(
        data_type=DataType.RGB_IMAGES,
        camera_id="test_camera",
        width=640,
        height=480,
    )


class TestSetLatestData:
    """Tests for set_latest_data out-of-order timestamp detection."""

    def test_stores_data(self, json_stream):
        """set_latest_data stores the data as latest."""
        data = JointData(value=1.0, timestamp=1.0)
        json_stream.set_latest_data(data)
        assert json_stream.get_latest_data() is data

    def test_no_warning_on_increasing_timestamps(self, json_stream):
        """No warning when timestamps arrive in order."""
        with patch("neuracore.core.streaming.data_stream.logger") as mock_logger:
            json_stream.set_latest_data(JointData(value=1.0, timestamp=1.0))
            json_stream.set_latest_data(JointData(value=2.0, timestamp=2.0))
            json_stream.set_latest_data(JointData(value=3.0, timestamp=3.0))
            mock_logger.warning.assert_not_called()
            assert json_stream._max_timestamp == 3.0

    def test_warning_on_out_of_order_timestamp(self, json_stream):
        """Warning on older or equal timestamps, max unchanged."""
        with patch("neuracore.core.streaming.data_stream.logger") as mock_logger:
            json_stream.set_latest_data(JointData(value=1.0, timestamp=10.0))
            json_stream.set_latest_data(JointData(value=2.0, timestamp=5.0))
            json_stream.set_latest_data(JointData(value=3.0, timestamp=5.0))
            assert mock_logger.warning.call_count == 2
            assert json_stream._max_timestamp == 10.0


class TestTimestampResetOnRecording:
    """Tests that _max_timestamp resets across recording lifecycles."""

    def test_no_warning_after_recording_restart(self, json_stream, recording_context):
        """After restarting recording, old timestamps don't trigger warnings."""
        json_stream.set_latest_data(JointData(value=1.0, timestamp=100.0))

        with patch.object(json_stream, "_handle_ensure_producer"):
            json_stream.start_recording(recording_context)

        with patch("neuracore.core.streaming.data_stream.logger") as mock_logger:
            json_stream.set_latest_data(JointData(value=2.0, timestamp=1.0))
            mock_logger.warning.assert_not_called()


class TestLogDelegatesToSetLatestData:
    """Tests that log methods delegate to set_latest_data."""

    def test_json_stream_log(self, json_stream):
        """JsonDataStream.log delegates to set_latest_data."""
        data = JointData(value=1.0, timestamp=1.0)
        with patch.object(json_stream, "set_latest_data") as mock_set:
            json_stream.log(data)
            mock_set.assert_called_once_with(data)

    def test_video_stream_log(self, video_stream):
        """VideoDataStream.log delegates to set_latest_data."""
        metadata = CameraData(camera_id="test_camera", timestamp=1.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch.object(video_stream, "set_latest_data") as mock_set:
            video_stream.log(metadata, frame)
            mock_set.assert_called_once_with(metadata)
