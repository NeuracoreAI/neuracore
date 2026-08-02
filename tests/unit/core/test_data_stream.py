from __future__ import annotations

import numpy as np
import pytest
from neuracore_types import DataType, JointData

from neuracore.core.streaming.data_stream import (
    DataRecordingContext,
    JointDataStream,
    RGBDataStream,
)


class _DummyCameraData:
    def __init__(self, timestamp: float) -> None:
        self.timestamp = timestamp
        self.frame = None

    def model_dump(self, mode: str = "json", exclude: set[str] | None = None) -> dict:
        del mode
        payload = {
            "timestamp": self.timestamp,
            "frame": self.frame,
        }
        for key in exclude or set():
            payload.pop(key, None)
        return payload


def _context(recording_id: str = "rec-1") -> DataRecordingContext:
    return DataRecordingContext(
        recording_id=recording_id,
        robot_id="robot-1",
        robot_name="robot",
        robot_instance=0,
        dataset_id="dataset-1",
        dataset_name="dataset",
    )


def test_stream_tracks_recording_state_and_latest_sample() -> None:
    """A stream owns no transport — the daemon interface lives at the logging
    layer (RecordingContext), so a stream only tracks state and latest data."""
    width, height = 4, 3
    stream = RGBDataStream("front_camera", width=width, height=height)
    stream.start_recording(_context())

    assert stream.is_recording() is True
    assert stream.get_recording_context() is not None

    metadata = _DummyCameraData(timestamp=1.0)
    frame = np.arange(width * height * 3, dtype=np.uint8).reshape((height, width, 3))
    stream.log(metadata, frame)

    assert stream.get_latest_data() is metadata

    stream.stop_recording()

    assert stream.is_recording() is False
    assert stream.get_recording_context() is None


def test_video_stream_rejects_non_increasing_timestamp() -> None:
    stream = RGBDataStream("front_camera", width=4, height=3)
    stream.start_recording(_context())
    frame = np.zeros((3, 4, 3), dtype=np.uint8)

    stream.log(_DummyCameraData(timestamp=1.0), frame)
    stream.log(_DummyCameraData(timestamp=2.0), frame)

    with pytest.raises(ValueError, match="Non-monotonic timestamp"):
        stream.log(_DummyCameraData(timestamp=2.0), frame)
    with pytest.raises(ValueError, match="Non-monotonic timestamp"):
        stream.log(_DummyCameraData(timestamp=1.5), frame)


def test_joint_stream_record_scalar_rejects_non_increasing_timestamp() -> None:
    stream = JointDataStream(data_type=DataType.JOINT_POSITIONS, data_type_name="j1")
    stream.start_recording(_context())

    stream.record_scalar(1.0, 0.5)
    stream.record_scalar(2.0, 0.6)

    with pytest.raises(ValueError, match="Non-monotonic timestamp"):
        stream.record_scalar(2.0, 0.7)


def test_joint_stream_log_rejects_non_increasing_timestamp() -> None:
    stream = JointDataStream(data_type=DataType.JOINT_POSITIONS, data_type_name="j1")
    stream.start_recording(_context())

    stream.log(JointData(timestamp=1.0, value=0.5))

    with pytest.raises(ValueError, match="Non-monotonic timestamp"):
        stream.log(JointData(timestamp=0.9, value=0.6))


def test_joint_stream_materialises_deferred_scalar_on_demand() -> None:
    stream = JointDataStream(data_type=DataType.JOINT_POSITIONS, data_type_name="j1")
    stream.start_recording(_context())

    stream.record_scalar(1.0, 0.5)

    latest = stream.get_latest_data()
    assert isinstance(latest, JointData)
    assert (latest.timestamp, latest.value) == (1.0, 0.5)


def test_monotonic_check_is_per_stream() -> None:
    """Each stream keeps its own timeline — sharing a timestamp is fine."""
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    front = RGBDataStream("front_camera", width=4, height=3)
    wrist = RGBDataStream("wrist_camera", width=4, height=3)
    front.start_recording(_context())
    wrist.start_recording(_context())

    front.log(_DummyCameraData(timestamp=1.0), frame)
    wrist.log(_DummyCameraData(timestamp=1.0), frame)
    front.log(_DummyCameraData(timestamp=2.0), frame)
    wrist.log(_DummyCameraData(timestamp=2.0), frame)


def test_monotonic_check_skipped_when_not_recording() -> None:
    """Outside a recording there is no timeline to enforce."""
    stream = RGBDataStream("front_camera", width=4, height=3)
    frame = np.zeros((3, 4, 3), dtype=np.uint8)

    stream.log(_DummyCameraData(timestamp=5.0), frame)
    stream.log(_DummyCameraData(timestamp=1.0), frame)


def test_start_recording_resets_monotonic_timeline() -> None:
    """A new recording is an independent timeline that may restart lower."""
    stream = RGBDataStream("front_camera", width=4, height=3)
    frame = np.zeros((3, 4, 3), dtype=np.uint8)

    stream.start_recording(_context("rec-1"))
    stream.log(_DummyCameraData(timestamp=5.0), frame)
    stream.stop_recording()

    stream.start_recording(_context("rec-2"))
    stream.log(_DummyCameraData(timestamp=1.0), frame)
