from __future__ import annotations

import numpy as np
import pytest
from neuracore_types import DataType, JointData

from neuracore.core.streaming.data_stream import JointDataStream, RGBDataStream

RECORDING = 1_000
NEXT_RECORDING = 2_000


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


def test_stream_tracks_latest_sample() -> None:
    """A stream owns no transport and no recording identity, so it tracks only
    its place in the open recording and the latest sample."""
    width, height = 4, 3
    stream = RGBDataStream("front_camera", width=width, height=height)

    metadata = _DummyCameraData(timestamp=1.0)
    frame = np.arange(width * height * 3, dtype=np.uint8).reshape((height, width, 3))
    stream.log(metadata, frame, recording_epoch=RECORDING)

    assert stream.get_latest_data() is metadata


def test_video_stream_rejects_non_increasing_timestamp() -> None:
    stream = RGBDataStream("front_camera", width=4, height=3)
    frame = np.zeros((3, 4, 3), dtype=np.uint8)

    stream.log(_DummyCameraData(timestamp=1.0), frame, recording_epoch=RECORDING)
    stream.log(_DummyCameraData(timestamp=2.0), frame, recording_epoch=RECORDING)

    with pytest.raises(ValueError, match="Non-monotonic timestamp"):
        stream.log(_DummyCameraData(timestamp=2.0), frame, recording_epoch=RECORDING)
    with pytest.raises(ValueError, match="Non-monotonic timestamp"):
        stream.log(_DummyCameraData(timestamp=1.5), frame, recording_epoch=RECORDING)


def test_joint_stream_record_scalar_rejects_non_increasing_timestamp() -> None:
    stream = JointDataStream(data_type=DataType.JOINT_POSITIONS, data_type_name="j1")

    stream.record_scalar(1.0, 0.5, RECORDING)
    stream.record_scalar(2.0, 0.6, RECORDING)

    with pytest.raises(ValueError, match="Non-monotonic timestamp"):
        stream.record_scalar(2.0, 0.7, RECORDING)


def test_joint_stream_log_rejects_non_increasing_timestamp() -> None:
    stream = JointDataStream(data_type=DataType.JOINT_POSITIONS, data_type_name="j1")

    stream.log(JointData(timestamp=1.0, value=0.5), recording_epoch=RECORDING)

    with pytest.raises(ValueError, match="Non-monotonic timestamp"):
        stream.log(JointData(timestamp=0.9, value=0.6), recording_epoch=RECORDING)


def test_joint_stream_materialises_deferred_scalar_on_demand() -> None:
    stream = JointDataStream(data_type=DataType.JOINT_POSITIONS, data_type_name="j1")

    stream.record_scalar(1.0, 0.5, RECORDING)

    latest = stream.get_latest_data()
    assert isinstance(latest, JointData)
    assert (latest.timestamp, latest.value) == (1.0, 0.5)


def test_monotonic_check_is_per_stream() -> None:
    """Each stream keeps its own timeline — sharing a timestamp is fine."""
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    front = RGBDataStream("front_camera", width=4, height=3)
    wrist = RGBDataStream("wrist_camera", width=4, height=3)

    front.log(_DummyCameraData(timestamp=1.0), frame, recording_epoch=RECORDING)
    wrist.log(_DummyCameraData(timestamp=1.0), frame, recording_epoch=RECORDING)
    front.log(_DummyCameraData(timestamp=2.0), frame, recording_epoch=RECORDING)
    wrist.log(_DummyCameraData(timestamp=2.0), frame, recording_epoch=RECORDING)


def test_monotonic_check_skipped_when_not_recording() -> None:
    """Outside a recording there is no timeline to enforce: nothing logged
    there reaches a trace, and a producer free-running between recordings must
    not be failed for it."""
    stream = RGBDataStream("front_camera", width=4, height=3)
    frame = np.zeros((3, 4, 3), dtype=np.uint8)

    stream.log(_DummyCameraData(timestamp=5.0), frame, recording_epoch=None)
    stream.log(_DummyCameraData(timestamp=1.0), frame, recording_epoch=None)


def test_a_new_recording_may_restart_the_timeline_lower() -> None:
    """The epoch is what tells one recording's timeline from the next, so a
    recording that legitimately starts below where the last one ended — an
    importer replaying episodes newest-first — is not a violation."""
    stream = RGBDataStream("front_camera", width=4, height=3)
    frame = np.zeros((3, 4, 3), dtype=np.uint8)

    stream.log(_DummyCameraData(timestamp=5.0), frame, recording_epoch=RECORDING)
    stream.log(_DummyCameraData(timestamp=1.0), frame, recording_epoch=NEXT_RECORDING)

    with pytest.raises(ValueError, match="Non-monotonic timestamp"):
        stream.log(
            _DummyCameraData(timestamp=0.5), frame, recording_epoch=NEXT_RECORDING
        )


def test_a_stale_timeline_does_not_survive_a_gap_outside_a_recording() -> None:
    """The reviewer's case on #994: a sample logged outside a recording must not
    be able to fail the first sample of the recording that follows it."""
    stream = RGBDataStream("front_camera", width=4, height=3)
    frame = np.zeros((3, 4, 3), dtype=np.uint8)

    stream.log(_DummyCameraData(timestamp=5.0), frame, recording_epoch=None)
    stream.log(_DummyCameraData(timestamp=1.0), frame, recording_epoch=RECORDING)
