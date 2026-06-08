from __future__ import annotations

import json
import struct

import numpy as np
from neuracore_types import DataType

from neuracore.core.streaming.data_stream import DataRecordingContext, RGBDataStream
from neuracore.data_daemon.communications_management.producer import (
    StreamPayload,
    StreamSession,
)


class _FakeCoordinator:
    """Records session registrations and enqueued payloads."""

    def __init__(self) -> None:
        self.register_calls: list[dict[str, object]] = []
        self.payloads: list[StreamPayload] = []

    def register_stream_session(self, **kwargs: object) -> StreamSession:
        self.register_calls.append(kwargs)
        return StreamSession(
            session_id=f"{kwargs['data_type'].value}:{kwargs['stream_name']}",
            data_type=kwargs["data_type"],  # type: ignore[arg-type]
            data_type_name=str(kwargs["stream_name"]),
            recording_id=str(kwargs["recording_id"]),
            robot_instance=int(kwargs["robot_instance"]),  # type: ignore[arg-type]
            robot_id=kwargs["robot_id"],  # type: ignore[arg-type]
            robot_name=kwargs["robot_name"],  # type: ignore[arg-type]
            dataset_id=kwargs["dataset_id"],  # type: ignore[arg-type]
            dataset_name=kwargs["dataset_name"],  # type: ignore[arg-type]
            trace_id="trace-1",
        )

    def enqueue_stream_payload(self, payload: StreamPayload) -> None:
        self.payloads.append(payload)


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


def test_rgb_stream_registers_video_session() -> None:
    coordinator = _FakeCoordinator()
    stream = RGBDataStream("front_camera", width=3840, height=2160)

    stream.start_recording(_context(), coordinator)

    assert len(coordinator.register_calls) == 1
    call = coordinator.register_calls[0]
    assert call["data_type"] == DataType.RGB_IMAGES
    assert call["stream_name"] == "front_camera"
    session = stream.get_stream_session()
    assert session is not None
    assert session.uses_video_transport()


def test_start_recording_replaces_stale_recording_session() -> None:
    coordinator = _FakeCoordinator()
    stream = RGBDataStream("front_camera", width=640, height=480)

    stream.start_recording(_context("rec-old"), coordinator)
    old_session = stream.get_stream_session()
    stream.start_recording(_context("rec-new"), coordinator)

    assert len(coordinator.register_calls) == 2
    assert coordinator.register_calls[0]["recording_id"] == "rec-old"
    assert coordinator.register_calls[1]["recording_id"] == "rec-new"
    current_context = stream.get_recording_context()
    assert current_context is not None
    assert current_context.recording_id == "rec-new"
    assert stream.get_stream_session() is not old_session


def test_rgb_stream_enqueues_frame_as_multipart_payload() -> None:
    coordinator = _FakeCoordinator()
    width, height = 4, 3
    stream = RGBDataStream("front_camera", width=width, height=height)
    stream.start_recording(_context(), coordinator)

    metadata = _DummyCameraData(timestamp=123.0)
    frame = np.arange(width * height * 3, dtype=np.uint8).reshape((height, width, 3))
    stream.log(metadata, frame)

    assert len(coordinator.payloads) == 1
    payload = coordinator.payloads[0]
    parts = payload.parts
    assert isinstance(parts, tuple)
    assert len(parts) == 3
    header, metadata_json, frame_view = parts

    expected_metadata = {
        "timestamp": 123.0,
        "width": width,
        "height": height,
        "frame_nbytes": frame.nbytes,
    }
    expected_metadata_json = json.dumps(expected_metadata).encode("utf-8")

    assert header == struct.pack("<I", len(expected_metadata_json))
    assert metadata_json == expected_metadata_json
    assert isinstance(frame_view, memoryview)
    assert len(frame_view) == frame.nbytes
    assert payload.total_bytes == len(header) + len(metadata_json) + frame.nbytes


def test_stream_mark_recording_stopped_clears_local_state() -> None:
    coordinator = _FakeCoordinator()
    stream = RGBDataStream("front_camera", width=640, height=480)
    stream.start_recording(_context(), coordinator)

    stream.mark_recording_stopped()

    assert stream.get_recording_context() is None
    assert stream.get_stream_session() is None
    assert stream.is_recording() is False
