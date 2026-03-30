from __future__ import annotations

import json
import struct

import numpy as np

from neuracore.core.streaming.data_stream import (
    DataRecordingContext,
    RGBDataStream,
)
from neuracore.data_daemon.const import (
    DEFAULT_VIDEO_CHUNK_SIZE,
    DEFAULT_VIDEO_RING_BUFFER_SIZE,
    DEFAULT_VIDEO_SEND_QUEUE_MAXSIZE,
)


class _FakeProducerChannel:
    instances: list["_FakeProducerChannel"] = []

    def __init__(
        self,
        id: str | None = None,
        recording_id: str | None = None,
        chunk_size: int | None = None,
        send_queue_maxsize: int | None = None,
        **_: object,
    ) -> None:
        self.id = id
        self.recording_id = recording_id
        self.chunk_size = chunk_size
        self.send_queue_maxsize = send_queue_maxsize
        self.init_ring_buffer_size: int | None = None
        self.reopened_ring_buffer_sizes: list[int] = []
        self.send_data_parts_calls: list[dict[str, object]] = []
        self.trace_id = None
        _FakeProducerChannel.instances.append(self)

    def initialize_new_producer_channel(self, ring_buffer_size: int) -> None:
        self.init_ring_buffer_size = ring_buffer_size

    def set_recording_id(self, recording_id: str | None) -> None:
        self.recording_id = recording_id

    def start_producer_channel(self) -> None:
        return

    def open_ring_buffer(self, size: int) -> None:
        self.reopened_ring_buffer_sizes.append(size)

    def start_new_trace(self) -> None:
        self.trace_id = "trace-1"

    def cleanup_producer_channel(self) -> None:
        return

    def send_data_parts(self, **kwargs: object) -> None:
        self.send_data_parts_calls.append(kwargs)


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


def test_rgb_stream_uses_video_specific_producer_settings(monkeypatch) -> None:
    _FakeProducerChannel.instances.clear()
    monkeypatch.setattr(
        "neuracore.core.streaming.data_stream.ProducerChannel",
        _FakeProducerChannel,
    )

    stream = RGBDataStream("front_camera", width=3840, height=2160)
    stream.start_recording(_context())

    producer = _FakeProducerChannel.instances[0]
    assert producer.chunk_size == DEFAULT_VIDEO_CHUNK_SIZE
    assert producer.send_queue_maxsize == DEFAULT_VIDEO_SEND_QUEUE_MAXSIZE
    assert producer.init_ring_buffer_size == DEFAULT_VIDEO_RING_BUFFER_SIZE


def test_rgb_stream_sends_frame_as_multipart_payload(monkeypatch) -> None:
    _FakeProducerChannel.instances.clear()
    monkeypatch.setattr(
        "neuracore.core.streaming.data_stream.ProducerChannel",
        _FakeProducerChannel,
    )

    width, height = 4, 3
    stream = RGBDataStream("front_camera", width=width, height=height)
    stream.start_recording(_context())

    metadata = _DummyCameraData(timestamp=123.0)
    frame = np.arange(width * height * 3, dtype=np.uint8).reshape((height, width, 3))
    stream.log(metadata, frame)

    producer = _FakeProducerChannel.instances[0]
    assert len(producer.send_data_parts_calls) == 1

    send_call = producer.send_data_parts_calls[0]
    parts = send_call["parts"]
    total_bytes = send_call["total_bytes"]

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
    assert total_bytes == len(header) + len(metadata_json) + frame.nbytes
