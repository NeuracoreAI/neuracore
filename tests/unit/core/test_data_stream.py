from __future__ import annotations

import json
import struct
import sys
import types

import numpy as np
import pytest
from neuracore_types import DataType

from neuracore.core.streaming.data_stream import (
    DataRecordingContext,
    JsonDataStream,
    RGBDataStream,
)
from neuracore.data_daemon.communications_management.producer import (
    native_producer_channel as native_adaptor,
)
from neuracore.data_daemon.communications_management.producer.producer_channel import (
    producer_transport_args_for_data_type,
)
from neuracore.data_daemon.const import (
    DEFAULT_VIDEO_CHUNK_SIZE,
    DEFAULT_VIDEO_SEND_QUEUE_MAXSIZE,
    DEFAULT_VIDEO_SLOT_SIZE,
)


class _FakeProducerChannel:
    instances: list[_FakeProducerChannel] = []

    def __init__(
        self,
        *,
        data_type: DataType,
        id: str | None = None,
        recording_id: str | None = None,
        chunk_size: int | None = None,
        send_queue_maxsize: int | None = None,
        shared_memory_size: int | None = None,
        **_: object,
    ) -> None:
        default_chunk_size, default_shared_memory_size, default_send_queue_maxsize = (
            producer_transport_args_for_data_type(data_type)
        )
        self.id = id
        self.recording_id = recording_id
        self.data_type = data_type
        self.chunk_size = default_chunk_size if chunk_size is None else chunk_size
        self.send_queue_maxsize = (
            default_send_queue_maxsize
            if send_queue_maxsize is None
            else send_queue_maxsize
        )
        self.init_shared_memory_size: int | None = None
        self.default_shared_memory_size = (
            default_shared_memory_size
            if shared_memory_size is None
            else shared_memory_size
        )
        self.opened_shared_memory_sizes: list[int] = []
        self.send_data_parts_calls: list[dict[str, object]] = []
        self.cleanup_wait_for_slot_drain_calls: list[bool] = []
        self.stop_wait_for_slot_drain_calls: list[bool] = []
        self.trace_id = None
        _FakeProducerChannel.instances.append(self)

    def start_recording_session(
        self,
        *,
        recording_id: str | None = None,
        shared_memory_size: int | None = None,
    ) -> None:
        if recording_id is not None:
            self.recording_id = recording_id
        self.trace_id = "trace-1"
        self.opened_shared_memory_sizes.append(
            self.default_shared_memory_size
            if shared_memory_size is None
            else shared_memory_size
        )

    def initialize_new_producer_channel(
        self, shared_memory_size: int | None = None
    ) -> None:
        self.init_shared_memory_size = (
            self.default_shared_memory_size
            if shared_memory_size is None
            else shared_memory_size
        )

    def set_recording_id(self, recording_id: str | None) -> None:
        self.recording_id = recording_id

    def start_producer_channel(self) -> None:
        return

    def stop_producer_channel(
        self,
        *,
        wait_for_slot_drain: bool = True,
    ) -> None:
        self.stop_wait_for_slot_drain_calls.append(wait_for_slot_drain)
        return

    def open_fixed_shared_slots(self, slot_size: int | None = None) -> None:
        self.opened_shared_memory_sizes.append(
            self.default_shared_memory_size if slot_size is None else slot_size
        )

    def start_new_trace(self) -> None:
        self.trace_id = "trace-1"

    def cleanup_producer_channel(
        self,
        *,
        stop_cutoff_sequence_number: int | None = None,
        wait_for_slot_drain: bool = True,
    ) -> None:
        del stop_cutoff_sequence_number
        self.cleanup_wait_for_slot_drain_calls.append(wait_for_slot_drain)
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
    assert producer.data_type == DataType.RGB_IMAGES
    assert producer.chunk_size == DEFAULT_VIDEO_CHUNK_SIZE
    assert producer.send_queue_maxsize == DEFAULT_VIDEO_SEND_QUEUE_MAXSIZE
    assert producer.opened_shared_memory_sizes == [DEFAULT_VIDEO_SLOT_SIZE]


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


def test_stream_stop_recording_wait_false_skips_slot_drain(monkeypatch) -> None:
    _FakeProducerChannel.instances.clear()
    monkeypatch.setattr(
        "neuracore.core.streaming.data_stream.ProducerChannel",
        _FakeProducerChannel,
    )

    stream = RGBDataStream("front_camera", width=640, height=480)
    stream.start_recording(_context())

    producer = _FakeProducerChannel.instances[0]
    stream.stop_recording(
        stop_cutoff_sequence_number=0,
        wait_for_producer_drain=False,
    )

    assert producer.cleanup_wait_for_slot_drain_calls == [False]
    assert producer.stop_wait_for_slot_drain_calls == [False]
    assert stream.get_recording_context() is None
    assert stream.get_producer_channel() is None
    assert stream.is_recording() is False


class _NativeProducerStub:
    """Records native producer entry-point calls for assertion."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def start_recording(self, recording_id: str) -> None:
        self.calls.append(("start_recording", (recording_id,)))

    def start_trace(
        self,
        recording_id: str,
        trace_id: str,
        data_type: str,
        data_type_name: str | None = None,
    ) -> None:
        self.calls.append(
            ("start_trace", (recording_id, trace_id, data_type, data_type_name))
        )

    def open_frame_stream(self, trace_id: str, width: int, height: int) -> None:
        self.calls.append(("open_frame_stream", (trace_id, width, height)))

    def send_data(
        self,
        trace_id: str,
        payload: bytes,
        timestamp_ns: int,
        timestamp_s: float | None = None,
    ) -> None:
        self.calls.append(
            ("send_data", (trace_id, bytes(payload), timestamp_ns, timestamp_s))
        )

    def end_trace(self, trace_id: str) -> None:
        self.calls.append(("end_trace", (trace_id,)))

    def stop_recording(self, recording_id: str) -> None:
        self.calls.append(("stop_recording", (recording_id,)))


@pytest.fixture
def native_runtime(monkeypatch: pytest.MonkeyPatch):
    """Force the data-stream module to pick the native producer this test."""
    stub = _NativeProducerStub()
    fake_module = types.ModuleType("neuracore.data_daemon._native_producer")
    for method_name in (
        "start_recording",
        "start_trace",
        "open_frame_stream",
        "send_data",
        "end_trace",
        "stop_recording",
    ):
        setattr(fake_module, method_name, getattr(stub, method_name))
    monkeypatch.setitem(
        sys.modules, "neuracore.data_daemon._native_producer", fake_module
    )
    monkeypatch.setattr(native_adaptor, "_NATIVE_MODULE", None)
    monkeypatch.setattr(
        "neuracore.core.streaming.data_stream.rust_daemon_enabled",
        lambda: True,
    )
    yield stub
    monkeypatch.setattr(native_adaptor, "_NATIVE_MODULE", None)


def test_rgb_stream_uses_native_producer_when_rust_daemon_enabled(
    native_runtime: _NativeProducerStub,
) -> None:
    width, height = 4, 3
    stream = RGBDataStream("front_camera", width=width, height=height)
    stream.start_recording(_context())

    sequence = [call[0] for call in native_runtime.calls]
    assert sequence == ["start_recording", "start_trace", "open_frame_stream"]

    start_trace_args = native_runtime.calls[1][1]
    assert start_trace_args[2] == DataType.RGB_IMAGES.value
    open_args = native_runtime.calls[2][1]
    assert open_args[1:] == (width, height)


def test_rgb_stream_native_log_sends_only_raw_frame_bytes(
    native_runtime: _NativeProducerStub,
) -> None:
    width, height = 4, 3
    stream = RGBDataStream("front_camera", width=width, height=height)
    stream.start_recording(_context())

    frame = np.arange(width * height * 3, dtype=np.uint8).reshape((height, width, 3))
    stream.log(_DummyCameraData(timestamp=1.0), frame)

    send_calls = [call for call in native_runtime.calls if call[0] == "send_data"]
    assert len(send_calls) == 1
    _, payload, _, _ = send_calls[0][1]
    assert payload == bytes(frame)


def test_json_stream_native_log_publishes_serialized_payload(
    native_runtime: _NativeProducerStub,
) -> None:
    class _Payload:
        def model_dump(self, mode: str = "json"):
            del mode
            return {"value": 1.5}

    stream = JsonDataStream(
        data_type=DataType.JOINT_POSITIONS, data_type_name="joint_x"
    )
    stream.start_recording(_context())

    stream.log(_Payload())

    send_calls = [call for call in native_runtime.calls if call[0] == "send_data"]
    assert len(send_calls) == 1
    _, payload, _, _ = send_calls[0][1]
    assert json.loads(payload.decode("utf-8")) == {"value": 1.5}


def test_native_stream_stop_recording_emits_end_trace(
    native_runtime: _NativeProducerStub,
) -> None:
    stream = JsonDataStream(
        data_type=DataType.JOINT_POSITIONS, data_type_name="joint_x"
    )
    stream.start_recording(_context())
    stream.stop_recording(stop_cutoff_sequence_number=0)

    assert any(call[0] == "end_trace" for call in native_runtime.calls)
    assert stream.get_producer_channel() is None
    assert stream.is_recording() is False
