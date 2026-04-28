import json
import struct
import threading
import time
from datetime import datetime, timedelta, timezone
from multiprocessing.shared_memory import SharedMemory

import pytest
from neuracore_types import DataType
import zmq

from neuracore.data_daemon.communications_management.channel_reader import (
    ChannelMessageReader,
)
from neuracore.data_daemon.communications_management.data_bridge import (
    ChannelState,
    Daemon,
)
from neuracore.data_daemon.communications_management.producer_channel import (
    ProducerChannel,
)
from neuracore.data_daemon.communications_management.ring_buffer import RingBuffer
from neuracore.data_daemon.communications_management.shared_slot_transport import (
    PacketTooLarge,
    QueuedSharedSlotPacket,
    SharedSlotRegistry,
    SharedSlotVideoWorker,
    parse_shared_frame_packet,
)
from neuracore.data_daemon.const import (
    DEFAULT_VIDEO_SEND_QUEUE_MAXSIZE,
    HEARTBEAT_TIMEOUT_SECS,
    SHARED_RING_RECORD_HEADER_FORMAT,
    SHARED_RING_RECORD_MAGIC,
)
from neuracore.data_daemon.models import (
    CommandType,
    CompleteMessage,
    DataChunkPayload,
    MessageEnvelope,
    SharedSlotCreditReturn,
    SharedSlotDescriptor,
    SharedSlotReadyModel,
)


class FakeSharedFrame:
    def __init__(self, payload: bytes) -> None:
        self.data = memoryview(payload)
        self.disposed = False

    def dispose(self) -> None:
        self.disposed = True
        self.data.release()


class FakeSharedReader:
    def __init__(self, packets: list[bytes]) -> None:
        self._packets = list(packets)
        self.frames: list[FakeSharedFrame] = []
        self.closed = False

    def read_frame(self, timeout: float = 0.0) -> FakeSharedFrame | None:
        del timeout
        if not self._packets:
            return None
        frame = FakeSharedFrame(self._packets.pop(0))
        self.frames.append(frame)
        return frame

    def close(self) -> None:
        self.closed = True


def _make_shared_ring_reader(*packets: bytes) -> RingBuffer:
    size = max((len(packet) for packet in packets), default=1)
    return RingBuffer(
        size=size,
        _shared_name="test-shared",
        _shared_reader=FakeSharedReader(list(packets)),
    )


def _decode_shared_ring_write(packet: bytes) -> tuple[dict, bytes]:
    _magic, metadata_len, chunk_len = struct.unpack(
        SHARED_RING_RECORD_HEADER_FORMAT,
        packet[: struct.calcsize(SHARED_RING_RECORD_HEADER_FORMAT)],
    )
    metadata_start = struct.calcsize(SHARED_RING_RECORD_HEADER_FORMAT)
    metadata_end = metadata_start + metadata_len
    metadata = json.loads(packet[metadata_start:metadata_end].decode("utf-8"))
    chunk = packet[metadata_end : metadata_end + chunk_len]
    return metadata, chunk


def _read_shared_slot_packet(envelope: MessageEnvelope) -> tuple[dict, bytes]:
    descriptor = SharedSlotDescriptor.from_dict(
        envelope.payload[CommandType.SHARED_SLOT_DESCRIPTOR.value]
    )
    shm = SharedMemory(name=descriptor.shm_name)
    try:
        packet = bytes(shm.buf[descriptor.offset : descriptor.offset + descriptor.length])
    finally:
        shm.close()
    return parse_shared_frame_packet(packet)


def test_message_envelope_round_trip() -> None:
    payload = {"open_ring_buffer": {"size": 2048}}
    envelope = MessageEnvelope(
        producer_id="producer-123",
        command=CommandType.OPEN_RING_BUFFER,
        payload=payload,
    )

    parsed = MessageEnvelope.from_bytes(envelope.to_bytes())

    assert parsed.producer_id == "producer-123"
    assert parsed.command == CommandType.OPEN_RING_BUFFER
    assert parsed.payload == payload


def test_data_chunk_payload_round_trip() -> None:
    chunk = DataChunkPayload(
        channel_id="chan-1",
        recording_id="rec-1",
        trace_id="42",
        chunk_index=1,
        total_chunks=2,
        data_type_name="custom",
        dataset_id=None,
        dataset_name=None,
        robot_name=None,
        robot_id=None,
        robot_instance=3,
        data_type=DataType.CUSTOM_1D,
        data=b"payload-bytes",
    )

    restored = DataChunkPayload.from_dict(chunk.to_dict())

    assert restored == chunk


def test_complete_message_batch_record_round_trip() -> None:
    record = CompleteMessage.from_bytes(
        "prod",
        "rec-1",
        True,
        "trace",
        DataType.CUSTOM_1D,
        "custom_data",
        0,
        b"hello",
        None,
        None,
        None,
        None,
    )

    raw = record.to_batch_record()
    restored = CompleteMessage.iter_batch_records(raw)

    assert len(restored) == 1
    parsed = restored[0]
    assert parsed.producer_id == "prod"
    assert parsed.trace_id == "trace"
    assert parsed.recording_id == "rec-1"
    assert parsed.dataset_id is None
    assert parsed.dataset_name is None
    assert parsed.robot_name is None
    assert parsed.robot_id is None
    assert parsed.data_type == DataType.CUSTOM_1D
    assert parsed.data_type_name == "custom_data"
    assert parsed.robot_instance == 0
    assert datetime.fromisoformat(parsed.received_at)
    assert parsed.data == b"hello"
    assert parsed.final_chunk is True


def test_channel_reader_reassembles_shared_ring_records() -> None:
    packets: list[bytes] = []

    for idx, chunk in enumerate((b"robot", b"ics")):
        metadata = {
            "trace_id": "trace-shared",
            "chunk_index": idx,
            "total_chunks": 2,
        }
        if idx == 0:
            metadata.update(
                {
                    "recording_id": "rec-shared",
                    "data_type": DataType.CUSTOM_1D.value,
                    "data_type_name": "custom",
                    "robot_instance": 1,
                    "robot_id": "robot-1",
                    "dataset_id": "dataset-1",
                }
            )
        metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        packets.append(
            struct.pack(
                SHARED_RING_RECORD_HEADER_FORMAT,
                SHARED_RING_RECORD_MAGIC,
                len(metadata_bytes),
                len(chunk),
            )
            + metadata_bytes
            + chunk
        )

    reader = ChannelMessageReader(_make_shared_ring_reader(*packets))

    assembled = reader.poll_one()
    while assembled is None:
        assembled = reader.poll_one()

    assert assembled == ("trace-shared", DataType.CUSTOM_1D, b"robotics")
    assert assembled.metadata is not None
    assert assembled.metadata.recording_id == "rec-shared"


def test_channel_reader_rejects_shared_ring_trailing_bytes() -> None:
    metadata_bytes = json.dumps(
        {
            "trace_id": "trace-shared",
            "recording_id": "rec-shared",
            "chunk_index": 0,
            "total_chunks": 1,
            "data_type": DataType.CUSTOM_1D.value,
            "data_type_name": "custom",
            "robot_instance": 1,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    payload = b"robotics"
    packet = (
        struct.pack(
            SHARED_RING_RECORD_HEADER_FORMAT,
            SHARED_RING_RECORD_MAGIC,
            len(metadata_bytes),
            len(payload),
        )
        + metadata_bytes
        + payload
        + b"!"
    )

    reader = ChannelMessageReader(_make_shared_ring_reader(packet))

    try:
        reader.poll_one()
    except ValueError as exc:
        assert "trailing bytes" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected trailing-byte packet to be rejected")


def test_channel_reader_rejects_short_shared_ring_packet() -> None:
    metadata_bytes = json.dumps(
        {
            "trace_id": "trace-shared",
            "recording_id": "rec-shared",
            "chunk_index": 0,
            "total_chunks": 1,
            "data_type": DataType.CUSTOM_1D.value,
            "data_type_name": "custom",
            "robot_instance": 1,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    payload = b"robotics"
    packet = (
        struct.pack(
            SHARED_RING_RECORD_HEADER_FORMAT,
            SHARED_RING_RECORD_MAGIC,
            len(metadata_bytes),
            len(payload) + 1,
        )
        + metadata_bytes
        + payload
    )

    reader = ChannelMessageReader(_make_shared_ring_reader(packet))

    try:
        reader.poll_one()
    except ValueError as exc:
        assert "shorter than declared lengths" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected short packet to be rejected")


class DummyComm:
    def __init__(self) -> None:
        self.messages = []
        self.cleaned = False
        self.socket_requested = False

    def create_producer_socket(self):
        self.socket_requested = True
        return object()

    def create_subscriber_socket(self):
        return None

    def send_message(self, message):
        self.messages.append(message)

    def cleanup_producer(self):
        self.cleaned = True


def _wait_for_messages(comm: DummyComm, expected: int, timeout: float = 1.0) -> None:
    """Wait for ProducerChannel's sender thread to flush messages to DummyComm."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(comm.messages) >= expected:
            return
        time.sleep(0.02)
    return


def _wait_for_envelopes(
    messages: list[MessageEnvelope], expected: int, timeout: float = 1.0
) -> None:
    """Wait for a stubbed producer transport to capture messages."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(messages) >= expected:
            return
        time.sleep(0.02)
    return


def _stub_producer_transport(monkeypatch) -> list[MessageEnvelope]:
    messages: list[MessageEnvelope] = []
    control_context = zmq.Context()
    control_sockets: dict[str, zmq.Socket] = {}
    control_endpoints: dict[str, str] = {}
    shared_memories: dict[str, SharedMemory] = {}

    monkeypatch.setattr(
        "neuracore.data_daemon.communications_management.communications_manager.CommunicationsManager.create_producer_socket",
        lambda self: None,
    )

    def _send_message(_self, message):
        messages.append(message)

        if message.command == CommandType.OPEN_FIXED_SHARED_SLOTS:
            payload = message.payload["open_fixed_shared_slots"]
            control_endpoint = payload["control_endpoint"]
            shm_name = f"neuracore-slots-test-{len(shared_memories)}"
            shm = SharedMemory(
                name=shm_name,
                create=True,
                size=int(payload["slot_size"]) * int(payload["slot_count"]),
            )
            shared_memories[shm_name] = shm
            control_endpoints[str(message.producer_id)] = control_endpoint
            socket_obj = control_sockets.get(control_endpoint)
            if socket_obj is None:
                socket_obj = control_context.socket(zmq.PUSH)
                socket_obj.setsockopt(zmq.LINGER, 0)
                socket_obj.connect(control_endpoint)
                control_sockets[control_endpoint] = socket_obj
            socket_obj.send(
                MessageEnvelope(
                    producer_id=None,
                    command=CommandType.SHARED_SLOT_READY,
                    payload={
                        CommandType.SHARED_SLOT_READY.value: SharedSlotReadyModel(
                            shm_name=shm_name,
                            slot_size=int(payload["slot_size"]),
                            slot_count=int(payload["slot_count"]),
                        ).model_dump()
                    },
                ).to_bytes()
            )
            return

        if message.command == CommandType.SHARED_SLOT_DESCRIPTOR:
            descriptor = SharedSlotDescriptor.from_dict(
                message.payload[CommandType.SHARED_SLOT_DESCRIPTOR.value]
            )
            control_endpoint = control_endpoints.get(str(message.producer_id))
            if control_endpoint is None:
                raise RuntimeError("Missing control endpoint for shared-slot descriptor")
            socket_obj = control_sockets[control_endpoint]
            socket_obj.send(
                MessageEnvelope(
                    producer_id=None,
                    command=CommandType.SHARED_SLOT_CREDIT_RETURN,
                    payload={
                        CommandType.SHARED_SLOT_CREDIT_RETURN.value: SharedSlotCreditReturn(
                            shm_name=descriptor.shm_name,
                            slot_id=descriptor.slot_id,
                            sequence_id=descriptor.sequence_id,
                        ).to_dict()
                    },
                ).to_bytes()
            )

    monkeypatch.setattr(
        "neuracore.data_daemon.communications_management.communications_manager.CommunicationsManager.send_message",
        _send_message,
    )
    def _cleanup_producer(_self) -> None:
        for socket_obj in control_sockets.values():
            socket_obj.close(0)
        control_sockets.clear()
        for shm in shared_memories.values():
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        shared_memories.clear()
        control_context.term()

    monkeypatch.setattr(
        "neuracore.data_daemon.communications_management.communications_manager.CommunicationsManager.cleanup_producer",
        _cleanup_producer,
    )

    return messages


def test_producer_open_ring_buffer_sends_payload(monkeypatch) -> None:
    messages = _stub_producer_transport(monkeypatch)
    producer = ProducerChannel(data_type=DataType.RGB_IMAGES)

    try:
        producer.open_ring_buffer(size=2048)
        _wait_for_envelopes(messages, 1)
    finally:
        producer.stop_producer_channel()

    assert len(messages) == 1
    envelope = messages[0]
    assert envelope.command == CommandType.OPEN_FIXED_SHARED_SLOTS
    payload = envelope.payload["open_fixed_shared_slots"]
    assert payload["slot_size"] == 2048
    assert payload["slot_count"] == 16
    assert payload["control_endpoint"].startswith("ipc://")


def test_producer_send_data_parts_lazily_opens_shared_ring_buffer(
    monkeypatch,
) -> None:
    messages = _stub_producer_transport(monkeypatch)

    producer = ProducerChannel(
        chunk_size=2,
        recording_id="rec-1",
        ring_buffer_size=2048,
        data_type=DataType.RGB_IMAGES,
    )

    try:
        producer.start_new_trace()
        producer.send_data(
            b"abcd",
            data_type=DataType.RGB_IMAGES,
            data_type_name="custom",
            robot_instance=2,
            robot_id="robot-1",
            robot_name="robot",
            dataset_id="dataset-1",
            dataset_name="dataset",
        )
        _wait_for_envelopes(messages, 3)
        first_metadata, first_chunk = _read_shared_slot_packet(messages[1])
        second_metadata, second_chunk = _read_shared_slot_packet(messages[2])
    finally:
        producer.stop_producer_channel()

    assert len(messages) == 3
    assert messages[0].command == CommandType.OPEN_FIXED_SHARED_SLOTS
    assert messages[1].command == CommandType.SHARED_SLOT_DESCRIPTOR
    assert messages[2].command == CommandType.SHARED_SLOT_DESCRIPTOR
    assert first_chunk == b"ab"
    assert second_chunk == b"cd"


def test_producer_send_data_parts_uses_socket_for_non_video(monkeypatch) -> None:
    messages = _stub_producer_transport(monkeypatch)
    producer = ProducerChannel(
        recording_id="rec-1",
        data_type=DataType.CUSTOM_1D,
    )

    try:
        producer.start_new_trace()
        producer.send_data(
            b"abcd",
            data_type=DataType.CUSTOM_1D,
            data_type_name="custom",
            robot_instance=2,
            robot_id="robot-1",
            robot_name="robot",
            dataset_id="dataset-1",
            dataset_name="dataset",
        )
        _wait_for_envelopes(messages, 1)
    finally:
        producer.stop_producer_channel()

    assert len(messages) == 1
    envelope = messages[0]
    assert envelope.command == CommandType.DATA_CHUNK
    payload = DataChunkPayload.from_dict(envelope.payload["data_chunk"])
    assert payload.channel_id == producer.channel_id
    assert payload.recording_id == "rec-1"
    assert payload.trace_id == producer.trace_id
    assert payload.chunk_index == 0
    assert payload.total_chunks == 1
    assert payload.data_type == DataType.CUSTOM_1D
    assert payload.data == b"abcd"


def test_producer_ensure_shared_ring_buffer_does_not_reannounce(
    monkeypatch,
) -> None:
    messages = _stub_producer_transport(monkeypatch)

    producer = ProducerChannel(
        recording_id="rec-1",
        ring_buffer_size=2048,
        data_type=DataType.RGB_IMAGES,
    )

    try:
        producer.start_new_trace()
        producer.open_ring_buffer(size=2048)
        _wait_for_envelopes(messages, 1)
        control_endpoint = messages[0].payload["open_fixed_shared_slots"]["control_endpoint"]
        trace_id = producer.trace_id

        producer.send_data(
            b"ab",
            data_type=DataType.RGB_IMAGES,
            data_type_name="custom",
            robot_instance=2,
            robot_id="robot-1",
            robot_name="robot",
            dataset_id="dataset-1",
            dataset_name="dataset",
        )
        _wait_for_envelopes(messages, 2)
        metadata, _chunk = _read_shared_slot_packet(messages[1])
    finally:
        producer.stop_producer_channel()

    assert len(messages) == 2
    assert messages[0].payload["open_fixed_shared_slots"]["control_endpoint"] == control_endpoint
    assert metadata["trace_id"] == trace_id


class DummySharedRingBuffer:
    def __init__(self, shared_name: str = "test-shared", size: int = 2048) -> None:
        self.shared_name = shared_name
        self.size = size
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, data: bytes | bytearray | memoryview) -> None:
        self.writes.append(bytes(data))

    def close(self) -> None:
        self.closed = True


def test_producer_send_data_parts_chunks_across_multiple_buffers(monkeypatch) -> None:
    messages = _stub_producer_transport(monkeypatch)

    producer = ProducerChannel(
        chunk_size=3,
        recording_id="rec-1",
        ring_buffer_size=2048,
        data_type=DataType.RGB_IMAGES,
    )

    try:
        producer.start_new_trace()
        # cspell:ignore cdef
        producer.send_data_parts(
            (b"ab", memoryview(b"cdef"), b"gh"),
            total_bytes=8,
            data_type=DataType.RGB_IMAGES,
            data_type_name="custom",
            robot_instance=2,
            robot_id="robot-1",
            robot_name="robot",
            dataset_id="dataset-1",
            dataset_name="dataset",
        )
        _wait_for_envelopes(messages, 4)
        packets = [_read_shared_slot_packet(packet) for packet in messages[1:]]
    finally:
        producer.stop_producer_channel()

    assert len(messages) == 4
    envelope = messages[0]
    assert envelope.command == CommandType.OPEN_FIXED_SHARED_SLOTS
    payload = envelope.payload["open_fixed_shared_slots"]
    assert payload["slot_size"] == 2048
    assert payload["control_endpoint"].startswith("ipc://")
    assert [chunk for _, chunk in packets] == [b"abc", b"def", b"gh"]
    assert packets[0][0]["recording_id"] == "rec-1"
    assert "recording_id" not in packets[1][0]
    assert "recording_id" not in packets[2][0]


def test_producer_transport_stats_include_sender_and_shared_ring_metrics(
    monkeypatch,
) -> None:
    monkeypatch.setenv("NDD_DEBUG", "true")
    _stub_producer_transport(monkeypatch)

    producer = ProducerChannel(
        chunk_size=2,
        recording_id="rec-1",
        ring_buffer_size=2048,
        data_type=DataType.RGB_IMAGES,
    )

    try:
        producer.start_new_trace()
        producer.send_data(
            b"abcd",
            data_type=DataType.RGB_IMAGES,
            data_type_name="custom",
            robot_instance=2,
            robot_id="robot-1",
            dataset_id="dataset-1",
        )
        time.sleep(0.1)
        stats = producer.get_transport_stats()
    finally:
        producer.stop_producer_channel()

    assert stats["channel_id"] == producer.channel_id
    assert stats["recording_id"] == "rec-1"
    assert stats["trace_id"] is not None
    assert str(stats["shared_ring_buffer_name"]).startswith("neuracore-slots-")
    assert stats["shared_ring_buffer_size"] == 2048 * 16
    assert stats["send_queue_qsize"] == 0
    assert stats["send_queue_maxsize"] == DEFAULT_VIDEO_SEND_QUEUE_MAXSIZE
    assert stats["last_enqueued_sequence_number"] >= 1
    assert stats["last_socket_sent_sequence_number"] >= 1
    assert stats["pending_sequence_count"] == 0
    assert stats["sender_thread_alive"] is True
    assert stats["shared_ring_open_count"] == 0
    assert stats["shared_ring_write_count"] == 0
    assert stats["shared_ring_dispatch_count"] == 0
    assert stats["socket_send_count"] >= 1
    assert stats["queue_put_count"] >= 3
    assert stats["send_error_count"] == 0
    assert stats["last_send_error"] is None


def test_producer_shared_ring_rejects_oversized_packet(monkeypatch) -> None:
    monkeypatch.setenv("NDD_DEBUG", "true")
    _stub_producer_transport(monkeypatch)

    producer = ProducerChannel(
        chunk_size=16,
        recording_id="rec-1",
        ring_buffer_size=8,
        data_type=DataType.RGB_IMAGES,
    )

    try:
        producer.start_new_trace()
        try:
            producer.send_data(
                b"abcdefgh",
                data_type=DataType.RGB_IMAGES,
                data_type_name="custom",
                robot_instance=2,
                robot_id="robot-1",
                dataset_id="dataset-1",
            )
        except PacketTooLarge:
            oversized = True
        else:
            oversized = False
    finally:
        producer.stop_producer_channel()

    assert oversized is True


def test_producer_sender_failure_stops_waiters(monkeypatch) -> None:
    monkeypatch.setenv("NDD_DEBUG", "true")
    sent = {"count": 0}
    control_context = zmq.Context()
    control_sockets: dict[str, zmq.Socket] = {}
    shared_memories: dict[str, SharedMemory] = {}

    def flaky_send(_self, message):
        sent["count"] += 1
        if message.command == CommandType.OPEN_FIXED_SHARED_SLOTS:
            payload = message.payload["open_fixed_shared_slots"]
            control_endpoint = payload["control_endpoint"]
            shm_name = "neuracore-slots-test-failure"
            shm = SharedMemory(
                name=shm_name,
                create=True,
                size=int(payload["slot_size"]) * int(payload["slot_count"]),
            )
            shared_memories[shm_name] = shm
            socket_obj = control_sockets.get(control_endpoint)
            if socket_obj is None:
                socket_obj = control_context.socket(zmq.PUSH)
                socket_obj.setsockopt(zmq.LINGER, 0)
                socket_obj.connect(control_endpoint)
                control_sockets[control_endpoint] = socket_obj
            socket_obj.send(
                MessageEnvelope(
                    producer_id=None,
                    command=CommandType.SHARED_SLOT_READY,
                    payload={
                        CommandType.SHARED_SLOT_READY.value: SharedSlotReadyModel(
                            shm_name=shm_name,
                            slot_size=int(payload["slot_size"]),
                            slot_count=int(payload["slot_count"]),
                        ).model_dump()
                    },
                ).to_bytes()
            )
            return
        if message.command == CommandType.SHARED_SLOT_DESCRIPTOR:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "neuracore.data_daemon.communications_management.communications_manager.CommunicationsManager.create_producer_socket",
        lambda self: None,
    )
    monkeypatch.setattr(
        "neuracore.data_daemon.communications_management.communications_manager.CommunicationsManager.send_message",
        flaky_send,
    )
    def _cleanup_producer(_self) -> None:
        for socket_obj in control_sockets.values():
            socket_obj.close(0)
        control_sockets.clear()
        for shm in shared_memories.values():
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        shared_memories.clear()
        control_context.term()

    monkeypatch.setattr(
        "neuracore.data_daemon.communications_management.communications_manager.CommunicationsManager.cleanup_producer",
        _cleanup_producer,
    )

    producer = ProducerChannel(
        chunk_size=2,
        recording_id="rec-1",
        ring_buffer_size=2048,
        data_type=DataType.RGB_IMAGES,
    )

    try:
        producer.start_new_trace()
        producer.send_data(
            b"abcd",
            data_type=DataType.RGB_IMAGES,
            data_type_name="custom",
            robot_instance=2,
            robot_id="robot-1",
            dataset_id="dataset-1",
        )
        deadline = time.monotonic() + 1.0
        while producer.get_transport_stats()["sender_thread_alive"] and time.monotonic() < deadline:
            time.sleep(0.02)
        try:
            producer._send(CommandType.HEARTBEAT, {})
        except RuntimeError:
            wait_result = False
        else:
            wait_result = True
    finally:
        producer.stop_producer_channel()

    assert sent["count"] >= 1
    assert wait_result is False


def test_shared_slot_video_worker_surfaces_background_failure() -> None:
    registry = SharedSlotRegistry.acquire(slot_size=2048, slot_count=2)
    worker = SharedSlotVideoWorker.acquire(registry)

    def raise_worker_error(_item) -> None:
        raise RuntimeError("boom")

    worker._process_item = raise_worker_error  # type: ignore[method-assign]

    packet = QueuedSharedSlotPacket(
        producer_id="producer-1",
        sender=None,  # type: ignore[arg-type]
        metadata_bytes=b"{}",
        chunk=b"x",
        packet_length=1,
    )

    try:
        worker.enqueue_packet(packet=packet)
        deadline = time.monotonic() + 1.0
        while worker._thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)

        with pytest.raises(RuntimeError, match="Shared-slot video worker failed"):
            worker.enqueue_packet(packet=packet)
    finally:
        SharedSlotVideoWorker.reset_shared_instance_for_tests()
        SharedSlotRegistry.reset_shared_instance_for_tests()


def test_shared_slot_timeout_clock_starts_after_socket_send() -> None:
    registry = SharedSlotRegistry.acquire(
        slot_size=2048,
        slot_count=2,
        ack_timeout_s=0.01,
        allocate_timeout_s=0.01,
    )
    shm = SharedMemory(name="test-credit-timeout", create=True, size=2048 * 2)

    try:
        registry._apply_ready_message(
            SharedSlotReadyModel(
                shm_name="test-credit-timeout",
                slot_size=2048,
                slot_count=2,
            )
        )
        slot_id, _offset = registry.allocate_slot()
        sequence_id = registry.mark_in_flight(slot_id)

        time.sleep(0.03)
        with registry._condition:
            registry._check_for_timeouts_locked()
        snapshot = registry.debug_snapshot()
        assert snapshot["healthy"] is True
        assert sequence_id in snapshot["in_flight_sequence_ids"]

        registry.mark_sent(sequence_id)
        time.sleep(0.03)
        with registry._condition:
            registry._check_for_timeouts_locked()
        assert registry.is_healthy() is False
    finally:
        SharedSlotRegistry.reset_shared_instance_for_tests()
        shm.close()
        shm.unlink()


def test_shared_slot_registry_runtime_starts_and_stops_cleanly() -> None:
    registry = SharedSlotRegistry.acquire(slot_size=2048, slot_count=2)

    try:
        assert registry.control_endpoint.startswith("ipc://")
        assert registry._runtime.control_thread.is_alive()
        assert registry._runtime.watchdog_thread.is_alive()
    finally:
        SharedSlotRegistry.reset_shared_instance_for_tests()

    assert not registry._runtime.control_thread.is_alive()
    assert not registry._runtime.watchdog_thread.is_alive()


def test_shared_slot_ready_message_populates_free_slots() -> None:
    registry = SharedSlotRegistry.acquire(slot_size=2048, slot_count=3)
    shm = SharedMemory(name="test-ready-populates-free-slots", create=True, size=2048 * 3)

    try:
        registry._apply_ready_message(
            SharedSlotReadyModel(
                shm_name="test-ready-populates-free-slots",
                slot_size=2048,
                slot_count=3,
            )
        )

        snapshot = registry.debug_snapshot()
        assert snapshot["ready"] is True
        assert snapshot["free_slot_count"] == 3
        assert registry.slot_size == 2048
        assert registry.slot_count == 3
    finally:
        SharedSlotRegistry.reset_shared_instance_for_tests()
        shm.close()
        shm.unlink()


def test_shared_slot_ready_message_adopts_daemon_slot_dimensions() -> None:
    registry = SharedSlotRegistry.acquire(slot_size=1024, slot_count=1)
    shm = SharedMemory(name="test-ready-adopts-slot-dimensions", create=True, size=4096 * 4)

    try:
        registry._apply_ready_message(
            SharedSlotReadyModel(
                shm_name="test-ready-adopts-slot-dimensions",
                slot_size=4096,
                slot_count=4,
            )
        )

        assert registry.slot_size == 4096
        assert registry.slot_count == 4
        assert registry.total_shared_memory_bytes == 4096 * 4
        assert registry.debug_snapshot()["free_slot_count"] == 4
    finally:
        SharedSlotRegistry.reset_shared_instance_for_tests()
        shm.close()
        shm.unlink()


def test_producer_sequences_follow_enqueue_order_under_concurrent_senders(
    monkeypatch,
) -> None:
    messages = _stub_producer_transport(monkeypatch)
    producer = ProducerChannel(
        recording_id="rec-1",
        data_type=DataType.CUSTOM_1D,
    )

    first_put_entered = threading.Event()
    allow_first_put = threading.Event()
    second_send_finished = threading.Event()
    thread_errors: list[BaseException] = []

    real_put = producer._send_queue.put

    def blocked_put(item):
        if item is not None and not first_put_entered.is_set():
            first_put_entered.set()
            allow_first_put.wait(timeout=5.0)
        return real_put(item)

    monkeypatch.setattr(producer._send_queue, "put", blocked_put)

    def send_heartbeat(mark_done: threading.Event | None = None) -> None:
        try:
            producer.heartbeat()
        except BaseException as exc:  # pragma: no cover - surfaced below
            thread_errors.append(exc)
        finally:
            if mark_done is not None:
                mark_done.set()

    first_sender = threading.Thread(target=send_heartbeat, daemon=True)
    second_sender = threading.Thread(
        target=send_heartbeat,
        kwargs={"mark_done": second_send_finished},
        daemon=True,
    )

    try:
        first_sender.start()
        assert first_put_entered.wait(timeout=5.0) is True

        second_sender.start()
        time.sleep(0.1)

        assert producer.get_last_enqueued_sequence_number() == 1
        assert second_send_finished.is_set() is False

        allow_first_put.set()

        first_sender.join(timeout=5.0)
        second_sender.join(timeout=5.0)
        _wait_for_envelopes(messages, 2)
    finally:
        allow_first_put.set()
        producer.stop_producer_channel()

    assert thread_errors == []
    assert [message.sequence_number for message in messages] == [1, 2]


class DummyRecordingDiskManager:
    """Minimal recording disk manager for tests."""

    def __init__(self) -> None:
        self.messages = []

    def enqueue(self, msg):
        self.messages.append(msg)


def test_daemon_registers_trace_from_shared_ring_metadata(emitter) -> None:
    rdm = DummyRecordingDiskManager()
    daemon = Daemon(
        comm_manager=DummyComm(),
        recording_disk_manager=rdm,
        emitter=emitter,
    )
    channel = ChannelState(producer_id="shared")
    daemon.channels["shared"] = channel

    metadata = {
        "trace_id": "trace-shared",
        "recording_id": "rec-shared",
        "chunk_index": 0,
        "total_chunks": 1,
        "data_type": DataType.CUSTOM_1D.value,
        "data_type_name": "custom",
        "robot_instance": 1,
        "robot_id": "robot-1",
        "dataset_id": "dataset-1",
    }
    payload = b"robotics"
    metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    channel.set_ring_buffer(
        _make_shared_ring_reader(
            struct.pack(
                SHARED_RING_RECORD_HEADER_FORMAT,
                SHARED_RING_RECORD_MAGIC,
                len(metadata_bytes),
                len(payload),
            )
            + metadata_bytes
            + payload
        )
    )

    daemon._drain_single_channel_messages(channel)

    assert daemon._trace_recordings["trace-shared"] == "rec-shared"
    assert (
        daemon._trace_metadata["trace-shared"]["data_type"] == DataType.CUSTOM_1D.value
    )
    assert len(rdm.messages) == 1


def test_cleanup_removes_channel_without_heartbeat(emitter) -> None:
    daemon = Daemon(
        comm_manager=DummyComm(),
        recording_disk_manager=DummyRecordingDiskManager(),
        emitter=emitter,
    )
    channel = ChannelState(
        producer_id="stale",
        last_heartbeat=datetime.now(timezone.utc)
        - timedelta(seconds=HEARTBEAT_TIMEOUT_SECS + 1),
    )
    daemon.channels["stale"] = channel

    daemon._cleanup_expired_channels()

    assert "stale" not in daemon.channels


def test_cleanup_keeps_recent_channel(emitter) -> None:
    daemon = Daemon(
        comm_manager=DummyComm(),
        recording_disk_manager=DummyRecordingDiskManager(),
        emitter=emitter,
    )
    channel = ChannelState(
        producer_id="active",
        last_heartbeat=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    daemon.channels["active"] = channel

    daemon._cleanup_expired_channels()

    assert "active" in daemon.channels
