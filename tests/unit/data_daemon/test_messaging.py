import json
import struct
import threading
import time
from datetime import datetime, timedelta, timezone

from neuracore_types import DataType

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
from neuracore.data_daemon.const import (
    CHUNK_HEADER_FORMAT,
    DATA_TYPE_FIELD_SIZE,
    HEARTBEAT_TIMEOUT_SECS,
    SHARED_RING_RECORD_HEADER_FORMAT,
    SHARED_RING_RECORD_MAGIC,
    TRACE_ID_FIELD_SIZE,
)
from neuracore.data_daemon.models import (
    CommandType,
    CompleteMessage,
    DataChunkPayload,
    MessageEnvelope,
)


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


def test_channel_reader_reassembles_chunks() -> None:
    ring = RingBuffer(size=2048)
    reader = ChannelMessageReader(ring)
    trace_id = "7"
    chunks = [b"robot", b"ics"]

    for idx, chunk in enumerate(chunks):
        trace_id_bytes = trace_id.encode("utf-8")
        trace_id_field = trace_id_bytes[:TRACE_ID_FIELD_SIZE].ljust(
            TRACE_ID_FIELD_SIZE, b"\x00"
        )
        data_type_bytes = DataType.CUSTOM_1D.value.encode("utf-8")
        data_type_field = data_type_bytes[:DATA_TYPE_FIELD_SIZE].ljust(
            DATA_TYPE_FIELD_SIZE, b"\x00"
        )
        header = struct.pack(
            CHUNK_HEADER_FORMAT,
            trace_id_field,
            data_type_field,
            idx,
            len(chunks),
            len(chunk),
        )
        ring.write(header + chunk)

    assembled = reader.poll_one()
    while assembled is None:
        assembled = reader.poll_one()

    assert assembled == (trace_id, DataType.CUSTOM_1D, b"robotics")


def test_channel_reader_reassembles_shared_ring_records() -> None:
    ring = RingBuffer(size=4096)
    reader = ChannelMessageReader(ring)

    for idx, chunk in enumerate((b"robot", b"ics")):
        metadata = {
            "trace_id": "trace-shared",
            "recording_id": "rec-shared",
            "chunk_index": idx,
            "total_chunks": 2,
            "data_type": DataType.CUSTOM_1D.value,
            "data_type_name": "custom",
            "robot_instance": 1,
            "robot_id": "robot-1",
            "dataset_id": "dataset-1",
        }
        metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        ring.write(
            struct.pack(
                SHARED_RING_RECORD_HEADER_FORMAT,
                SHARED_RING_RECORD_MAGIC,
                len(metadata_bytes),
                len(chunk),
            )
            + metadata_bytes
            + chunk
        )

    assembled = reader.poll_one()
    while assembled is None:
        assembled = reader.poll_one()

    assert assembled == ("trace-shared", DataType.CUSTOM_1D, b"robotics")
    assert assembled.metadata["recording_id"] == "rec-shared"


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

    monkeypatch.setattr(
        "neuracore.data_daemon.communications_management.communications_manager.CommunicationsManager.create_producer_socket",
        lambda self: None,
    )
    monkeypatch.setattr(
        "neuracore.data_daemon.communications_management.communications_manager.CommunicationsManager.send_message",
        lambda self, message: messages.append(message),
    )
    monkeypatch.setattr(
        "neuracore.data_daemon.communications_management.communications_manager.CommunicationsManager.cleanup_producer",
        lambda self: None,
    )

    return messages


def test_producer_open_ring_buffer_sends_payload(monkeypatch) -> None:
    messages = _stub_producer_transport(monkeypatch)
    producer = ProducerChannel(data_type=DataType.CUSTOM_1D)

    try:
        with monkeypatch.context() as ring_patch:
            ring_patch.setattr(
                "neuracore.data_daemon.communications_management.producer_channel.RingBuffer.create_shared",
                lambda size: DummySharedRingBuffer(shared_name="rb-open"),
            )
            producer.open_ring_buffer(size=2048)
    finally:
        producer.stop_producer_channel()

    assert len(messages) == 1
    envelope = messages[0]
    assert envelope.command == CommandType.OPEN_RING_BUFFER
    assert envelope.payload == {
        "open_ring_buffer": {"size": 2048, "shared_memory_name": "rb-open"}
    }


def test_producer_send_data_parts_lazily_opens_shared_ring_buffer(
    monkeypatch,
) -> None:
    messages = _stub_producer_transport(monkeypatch)
    shared_ring = DummySharedRingBuffer()

    monkeypatch.setattr(
        "neuracore.data_daemon.communications_management.producer_channel.RingBuffer.create_shared",
        lambda size: shared_ring,
    )

    producer = ProducerChannel(
        chunk_size=2,
        recording_id="rec-1",
        ring_buffer_size=2048,
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

        deadline = time.monotonic() + 1.0
        while len(shared_ring.writes) < 6 and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        producer.stop_producer_channel()

    assert len(messages) == 1
    envelope = messages[0]
    assert envelope.command == CommandType.OPEN_RING_BUFFER
    assert envelope.payload == {
        "open_ring_buffer": {"size": 2048, "shared_memory_name": "test-shared"}
    }
    assert shared_ring.writes[2::3] == [b"ab", b"cd"]


class DummySharedRingBuffer:
    def __init__(self, shared_name: str = "test-shared") -> None:
        self.shared_name = shared_name
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, data: bytes | bytearray | memoryview) -> None:
        self.writes.append(bytes(data))

    def close(self) -> None:
        self.closed = True


def test_producer_send_data_parts_chunks_across_multiple_buffers(monkeypatch) -> None:
    messages = _stub_producer_transport(monkeypatch)
    shared_ring = DummySharedRingBuffer()

    monkeypatch.setattr(
        "neuracore.data_daemon.communications_management.producer_channel.RingBuffer.create_shared",
        lambda size: shared_ring,
    )

    producer = ProducerChannel(
        chunk_size=3,
        recording_id="rec-1",
        ring_buffer_size=2048,
        data_type=DataType.CUSTOM_1D,
    )

    try:
        producer.start_new_trace()
        # cspell:ignore cdef
        producer.send_data_parts(
            (b"ab", memoryview(b"cdef"), b"gh"),
            total_bytes=8,
            data_type=DataType.CUSTOM_1D,
            data_type_name="custom",
            robot_instance=2,
            robot_id="robot-1",
            robot_name="robot",
            dataset_id="dataset-1",
            dataset_name="dataset",
        )

        deadline = time.monotonic() + 1.0
        while len(shared_ring.writes) < 9 and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        producer.stop_producer_channel()

    assert len(messages) == 1
    envelope = messages[0]
    assert envelope.command == CommandType.OPEN_RING_BUFFER
    assert envelope.payload == {
        "open_ring_buffer": {"size": 2048, "shared_memory_name": "test-shared"}
    }
    assert shared_ring.writes[2::3] == [b"abc", b"def", b"gh"]


def test_producer_transport_stats_include_sender_and_shared_ring_metrics(
    monkeypatch,
) -> None:
    monkeypatch.setenv("NDD_DEBUG", "true")
    _stub_producer_transport(monkeypatch)
    shared_ring = DummySharedRingBuffer()

    monkeypatch.setattr(
        "neuracore.data_daemon.communications_management.producer_channel.RingBuffer.create_shared",
        lambda size: shared_ring,
    )

    producer = ProducerChannel(
        chunk_size=2,
        recording_id="rec-1",
        ring_buffer_size=2048,
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
            dataset_id="dataset-1",
        )
        deadline = time.monotonic() + 1.0
        while len(shared_ring.writes) < 6 and time.monotonic() < deadline:
            time.sleep(0.02)
        stats = producer.get_transport_stats()
    finally:
        producer.stop_producer_channel()

    assert stats["channel_id"] == producer.channel_id
    assert stats["recording_id"] == "rec-1"
    assert stats["trace_id"] is not None
    assert stats["shared_ring_buffer_name"] == "test-shared"
    assert stats["shared_ring_buffer_size"] == 2048
    assert stats["send_queue_qsize"] == 0
    assert stats["send_queue_maxsize"] == 0
    assert stats["last_enqueued_sequence_number"] == 1
    assert stats["last_socket_sent_sequence_number"] == 1
    assert stats["pending_sequence_count"] == 0
    assert stats["sender_thread_alive"] is True
    assert stats["shared_ring_open_count"] == 1
    assert stats["shared_ring_write_count"] == 2
    assert stats["shared_ring_dispatch_count"] == 2
    assert stats["socket_send_count"] == 1
    assert stats["queue_put_count"] == 3
    assert stats["send_error_count"] == 0
    assert stats["last_send_error"] is None


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
    channel.set_ring_buffer(RingBuffer(size=4096))

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
    channel.ring_buffer.write(
        struct.pack(
            SHARED_RING_RECORD_HEADER_FORMAT,
            SHARED_RING_RECORD_MAGIC,
            len(metadata_bytes),
            len(payload),
        )
        + metadata_bytes
        + payload
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
