import base64
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


def test_trace_record_serialization_encodes_data() -> None:
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

    assert record.data == b"hello"

    as_map = record.to_dict()
    assert as_map["producer_id"] == "prod"
    assert as_map["trace_id"] == "trace"
    assert as_map["recording_id"] == "rec-1"
    assert as_map["dataset_id"] is None
    assert as_map["dataset_name"] is None
    assert as_map["robot_name"] is None
    assert as_map["robot_id"] is None
    assert as_map["data_type"] == DataType.CUSTOM_1D.value
    assert as_map["data_type_name"] == "custom_data"
    assert as_map["robot_instance"] == 0
    assert datetime.fromisoformat(as_map["received_at"])
    assert as_map["data"] == base64.b64encode(b"hello").decode("ascii")


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


def test_producer_open_ring_buffer_sends_payload() -> None:
    comm = DummyComm()
    producer = ProducerChannel(comm_manager=comm)

    assert comm.socket_requested is True
    producer.open_ring_buffer(size=2048)
    _wait_for_messages(comm, 1)

    assert len(comm.messages) == 1
    envelope = comm.messages[0]
    assert envelope.command == CommandType.OPEN_RING_BUFFER
    assert envelope.payload == {"open_ring_buffer": {"size": 2048}}


def test_producer_send_data_chunks_and_base64() -> None:
    comm = DummyComm()
    producer = ProducerChannel(comm_manager=comm, chunk_size=2, recording_id="rec-1")

    assert comm.socket_requested is True
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
    _wait_for_messages(comm, 2)

    assert len(comm.messages) == 2
    for idx, envelope in enumerate(comm.messages):
        assert envelope.command == CommandType.DATA_CHUNK
        payload = envelope.payload.get("data_chunk")
        assert payload["trace_id"] == producer.trace_id
        assert payload["chunk_index"] == idx
        assert payload["total_chunks"] == 2
        assert payload["dataset_id"] == "dataset-1"
        assert payload["dataset_name"] == "dataset"
        assert payload["robot_name"] == "robot"
        assert payload["robot_id"] == "robot-1"
        assert payload["robot_instance"] == 2
        assert payload["data_type"] == DataType.CUSTOM_1D.value
        decoded = base64.b64decode(payload["data"])
        assert decoded == (b"ab" if idx == 0 else b"cd")


def test_producer_send_data_parts_chunks_across_multiple_buffers() -> None:
    comm = DummyComm()
    producer = Producer(comm_manager=comm, chunk_size=3, recording_id="rec-1")

    assert comm.socket_requested is True
    producer.start_new_trace()
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
    _wait_for_messages(comm, 3)

    assert len(comm.messages) == 3
    decoded_chunks = [
        base64.b64decode(envelope.payload["data_chunk"]["data"])
        for envelope in comm.messages
    ]
    assert decoded_chunks == [b"abc", b"def", b"gh"]


def test_producer_sequences_follow_enqueue_order_under_concurrent_senders(
    monkeypatch,
) -> None:
    comm = DummyComm()
    producer = Producer(comm_manager=comm, recording_id="rec-1")

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

        # The second sender must not advance sequence numbering while the first
        # send is still blocked before entering the queue.
        assert producer.get_last_enqueued_sequence_number() == 1
        assert second_send_finished.is_set() is False

        allow_first_put.set()

        first_sender.join(timeout=5.0)
        second_sender.join(timeout=5.0)
        _wait_for_messages(comm, 2)
    finally:
        allow_first_put.set()
        producer.stop_producer()

    assert thread_errors == []
    assert [message.sequence_number for message in comm.messages] == [1, 2]


class DummyRecordingDiskManager:
    """Minimal recording disk manager for tests."""

    def enqueue(self, msg):
        pass


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
