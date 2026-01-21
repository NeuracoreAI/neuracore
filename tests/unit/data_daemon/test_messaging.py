import base64
import struct
from datetime import datetime, timedelta, timezone

from neuracore_types import DataType

from neuracore.data_daemon.communications_management.channel_reader import (
    ChannelMessageReader,
)
from neuracore.data_daemon.communications_management.data_bridge import (
    ChannelState,
    Daemon,
)
from neuracore.data_daemon.communications_management.producer import Producer
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

    def send_message(self, socket, message):
        self.messages.append(message)

    def cleanup_producer(self):
        self.cleaned = True


def test_producer_open_ring_buffer_sends_payload() -> None:
    comm = DummyComm()
    producer = Producer(comm_manager=comm)

    assert comm.socket_requested is True
    producer.open_ring_buffer(size=2048)

    assert len(comm.messages) == 1
    envelope = comm.messages[0]
    assert envelope.command == CommandType.OPEN_RING_BUFFER
    assert envelope.payload == {"open_ring_buffer": {"size": 2048}}


def test_producer_send_data_chunks_and_base64() -> None:
    comm = DummyComm()
    producer = Producer(comm_manager=comm, chunk_size=2, recording_id="rec-1")

    assert comm.socket_requested is True
    producer.send_data(
        b"abcd",
        trace_id="7",
        data_type=DataType.CUSTOM_1D,
        data_type_name="custom",
        robot_instance=2,
        robot_id="robot-1",
        robot_name="robot",
        dataset_id="dataset-1",
        dataset_name="dataset",
    )

    assert len(comm.messages) == 2
    for idx, envelope in enumerate(comm.messages):
        assert envelope.command == CommandType.DATA_CHUNK
        payload = envelope.payload
        assert payload["trace_id"] == "7"
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


def test_cleanup_removes_channel_without_heartbeat() -> None:
    daemon = Daemon()
    channel = ChannelState(
        producer_id="stale",
        last_heartbeat=datetime.now(timezone.utc)
        - timedelta(seconds=HEARTBEAT_TIMEOUT_SECS + 1),
    )
    daemon.channels["stale"] = channel

    daemon._cleanup_expired_channels()

    assert "stale" not in daemon.channels


def test_cleanup_keeps_recent_channel() -> None:
    daemon = Daemon()
    channel = ChannelState(
        producer_id="active",
        last_heartbeat=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    daemon.channels["active"] = channel

    daemon._cleanup_expired_channels()

    assert "active" in daemon.channels
