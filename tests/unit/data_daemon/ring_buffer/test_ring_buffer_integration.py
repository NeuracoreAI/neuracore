"""Level 3: Ring Buffer Interface Integration Tests.

Tests for ring buffer integration with Daemon components:
- Daemon write path (buffer creation, chunk writing)
- Header construction (padding, truncation)
- Multi-channel isolation
- Channel lifecycle
"""

from __future__ import annotations

import base64
import struct

from neuracore_types import DataType

from neuracore.data_daemon.communications_management.data_bridge import (
    ChannelState,
    Daemon,
)
from neuracore.data_daemon.const import (
    CHUNK_HEADER_FORMAT,
    CHUNK_HEADER_SIZE,
    DATA_TYPE_FIELD_SIZE,
    TRACE_ID_FIELD_SIZE,
)
from neuracore.data_daemon.models import CommandType, CompleteMessage, MessageEnvelope


class MockRDM:
    """Mock RecordingDiskManager to capture enqueued messages."""

    def __init__(self) -> None:
        self.enqueued: list[CompleteMessage] = []

    def enqueue(self, message: CompleteMessage) -> None:
        self.enqueued.append(message)


class MockComm:
    """Mock CommunicationsManager for testing."""

    def __init__(self) -> None:
        self.started_consumer = False
        self.started_publisher = False
        self.cleaned = False

    def start_consumer(self) -> None:
        self.started_consumer = True

    def start_publisher(self) -> None:
        self.started_publisher = True

    def cleanup_daemon(self) -> None:
        self.cleaned = True

    def receive_message(self) -> MessageEnvelope:
        raise KeyboardInterrupt()


def test_daemon_creates_ring_buffer_on_open() -> None:
    """OPEN_RING_BUFFER creates buffer.

    Buffer created on-demand when producer requests it. Lazy initialization
    saves memory.
    """
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        recording_disk_manager=mock_rdm,
    )

    channel = ChannelState(producer_id="test-producer", recording_id="rec-123")
    daemon.channels["test-producer"] = channel

    assert channel.ring_buffer is None

    message = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.OPEN_RING_BUFFER,
        payload={"open_ring_buffer": {"size": 4096}},
    )
    daemon._handle_open_ring_buffer(channel, message)

    assert channel.ring_buffer is not None
    assert channel.ring_buffer.size == 4096
    assert channel.reader is not None


def test_daemon_writes_chunk_to_ring_buffer() -> None:
    """DATA_CHUNK writes to buffer.

    Data path works: producer sends chunk, daemon writes to buffer. Core
    functionality.
    """
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        recording_disk_manager=mock_rdm,
    )

    channel = ChannelState(producer_id="test-producer", recording_id="rec-123")
    daemon.channels["test-producer"] = channel

    open_msg = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.OPEN_RING_BUFFER,
        payload={"open_ring_buffer": {"size": 4096}},
    )
    daemon._handle_open_ring_buffer(channel, open_msg)

    initial_available = channel.ring_buffer.available()
    assert initial_available == 0

    data = b"test-payload-data"
    data_msg = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.DATA_CHUNK,
        payload={
            "trace_id": "trace-123",
            "data_type": DataType.CUSTOM_1D.value,
            "chunk_index": 0,
            "total_chunks": 1,
            "data": base64.b64encode(data).decode("ascii"),
            "robot_instance": 0,
        },
    )
    daemon._handle_write_data_chunk(channel, data_msg)

    assert channel.ring_buffer.available() > 0


def test_daemon_writes_correct_header_format() -> None:
    """Header format matches spec.

    Daemon must write headers that reader can parse. Format mismatch breaks
    everything.
    """
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        recording_disk_manager=mock_rdm,
    )

    channel = ChannelState(producer_id="test-producer", recording_id="rec-123")
    daemon.channels["test-producer"] = channel

    open_msg = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.OPEN_RING_BUFFER,
        payload={"open_ring_buffer": {"size": 4096}},
    )
    daemon._handle_open_ring_buffer(channel, open_msg)

    data = b"payload"
    data_msg = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.DATA_CHUNK,
        payload={
            "trace_id": "trace-abc",
            "data_type": DataType.JOINT_POSITIONS.value,
            "chunk_index": 0,
            "total_chunks": 1,
            "data": base64.b64encode(data).decode("ascii"),
            "robot_instance": 0,
        },
    )
    daemon._handle_write_data_chunk(channel, data_msg)

    header_bytes = channel.ring_buffer.peek(CHUNK_HEADER_SIZE)
    assert header_bytes is not None

    raw_trace_id, raw_data_type, chunk_index, total_chunks, chunk_len = struct.unpack(
        CHUNK_HEADER_FORMAT, header_bytes
    )

    trace_id = raw_trace_id.rstrip(b"\x00").decode("utf-8")
    data_type_str = raw_data_type.rstrip(b"\x00").decode("utf-8")

    assert trace_id == "trace-abc"
    assert data_type_str == DataType.JOINT_POSITIONS.value
    assert chunk_index == 0
    assert total_chunks == 1
    assert chunk_len == len(data)


def test_daemon_handles_multiple_channels() -> None:
    """Separate buffers per producer.

    Channel isolation: producer A's data never appears in producer B's buffer.
    """
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        recording_disk_manager=mock_rdm,
    )

    channel_a = ChannelState(producer_id="producer-A", recording_id="rec-A")
    channel_b = ChannelState(producer_id="producer-B", recording_id="rec-B")
    daemon.channels["producer-A"] = channel_a
    daemon.channels["producer-B"] = channel_b

    for channel, producer_id in [(channel_a, "producer-A"), (channel_b, "producer-B")]:
        open_msg = MessageEnvelope(
            producer_id=producer_id,
            command=CommandType.OPEN_RING_BUFFER,
            payload={"open_ring_buffer": {"size": 2048}},
        )
        daemon._handle_open_ring_buffer(channel, open_msg)

    data_msg = MessageEnvelope(
        producer_id="producer-A",
        command=CommandType.DATA_CHUNK,
        payload={
            "trace_id": "trace-A",
            "data_type": DataType.CUSTOM_1D.value,
            "chunk_index": 0,
            "total_chunks": 1,
            "data": base64.b64encode(b"data-for-A").decode("ascii"),
            "robot_instance": 0,
        },
    )
    daemon._handle_write_data_chunk(channel_a, data_msg)

    assert channel_a.ring_buffer.available() > 0

    assert channel_b.ring_buffer.available() == 0


def test_daemon_header_trace_id_padding() -> None:
    """trace_id padded to 36 bytes.

    Short trace_ids must be null-padded to fill fixed field size.
    """
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        recording_disk_manager=mock_rdm,
    )

    channel = ChannelState(producer_id="test-producer", recording_id="rec-123")
    daemon.channels["test-producer"] = channel

    open_msg = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.OPEN_RING_BUFFER,
        payload={"open_ring_buffer": {"size": 4096}},
    )
    daemon._handle_open_ring_buffer(channel, open_msg)

    data_msg = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.DATA_CHUNK,
        payload={
            "trace_id": "short",
            "data_type": DataType.CUSTOM_1D.value,
            "chunk_index": 0,
            "total_chunks": 1,
            "data": base64.b64encode(b"x").decode("ascii"),
            "robot_instance": 0,
        },
    )
    daemon._handle_write_data_chunk(channel, data_msg)

    header = channel.ring_buffer.peek(CHUNK_HEADER_SIZE)
    raw_trace_id = header[:TRACE_ID_FIELD_SIZE]

    assert len(raw_trace_id) == TRACE_ID_FIELD_SIZE
    assert raw_trace_id.rstrip(b"\x00") == b"short"


def test_daemon_header_data_type_padding() -> None:
    """data_type padded to 32 bytes.

    Same padding requirement for data_type field. Fixed format requires
    padding.
    """
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        recording_disk_manager=mock_rdm,
    )

    channel = ChannelState(producer_id="test-producer", recording_id="rec-123")
    daemon.channels["test-producer"] = channel

    open_msg = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.OPEN_RING_BUFFER,
        payload={"open_ring_buffer": {"size": 4096}},
    )
    daemon._handle_open_ring_buffer(channel, open_msg)

    data_msg = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.DATA_CHUNK,
        payload={
            "trace_id": "trace-1",
            "data_type": DataType.CUSTOM_1D.value,
            "chunk_index": 0,
            "total_chunks": 1,
            "data": base64.b64encode(b"x").decode("ascii"),
            "robot_instance": 0,
        },
    )
    daemon._handle_write_data_chunk(channel, data_msg)

    header = channel.ring_buffer.peek(CHUNK_HEADER_SIZE)
    raw_data_type = header[
        TRACE_ID_FIELD_SIZE : TRACE_ID_FIELD_SIZE + DATA_TYPE_FIELD_SIZE
    ]

    assert len(raw_data_type) == DATA_TYPE_FIELD_SIZE


def test_daemon_header_with_long_trace_id() -> None:
    """Long trace_id truncated.

    Truncation must happen silently. Long IDs shouldn't corrupt header or
    crash.
    """
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        recording_disk_manager=mock_rdm,
    )

    channel = ChannelState(producer_id="test-producer", recording_id="rec-123")
    daemon.channels["test-producer"] = channel

    open_msg = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.OPEN_RING_BUFFER,
        payload={"open_ring_buffer": {"size": 4096}},
    )
    daemon._handle_open_ring_buffer(channel, open_msg)

    long_trace_id = "x" * 100
    data_msg = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.DATA_CHUNK,
        payload={
            "trace_id": long_trace_id,
            "data_type": DataType.CUSTOM_1D.value,
            "chunk_index": 0,
            "total_chunks": 1,
            "data": base64.b64encode(b"payload").decode("ascii"),
            "robot_instance": 0,
        },
    )
    daemon._handle_write_data_chunk(channel, data_msg)

    header = channel.ring_buffer.peek(CHUNK_HEADER_SIZE)
    assert header is not None

    unpacked = struct.unpack(CHUNK_HEADER_FORMAT, header)
    raw_trace_id = unpacked[0]
    assert len(raw_trace_id) == TRACE_ID_FIELD_SIZE


def test_new_channel_fresh_buffer() -> None:
    """New channel gets clean buffer.

    Each channel starts fresh. No cross-contamination from previous channels.
    """
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        recording_disk_manager=mock_rdm,
    )

    channel1 = ChannelState(producer_id="producer-1", recording_id="rec-1")
    daemon.channels["producer-1"] = channel1

    open_msg1 = MessageEnvelope(
        producer_id="producer-1",
        command=CommandType.OPEN_RING_BUFFER,
        payload={"open_ring_buffer": {"size": 2048}},
    )
    daemon._handle_open_ring_buffer(channel1, open_msg1)

    data_msg = MessageEnvelope(
        producer_id="producer-1",
        command=CommandType.DATA_CHUNK,
        payload={
            "trace_id": "trace-1",
            "data_type": DataType.CUSTOM_1D.value,
            "chunk_index": 0,
            "total_chunks": 1,
            "data": base64.b64encode(b"channel1-data").decode("ascii"),
            "robot_instance": 0,
        },
    )
    daemon._handle_write_data_chunk(channel1, data_msg)

    channel2 = ChannelState(producer_id="producer-2", recording_id="rec-2")
    daemon.channels["producer-2"] = channel2

    open_msg2 = MessageEnvelope(
        producer_id="producer-2",
        command=CommandType.OPEN_RING_BUFFER,
        payload={"open_ring_buffer": {"size": 2048}},
    )
    daemon._handle_open_ring_buffer(channel2, open_msg2)

    assert channel2.ring_buffer is not None
    assert channel2.ring_buffer.available() == 0
    assert channel2.ring_buffer.write_pos == 0
    assert channel2.ring_buffer.read_pos == 0
