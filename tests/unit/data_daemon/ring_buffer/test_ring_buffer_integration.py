"""Level 3: Ring Buffer Interface Integration Tests.

Tests for ring buffer integration with Daemon components:
- Daemon write path (buffer creation, chunk writing)
- Header construction (padding, truncation)
- Multi-channel isolation
- Channel lifecycle
"""

from __future__ import annotations

import struct
from typing import Any

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
from tests.unit.data_daemon.helpers import MockConfigManager


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


# =============================================================================
# L3-001 to L3-004: Daemon → RingBuffer Write Path
# =============================================================================


def test_daemon_creates_ring_buffer_on_open(tmp_path: Any) -> None:
    """OPEN_RING_BUFFER creates buffer.

    Buffer created on-demand when producer requests it. Lazy initialization
    saves memory.
    """
    mock_config = MockConfigManager().path_to_store_record_from(tmp_path)
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        config_manager=mock_config,
        recording_disk_manager=mock_rdm,
    )

    channel = ChannelState(producer_id="test-producer", recording_id="rec-123")
    daemon.channels["test-producer"] = channel

    # Initially no ring buffer
    assert channel.ring_buffer is None

    # Send OPEN_RING_BUFFER command
    message = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.OPEN_RING_BUFFER,
        payload={"open_ring_buffer": {"size": 4096}},
    )
    daemon._handle_open_ring_buffer(channel, message)

    # Now buffer should exist
    assert channel.ring_buffer is not None
    assert channel.ring_buffer.size == 4096
    assert channel.reader is not None


def test_daemon_writes_chunk_to_ring_buffer(tmp_path: Any) -> None:
    """DATA_CHUNK writes to buffer.

    Data path works: producer sends chunk, daemon writes to buffer. Core
    functionality.
    """
    mock_config = MockConfigManager().path_to_store_record_from(tmp_path)
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        config_manager=mock_config,
        recording_disk_manager=mock_rdm,
    )

    channel = ChannelState(producer_id="test-producer", recording_id="rec-123")
    daemon.channels["test-producer"] = channel

    # Open ring buffer first
    open_msg = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.OPEN_RING_BUFFER,
        payload={"open_ring_buffer": {"size": 4096}},
    )
    daemon._handle_open_ring_buffer(channel, open_msg)

    initial_available = channel.ring_buffer.available()
    assert initial_available == 0

    # Send data chunk
    import base64

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

    # Buffer should now have data
    assert channel.ring_buffer.available() > 0


def test_daemon_writes_correct_header_format(tmp_path: Any) -> None:
    """Header format matches spec.

    Daemon must write headers that reader can parse. Format mismatch breaks
    everything.
    """
    mock_config = MockConfigManager().path_to_store_record_from(tmp_path)
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        config_manager=mock_config,
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

    import base64

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

    # Peek the header from buffer
    header_bytes = channel.ring_buffer.peek(CHUNK_HEADER_SIZE)
    assert header_bytes is not None

    # Should be able to parse
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


def test_daemon_handles_multiple_channels(tmp_path: Any) -> None:
    """Separate buffers per producer.

    Channel isolation: producer A's data never appears in producer B's buffer.
    """
    mock_config = MockConfigManager().path_to_store_record_from(tmp_path)
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        config_manager=mock_config,
        recording_disk_manager=mock_rdm,
    )

    # Create two channels
    channel_a = ChannelState(producer_id="producer-A", recording_id="rec-A")
    channel_b = ChannelState(producer_id="producer-B", recording_id="rec-B")
    daemon.channels["producer-A"] = channel_a
    daemon.channels["producer-B"] = channel_b

    # Open buffers for both
    for channel, producer_id in [(channel_a, "producer-A"), (channel_b, "producer-B")]:
        open_msg = MessageEnvelope(
            producer_id=producer_id,
            command=CommandType.OPEN_RING_BUFFER,
            payload={"open_ring_buffer": {"size": 2048}},
        )
        daemon._handle_open_ring_buffer(channel, open_msg)

    import base64

    # Write to channel A only
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

    # Channel A should have data
    assert channel_a.ring_buffer.available() > 0

    # Channel B should be empty
    assert channel_b.ring_buffer.available() == 0


# =============================================================================
# L3-005 to L3-008: Header Construction and Channel Lifecycle
# =============================================================================


def test_daemon_header_trace_id_padding(tmp_path: Any) -> None:
    """trace_id padded to 36 bytes.

    Short trace_ids must be null-padded to fill fixed field size.
    """
    mock_config = MockConfigManager().path_to_store_record_from(tmp_path)
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        config_manager=mock_config,
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

    import base64

    # Short trace_id
    data_msg = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.DATA_CHUNK,
        payload={
            "trace_id": "short",  # Only 5 bytes
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

    # Should be padded to 36 bytes
    assert len(raw_trace_id) == TRACE_ID_FIELD_SIZE
    assert raw_trace_id.rstrip(b"\x00") == b"short"


def test_daemon_header_data_type_padding(tmp_path: Any) -> None:
    """data_type padded to 32 bytes.

    Same padding requirement for data_type field. Fixed format requires
    padding.
    """
    mock_config = MockConfigManager().path_to_store_record_from(tmp_path)
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        config_manager=mock_config,
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

    import base64

    data_msg = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.DATA_CHUNK,
        payload={
            "trace_id": "trace-1",
            "data_type": DataType.CUSTOM_1D.value,  # "custom_1d"
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

    # Should be padded to 32 bytes
    assert len(raw_data_type) == DATA_TYPE_FIELD_SIZE


def test_daemon_header_with_long_trace_id(tmp_path: Any) -> None:
    """Long trace_id truncated.

    Truncation must happen silently. Long IDs shouldn't corrupt header or
    crash.
    """
    mock_config = MockConfigManager().path_to_store_record_from(tmp_path)
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        config_manager=mock_config,
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

    import base64

    long_trace_id = "x" * 100  # Much longer than 36 bytes
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

    # Should not crash, and header should be able to parse
    header = channel.ring_buffer.peek(CHUNK_HEADER_SIZE)
    assert header is not None

    # Verify it can be unpacked
    unpacked = struct.unpack(CHUNK_HEADER_FORMAT, header)
    raw_trace_id = unpacked[0]
    assert len(raw_trace_id) == TRACE_ID_FIELD_SIZE


def test_new_channel_fresh_buffer(tmp_path: Any) -> None:
    """New channel gets clean buffer.

    Each channel starts fresh. No cross-contamination from previous channels.
    """
    mock_config = MockConfigManager().path_to_store_record_from(tmp_path)
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        config_manager=mock_config,
        recording_disk_manager=mock_rdm,
    )

    # Create and setup first channel
    channel1 = ChannelState(producer_id="producer-1", recording_id="rec-1")
    daemon.channels["producer-1"] = channel1

    open_msg1 = MessageEnvelope(
        producer_id="producer-1",
        command=CommandType.OPEN_RING_BUFFER,
        payload={"open_ring_buffer": {"size": 2048}},
    )
    daemon._handle_open_ring_buffer(channel1, open_msg1)

    import base64

    # Write some data to first channel
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

    # Now create a new channel
    channel2 = ChannelState(producer_id="producer-2", recording_id="rec-2")
    daemon.channels["producer-2"] = channel2

    open_msg2 = MessageEnvelope(
        producer_id="producer-2",
        command=CommandType.OPEN_RING_BUFFER,
        payload={"open_ring_buffer": {"size": 2048}},
    )
    daemon._handle_open_ring_buffer(channel2, open_msg2)

    # New channel should be completely fresh
    assert channel2.ring_buffer is not None
    assert channel2.ring_buffer.available() == 0
    assert channel2.ring_buffer.write_pos == 0
    assert channel2.ring_buffer.read_pos == 0
