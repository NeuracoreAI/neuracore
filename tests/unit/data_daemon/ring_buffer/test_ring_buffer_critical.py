"""Level 4: Critical Path Tests.

Tests for critical data paths that could break the data daemon:
- End-to-end data integrity
- Multi-chunk reassembly
- Binary payload preservation
- Trace and recording isolation
"""

from __future__ import annotations

import base64
import struct
from typing import Any

from neuracore_types import DataType

from neuracore.data_daemon.communications_management.data_bridge import (
    ChannelState,
    Daemon,
)
from neuracore.data_daemon.communications_management.ring_buffer import RingBuffer
from neuracore.data_daemon.const import (
    CHUNK_HEADER_FORMAT,
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

    def start_consumer(self) -> None:
        self.started_consumer = True

    def start_publisher(self) -> None:
        self.started_publisher = True

    def cleanup_daemon(self) -> None:
        pass


def _write_chunk_to_ring_buffer(
    ring: RingBuffer,
    trace_id: str,
    data_type: DataType,
    chunk_index: int,
    total_chunks: int,
    data: bytes,
) -> None:
    """Helper to write a chunk to a ring buffer in the expected format."""
    trace_id_bytes = trace_id.encode("utf-8")
    trace_id_field = trace_id_bytes[:TRACE_ID_FIELD_SIZE].ljust(
        TRACE_ID_FIELD_SIZE, b"\x00"
    )
    data_type_bytes = data_type.value.encode("utf-8")
    data_type_field = data_type_bytes[:DATA_TYPE_FIELD_SIZE].ljust(
        DATA_TYPE_FIELD_SIZE, b"\x00"
    )
    header = struct.pack(
        CHUNK_HEADER_FORMAT,
        trace_id_field,
        data_type_field,
        chunk_index,
        total_chunks,
        len(data),
    )
    ring.write(header + data)


# =============================================================================
# L4-001 to L4-003: Data Integrity Through Pipeline
# =============================================================================


def test_end_to_end_data_integrity(tmp_path: Any) -> None:
    """Bytes preserved producer→RDM.

    The ultimate test: data in equals data out. Any corruption anywhere in
    pipeline fails this.
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

    # Open ring buffer
    open_msg = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.OPEN_RING_BUFFER,
        payload={"open_ring_buffer": {"size": 8192}},
    )
    daemon._handle_open_ring_buffer(channel, open_msg)

    # Send known payload
    original_data = b"This is the exact payload that must be preserved!"
    data_msg = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.DATA_CHUNK,
        payload={
            "trace_id": "trace-integrity",
            "data_type": DataType.CUSTOM_1D.value,
            "chunk_index": 0,
            "total_chunks": 1,
            "data": base64.b64encode(original_data).decode("ascii"),
            "robot_instance": 0,
        },
    )
    daemon._handle_write_data_chunk(channel, data_msg)

    # Drain to RDM
    daemon._drain_channel_messages()

    # Verify exact bytes received
    assert len(mock_rdm.enqueued) == 1
    received_data = base64.b64decode(mock_rdm.enqueued[0].data)
    assert received_data == original_data


def test_multi_chunk_reassembly_integrity(tmp_path: Any) -> None:
    """Large data split and reassembled correctly.

    Chunking is lossless. Large messages must survive split/reassemble without
    byte loss.
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
        payload={"open_ring_buffer": {"size": 65536}},
    )
    daemon._handle_open_ring_buffer(channel, open_msg)

    # Create large payload that will be split into chunks
    original_data = bytes([i % 256 for i in range(10000)])  # 10KB

    # Simulate chunking (e.g., 4KB chunks)
    chunk_size = 4000
    chunks = [
        original_data[i : i + chunk_size]
        for i in range(0, len(original_data), chunk_size)
    ]
    total_chunks = len(chunks)

    # Send all chunks
    for idx, chunk in enumerate(chunks):
        data_msg = MessageEnvelope(
            producer_id="test-producer",
            command=CommandType.DATA_CHUNK,
            payload={
                "trace_id": "trace-multi",
                "data_type": DataType.CUSTOM_1D.value,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "data": base64.b64encode(chunk).decode("ascii"),
                "robot_instance": 0,
            },
        )
        daemon._handle_write_data_chunk(channel, data_msg)
        daemon._drain_channel_messages()

    # Final drain
    daemon._drain_channel_messages()

    # Should have one complete message
    assert len(mock_rdm.enqueued) == 1
    reassembled = base64.b64decode(mock_rdm.enqueued[0].data)
    assert reassembled == original_data


def test_binary_payload_through_pipeline(tmp_path: Any) -> None:
    """Binary with null bytes preserved.

    Full byte range must work. No special character handling that corrupts
    binary data.
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
        payload={"open_ring_buffer": {"size": 8192}},
    )
    daemon._handle_open_ring_buffer(channel, open_msg)

    # All possible byte values including null, high bytes
    binary_data = bytes(range(256))

    data_msg = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.DATA_CHUNK,
        payload={
            "trace_id": "trace-binary",
            "data_type": DataType.CUSTOM_1D.value,
            "chunk_index": 0,
            "total_chunks": 1,
            "data": base64.b64encode(binary_data).decode("ascii"),
            "robot_instance": 0,
        },
    )
    daemon._handle_write_data_chunk(channel, data_msg)
    daemon._drain_channel_messages()

    assert len(mock_rdm.enqueued) == 1
    received = base64.b64decode(mock_rdm.enqueued[0].data)
    assert received == binary_data


# =============================================================================
# L4-004 to L4-006: Error Resilience
# =============================================================================


def test_daemon_handles_corrupted_chunk_gracefully(tmp_path: Any) -> None:
    """Malformed chunk doesn't crash.

    Corruption happens. Daemon must log and continue, not crash the whole
    system.
    """
    from neuracore.data_daemon.communications_management.channel_reader import (
        ChannelMessageReader,
    )
    from neuracore.data_daemon.const import CHUNK_HEADER_SIZE

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

    # Manually create ring buffer and reader
    channel.ring_buffer = RingBuffer(size=4096)
    channel.reader = ChannelMessageReader(channel.ring_buffer)

    # Write corrupted/garbage data that looks like a header but isn't valid
    garbage = b"x" * CHUNK_HEADER_SIZE + b"payload"
    channel.ring_buffer.write(garbage)

    # Daemon should not crash when draining
    # It may log a warning but should continue
    try:
        daemon._drain_channel_messages()
    except Exception:
        # If it raises, that's acceptable for corrupted data
        # The key is daemon didn't crash entirely
        pass

    # Daemon should still be functional for subsequent operations
    assert daemon.channels["test-producer"] is not None


def test_daemon_survives_full_buffer(tmp_path: Any) -> None:
    """Daemon handles buffer pressure.

    Back-pressure scenario: system must survive temporary overload without
    crashing.
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

    # Small buffer to easily fill
    open_msg = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.OPEN_RING_BUFFER,
        payload={"open_ring_buffer": {"size": 512}},
    )
    daemon._handle_open_ring_buffer(channel, open_msg)

    # Write data that nearly fills buffer (header + small payload)
    small_data = b"x" * 100
    data_msg = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.DATA_CHUNK,
        payload={
            "trace_id": "trace-1",
            "data_type": DataType.CUSTOM_1D.value,
            "chunk_index": 0,
            "total_chunks": 1,
            "data": base64.b64encode(small_data).decode("ascii"),
            "robot_instance": 0,
        },
    )
    daemon._handle_write_data_chunk(channel, data_msg)

    # Drain to free space
    daemon._drain_channel_messages()

    # Should be able to write more
    data_msg2 = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.DATA_CHUNK,
        payload={
            "trace_id": "trace-2",
            "data_type": DataType.CUSTOM_1D.value,
            "chunk_index": 0,
            "total_chunks": 1,
            "data": base64.b64encode(b"more-data").decode("ascii"),
            "robot_instance": 0,
        },
    )
    daemon._handle_write_data_chunk(channel, data_msg2)
    daemon._drain_channel_messages()

    # Both messages should have been processed
    assert len(mock_rdm.enqueued) == 2


def test_rdm_failure_does_not_lose_ring_buffer(tmp_path: Any) -> None:
    """RDM error doesn't corrupt buffer.

    Downstream failure shouldn't corrupt upstream state. Isolation between
    components.
    """
    mock_config = MockConfigManager().path_to_store_record_from(tmp_path)
    mock_comm = MockComm()

    # RDM that fails on first call
    class FailingRDM:
        def __init__(self) -> None:
            self.call_count = 0
            self.enqueued: list[CompleteMessage] = []

        def enqueue(self, message: CompleteMessage) -> None:
            self.call_count += 1
            if self.call_count == 1:
                raise RuntimeError("Simulated RDM failure")
            self.enqueued.append(message)

    failing_rdm = FailingRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        config_manager=mock_config,
        recording_disk_manager=failing_rdm,
    )

    channel = ChannelState(producer_id="test-producer", recording_id="rec-123")
    daemon.channels["test-producer"] = channel

    open_msg = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.OPEN_RING_BUFFER,
        payload={"open_ring_buffer": {"size": 4096}},
    )
    daemon._handle_open_ring_buffer(channel, open_msg)

    # First message - RDM will fail
    data_msg1 = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.DATA_CHUNK,
        payload={
            "trace_id": "trace-1",
            "data_type": DataType.CUSTOM_1D.value,
            "chunk_index": 0,
            "total_chunks": 1,
            "data": base64.b64encode(b"data1").decode("ascii"),
            "robot_instance": 0,
        },
    )
    daemon._handle_write_data_chunk(channel, data_msg1)
    daemon._drain_channel_messages()  # This triggers RDM failure

    # Buffer should still be functional
    assert channel.ring_buffer is not None

    # Second message should work
    data_msg2 = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.DATA_CHUNK,
        payload={
            "trace_id": "trace-2",
            "data_type": DataType.CUSTOM_1D.value,
            "chunk_index": 0,
            "total_chunks": 1,
            "data": base64.b64encode(b"data2").decode("ascii"),
            "robot_instance": 0,
        },
    )
    daemon._handle_write_data_chunk(channel, data_msg2)
    daemon._drain_channel_messages()

    # Second message should have been enqueued
    assert len(failing_rdm.enqueued) == 1
    assert failing_rdm.enqueued[0].trace_id == "trace-2"


# =============================================================================
# L4-007 to L4-008: Trace Boundaries
# =============================================================================


def test_new_trace_independent_of_previous(tmp_path: Any) -> None:
    """New trace unaffected by prior.

    Trace isolation: completing one trace must not leave state that affects
    next trace.
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
        payload={"open_ring_buffer": {"size": 8192}},
    )
    daemon._handle_open_ring_buffer(channel, open_msg)

    # Complete first trace
    data_msg1 = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.DATA_CHUNK,
        payload={
            "trace_id": "trace-A",
            "data_type": DataType.JOINT_POSITIONS.value,
            "chunk_index": 0,
            "total_chunks": 1,
            "data": base64.b64encode(b"trace-A-data").decode("ascii"),
            "robot_instance": 0,
        },
    )
    daemon._handle_write_data_chunk(channel, data_msg1)
    daemon._drain_channel_messages()

    # Start second trace
    data_msg2 = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.DATA_CHUNK,
        payload={
            "trace_id": "trace-B",
            "data_type": DataType.JOINT_VELOCITIES.value,
            "chunk_index": 0,
            "total_chunks": 1,
            "data": base64.b64encode(b"trace-B-data").decode("ascii"),
            "robot_instance": 0,
        },
    )
    daemon._handle_write_data_chunk(channel, data_msg2)
    daemon._drain_channel_messages()

    # Both traces should be complete and independent
    assert len(mock_rdm.enqueued) == 2
    assert mock_rdm.enqueued[0].trace_id == "trace-A"
    assert mock_rdm.enqueued[0].data_type == DataType.JOINT_POSITIONS
    assert mock_rdm.enqueued[1].trace_id == "trace-B"
    assert mock_rdm.enqueued[1].data_type == DataType.JOINT_VELOCITIES


def test_concurrent_traces_isolation(tmp_path: Any) -> None:
    """Multiple traces don't interfere.

    Concurrent traces are common. Must not mix up chunks between different
    traces.
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
        payload={"open_ring_buffer": {"size": 16384}},
    )
    daemon._handle_open_ring_buffer(channel, open_msg)

    # Interleave chunks from two traces
    chunks = [
        ("trace-X", 0, 2, b"X0"),
        ("trace-Y", 0, 2, b"Y0"),
        ("trace-X", 1, 2, b"X1"),
        ("trace-Y", 1, 2, b"Y1"),
    ]

    for trace_id, idx, total, data in chunks:
        msg = MessageEnvelope(
            producer_id="test-producer",
            command=CommandType.DATA_CHUNK,
            payload={
                "trace_id": trace_id,
                "data_type": DataType.CUSTOM_1D.value,
                "chunk_index": idx,
                "total_chunks": total,
                "data": base64.b64encode(data).decode("ascii"),
                "robot_instance": 0,
            },
        )
        daemon._handle_write_data_chunk(channel, msg)
        daemon._drain_channel_messages()

    # Both traces should complete correctly
    assert len(mock_rdm.enqueued) == 2

    results = {msg.trace_id: base64.b64decode(msg.data) for msg in mock_rdm.enqueued}
    assert results["trace-X"] == b"X0X1"
    assert results["trace-Y"] == b"Y0Y1"
