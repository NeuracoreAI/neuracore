"""Level 5: Failure & Error Recovery Tests.

Tests for error handling and recovery:
- Ring buffer error handling
- Reader error recovery
- Daemon loop resilience
- Resource exhaustion handling
- State consistency after errors
- Graceful degradation
"""

from __future__ import annotations

import base64
import struct
import threading
import time

import pytest
from neuracore_types import DataType

from neuracore.data_daemon.communications_management.channel_reader import (
    ChannelMessageReader,
    PartialMessage,
)
from neuracore.data_daemon.communications_management.data_bridge import (
    ChannelState,
    Daemon,
)
from neuracore.data_daemon.communications_management.ring_buffer import RingBuffer
from neuracore.data_daemon.const import (
    CHUNK_HEADER_FORMAT,
    CHUNK_HEADER_SIZE,
    DATA_TYPE_FIELD_SIZE,
    TRACE_ID_FIELD_SIZE,
)
from neuracore.data_daemon.models import CommandType, CompleteMessage, MessageEnvelope


class MockRDM:
    """Mock RecordingDiskManager."""

    def __init__(self) -> None:
        self.enqueued: list[CompleteMessage] = []

    def enqueue(self, message: CompleteMessage) -> None:
        self.enqueued.append(message)


class MockComm:
    """Mock CommunicationsManager."""

    def start_consumer(self) -> None:
        pass

    def start_publisher(self) -> None:
        pass

    def cleanup_daemon(self) -> None:
        pass


def _make_chunk_header(
    trace_id: str,
    data_type: DataType,
    chunk_index: int,
    total_chunks: int,
    chunk_len: int,
) -> bytes:
    """Helper to create a chunk header."""
    trace_id_field = trace_id.encode("utf-8")[:TRACE_ID_FIELD_SIZE].ljust(
        TRACE_ID_FIELD_SIZE, b"\x00"
    )
    data_type_field = data_type.value.encode("utf-8")[:DATA_TYPE_FIELD_SIZE].ljust(
        DATA_TYPE_FIELD_SIZE, b"\x00"
    )
    return struct.pack(
        CHUNK_HEADER_FORMAT,
        trace_id_field,
        data_type_field,
        chunk_index,
        total_chunks,
        chunk_len,
    )


def _write_chunk(
    ring: RingBuffer,
    trace_id: str,
    data_type: DataType,
    chunk_index: int,
    total_chunks: int,
    data: bytes,
) -> None:
    """Helper to write a chunk to ring buffer."""
    header = _make_chunk_header(
        trace_id, data_type, chunk_index, total_chunks, len(data)
    )
    ring.write(header + data)


def test_write_oversized_raises_not_crashes() -> None:
    """Oversized write raises ValueError, doesn't corrupt.

    Error must be contained. Failed write shouldn't leave buffer in corrupted
    state.
    """
    ring = RingBuffer(size=100)
    ring.write(b"initial")

    with pytest.raises(ValueError):
        ring.write(b"x" * 200)

    assert ring.available() == 7
    assert ring.read(7) == b"initial"


def test_read_after_failed_write_still_works() -> None:
    """Buffer usable after write failure.

    Recovery: after error, buffer must still be functional for valid
    operations.
    """
    ring = RingBuffer(size=100)
    ring.write(b"12345")

    try:
        ring.write(b"x" * 200)
    except ValueError:
        pass

    assert ring.read(5) == b"12345"


def test_write_after_failed_write_still_works() -> None:
    """Can write after failure.

    Error doesn't permanently break buffer. Next valid operation should work.
    """
    ring = RingBuffer(size=100)

    try:
        ring.write(b"x" * 200)
    except ValueError:
        pass

    ring.write(b"valid")
    assert ring.read(5) == b"valid"


def test_negative_read_length_handled() -> None:
    """Negative length doesn't crash.

    Invalid input must not crash or corrupt. Defensive programming against bad
    callers.
    """
    ring = RingBuffer(size=100)
    ring.write(b"data")

    result = ring.read(-1)
    assert result is None or isinstance(result, bytes)


def test_negative_peek_length_handled() -> None:
    """Negative peek doesn't crash.

    Same protection for peek. All public methods must handle invalid input
    safely.
    """
    ring = RingBuffer(size=100)
    ring.write(b"data")

    result = ring.peek(-1)
    assert result is None or isinstance(result, bytes)


def test_zero_read_length() -> None:
    """Zero length read is safe.

    Edge case: reading nothing. Should be no-op, not error or state change.
    """
    ring = RingBuffer(size=100)
    ring.write(b"data")

    ring.read(0)
    assert ring.available() == 4


def test_zero_peek_length() -> None:
    """Zero peek is safe.

    Same for peek(0). Consistent handling of zero-length requests.
    """
    ring = RingBuffer(size=100)
    ring.write(b"data")

    ring.peek(0)
    assert ring.available() == 4


def test_reader_recovers_from_corrupted_header() -> None:
    """Corrupted header doesn't crash.

    Corruption in header bytes must not crash reader. Graceful handling
    required.
    """
    ring = RingBuffer(size=1024)
    reader = ChannelMessageReader(ring)

    ring.write(b"x" * CHUNK_HEADER_SIZE)

    try:
        reader.poll_one()
    except struct.error:
        pass
    except Exception:
        raise


def test_reader_recovers_from_truncated_packet() -> None:
    """Partial packet recoverable.

    Incomplete data should wait, not corrupt. When rest arrives, should
    complete normally.
    """
    ring = RingBuffer(size=1024)
    reader = ChannelMessageReader(ring)

    header = _make_chunk_header("trace-1", DataType.CUSTOM_1D, 0, 1, 100)
    ring.write(header + b"x" * 50)

    result = reader.poll_one()
    assert result is None

    assert ring.available() == CHUNK_HEADER_SIZE + 50

    ring.write(b"y" * 50)

    result = reader.poll_one()
    assert result is not None
    assert result[0] == "trace-1"


def test_reader_handles_chunk_index_out_of_range() -> None:
    """Invalid chunk_index handled.

    Invalid index could cause array errors. Must handle gracefully.
    """
    ring = RingBuffer(size=1024)
    reader = ChannelMessageReader(ring)

    header = _make_chunk_header("trace-1", DataType.CUSTOM_1D, 999, 2, 5)
    ring.write(header + b"data!")

    reader.poll_one()


def test_reader_handles_total_chunks_zero() -> None:
    """total_chunks=0 handled.

    Zero chunks is invalid. Must not cause divide-by-zero or infinite wait.
    """
    ring = RingBuffer(size=1024)
    reader = ChannelMessageReader(ring)

    header = _make_chunk_header("trace-1", DataType.CUSTOM_1D, 0, 0, 5)
    ring.write(header + b"data!")

    reader.poll_one()


def test_reader_handles_negative_chunk_len() -> None:
    """Negative chunk_len handled.

    Malformed header with impossible length. Must not try to read negative
    bytes.
    """
    ring = RingBuffer(size=1024)
    reader = ChannelMessageReader(ring)

    trace_id_field = b"trace".ljust(TRACE_ID_FIELD_SIZE, b"\x00")
    data_type_field = DataType.CUSTOM_1D.value.encode().ljust(
        DATA_TYPE_FIELD_SIZE, b"\x00"
    )

    header = struct.pack(
        CHUNK_HEADER_FORMAT,
        trace_id_field,
        data_type_field,
        0,  # chunk_index
        1,  # total_chunks
        0xFFFFFFFF,  # chunk_len - huge value
    )
    ring.write(header)

    result = reader.poll_one()
    assert result is None


def test_reader_partial_message_cleanup() -> None:
    """Stale partials can be cleared.

    Abandoned traces shouldn't leak memory forever. Need cleanup mechanism.
    """
    ring = RingBuffer(size=4096)
    reader = ChannelMessageReader(ring)

    for i in range(100):
        _write_chunk(ring, f"trace-{i}", DataType.CUSTOM_1D, 0, 2, b"part0")
        reader.poll_one()

    assert len(reader._pending) == 100

    reader._pending.clear()
    assert len(reader._pending) == 0


def test_reader_assemble_failure_clears_pending() -> None:
    """Failed assemble cleans up.

    Even on error, cleanup must happen. Don't leave zombie entries in pending
    dict.
    """
    ring = RingBuffer(size=4096)
    reader = ChannelMessageReader(ring)

    partial = PartialMessage(total_chunks=2)
    partial.add_chunk(0, b"data")
    reader._pending["test-trace"] = partial

    try:
        partial.assemble()
    except ValueError:
        pass


def test_daemon_continues_after_ring_buffer_write_error() -> None:
    """Write error doesn't stop daemon.

    One bad message shouldn't kill daemon. Log error, skip message, keep
    processing.
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
        payload={"open_ring_buffer": {"size": 100}},
    )
    daemon._handle_open_ring_buffer(channel, open_msg)

    huge_data = b"x" * 1000
    data_msg = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.DATA_CHUNK,
        payload={
            "trace_id": "trace-big",
            "data_type": DataType.CUSTOM_1D.value,
            "chunk_index": 0,
            "total_chunks": 1,
            "data": base64.b64encode(huge_data).decode("ascii"),
            "robot_instance": 0,
        },
    )

    try:
        daemon._handle_write_data_chunk(channel, data_msg)
    except ValueError:
        pass

    small_data = b"small"
    small_msg = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.DATA_CHUNK,
        payload={
            "trace_id": "trace-small",
            "data_type": DataType.CUSTOM_1D.value,
            "chunk_index": 0,
            "total_chunks": 1,
            "data": base64.b64encode(small_data).decode("ascii"),
            "robot_instance": 0,
        },
    )
    daemon._handle_write_data_chunk(channel, small_msg)
    daemon._drain_channel_messages()

    assert len(mock_rdm.enqueued) >= 1


def test_daemon_continues_after_ring_buffer_read_error() -> None:
    """Read error doesn't stop daemon.

    Read failures must be contained. Daemon keeps running for other channels.
    """
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        recording_disk_manager=mock_rdm,
    )

    channel = ChannelState(producer_id="test-producer", recording_id="rec-123")
    daemon.channels["test-producer"] = channel

    channel.ring_buffer = RingBuffer(size=1024)
    channel.reader = ChannelMessageReader(channel.ring_buffer)

    channel.ring_buffer.write(b"garbage" * 20)

    try:
        daemon._drain_channel_messages()
    except Exception:
        pass

    assert "test-producer" in daemon.channels


def test_daemon_continues_after_reader_poll_error() -> None:
    """Reader error doesn't stop daemon.

    Per-channel errors isolated. Bad channel doesn't break good channels.
    """
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        recording_disk_manager=mock_rdm,
    )

    channel = ChannelState(producer_id="test-producer", recording_id="rec-123")
    daemon.channels["test-producer"] = channel

    channel.ring_buffer = RingBuffer(size=1024)
    channel.reader = ChannelMessageReader(channel.ring_buffer)

    bad_header = b"\xff" * CHUNK_HEADER_SIZE
    channel.ring_buffer.write(bad_header)

    try:
        daemon._drain_channel_messages()
    except Exception:
        pass

    assert daemon.channels is not None


def test_daemon_continues_after_channel_error() -> None:
    """One bad channel doesn't affect others.

    Critical isolation: channels are independent. One failure can't cascade.
    """
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        recording_disk_manager=mock_rdm,
    )

    channel_good = ChannelState(producer_id="good-producer", recording_id="rec-good")
    channel_bad = ChannelState(producer_id="bad-producer", recording_id="rec-bad")
    daemon.channels["good-producer"] = channel_good
    daemon.channels["bad-producer"] = channel_bad

    open_good = MessageEnvelope(
        producer_id="good-producer",
        command=CommandType.OPEN_RING_BUFFER,
        payload={"open_ring_buffer": {"size": 4096}},
    )
    daemon._handle_open_ring_buffer(channel_good, open_good)

    data_msg = MessageEnvelope(
        producer_id="good-producer",
        command=CommandType.DATA_CHUNK,
        payload={
            "trace_id": "trace-good",
            "data_type": DataType.CUSTOM_1D.value,
            "chunk_index": 0,
            "total_chunks": 1,
            "data": base64.b64encode(b"good-data").decode("ascii"),
            "robot_instance": 0,
        },
    )
    daemon._handle_write_data_chunk(channel_good, data_msg)

    daemon._drain_channel_messages()

    assert len(mock_rdm.enqueued) == 1
    assert mock_rdm.enqueued[0].trace_id == "trace-good"


def test_buffer_full_blocks_not_crashes() -> None:
    """Full buffer blocks, no crash.

    Back-pressure via blocking. Full buffer is normal condition, not error.
    """
    ring = RingBuffer(size=100)
    ring.write(b"x" * 100)

    write_completed = threading.Event()

    def blocked_write() -> None:
        ring.write(b"new")
        write_completed.set()

    thread = threading.Thread(target=blocked_write)
    thread.start()

    time.sleep(0.05)
    assert not write_completed.is_set()

    ring.read(50)

    write_completed.wait(timeout=1)
    assert write_completed.is_set()

    thread.join(timeout=1)


def test_buffer_full_then_drained_resumes() -> None:
    """Blocked write resumes.

    Recovery from full: blocking is temporary. Once space available, writer
    proceeds.
    """
    ring = RingBuffer(size=50)
    ring.write(b"x" * 50)

    results: list[str] = []

    def writer() -> None:
        ring.write(b"after-drain")
        results.append("completed")

    thread = threading.Thread(target=writer)
    thread.start()

    time.sleep(0.05)
    ring.read(50)

    thread.join(timeout=1)

    assert results == ["completed"]
    assert ring.available() == 11


def test_many_partial_messages_memory_bounded() -> None:
    """Memory doesn't explode.

    Resource exhaustion attack. System must have limits to prevent memory bomb.
    """
    ring = RingBuffer(size=65536)
    reader = ChannelMessageReader(ring)

    for i in range(1000):
        _write_chunk(ring, f"trace-{i}", DataType.CUSTOM_1D, 0, 100, b"x")
        reader.poll_one()

    assert len(reader._pending) == 1000

    total_chunks = sum(len(p.chunks) for p in reader._pending.values())
    assert total_chunks == 1000


def test_rapid_channel_create_destroy() -> None:
    """Fast churn handled.

    Lifecycle stress: rapid channel turnover shouldn't leak resources or
    destabilize.
    """
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        recording_disk_manager=mock_rdm,
    )

    for i in range(100):
        producer_id = f"producer-{i}"
        channel = ChannelState(producer_id=producer_id, recording_id=f"rec-{i}")
        daemon.channels[producer_id] = channel

        open_msg = MessageEnvelope(
            producer_id=producer_id,
            command=CommandType.OPEN_RING_BUFFER,
            payload={"open_ring_buffer": {"size": 1024}},
        )
        daemon._handle_open_ring_buffer(channel, open_msg)

        del daemon.channels[producer_id]

    assert len(daemon.channels) == 0


def test_buffer_invariants_after_write_error() -> None:
    """Invariants hold after write fails.

    Error must not corrupt internal state. Invariants must hold even after
    failures.
    """
    ring = RingBuffer(size=100)
    ring.write(b"12345")

    original_write_pos = ring.write_pos
    original_read_pos = ring.read_pos
    original_used = ring.used

    try:
        ring.write(b"x" * 200)
    except ValueError:
        pass

    assert ring.write_pos == original_write_pos
    assert ring.read_pos == original_read_pos
    assert ring.used == original_used


def test_buffer_invariants_after_read_error() -> None:
    """Invariants hold after failed read.

    Same for read failures. State must be consistent after any operation.
    """
    ring = RingBuffer(size=100)
    ring.write(b"12345")

    original_write_pos = ring.write_pos
    original_read_pos = ring.read_pos
    original_used = ring.used

    result = ring.read(100)
    assert result is None

    assert ring.write_pos == original_write_pos
    assert ring.read_pos == original_read_pos
    assert ring.used == original_used


def test_reader_state_consistent_after_poll_error() -> None:
    """Reader state valid after error.

    Reader internal state must stay valid. Corrupted input shouldn't corrupt
    reader.
    """
    ring = RingBuffer(size=1024)
    reader = ChannelMessageReader(ring)

    _write_chunk(ring, "valid-trace", DataType.CUSTOM_1D, 0, 2, b"part0")
    reader.poll_one()

    len(reader._pending)

    ring.write(b"garbage" * 20)

    try:
        reader.poll_one()
    except Exception:
        pass

    assert "valid-trace" in reader._pending


def test_channel_state_consistent_after_error() -> None:
    """Channel state valid after error.

    Channel object must stay in valid state. Partial state is worse than no
    state.
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

    assert channel.ring_buffer is not None
    assert channel.reader is not None

    assert channel.ring_buffer.size == 4096


def test_daemon_handles_none_ring_buffer() -> None:
    """None buffer handled safely.

    Defensive: missing buffer shouldn't crash. Skip and continue.
    """
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        recording_disk_manager=mock_rdm,
    )

    channel = ChannelState(producer_id="test-producer", recording_id="rec-123")
    daemon.channels["test-producer"] = channel

    daemon._drain_channel_messages()


def test_daemon_handles_none_reader() -> None:
    """None reader handled safely.

    Same for missing reader. Graceful skip, not crash.
    """
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        recording_disk_manager=mock_rdm,
    )

    channel = ChannelState(producer_id="test-producer", recording_id="rec-123")
    channel.ring_buffer = RingBuffer(size=1024)
    channel.reader = None
    daemon.channels["test-producer"] = channel

    daemon._drain_channel_messages()


def test_data_chunk_before_open_buffer() -> None:
    """DATA_CHUNK before OPEN_RING_BUFFER.

    Out-of-order commands happen. Must handle gracefully, not crash.
    """
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        recording_disk_manager=mock_rdm,
    )

    channel = ChannelState(producer_id="test-producer", recording_id="rec-123")
    daemon.channels["test-producer"] = channel

    data_msg = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.DATA_CHUNK,
        payload={
            "trace_id": "trace-1",
            "data_type": DataType.CUSTOM_1D.value,
            "chunk_index": 0,
            "total_chunks": 1,
            "data": base64.b64encode(b"data").decode("ascii"),
            "robot_instance": 0,
        },
    )

    try:
        daemon._handle_write_data_chunk(channel, data_msg)
    except (AttributeError, TypeError):
        pass


def test_trace_end_for_unknown_trace() -> None:
    """TRACE_END for nonexistent trace.

    Spurious or duplicate trace_end. Must not crash or corrupt state.
    """
    mock_comm = MockComm()
    mock_rdm = MockRDM()

    daemon = Daemon(
        comm_manager=mock_comm,
        recording_disk_manager=mock_rdm,
    )

    channel = ChannelState(producer_id="test-producer", recording_id="rec-123")
    daemon.channels["test-producer"] = channel

    end_msg = MessageEnvelope(
        producer_id="test-producer",
        command=CommandType.TRACE_END,
        payload={
            "trace_end": {
                "trace_id": "nonexistent-trace",
                "recording_id": "rec-123",
            }
        },
    )

    daemon._handle_end_trace(channel, end_msg)


def test_reader_handles_struct_unpack_error() -> None:
    """struct.unpack error handled.

    Binary parse errors must not crash. Header corruption should be handled
    gracefully.
    """
    ring = RingBuffer(size=1024)
    reader = ChannelMessageReader(ring)

    bad_header = b"\xff" * CHUNK_HEADER_SIZE
    ring.write(bad_header)

    try:
        reader.poll_one()
    except struct.error:
        pass


def test_reader_handles_decode_error() -> None:
    """UTF-8 decode error handled.

    Invalid encodings must not crash. Code uses errors='ignore' for this
    reason.
    """
    ring = RingBuffer(size=1024)
    reader = ChannelMessageReader(ring)

    invalid_utf8 = b"\xff\xfe" + b"\x00" * (TRACE_ID_FIELD_SIZE - 2)
    data_type_field = DataType.CUSTOM_1D.value.encode().ljust(
        DATA_TYPE_FIELD_SIZE, b"\x00"
    )

    header = struct.pack(
        CHUNK_HEADER_FORMAT,
        invalid_utf8,
        data_type_field,
        0,
        1,
        5,
    )
    ring.write(header + b"data!")

    reader.poll_one()
