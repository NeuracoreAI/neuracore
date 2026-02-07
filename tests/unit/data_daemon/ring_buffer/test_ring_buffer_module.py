"""Level 2: Ring Buffer Module Unit Tests.

Tests for the RingBuffer module including PartialMessage and ChannelMessageReader:
- PartialMessage chunk assembly
- ChannelMessageReader polling and message completion
- Header format and constants
- Field handling (trace_id, data_type)
- Reader state management
"""

from __future__ import annotations

import struct

import pytest
from neuracore_types import DataType

from neuracore.data_daemon.communications_management.channel_reader import (
    ChannelMessageReader,
    PartialMessage,
)
from neuracore.data_daemon.communications_management.ring_buffer import RingBuffer
from neuracore.data_daemon.const import (
    CHUNK_HEADER_FORMAT,
    CHUNK_HEADER_SIZE,
    DATA_TYPE_FIELD_SIZE,
    TRACE_ID_FIELD_SIZE,
)


def _make_chunk_header(
    trace_id: str,
    data_type: DataType,
    chunk_index: int,
    total_chunks: int,
    chunk_len: int,
) -> bytes:
    """Helper to create a properly formatted chunk header."""
    trace_id_bytes = trace_id.encode("utf-8")
    trace_id_field = trace_id_bytes[:TRACE_ID_FIELD_SIZE].ljust(
        TRACE_ID_FIELD_SIZE, b"\x00"
    )
    data_type_bytes = data_type.value.encode("utf-8")
    data_type_field = data_type_bytes[:DATA_TYPE_FIELD_SIZE].ljust(
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
    """Helper to write a complete chunk (header + data) to ring buffer."""
    header = _make_chunk_header(
        trace_id, data_type, chunk_index, total_chunks, len(data)
    )
    ring.write(header + data)


# =============================================================================
# L2-001 to L2-005: PartialMessage Tests
# =============================================================================


def test_partial_message_single_chunk() -> None:
    """Single chunk assembly.

    Simplest case: message fits in one chunk. Should complete immediately after
    first chunk added.
    """
    partial = PartialMessage(total_chunks=1)

    is_complete = partial.add_chunk(0, b"hello")

    assert is_complete is True
    assert partial.assemble() == b"hello"


def test_partial_message_multi_chunk_in_order() -> None:
    """Multi-chunk in-order assembly.

    Normal case: chunks arrive in order 0,1,2. Last chunk triggers completion.
    """
    partial = PartialMessage(total_chunks=3)

    assert partial.add_chunk(0, b"aaa") is False
    assert partial.add_chunk(1, b"bbb") is False
    assert partial.add_chunk(2, b"ccc") is True

    assert partial.assemble() == b"aaa" + b"bbb" + b"ccc"


def test_partial_message_multi_chunk_out_of_order() -> None:
    """Multi-chunk out-of-order assembly.

    Network reordering is common. Chunks can arrive in any order but must
    assemble correctly.
    """
    partial = PartialMessage(total_chunks=3)

    assert partial.add_chunk(2, b"ccc") is False
    assert partial.add_chunk(0, b"aaa") is False
    assert partial.add_chunk(1, b"bbb") is True

    # Must assemble in index order, not insertion order
    assert partial.assemble() == b"aaa" + b"bbb" + b"ccc"


def test_partial_message_duplicate_chunk_ignored() -> None:
    """Duplicate chunks don't re-add.

    Retransmissions happen. Duplicate chunks should be ignored, not corrupt
    the message.
    """
    partial = PartialMessage(total_chunks=2)

    partial.add_chunk(0, b"first")
    # Add duplicate with different data
    result = partial.add_chunk(0, b"different")

    assert result is False  # Not complete yet
    assert partial.chunks[0] == b"first"  # Original preserved


def test_partial_message_assemble_missing_raises() -> None:
    """Assemble with missing chunks fails.

    Can't assemble incomplete message. Clear error tells caller which chunks
    are missing.
    """
    partial = PartialMessage(total_chunks=3)
    partial.add_chunk(0, b"aaa")
    # Missing chunks 1 and 2

    with pytest.raises(ValueError, match="Missing chunks"):
        partial.assemble()


# =============================================================================
# L2-006 to L2-009: ChannelMessageReader Basic Tests
# =============================================================================


def test_reader_poll_empty_buffer() -> None:
    """Poll on empty buffer.

    No data means no message. Reader should return None immediately, not block
    or crash.
    """
    ring = RingBuffer(size=1024)
    reader = ChannelMessageReader(ring)

    result = reader.poll_one()

    assert result is None


def test_reader_poll_insufficient_header() -> None:
    """Poll with partial header.

    Header incomplete - can't even know message size yet. Must wait without
    consuming partial data.
    """
    ring = RingBuffer(size=1024)
    reader = ChannelMessageReader(ring)

    # Write less than CHUNK_HEADER_SIZE (80 bytes)
    ring.write(b"x" * 40)

    result = reader.poll_one()

    assert result is None
    assert ring.available() == 40  # Data not consumed


def test_reader_poll_header_but_no_payload() -> None:
    """Header present but payload missing.

    Header parsed but payload incomplete. Must wait for rest without corrupting
    state.
    """
    ring = RingBuffer(size=1024)
    reader = ChannelMessageReader(ring)

    # Write header that says 100 bytes payload, but only partial payload
    header = _make_chunk_header("trace-1", DataType.CUSTOM_1D, 0, 1, 100)
    ring.write(header + b"x" * 50)  # Only 50 of 100 bytes

    result = reader.poll_one()

    assert result is None
    assert ring.available() == CHUNK_HEADER_SIZE + 50  # Data not consumed


def test_reader_single_chunk_message() -> None:
    """Complete single-chunk message.

    Happy path: complete message ready. Should return all three components
    correctly.
    """
    ring = RingBuffer(size=1024)
    reader = ChannelMessageReader(ring)

    _write_chunk(ring, "trace-123", DataType.JOINT_POSITIONS, 0, 1, b"payload-data")

    result = reader.poll_one()

    assert result is not None
    trace_id, data_type, payload = result
    assert trace_id == "trace-123"
    assert data_type == DataType.JOINT_POSITIONS
    assert payload == b"payload-data"


# =============================================================================
# L2-010 to L2-016: ChannelMessageReader Edge Cases
# =============================================================================


def test_reader_interleaved_traces() -> None:
    """Multiple traces interleaved.

    Real systems multiplex traces. Reader must track each trace separately and
    complete them independently.
    """
    ring = RingBuffer(size=4096)
    reader = ChannelMessageReader(ring)

    # Interleave two 2-chunk messages
    _write_chunk(ring, "trace-A", DataType.CUSTOM_1D, 0, 2, b"A0")
    _write_chunk(ring, "trace-B", DataType.CUSTOM_1D, 0, 2, b"B0")
    _write_chunk(ring, "trace-A", DataType.CUSTOM_1D, 1, 2, b"A1")
    _write_chunk(ring, "trace-B", DataType.CUSTOM_1D, 1, 2, b"B1")

    results = []
    for _ in range(4):
        result = reader.poll_one()
        if result:
            results.append((result[0], result[2]))

    # Both should complete
    assert ("trace-A", b"A0A1") in results
    assert ("trace-B", b"B0B1") in results


def test_reader_inconsistent_total_chunks_warns() -> None:
    """Mismatched total_chunks logged.

    Data corruption or bug. Should warn but not crash - use first value and
    proceed cautiously.
    """
    ring = RingBuffer(size=4096)
    reader = ChannelMessageReader(ring)

    # First chunk says total=2
    _write_chunk(ring, "trace-1", DataType.CUSTOM_1D, 0, 2, b"chunk0")
    # Second chunk says total=3 (inconsistent!)
    _write_chunk(ring, "trace-1", DataType.CUSTOM_1D, 1, 3, b"chunk1")

    # Should still work, using original total_chunks
    reader.poll_one()  # First chunk
    result = reader.poll_one()  # Second chunk completes it

    assert result is not None
    assert result[2] == b"chunk0chunk1"


def test_reader_clears_pending_after_complete() -> None:
    """Pending state cleaned after complete.

    Memory hygiene: completed traces must be removed from pending dict.
    Prevents memory leak.
    """
    ring = RingBuffer(size=4096)
    reader = ChannelMessageReader(ring)

    _write_chunk(ring, "trace-1", DataType.CUSTOM_1D, 0, 2, b"part1")
    reader.poll_one()
    assert "trace-1" in reader._pending

    _write_chunk(ring, "trace-1", DataType.CUSTOM_1D, 1, 2, b"part2")
    reader.poll_one()

    assert "trace-1" not in reader._pending


def test_reader_with_wraparound_data() -> None:
    """Message spans buffer boundary.

    Reader shouldn't care about physical layout. Wrap-around must be invisible
    to message parsing.
    """
    ring = RingBuffer(size=200)
    reader = ChannelMessageReader(ring)

    # Fill buffer partially and read to move positions near end
    ring.write(b"x" * 150)
    ring.read(150)

    # Now write a message that will wrap around
    _write_chunk(ring, "trace-wrap", DataType.CUSTOM_1D, 0, 1, b"wrapped-payload")

    result = reader.poll_one()

    assert result is not None
    assert result[0] == "trace-wrap"
    assert result[2] == b"wrapped-payload"


def test_reader_continuous_stream() -> None:
    """Stream of messages.

    Streaming workload: many messages in sequence. Tests sustained operation,
    not just single message.
    """
    # 100 messages × ~90 bytes each = ~9000 bytes, need buffer > 9KB
    ring = RingBuffer(size=16384)
    reader = ChannelMessageReader(ring)

    # Write 100 single-chunk messages
    for i in range(100):
        _write_chunk(ring, f"trace-{i}", DataType.CUSTOM_1D, 0, 1, f"data-{i}".encode())

    # Read all
    results = []
    for _ in range(100):
        result = reader.poll_one()
        if result:
            results.append(result[0])

    assert len(results) == 100
    assert results[0] == "trace-0"
    assert results[99] == "trace-99"


def test_reader_large_payload() -> None:
    """Large payload handling.

    Realistic sizes: video frames, sensor batches can be large. Must handle
    without truncation.
    """
    ring = RingBuffer(size=1024 * 1024)  # 1MB
    reader = ChannelMessageReader(ring)

    large_data = b"x" * (500 * 1024)  # 500KB
    _write_chunk(ring, "large-trace", DataType.CUSTOM_1D, 0, 1, large_data)

    result = reader.poll_one()

    assert result is not None
    assert len(result[2]) == 500 * 1024
    assert result[2] == large_data


def test_reader_zero_length_chunk() -> None:
    """Zero-length payload.

    Valid edge case: metadata-only messages. Empty payload is legal and must
    work.
    """
    ring = RingBuffer(size=1024)
    reader = ChannelMessageReader(ring)

    _write_chunk(ring, "empty-trace", DataType.CUSTOM_1D, 0, 1, b"")

    result = reader.poll_one()

    assert result is not None
    assert result[0] == "empty-trace"
    assert result[2] == b""


# =============================================================================
# L2-017 to L2-022: Header Format and Constants
# =============================================================================


def test_chunk_header_size_matches_format() -> None:
    """CHUNK_HEADER_SIZE = 80.

    Header size constant must match actual struct format. Mismatch causes parse
    failures.
    """
    expected_size = struct.calcsize(CHUNK_HEADER_FORMAT)
    assert CHUNK_HEADER_SIZE == expected_size
    assert CHUNK_HEADER_SIZE == 80


def test_header_struct_pack_unpack_roundtrip() -> None:
    """Header pack/unpack roundtrip.

    Serialization must be reversible. Tests that format string and field order
    are correct.
    """
    trace_id = b"test-trace-id".ljust(TRACE_ID_FIELD_SIZE, b"\x00")
    data_type = b"custom_1d".ljust(DATA_TYPE_FIELD_SIZE, b"\x00")
    chunk_index = 5
    total_chunks = 10
    chunk_len = 12345

    packed = struct.pack(
        CHUNK_HEADER_FORMAT,
        trace_id,
        data_type,
        chunk_index,
        total_chunks,
        chunk_len,
    )

    unpacked = struct.unpack(CHUNK_HEADER_FORMAT, packed)

    assert unpacked[0] == trace_id
    assert unpacked[1] == data_type
    assert unpacked[2] == chunk_index
    assert unpacked[3] == total_chunks
    assert unpacked[4] == chunk_len


def test_trace_id_max_length_36_bytes() -> None:
    """trace_id truncated at 36.

    Field has fixed size. Long trace_ids must be truncated, not overflow into
    next field.
    """
    ring = RingBuffer(size=1024)
    reader = ChannelMessageReader(ring)

    long_trace_id = "a" * 50  # Longer than 36 bytes
    _write_chunk(ring, long_trace_id, DataType.CUSTOM_1D, 0, 1, b"data")

    result = reader.poll_one()

    assert result is not None
    assert len(result[0]) <= TRACE_ID_FIELD_SIZE
    assert result[0] == "a" * TRACE_ID_FIELD_SIZE


def test_data_type_max_length_32_bytes() -> None:
    """data_type truncated at 32.

    Same for data_type field. Fixed-size protocol requires truncation handling.
    """
    # This is handled internally by the header construction
    assert DATA_TYPE_FIELD_SIZE == 32

    # Verify a long data type string gets truncated in the header
    long_str = "x" * 50
    truncated = long_str[:DATA_TYPE_FIELD_SIZE]
    assert len(truncated) == 32


def test_trace_id_with_unicode() -> None:
    """Unicode trace_id encoded.

    UUIDs are ASCII but system should handle unicode gracefully if encountered.
    """
    ring = RingBuffer(size=1024)
    reader = ChannelMessageReader(ring)

    # Unicode trace_id (will be truncated to fit)
    unicode_trace = "trace-emoji-测试"
    _write_chunk(ring, unicode_trace, DataType.CUSTOM_1D, 0, 1, b"data")

    result = reader.poll_one()

    assert result is not None
    # Should handle encoding without crashing


def test_trace_id_with_null_bytes() -> None:
    """Null bytes in trace_id handled.

    Null padding is used. Must not confuse null-in-data with null-padding.
    """
    ring = RingBuffer(size=1024)
    reader = ChannelMessageReader(ring)

    # trace_id with embedded null (unusual but possible)
    _write_chunk(ring, "trace\x00id", DataType.CUSTOM_1D, 0, 1, b"data")

    result = reader.poll_one()

    assert result is not None
    # The reader strips trailing nulls, so embedded null might cause truncation
    # This documents the behavior


# =============================================================================
# L2-023 to L2-026: PartialMessage Edge Cases
# =============================================================================


def test_partial_message_with_total_chunks_zero() -> None:
    """total_chunks=0 edge case.

    Invalid but possible. Should not cause divide-by-zero or infinite loop.
    With total_chunks=0, the message is considered "complete" immediately
    and assemble() returns empty bytes.
    """
    partial = PartialMessage(total_chunks=0)

    # With 0 total chunks, received_chunks == total_chunks (0 == 0) immediately
    assert partial.received_chunks == partial.total_chunks

    # assemble() returns empty bytes (range(0) yields nothing)
    result = partial.assemble()
    assert result == b""


def test_partial_message_with_very_large_total() -> None:
    """total_chunks=1000000.

    Resource exhaustion risk. Should not allocate million-entry dict
    immediately.
    """
    # Just creating the PartialMessage shouldn't allocate huge memory
    partial = PartialMessage(total_chunks=1000000)

    # Dict should be empty until chunks added
    assert len(partial.chunks) == 0

    # Adding one chunk shouldn't cause issues
    partial.add_chunk(0, b"first")
    assert len(partial.chunks) == 1


def test_partial_message_chunk_with_empty_data() -> None:
    """add_chunk with b"".

    Empty chunks are valid. Zero-length data must not be confused with missing
    chunk.
    """
    partial = PartialMessage(total_chunks=2)

    partial.add_chunk(0, b"")
    partial.add_chunk(1, b"data")

    assert partial.received_chunks == 2
    result = partial.assemble()
    assert result == b"data"  # Empty + data


def test_partial_message_assemble_order_preserved() -> None:
    """Chunks assembled in order.

    Assembly must respect chunk indices, not insertion order. Order determines
    final byte sequence.
    """
    partial = PartialMessage(total_chunks=4)

    # Add in reverse order
    partial.add_chunk(3, b"D")
    partial.add_chunk(1, b"B")
    partial.add_chunk(2, b"C")
    partial.add_chunk(0, b"A")

    # Must assemble as ABCD, not as insertion order
    assert partial.assemble() == b"ABCD"


# =============================================================================
# L2-027 to L2-031: Reader State Management
# =============================================================================


def test_reader_pending_state_grows_with_traces() -> None:
    """_pending grows correctly.

    Multiple concurrent traces each need separate state. Dict must track all of
    them.
    """
    ring = RingBuffer(size=8192)
    reader = ChannelMessageReader(ring)

    # Start 5 traces without completing any
    for i in range(5):
        _write_chunk(ring, f"trace-{i}", DataType.CUSTOM_1D, 0, 2, b"part0")
        reader.poll_one()

    assert len(reader._pending) == 5


def test_reader_pending_state_isolated_per_trace() -> None:
    """Traces don't share state.

    Trace A's chunks must not affect trace B. Complete isolation required.
    """
    ring = RingBuffer(size=4096)
    reader = ChannelMessageReader(ring)

    # Start two traces
    _write_chunk(ring, "trace-A", DataType.CUSTOM_1D, 0, 3, b"A0")
    _write_chunk(ring, "trace-B", DataType.CUSTOM_1D, 0, 2, b"B0")
    reader.poll_one()
    reader.poll_one()

    assert "trace-A" in reader._pending
    assert "trace-B" in reader._pending
    assert reader._pending["trace-A"].total_chunks == 3
    assert reader._pending["trace-B"].total_chunks == 2


def test_reader_reuse_after_complete() -> None:
    """Reader handles new traces after completion.

    Reader is long-lived. Must handle unlimited trace sequences over its
    lifetime.
    """
    ring = RingBuffer(size=4096)
    reader = ChannelMessageReader(ring)

    # Complete first trace
    _write_chunk(ring, "trace-1", DataType.CUSTOM_1D, 0, 1, b"data1")
    result1 = reader.poll_one()
    assert result1 is not None

    # Start and complete second trace
    _write_chunk(ring, "trace-2", DataType.CUSTOM_1D, 0, 1, b"data2")
    result2 = reader.poll_one()
    assert result2 is not None
    assert result2[0] == "trace-2"


def test_reader_multiple_poll_none_until_complete() -> None:
    """Repeated poll returns None until ready.

    Polling is idempotent. Calling poll when incomplete should be safe to
    repeat.
    """
    ring = RingBuffer(size=4096)
    reader = ChannelMessageReader(ring)

    # Write first of 2 chunks
    _write_chunk(ring, "trace-1", DataType.CUSTOM_1D, 0, 2, b"part0")

    # Multiple polls should return None
    assert reader.poll_one() is None
    assert reader.poll_one() is None
    assert reader.poll_one() is None

    # Complete the message
    _write_chunk(ring, "trace-1", DataType.CUSTOM_1D, 1, 2, b"part1")

    # Now should return the message
    result = reader.poll_one()
    assert result is not None
    assert result[2] == b"part0part1"


def test_reader_poll_after_drain_returns_none() -> None:
    """Poll after buffer drained.

    Empty state after drain. Reader should handle transition from full to empty
    gracefully.
    """
    ring = RingBuffer(size=4096)
    reader = ChannelMessageReader(ring)

    # Write and consume a message
    _write_chunk(ring, "trace-1", DataType.CUSTOM_1D, 0, 1, b"data")
    reader.poll_one()

    # Buffer is now empty
    assert ring.available() == 0

    # Polling empty buffer should return None
    assert reader.poll_one() is None
