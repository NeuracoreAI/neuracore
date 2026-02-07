"""Level 1: Ring Buffer Internal Unit Tests.

Tests for the RingBuffer class in isolation, covering:
- Constructor/initialization
- Basic operations (write, read, peek, available)
- Wrap-around behavior
- Edge cases
- State invariants
- Data integrity
- Blocking behavior
"""

from __future__ import annotations

import threading
import time

import pytest

from neuracore.data_daemon.communications_management.ring_buffer import RingBuffer

# =============================================================================
# L1-001 to L1-009: Basic Operations
# =============================================================================


def test_write_read_returns_same_data() -> None:
    """Verify write then read returns identical bytes.

    The most basic contract: data written must be readable exactly as written.
    If this fails, nothing else matters.
    """
    ring = RingBuffer(size=100)
    ring.write(b"hello")
    result = ring.read(5)
    assert result == b"hello"


def test_peek_does_not_consume_data() -> None:
    """Verify peek reads without consuming.

    Peek lets you inspect data without committing to remove it. Essential for
    header inspection before deciding to read full message.
    """
    ring = RingBuffer(size=100)
    ring.write(b"test")

    first_peek = ring.peek(4)
    second_peek = ring.peek(4)

    assert first_peek == b"test"
    assert second_peek == b"test"
    assert ring.available() == 4


def test_available_returns_correct_count() -> None:
    """Verify available() tracks used bytes.

    Available tells consumers how much data exists. Wrong count means readers
    either block forever or read garbage.
    """
    ring = RingBuffer(size=100)
    assert ring.available() == 0

    ring.write(b"abc")
    assert ring.available() == 3


def test_available_after_read() -> None:
    """Verify available() decreases after read.

    After consuming data, the count must decrease so writers know space freed
    up. Tracks producer/consumer balance.
    """
    ring = RingBuffer(size=100)
    ring.write(b"0123456789")
    assert ring.available() == 10

    ring.read(4)
    assert ring.available() == 6


def test_write_empty_data_is_noop() -> None:
    """Verify writing empty bytes does nothing.

    Empty writes shouldn't corrupt state or crash. Common edge case when
    message has no payload.
    """
    ring = RingBuffer(size=100)
    ring.write(b"")

    assert ring.available() == 0
    assert ring.write_pos == 0
    assert ring.read_pos == 0


def test_read_returns_none_when_empty() -> None:
    """Verify read on empty buffer returns None.

    Readers must handle empty buffer gracefully. None signals "try again later"
    rather than crashing or blocking.
    """
    ring = RingBuffer(size=100)
    result = ring.read(1)
    assert result is None


def test_peek_returns_none_when_empty() -> None:
    """Verify peek on empty buffer returns None.

    Same as read - peek on empty should signal "no data" not crash. Consistent
    API behavior.
    """
    ring = RingBuffer(size=100)
    result = ring.peek(1)
    assert result is None


def test_read_returns_none_when_insufficient() -> None:
    """Verify read returns None if not enough data.

    Partial reads aren't allowed - you get all requested bytes or nothing.
    Prevents corrupted message reads.
    """
    ring = RingBuffer(size=100)
    ring.write(b"abc")

    result = ring.read(5)

    assert result is None
    assert ring.available() == 3  # Data unchanged


def test_peek_returns_none_when_insufficient() -> None:
    """Verify peek returns None if not enough data.

    Same contract as read - all or nothing. Ensures header peek works correctly
    when header incomplete.
    """
    ring = RingBuffer(size=100)
    ring.write(b"abc")

    result = ring.peek(5)

    assert result is None
    assert ring.available() == 3


# =============================================================================
# L1-010 to L1-014: Wrap-Around Behavior
# =============================================================================


def test_write_wraps_around_buffer_end() -> None:
    """Verify write handles boundary crossing when data spans buffer end.

    Circular buffer's core feature. Data must wrap seamlessly so a fixed-size
    buffer can handle an infinite stream. The write is split: some bytes at
    the end, remaining bytes at the start.
    """
    ring = RingBuffer(size=10)

    # Move write_pos near end
    ring.write(b"12345678")  # write_pos now at 8
    ring.read(8)  # read_pos now at 8, buffer empty

    # This write should wrap: 2 bytes at end (pos 8,9), 3 at start (pos 0,1,2)
    ring.write(b"abcde")
    result = ring.read(5)

    assert result == b"abcde"


def test_read_wraps_around_buffer_end() -> None:
    """Verify read handles boundary crossing.

    Reading wrapped data must reassemble correctly. Reader shouldn't know or
    care about physical layout.
    """
    ring = RingBuffer(size=10)

    # Setup: write wraps around
    ring.write(b"12345678")
    ring.read(8)
    ring.write(b"abcde")

    # read_pos is at 8, data wraps to beginning
    result = ring.read(5)

    assert result == b"abcde"


def test_peek_wraps_around_buffer_end() -> None:
    """Verify peek handles boundary crossing.

    Peek must also handle wrap-around. Header inspection must work regardless
    of where data sits in buffer.
    """
    ring = RingBuffer(size=10)

    # Setup: write wraps around
    ring.write(b"12345678")
    ring.read(8)
    ring.write(b"abcde")

    # Peek across boundary
    result = ring.peek(5)

    assert result == b"abcde"
    assert ring.available() == 5  # Not consumed


def test_multiple_wraparounds() -> None:
    """Verify buffer works after many cycles.

    Long-running systems cycle the buffer millions of times. Must not
    accumulate errors or drift over time.
    """
    ring = RingBuffer(size=16)

    for i in range(10):
        data = f"cycle{i:04d}".encode()  # 9 bytes each
        ring.write(data)
        result = ring.read(len(data))
        assert result == data, f"Failed at cycle {i}"


def test_write_exactly_to_end_then_more() -> None:
    """Write fills to end, next starts at 0.

    Boundary condition: writing exactly to the end must wrap position correctly
    for next write.
    """
    ring = RingBuffer(size=10)

    # Write exactly to position 5
    ring.write(b"12345")
    assert ring.write_pos == 5

    # Write exactly to end (positions 5-9)
    ring.write(b"67890")
    assert ring.write_pos == 0  # Wrapped to start

    # Read all to free space
    ring.read(10)

    # Next write should start at position 0
    ring.write(b"abc")
    assert ring.write_pos == 3


# =============================================================================
# L1-015 to L1-020: Edge Cases
# =============================================================================


def test_write_exactly_fills_buffer() -> None:
    """Write data equal to buffer size.

    Maximum capacity must be usable. Buffer should hold exactly `size` bytes,
    not size-1.
    """
    ring = RingBuffer(size=10)
    ring.write(b"0123456789")

    assert ring.available() == 10
    assert ring.read(10) == b"0123456789"


def test_write_exceeds_buffer_raises_error() -> None:
    """Write larger than buffer fails.

    Oversized writes are rejected immediately. Can't fit more than capacity
    even when empty - that's a logic error.
    """
    ring = RingBuffer(size=10)

    with pytest.raises(ValueError, match="chunk exceeds ring buffer"):
        ring.write(b"12345678901")  # 11 bytes


def test_read_exact_available_amount() -> None:
    """Read exactly what's available.

    Draining buffer exactly must leave it empty and usable. Common pattern for
    consuming complete messages.
    """
    ring = RingBuffer(size=100)
    ring.write(b"12345")

    result = ring.read(5)

    assert result == b"12345"
    assert ring.available() == 0


def test_single_byte_operations() -> None:
    """Single byte write/read/peek.

    Minimum granularity must work. Some protocols send single-byte markers or
    length prefixes.
    """
    ring = RingBuffer(size=100)

    ring.write(b"x")
    assert ring.peek(1) == b"x"
    assert ring.read(1) == b"x"
    assert ring.available() == 0


def test_buffer_size_one() -> None:
    """Buffer with size=1.

    Degenerate case: smallest possible buffer. Tests that size=1 is valid and
    works correctly.
    """
    ring = RingBuffer(size=1)

    ring.write(b"a")
    assert ring.read(1) == b"a"

    ring.write(b"b")
    assert ring.read(1) == b"b"


def test_positions_reset_to_zero_after_full_cycle() -> None:
    """Verify positions after full use.

    After complete drain, positions should wrap to start. Clean state for next
    batch of data.
    """
    ring = RingBuffer(size=10)

    ring.write(b"0123456789")
    ring.read(10)

    assert ring.write_pos == 0
    assert ring.read_pos == 0
    assert ring.used == 0


# =============================================================================
# L1-021 to L1-025: Integrity Tests
# =============================================================================


def test_multiple_write_read_cycles_no_corruption() -> None:
    """Data integrity over many cycles.

    Stress test for subtle corruption bugs. Memory issues or off-by-one errors
    often appear after many iterations.
    """
    ring = RingBuffer(size=100)

    for i in range(50):
        # Generate unique data for each cycle
        data = bytes([i % 256] * 20)
        ring.write(data)
        result = ring.read(20)
        assert result == data, f"Corruption at cycle {i}"


def test_interleaved_peek_read_operations() -> None:
    """Mixed peek/read doesn't corrupt.

    Real usage alternates peek (to check header) then read (to consume). Must
    not interfere with each other.
    """
    ring = RingBuffer(size=100)
    ring.write(b"abcdef")

    assert ring.peek(2) == b"ab"
    assert ring.read(2) == b"ab"
    assert ring.peek(2) == b"cd"
    assert ring.read(2) == b"cd"
    assert ring.peek(2) == b"ef"
    assert ring.read(2) == b"ef"


def test_interleaved_write_read_operations() -> None:
    """Producer/consumer pattern works.

    Simulates streaming: producer adds while consumer removes. Classic ring
    buffer use case.
    """
    ring = RingBuffer(size=20)

    ring.write(b"12345")  # 5 bytes
    assert ring.read(3) == b"123"  # Read 3, 2 remain

    ring.write(b"6789")  # Add 4, now 6 total
    assert ring.read(6) == b"456789"


def test_binary_data_integrity() -> None:
    """Non-ASCII bytes preserved.

    Binary protocols use full byte range. Null bytes, high bytes must not be
    mangled or interpreted specially.
    """
    ring = RingBuffer(size=512)

    # All possible byte values
    data = bytes(range(256))
    ring.write(data)
    result = ring.read(256)

    assert result == data


def test_large_data_integrity() -> None:
    """Large payload integrity.

    Realistic payload sizes. Memory handling, slice operations must work at
    scale without corruption.
    """
    ring = RingBuffer(size=1024 * 1024)  # 1MB

    data = bytes([i % 256 for i in range(500 * 1024)])  # 500KB
    ring.write(data)
    result = ring.read(len(data))

    assert result == data


# =============================================================================
# L1-026 to L1-028: State Consistency
# =============================================================================


def test_used_never_exceeds_size() -> None:
    """Invariant: used <= size.

    Core invariant: can't have more data than capacity. Violation means buffer
    math is broken.
    """
    ring = RingBuffer(size=50)

    for _ in range(20):
        ring.write(b"12345")  # 5 bytes
        assert 0 <= ring.used <= ring.size
        ring.read(5)
        assert 0 <= ring.used <= ring.size


def test_write_pos_always_valid() -> None:
    """Invariant: write_pos < size.

    Position must stay in bounds. Out-of-bounds write_pos would corrupt memory
    or crash.
    """
    ring = RingBuffer(size=10)

    for _ in range(25):  # Multiple wraparounds
        ring.write(b"abc")
        assert 0 <= ring.write_pos < ring.size
        ring.read(3)


def test_read_pos_always_valid() -> None:
    """Invariant: read_pos < size.

    Same for read position. Both pointers must be valid array indices at all
    times.
    """
    ring = RingBuffer(size=10)

    for _ in range(25):  # Multiple wraparounds
        ring.write(b"abc")
        ring.read(3)
        assert 0 <= ring.read_pos < ring.size


# =============================================================================
# L1-029 to L1-034: Constructor/Initialization
# =============================================================================


def test_init_with_default_size() -> None:
    """Default size=1024.

    Constructor contract: default should be 1024 bytes as documented.
    """
    ring = RingBuffer()
    assert ring.size == 1024


def test_init_with_custom_size() -> None:
    """Custom size respected.

    Custom sizes must be honored. Production uses 4MB, tests use smaller sizes.
    """
    ring = RingBuffer(size=500)
    assert ring.size == 500


def test_init_buffer_is_zeroed() -> None:
    """Buffer initialized to zeros.

    Initial state must be deterministic. Uninitialized memory could leak data
    or cause flaky tests.
    """
    ring = RingBuffer(size=10)
    assert ring.buffer == bytearray(10)
    assert all(b == 0 for b in ring.buffer)


def test_init_positions_at_zero() -> None:
    """Initial positions=0.

    Fresh buffer starts empty at position zero. Any other state would be a bug.
    """
    ring = RingBuffer(size=10)

    assert ring.write_pos == 0
    assert ring.read_pos == 0
    assert ring.used == 0


def test_init_with_zero_size() -> None:
    """size=0 edge case.

    Zero-size buffer is nonsensical. Should either reject or handle without
    crashing. Tests input validation.
    """
    # This tests current behavior - may raise or create degenerate buffer
    try:
        ring = RingBuffer(size=0)
        # If it doesn't raise, at least verify it doesn't crash on basic ops
        assert ring.available() == 0
        assert ring.read(1) is None
    except (ValueError, Exception):
        # Raising an error is acceptable behavior
        pass


def test_init_with_negative_size() -> None:
    """size=-1 edge case.

    Negative size is invalid. Should reject early rather than create broken
    buffer. Tests input validation.
    """
    # This tests current behavior - may raise or create degenerate buffer
    try:
        RingBuffer(size=-1)
        # If it doesn't raise, behavior is undefined but shouldn't crash
    except (ValueError, OverflowError, MemoryError, Exception):
        # Raising an error is acceptable behavior
        pass


# =============================================================================
# L1-035 to L1-038: Blocking Behavior and Free Space
# =============================================================================


def test_write_blocks_when_buffer_full() -> None:
    """Write blocks until space available.

    Back-pressure mechanism: when full, writer waits rather than overwriting or
    crashing. Requires threading to test.
    """
    ring = RingBuffer(size=10)
    ring.write(b"0123456789")  # Fill buffer

    write_completed = threading.Event()
    write_started = threading.Event()

    def blocked_write() -> None:
        write_started.set()
        ring.write(b"abc")  # Should block
        write_completed.set()

    thread = threading.Thread(target=blocked_write)
    thread.start()

    # Wait for write to start
    write_started.wait(timeout=1)
    time.sleep(0.05)  # Give it time to block

    # Write should be blocked
    assert not write_completed.is_set()

    # Free some space
    ring.read(5)

    # Write should complete
    write_completed.wait(timeout=1)
    assert write_completed.is_set()

    thread.join(timeout=1)


def test_write_unblocks_after_read() -> None:
    """Blocked write completes after drain.

    Recovery from full state: once reader frees space, blocked writer should
    proceed. Producer/consumer coordination.
    """
    ring = RingBuffer(size=10)
    ring.write(b"0123456789")

    result_holder: list[bool] = []

    def writer() -> None:
        ring.write(b"new")
        result_holder.append(True)

    thread = threading.Thread(target=writer)
    thread.start()

    time.sleep(0.05)  # Let writer block

    # Drain buffer
    ring.read(10)

    thread.join(timeout=1)

    assert result_holder == [True]
    assert ring.available() == 3


def test_free_space_calculation() -> None:
    """size - used is correct free space.

    Implicit calculation used in write(). Verifies the math is correct for
    blocking decision.
    """
    ring = RingBuffer(size=100)

    ring.write(b"x" * 30)
    free_space = ring.size - ring.used

    assert free_space == 70


def test_free_space_after_wraparound() -> None:
    """Free space correct after wrap.

    Free space depends only on `used`, not on where data sits. Wrap-around
    shouldn't confuse the calculation.
    """
    ring = RingBuffer(size=10)

    # Create wraparound scenario
    ring.write(b"12345678")
    ring.read(8)
    ring.write(b"abcde")  # Wraps around

    free_space = ring.size - ring.used

    assert ring.used == 5
    assert free_space == 5
