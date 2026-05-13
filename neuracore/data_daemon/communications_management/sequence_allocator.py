"""Thread-safe channel-scoped sequence number allocation."""

from __future__ import annotations

import threading


class ChannelSequenceAllocator:
    """Reserve monotonically increasing sequence numbers for one channel."""

    def __init__(self, start: int = 1) -> None:
        """Initialize the sequence allocator."""
        self._next_sequence_number = int(start)
        self._lock = threading.Lock()

    def reserve(self) -> int:
        """Reserve and return the next sequence number."""
        with self._lock:
            sequence_number = self._next_sequence_number
            self._next_sequence_number += 1
            return sequence_number

    def get_last_reserved_sequence_number(self) -> int:
        """Return the most recently reserved sequence number."""
        with self._lock:
            return self._next_sequence_number - 1
