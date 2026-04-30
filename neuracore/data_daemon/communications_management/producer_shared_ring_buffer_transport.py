"""Shared ring buffer service for producer channels."""

from __future__ import annotations

import json
import struct
import threading

from neuracore.data_daemon.const import (
    SHARED_RING_RECORD_HEADER_FORMAT,
    SHARED_RING_RECORD_MAGIC,
)

from .producer_transport_debug_helper import ProducerTransportDebugHelper
from .producer_transport_debug_models import ProducerSharedRingBufferDebugStats
from .ring_buffer import RingBuffer


class ProducerSharedRingBufferTransport:
    """Producer-side shared ring buffer state and operations."""

    def __init__(self, default_size: int) -> None:
        """Initialize the shared-ring transport with its default buffer size."""
        self._default_size = int(default_size)
        self._ring_buffer: RingBuffer | None = None
        self._configured_size = self._default_size
        self._stats_lock = threading.Lock()
        self._debug_helper = ProducerTransportDebugHelper()

    def close(self) -> None:
        """Close the producer's current shared ring buffer handle."""
        with self._stats_lock:
            ring_buffer = self._ring_buffer
            self._ring_buffer = None
        if ring_buffer is None:
            return
        ring_buffer.close()

    def is_open(self) -> bool:
        """Return True when a shared ring buffer has been created."""
        return self._ring_buffer is not None

    def open(self, size: int | None = None) -> dict[str, str | int]:
        """Create a new shared ring buffer and return its open payload."""
        effective_size = self._default_size if size is None else int(size)
        self.close()
        started_at = self._debug_helper.start_timer()
        ring_buffer = RingBuffer.create_shared(effective_size)
        shared_name = ring_buffer.shared_name
        if shared_name is None:
            raise RuntimeError("Shared ring buffer did not expose a shared name")
        with self._stats_lock:
            self._ring_buffer = ring_buffer
            self._configured_size = effective_size
        self._debug_helper.record_shared_ring_open(started_at)
        return {
            "size": effective_size,
            "shared_memory_name": shared_name,
        }

    def ensure_open(self) -> None:
        """Raise when the shared ring buffer is not currently initialized."""
        with self._stats_lock:
            ring_buffer = self._ring_buffer
        if ring_buffer is None:
            raise RuntimeError("Shared ring buffer not initialized")

    def write_record(
        self,
        metadata: dict[str, str | int | None],
        chunk: bytes | bytearray | memoryview,
    ) -> None:
        """Write a self-describing record into the shared ring buffer."""
        self.ensure_open()
        with self._stats_lock:
            ring_buffer = self._ring_buffer
        if ring_buffer is None:
            raise RuntimeError("Shared ring buffer not initialized")

        metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        header = struct.pack(
            SHARED_RING_RECORD_HEADER_FORMAT,
            SHARED_RING_RECORD_MAGIC,
            len(metadata_bytes),
            len(chunk),
        )
        record_size = len(header) + len(metadata_bytes) + len(chunk)
        started_at = self._debug_helper.start_timer()
        ring_buffer.write(header)
        ring_buffer.write(metadata_bytes)
        ring_buffer.write(chunk)
        self._debug_helper.record_shared_ring_write(
            started_at=started_at,
            bytes_written=record_size,
        )

    def get_stats(self) -> ProducerSharedRingBufferDebugStats:
        """Return a lightweight snapshot of shared-ring transport state."""
        with self._stats_lock:
            ring_buffer = self._ring_buffer
            return self._debug_helper.shared_ring_stats(
                shared_ring_buffer_name=(
                    ring_buffer.shared_name if ring_buffer is not None else None
                ),
                shared_ring_buffer_size=self._configured_size,
            )
