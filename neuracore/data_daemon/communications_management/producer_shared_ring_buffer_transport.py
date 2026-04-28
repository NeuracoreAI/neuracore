"""Shared ring buffer service for producer channels."""

from __future__ import annotations

import json
import struct
import threading
import uuid

from neuracore.data_daemon.const import (
    SHARED_RING_RECORD_HEADER_FORMAT,
    SHARED_RING_RECORD_HEADER_SIZE,
    SHARED_RING_RECORD_MAGIC,
)
from neuracore.data_daemon.models import OpenRingBufferModel

from .producer_transport_debug_helper import ProducerTransportDebugHelper
from .producer_transport_debug_models import ProducerSharedRingBufferDebugStats
from .ring_buffer import RingBuffer


class ProducerSharedRingBufferTransport:
    """Producer-side shared ring buffer state and operations."""

    def __init__(self, default_size: int) -> None:
        """Initialize the shared-ring transport with its default buffer size."""
        self._default_size = int(default_size)
        self._ring_buffer: RingBuffer | None = None
        self._shared_ring_name: str | None = None
        self._configured_size = self._default_size
        self._stats_lock = threading.Lock()
        self._debug_helper = ProducerTransportDebugHelper()

    def close(self) -> None:
        """Reset the producer's current shared ring buffer state."""
        with self._stats_lock:
            ring_buffer = self._ring_buffer
            self._ring_buffer = None
            self._shared_ring_name = None
        if ring_buffer is None:
            return
        ring_buffer.close()

    def is_announced(self) -> bool:
        """Return True when a shared ring buffer name has been announced."""
        return self._shared_ring_name is not None

    def is_open(self) -> bool:
        """Return True when the producer writer handle is attached."""
        return isinstance(self._ring_buffer, RingBuffer)

    def open(self, size: int | None = None) -> OpenRingBufferModel:
        """Reserve a new shared ring buffer name and return its open payload."""
        effective_size = self._default_size if size is None else int(size)
        self.close()
        shared_name = f"neuracore-ring-buffer-{uuid.uuid4().hex}"
        with self._stats_lock:
            self._shared_ring_name = shared_name
            self._configured_size = effective_size
        return OpenRingBufferModel(
            size=effective_size,
            shared_memory_name=shared_name,
        )

    def ensure_open(self) -> None:
        """Open the producer-side writer handle after the daemon creates the reader."""
        with self._stats_lock:
            ring_buffer = self._ring_buffer
            shared_ring_name = self._shared_ring_name
            configured_size = self._configured_size
        if ring_buffer is not None:
            return
        if shared_ring_name is None:
            raise RuntimeError("Shared ring buffer not announced")

        started_at = self._debug_helper.start_timer()
        ring_buffer = RingBuffer.open_shared(shared_ring_name, configured_size)
        if ring_buffer is None:
            raise RuntimeError("Shared ring buffer not initialized")
        with self._stats_lock:
            self._ring_buffer = ring_buffer
        self._debug_helper.record_shared_ring_open(started_at)

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
        chunk_len = len(chunk)
        packet_len = SHARED_RING_RECORD_HEADER_SIZE + len(metadata_bytes) + chunk_len
        if packet_len > ring_buffer.size:
            raise ValueError(
                "Shared-ring record exceeds capacity: "
                f"packet_len={packet_len} capacity={ring_buffer.size} "
                f"metadata_len={len(metadata_bytes)} chunk_len={chunk_len}"
            )
        packet = (
            struct.pack(
                SHARED_RING_RECORD_HEADER_FORMAT,
                SHARED_RING_RECORD_MAGIC,
                len(metadata_bytes),
                chunk_len,
            )
            + metadata_bytes
            + bytes(chunk)
        )
        started_at = self._debug_helper.start_timer()
        ring_buffer.write(packet)
        self._debug_helper.record_shared_ring_write(
            started_at=started_at,
            bytes_written=packet_len,
        )

    def get_stats(self) -> ProducerSharedRingBufferDebugStats:
        """Return a lightweight snapshot of shared-ring transport state."""
        with self._stats_lock:
            ring_buffer = self._ring_buffer
            return self._debug_helper.shared_ring_stats(
                shared_ring_buffer_name=(
                    ring_buffer.shared_name
                    if ring_buffer is not None
                    else self._shared_ring_name
                ),
                shared_ring_buffer_size=self._configured_size,
            )
