"""Reads from a RingBuffer and yields complete logical messages."""

from __future__ import annotations

import json
import logging
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field

from neuracore_types import DataType

from neuracore.data_daemon.communications_management.ring_buffer import RingBuffer
from neuracore.data_daemon.const import (
    SHARED_RING_RECORD_HEADER_FORMAT,
    SHARED_RING_RECORD_HEADER_SIZE,
    SHARED_RING_RECORD_MAGIC,
)
from neuracore.data_daemon.models import SharedRingChunkMetadata, TraceTransportMetadata

logger = logging.getLogger(__name__)


@dataclass
class PartialMessage:
    """Represents a partial logical message."""

    total_chunks: int
    received_chunks: int = 0
    chunks: dict[int, bytes] = field(default_factory=dict)
    metadata: TraceTransportMetadata | None = None

    def add_chunk(self, index: int, data: bytes) -> bool:
        """Add a chunk to the partial message."""
        if index in self.chunks:
            return self.received_chunks == self.total_chunks

        self.chunks[index] = data
        self.received_chunks += 1
        return self.received_chunks == self.total_chunks

    def assemble(self) -> bytes:
        """Assemble a complete logical message from the partial message."""
        missing = [i for i in range(self.total_chunks) if i not in self.chunks]
        if missing:
            raise ValueError(f"Missing chunks: {missing}")
        return b"".join(self.chunks[i] for i in range(self.total_chunks))

    def register_metadata(
        self, trace_id: str, metadata: TraceTransportMetadata | None
    ) -> None:
        """Remember the trace-level metadata associated with this message."""
        if metadata is None:
            return
        if self.metadata is None:
            self.metadata = metadata
            return

        merged_metadata, mismatches = self.metadata.merged_with(metadata)
        self.metadata = merged_metadata
        for key, (existing, incoming) in mismatches.items():
            logger.warning(
                "Metadata mismatch for trace_id=%s field=%s (%s -> %s)",
                trace_id,
                key,
                existing,
                incoming,
            )


@dataclass
class CompletedChannelMessage:
    """A fully assembled logical message plus optional transport metadata."""

    trace_id: str
    data_type: DataType
    payload: bytes
    metadata: TraceTransportMetadata | None = None

    def __iter__(self) -> Iterator[str | DataType | bytes]:
        """Yield the trace ID, data type, and payload of the completed channel message.

        This iterator is used to unpack the completed channel message into its
        constituent parts. The yield order is:

        1. trace_id (str)
        2. data_type (DataType)
        3. payload (bytes)

        :yields: str, DataType, bytes
        :rtype: Iterator[str, DataType, bytes]
        """
        yield self.trace_id
        yield self.data_type
        yield self.payload

    def __getitem__(self, index: int) -> str | DataType | bytes:
        """Return the component of the completed channel message at the given index.

        The index maps to the following components:

        0: trace_id (str)
        1: data_type (DataType)
        2: payload (bytes)

        :param index: The index of the component to return.
        :type index: int
        :return: The component at the given index.
        :rtype: str | DataType | bytes
        """
        return (self.trace_id, self.data_type, self.payload)[index]

    def __len__(self) -> int:
        """Return the length of the completed channel message.

        The length of the completed channel message is 3, corresponding to the
        three components of the message: trace_id, data_type, and payload.

        :return: The length of the completed channel message.
        :rtype: int
        """
        return 3

    def __eq__(self, other: object) -> bool:
        """Compare the completed channel message to another object.

        If the other object is a tuple, compare the trace ID, data type, and payload
        of the completed channel message to the corresponding components of the tuple.
        Otherwise, compare the objects using the standard equality comparison.

        :param other: The object to compare to.
        :type other: object
        :return: True if the objects are equal, False otherwise.
        :rtype: bool
        """
        if isinstance(other, tuple):
            return (self.trace_id, self.data_type, self.payload) == other
        return super().__eq__(other)


class ChannelMessageReader:
    """Reads from a RingBuffer and yields complete logical messages."""

    def __init__(self, ring_buffer: RingBuffer) -> None:
        """Initialize the ChannelMessageReader.

        :param ring_buffer: The RingBuffer instance to read from.
        :type ring_buffer: RingBuffer
        """
        self._ring_buffer = ring_buffer
        self._pending: dict[str, PartialMessage] = {}

    def has_pending_trace(self, trace_id: str) -> bool:
        """Return True when the trace still has partially assembled chunks."""
        return trace_id in self._pending

    def poll_one(self) -> CompletedChannelMessage | None:
        """Try to read and assemble one complete message."""
        packet = self._ring_buffer.read_frame_packet()
        if packet is None:
            return None

        if len(packet) < SHARED_RING_RECORD_HEADER_SIZE:
            raise ValueError(
                "Shared ring packet shorter than header: "
                f"packet_len={len(packet)} header_len={SHARED_RING_RECORD_HEADER_SIZE}"
            )

        magic, metadata_len, chunk_len = struct.unpack(
            SHARED_RING_RECORD_HEADER_FORMAT,
            packet[:SHARED_RING_RECORD_HEADER_SIZE],
        )
        if magic != SHARED_RING_RECORD_MAGIC:
            raise ValueError(
                "Invalid shared ring magic: "
                f"expected={SHARED_RING_RECORD_MAGIC!r} actual={magic!r}"
            )

        required = SHARED_RING_RECORD_HEADER_SIZE + metadata_len + chunk_len
        if len(packet) != required:
            if len(packet) < required:
                raise ValueError(
                    "Shared ring packet shorter than declared lengths: "
                    f"packet_len={len(packet)} required={required} "
                    f"metadata_len={metadata_len} chunk_len={chunk_len}"
                )
            raise ValueError(
                "Shared ring packet has trailing bytes: "
                f"packet_len={len(packet)} required={required} "
                f"trailing_bytes={len(packet) - required}"
            )

        metadata_start = SHARED_RING_RECORD_HEADER_SIZE
        metadata_end = metadata_start + metadata_len
        chunk_end = metadata_end + chunk_len
        metadata_bytes = packet[metadata_start:metadata_end]
        chunk_data = packet[metadata_end:chunk_end]
        chunk_metadata = SharedRingChunkMetadata.from_dict(
            json.loads(metadata_bytes.decode("utf-8"))
        )
        trace_id = chunk_metadata.trace_id
        chunk_index = chunk_metadata.chunk_index
        total_chunks = chunk_metadata.total_chunks

        partial_message = self._pending.get(trace_id)
        if partial_message is None:
            partial_message = PartialMessage(total_chunks=total_chunks)
            self._pending[trace_id] = partial_message
        elif partial_message.total_chunks != total_chunks:
            logger.warning(
                "Inconsistent total_chunks for trace_id=%s (existing=%d, new=%d)",
                trace_id,
                partial_message.total_chunks,
                total_chunks,
            )

        partial_message.register_metadata(
            trace_id,
            None if chunk_metadata is None else chunk_metadata.trace_metadata,
        )

        complete = partial_message.add_chunk(chunk_index, chunk_data)
        if not complete:
            return None

        try:
            payload = partial_message.assemble()
        except ValueError as exc:
            logger.error("Failed to assemble trace_id=%s: %s", trace_id, exc)
            del self._pending[trace_id]
            return None

        trace_metadata = partial_message.metadata
        if trace_metadata is not None:
            data_type = trace_metadata.data_type
        else:
            raise ValueError(
                f"Missing trace metadata for shared-ring trace_id={trace_id}."
            )

        del self._pending[trace_id]
        return CompletedChannelMessage(
            trace_id=trace_id,
            data_type=data_type,
            payload=payload,
            metadata=trace_metadata,
        )
