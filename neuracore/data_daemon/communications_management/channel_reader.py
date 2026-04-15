"""Reads from a RingBuffer and yields complete logical messages."""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass, field

from neuracore_types import DataType

from neuracore.data_daemon.communications_management.ring_buffer import RingBuffer
from neuracore.data_daemon.const import (
    CHUNK_HEADER_FORMAT,
    CHUNK_HEADER_SIZE,
    DATA_TYPE_FIELD_SIZE,
    SHARED_RING_RECORD_HEADER_FORMAT,
    SHARED_RING_RECORD_HEADER_SIZE,
    SHARED_RING_RECORD_MAGIC,
)

logger = logging.getLogger(__name__)


@dataclass
class PartialMessage:
    """Represents a partial logical message."""

    total_chunks: int
    received_chunks: int = 0
    chunks: dict[int, bytes] = field(default_factory=dict)

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


@dataclass
class CompletedChannelMessage:
    """A fully assembled logical message plus optional transport metadata."""

    trace_id: str
    data_type: DataType
    payload: bytes
    metadata: dict[str, str | int | None] = field(default_factory=dict)

    def __iter__(self):
        yield self.trace_id
        yield self.data_type
        yield self.payload

    def __getitem__(self, index: int):
        return (self.trace_id, self.data_type, self.payload)[index]

    def __len__(self) -> int:
        return 3

    def __eq__(self, other: object) -> bool:
        if isinstance(other, tuple):
            return (self.trace_id, self.data_type, self.payload) == other
        return super().__eq__(other)


class ChannelMessageReader:
    """Reads from a RingBuffer and yields complete logical messages."""

    def __init__(self, ring_buffer: RingBuffer) -> None:
        self._ring_buffer = ring_buffer
        self._pending: dict[str, PartialMessage] = {}

    def has_pending_trace(self, trace_id: str) -> bool:
        """Return True when the trace still has partially assembled chunks."""
        return trace_id in self._pending

    def poll_one(self) -> CompletedChannelMessage | None:
        """Try to read and assemble one complete message."""
        available = self._ring_buffer.available()
        if available < 4:
            return None

        prefix = self._ring_buffer.peek(4)
        if prefix is None:
            return None

        if prefix == SHARED_RING_RECORD_MAGIC:
            if available < SHARED_RING_RECORD_HEADER_SIZE:
                return None

            header_bytes = self._ring_buffer.peek(SHARED_RING_RECORD_HEADER_SIZE)
            if header_bytes is None:
                return None

            _magic, metadata_len, chunk_len = struct.unpack(
                SHARED_RING_RECORD_HEADER_FORMAT,
                header_bytes,
            )
            required = SHARED_RING_RECORD_HEADER_SIZE + metadata_len + chunk_len
            if available < required:
                return None

            packet = self._ring_buffer.read(required)
            if packet is None:
                return None

            metadata_bytes = packet[
                SHARED_RING_RECORD_HEADER_SIZE : SHARED_RING_RECORD_HEADER_SIZE
                + metadata_len
            ]
            metadata = json.loads(metadata_bytes.decode("utf-8"))
            trace_id = str(metadata["trace_id"])
            data_type = DataType(str(metadata["data_type"]))
            chunk_index = int(metadata["chunk_index"])
            total_chunks = int(metadata["total_chunks"])
            chunk_data = packet[SHARED_RING_RECORD_HEADER_SIZE + metadata_len :]
        else:
            if available < CHUNK_HEADER_SIZE:
                return None

            header_bytes = self._ring_buffer.peek(CHUNK_HEADER_SIZE)
            if header_bytes is None:
                return None

            raw_trace_id, raw_data_type, chunk_index, total_chunks, chunk_len = (
                struct.unpack(CHUNK_HEADER_FORMAT, header_bytes)
            )
            trace_id = raw_trace_id.rstrip(b"\x00").decode("utf-8", errors="ignore")
            data_type_str = raw_data_type.rstrip(b"\x00").decode("utf-8", errors="ignore")
            if len(data_type_str) > DATA_TYPE_FIELD_SIZE:
                data_type_str = data_type_str[:DATA_TYPE_FIELD_SIZE]
            try:
                data_type = DataType(data_type_str)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown data_type '{data_type_str}' for trace_id={trace_id}. "
                ) from exc

            required = CHUNK_HEADER_SIZE + chunk_len
            if available < required:
                return None

            packet = self._ring_buffer.read(required)
            if packet is None:
                return None

            chunk_data = packet[CHUNK_HEADER_SIZE:]
            metadata = {}

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

        complete = partial_message.add_chunk(chunk_index, chunk_data)
        if not complete:
            return None

        try:
            payload = partial_message.assemble()
        except ValueError as exc:
            logger.error("Failed to assemble trace_id=%s: %s", trace_id, exc)
            del self._pending[trace_id]
            return None

        del self._pending[trace_id]
        return CompletedChannelMessage(
            trace_id=trace_id,
            data_type=data_type,
            payload=payload,
            metadata=metadata,
        )
