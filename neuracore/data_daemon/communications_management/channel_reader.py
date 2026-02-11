"""Reads from a RingBuffer and yields complete logical messages."""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field

from neuracore_types import DataType

from neuracore.data_daemon.communications_management.ring_buffer import RingBuffer
from neuracore.data_daemon.const import (
    CHUNK_HEADER_FORMAT,
    CHUNK_HEADER_SIZE,
    DATA_TYPE_FIELD_SIZE,
)

logger = logging.getLogger(__name__)


@dataclass
class PartialMessage:
    """Represents a partial logical message."""

    total_chunks: int
    received_chunks: int = 0
    chunks: dict[int, bytes] = field(default_factory=dict)

    def add_chunk(self, index: int, data: bytes) -> bool:
        """Add a chunk to the partial message.

        If the chunk is already present in the partial message, returns
        True if the partial message is complete, False otherwise.

        If the chunk is not present, adds it to the partial message and
        returns True if the partial message is complete, False otherwise.

        :param index: index of the chunk
        :param data: the chunk data
        :return: whether the partial message is complete
        """
        if index in self.chunks:
            return self.received_chunks == self.total_chunks

        self.chunks[index] = data
        self.received_chunks += 1
        return self.received_chunks == self.total_chunks

    def assemble(self) -> bytes:
        """Assemble a complete logical message from the partial message.

        If the partial message is missing any chunks, raises a ValueError
        with a list of the missing chunks.

        :return: the complete logical message as bytes
        """
        missing = [i for i in range(self.total_chunks) if i not in self.chunks]
        if missing:
            raise ValueError(f"Missing chunks: {missing}")
        return b"".join(self.chunks[i] for i in range(self.total_chunks))


class ChannelMessageReader:
    """Reads from a RingBuffer and yields complete logical messages.

    - Uses the [header][chunk] format written by the daemon.
    - Keeps PartialMessage state in memory while assembling.
    """

    def __init__(self, ring_buffer: RingBuffer) -> None:
        """Initialize the ChannelMessageReader."""
        self._ring_buffer = ring_buffer
        self._pending: dict[str, PartialMessage] = {}

    def poll_one(self) -> tuple[str, DataType, bytes] | None:
        """Try to read and assemble one complete message.

        Returns:
            (trace_id, data_type, payload_bytes) if a complete message is ready,
            or None if not enough data yet.
        """
        # Need at least a header to proceed
        if self._ring_buffer.available() < CHUNK_HEADER_SIZE:
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
        except ValueError:
            raise ValueError(
                f"Unknown data_type '{data_type_str}' for trace_id={trace_id}. "
            )

        # Now check if we have header + the full chunk
        required = CHUNK_HEADER_SIZE + chunk_len
        if self._ring_buffer.available() < required:
            # Not enough data yet; do not consume anything
            return None

        # Consume header + chunk in one go
        packet = self._ring_buffer.read(required)
        if packet is None:
            return None

        chunk_data = packet[CHUNK_HEADER_SIZE:]

        # Build or update PartialMessage
        partial_message = self._pending.get(trace_id)
        if partial_message is None:
            partial_message = PartialMessage(total_chunks=total_chunks)
            self._pending[trace_id] = partial_message
        else:
            if partial_message.total_chunks != total_chunks:
                logger.warning(
                    "Inconsistent total_chunks for trace_id=%s (existing=%d, new=%d)",
                    trace_id,
                    partial_message.total_chunks,
                    total_chunks,
                )

        complete = partial_message.add_chunk(chunk_index, chunk_data)
        if not complete:
            return None
        # Full message assembled
        try:
            payload = partial_message.assemble()
        except ValueError as exc:
            logger.error(
                "Failed to assemble trace_id=%s: %s",
                trace_id,
                exc,
            )
            del self._pending[trace_id]
            return None

        del self._pending[trace_id]
        return trace_id, data_type, payload
