"""Helpers for assembling complete channel messages from transport chunks."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

from neuracore_types import DataType

from neuracore.data_daemon.models import TraceTransportMetadata

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

