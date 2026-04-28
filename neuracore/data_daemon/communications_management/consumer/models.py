"""Consumer-side state and work models for daemon bridge processing."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

from neuracore_types import DataType

from neuracore.data_daemon.const import (
    HEARTBEAT_TIMEOUT_SECS,
    NEVER_OPENED_TIMEOUT_SECS,
)
from neuracore.data_daemon.helpers import utc_now
from neuracore.data_daemon.models import DataChunkPayload, TraceTransportMetadata

from .bridge_chunk_spool import BridgeChunkSpool, ChunkSpoolRef

logger = logging.getLogger(__name__)


@dataclass
class PartialMessage:
    """Represents a partial logical message."""

    total_chunks: int
    received_chunks: int = 0
    chunks: dict[int, bytes] = field(default_factory=dict)
    metadata: TraceTransportMetadata | None = None

    def add_chunk(self, index: int, data: bytes) -> bool:
        """Store one socket chunk and report whether the message is complete."""
        if index in self.chunks:
            return self.received_chunks == self.total_chunks

        self.chunks[index] = data
        self.received_chunks += 1
        return self.received_chunks == self.total_chunks

    def assemble(self) -> bytes:
        """Assemble all received chunks into one contiguous payload."""
        missing = [i for i in range(self.total_chunks) if i not in self.chunks]
        if missing:
            raise ValueError(f"Missing chunks: {missing}")
        return b"".join(self.chunks[i] for i in range(self.total_chunks))

    def register_metadata(
        self, trace_id: str, metadata: TraceTransportMetadata | None
    ) -> None:
        """Merge trace metadata observed across chunks for one logical message."""
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
        """Iterate as `(trace_id, data_type, payload)` for compatibility."""
        yield self.trace_id
        yield self.data_type
        yield self.payload

    def __getitem__(self, index: int) -> str | DataType | bytes:
        """Return one tuple-style field by positional index."""
        return (self.trace_id, self.data_type, self.payload)[index]

    def __len__(self) -> int:
        """Return the tuple-style field count."""
        return 3

    def __eq__(self, other: object) -> bool:
        """Compare equal to compatible tuples for legacy callers."""
        if isinstance(other, tuple):
            return (self.trace_id, self.data_type, self.payload) == other
        return super().__eq__(other)


class TransportMode(str, Enum):
    """Available channel transport modes."""

    NONE = "none"
    SOCKET = "socket"
    SHARED_MEMORY = "shared_memory"


@dataclass
class SharedSlotTransportState:
    """Mutable daemon-side shared-slot attachment state for one channel."""

    control_endpoint: str | None = None
    shm_name: str | None = None

    def reset(self) -> None:
        """Clear any shared-slot attachment details for the channel."""
        self.control_endpoint = None
        self.shm_name = None


@dataclass(frozen=True)
class CompletionChunkWork:
    """Completion-worker input for one spooled chunk of trace data."""

    producer_id: str
    trace_id: str
    recording_id: str
    chunk_index: int
    total_chunks: int
    chunk_spool: BridgeChunkSpool
    chunk_spool_ref: ChunkSpoolRef
    trace_metadata: TraceTransportMetadata | None = None
    fallback_data_type: DataType | None = None


@dataclass
class SpoolPartialMessage:
    """Partial message assembled from chunk spool references."""

    total_chunks: int
    received_chunks: int = 0
    chunks: dict[int, tuple[BridgeChunkSpool, ChunkSpoolRef]] = field(
        default_factory=dict
    )
    metadata: TraceTransportMetadata | None = None

    def add_chunk(
        self,
        index: int,
        spool: BridgeChunkSpool,
        ref: ChunkSpoolRef,
    ) -> bool:
        """Store one spooled chunk reference and report completion status."""
        if index in self.chunks:
            return self.received_chunks == self.total_chunks

        self.chunks[index] = (spool, ref)
        self.received_chunks += 1
        return self.received_chunks == self.total_chunks

    def ordered_refs(self) -> list[tuple[BridgeChunkSpool, ChunkSpoolRef]]:
        """Return chunk references ordered by their original chunk index."""
        missing = [i for i in range(self.total_chunks) if i not in self.chunks]
        if missing:
            raise ValueError(f"Missing chunks: {missing}")
        return [self.chunks[i] for i in range(self.total_chunks)]

    def register_metadata(
        self, trace_id: str, metadata: TraceTransportMetadata | None
    ) -> None:
        """Merge trace metadata observed while spooling chunk descriptors."""
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


@dataclass(frozen=True)
class FinalTraceWork:
    """Completion-worker input representing an explicit end-of-trace marker."""

    producer_id: str
    trace_id: str
    recording_id: str
    data_type: DataType
    metadata: dict[str, str | int | None]


@dataclass(frozen=True)
class SpoolDescriptorWork:
    """Spool-worker input for one shared-slot descriptor payload."""

    channel: ChannelState
    descriptor_payload: dict


@dataclass(frozen=True)
class RecordingDataDropRequest:
    """Context used to decide whether recording data should be discarded."""

    channel: ChannelState
    recording_id: str
    trace_id: str
    sequence_number: int | None


@dataclass(frozen=True)
class TraceMetadataSnapshot:
    """Normalized per-trace metadata stored by the daemon bridge."""

    dataset_id: str | None = None
    dataset_name: str | None = None
    robot_name: str | None = None
    robot_id: str | None = None
    robot_instance: int | None = None
    data_type: str | None = None
    data_type_name: str | None = None

    def to_dict(self) -> dict[str, str | int | None]:
        """Serialize trace metadata to a plain dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "robot_name": self.robot_name,
            "robot_id": self.robot_id,
            "robot_instance": self.robot_instance,
            "data_type": self.data_type,
            "data_type_name": self.data_type_name,
        }


@dataclass(frozen=True)
class TraceMetadataRegistrationRequest:
    """Request to merge metadata into the trace metadata registry."""

    trace_id: str
    metadata: TraceMetadataSnapshot


@dataclass(frozen=True)
class TraceRecordingLookupRequest:
    """Lookup request for resolving a trace's recording identifier."""

    trace_id: str


@dataclass(frozen=True)
class SharedSlotSequenceProgressRequest:
    """Request to update shared-slot descriptor progress for one producer."""

    producer_id: str
    sequence_number: int


@dataclass
class ChannelRegistry:
    """Registry of active producer channels keyed by producer id."""

    _channels: dict[str, ChannelState] = field(default_factory=dict)

    def get(self, producer_id: str) -> ChannelState | None:
        """Return the active channel for one producer, if present."""
        return self._channels.get(producer_id)

    def add(self, channel: ChannelState) -> None:
        """Register or replace one active producer channel."""
        self._channels[channel.producer_id] = channel

    def remove(self, producer_id: str) -> ChannelState | None:
        """Remove and return one producer channel, if present."""
        return self._channels.pop(producer_id, None)

    def items(self) -> Iterator[tuple[str, ChannelState]]:
        """Iterate over `(producer_id, channel)` pairs."""
        return iter(self._channels.items())

    def values(self) -> Iterator[ChannelState]:
        """Iterate over registered channel states."""
        return iter(self._channels.values())


@dataclass
class ClosedProducerRegistry:
    """Track producers that have been explicitly closed."""

    _producer_ids: set[str] = field(default_factory=set)

    def add(self, producer_id: str) -> None:
        """Mark a producer as closed."""
        self._producer_ids.add(producer_id)

    def discard(self, producer_id: str) -> None:
        """Remove a producer from the closed set when it reconnects."""
        self._producer_ids.discard(producer_id)

    def contains(self, producer_id: str) -> bool:
        """Return whether the producer is currently marked closed."""
        return producer_id in self._producer_ids


@dataclass
class ProducerSequenceRegistry:
    """Track latest observed sequence numbers per producer."""

    _last_sequence_numbers: dict[str, int] = field(default_factory=dict)

    def update(self, producer_id: str, sequence_number: int) -> None:
        """Store the latest observed sequence number for one producer."""
        self._last_sequence_numbers[producer_id] = sequence_number

    def get(self, producer_id: str) -> int | None:
        """Return the latest observed sequence number for one producer."""
        return self._last_sequence_numbers.get(producer_id)

    def set_max(self, producer_id: str, sequence_number: int) -> None:
        """Raise the stored sequence number only when the new value is larger."""
        self._last_sequence_numbers[producer_id] = max(
            self._last_sequence_numbers.get(producer_id, 0),
            sequence_number,
        )


@dataclass
class PendingSharedSlotSequenceRegistry:
    """Track shared-slot descriptors still pending completion per producer."""

    _pending_by_producer: dict[str, set[int]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, producer_id: str, sequence_number: int) -> None:
        """Record a descriptor sequence as pending completion."""
        with self._lock:
            self._pending_by_producer.setdefault(producer_id, set()).add(
                sequence_number
            )

    def complete(self, producer_id: str, sequence_number: int) -> None:
        """Mark a pending descriptor sequence as completed."""
        with self._lock:
            pending = self._pending_by_producer.get(producer_id)
            if pending is None:
                return
            pending.discard(sequence_number)
            if not pending:
                self._pending_by_producer.pop(producer_id, None)

    def has_pending_at_or_before(
        self,
        producer_id: str,
        cutoff_sequence_number: int,
    ) -> bool:
        """Return whether any pending sequence is at or before the cutoff."""
        with self._lock:
            pending = self._pending_by_producer.get(producer_id, set())
            return any(
                sequence_number <= cutoff_sequence_number for sequence_number in pending
            )


@dataclass
class ChannelState:
    """Mutable transport and heartbeat state for one producer channel."""

    producer_id: str
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_sequence_number: int = 0
    opened_at: datetime | None = None
    heartbeat_expired_at: datetime | None = None
    transport_mode: TransportMode = TransportMode.NONE
    socket_pending_messages: dict[str, PartialMessage] = field(default_factory=dict)
    shared_slot: SharedSlotTransportState = field(
        default_factory=SharedSlotTransportState
    )

    def is_opened(self) -> bool:
        """Return whether any transport for this channel has been opened."""
        return self.transport_mode is not TransportMode.NONE

    def touch(self) -> None:
        """Refresh heartbeat tracking for this channel."""
        self.last_heartbeat = datetime.now(timezone.utc)
        self.heartbeat_expired_at = None

    def is_open(self) -> bool:
        """Return whether the channel currently has an active transport."""
        return self.transport_mode is not TransportMode.NONE

    def mark_socket_transport_open(self) -> None:
        """Mark the channel as using socket-based chunk transport."""
        self.transport_mode = TransportMode.SOCKET
        if self.opened_at is None:
            self.opened_at = datetime.now(timezone.utc)

    def mark_shared_slot_transport_open(
        self,
        *,
        control_endpoint: str,
        shm_name: str,
    ) -> None:
        """Mark the channel as using shared-slot transport."""
        self.transport_mode = TransportMode.SHARED_MEMORY
        self.shared_slot.control_endpoint = control_endpoint
        self.shared_slot.shm_name = shm_name
        self.opened_at = datetime.now(timezone.utc)

    def mark_shared_slot_descriptor_seen(self, *, shm_name: str) -> None:
        """Record that shared-slot traffic has been observed for this channel."""
        self.transport_mode = TransportMode.SHARED_MEMORY
        self.shared_slot.shm_name = shm_name
        if self.opened_at is None:
            self.opened_at = datetime.now(timezone.utc)

    def uses_shared_memory_transport(self) -> bool:
        """Return whether the channel is currently using shared-slot transport."""
        return self.transport_mode is TransportMode.SHARED_MEMORY

    def clear_transport_state(self) -> None:
        """Reset transport-specific state after a trace finishes or closes."""
        self.opened_at = None
        self.transport_mode = TransportMode.NONE
        self.shared_slot.reset()
        self.socket_pending_messages.clear()

    def add_socket_data_chunk(
        self, data_chunk: DataChunkPayload
    ) -> CompletedChannelMessage | None:
        """Add one socket chunk and return a completed message when ready."""
        self.mark_socket_transport_open()
        return self.add_transport_chunk(
            trace_id=data_chunk.trace_id,
            chunk_index=data_chunk.chunk_index,
            total_chunks=data_chunk.total_chunks,
            chunk_data=data_chunk.data,
            trace_metadata=data_chunk.trace_metadata,
            fallback_data_type=data_chunk.data_type,
        )

    def add_transport_chunk(
        self,
        *,
        trace_id: str,
        chunk_index: int,
        total_chunks: int,
        chunk_data: bytes,
        trace_metadata: TraceTransportMetadata | None,
        fallback_data_type: DataType | None = None,
    ) -> CompletedChannelMessage | None:
        """Add one chunk to the active logical message for a trace."""
        partial_message = self.socket_pending_messages.get(trace_id)
        if partial_message is None:
            partial_message = PartialMessage(total_chunks=total_chunks)
            self.socket_pending_messages[trace_id] = partial_message
        elif partial_message.total_chunks != total_chunks:
            logger.warning(
                "Inconsistent total_chunks for trace_id=%s (existing=%d, new=%d)",
                trace_id,
                partial_message.total_chunks,
                total_chunks,
            )

        partial_message.register_metadata(trace_id, trace_metadata)
        complete = partial_message.add_chunk(chunk_index, chunk_data)
        if not complete:
            return None

        self.socket_pending_messages.pop(trace_id, None)
        return self._assemble_completed_message(
            trace_id=trace_id,
            partial_message=partial_message,
            fallback_data_type=fallback_data_type,
        )

    def _assemble_completed_message(
        self,
        *,
        trace_id: str,
        partial_message: PartialMessage,
        fallback_data_type: DataType | None,
    ) -> CompletedChannelMessage | None:
        try:
            payload = partial_message.assemble()
        except ValueError as exc:
            logger.error("Failed to assemble trace_id=%s: %s", trace_id, exc)
            return None

        metadata = partial_message.metadata
        if metadata is not None:
            data_type = metadata.data_type
        elif fallback_data_type is not None:
            data_type = fallback_data_type
        else:
            raise ValueError(f"Missing data_type in metadata for trace_id={trace_id}.")

        return CompletedChannelMessage(
            trace_id=trace_id,
            data_type=data_type,
            payload=payload,
            metadata=metadata,
        )

    def has_missed_heartbeat(
        self,
        now: datetime,
        heartbeat_timeout: timedelta | None = None,
    ) -> bool:
        """Return whether the channel heartbeat has expired."""
        if heartbeat_timeout is None:
            heartbeat_timeout = timedelta(seconds=HEARTBEAT_TIMEOUT_SECS)
        return now - self.last_heartbeat > heartbeat_timeout

    def is_stale_unopened(
        self,
        now: datetime,
        never_opened_timeout: timedelta | None = None,
    ) -> bool:
        """Return whether a never-opened channel has gone stale."""
        if never_opened_timeout is None:
            never_opened_timeout = timedelta(seconds=NEVER_OPENED_TIMEOUT_SECS)
        return (not self.is_open()) and (now - self.created_at > never_opened_timeout)

    def should_expire(self) -> bool:
        """Return whether the daemon should expire this channel."""
        now = utc_now()
        return self.has_missed_heartbeat(now) or self.is_stale_unopened(now)

    def set_trace_id(self, trace_id: str) -> None:
        """Update the channel's active trace identifier."""
        if trace_id != self.trace_id:
            self.trace_id = trace_id


@dataclass
class PendingTraceEnd:
    """Deferred finalization state for a trace end awaiting transport drain."""

    producer_id: str
    recording_id: str
    trace_id: str
    data_type: DataType
    sequence_number: int | None


@dataclass
class RecordingClosingState:
    """Recording shutdown state including producer stop cutoffs."""

    producer_stop_sequence_numbers: dict[str, int]
    stop_requested_at: datetime


@dataclass
class PendingTraceEndRegistry:
    """Registry of trace-end work waiting for safe finalization."""

    _pending_by_trace: dict[str, PendingTraceEnd] = field(default_factory=dict)

    def add(self, pending_trace_end: PendingTraceEnd) -> None:
        """Store pending trace-end state for one trace."""
        self._pending_by_trace[pending_trace_end.trace_id] = pending_trace_end

    def get(self, trace_id: str) -> PendingTraceEnd | None:
        """Return pending trace-end state for one trace, if present."""
        return self._pending_by_trace.get(trace_id)

    def pop(self, trace_id: str) -> PendingTraceEnd | None:
        """Remove and return pending trace-end state for one trace."""
        return self._pending_by_trace.pop(trace_id, None)


@dataclass
class FinalChunkRegistry:
    """Track traces that have already enqueued their final chunk."""

    _trace_ids: set[str] = field(default_factory=set)

    def add(self, trace_id: str) -> None:
        """Mark a trace as having enqueued its final chunk."""
        self._trace_ids.add(trace_id)

    def discard(self, trace_id: str) -> None:
        """Remove a trace from the final-chunk set."""
        self._trace_ids.discard(trace_id)

    def contains(self, trace_id: str) -> bool:
        """Return whether the trace has already enqueued a final chunk."""
        return trace_id in self._trace_ids


@dataclass
class TraceMetadataRegistry:
    """Registry of normalized metadata keyed by trace id."""

    _metadata_by_trace: dict[str, dict[str, str | int | None]] = field(
        default_factory=dict
    )

    def get(self, trace_id: str) -> dict[str, str | int | None]:
        """Return stored metadata for one trace."""
        return self._metadata_by_trace.get(trace_id, {})

    def pop(self, trace_id: str) -> dict[str, str | int | None] | None:
        """Remove and return stored metadata for one trace."""
        return self._metadata_by_trace.pop(trace_id, None)

    def register(
        self,
        request: TraceMetadataRegistrationRequest,
    ) -> list[tuple[str, str | int | None, str | int | None]]:
        """Merge incoming metadata and return any conflicting field changes."""
        trace_id = request.trace_id
        metadata = request.metadata.to_dict()
        existing = self._metadata_by_trace.get(trace_id)
        if existing is None:
            self._metadata_by_trace[trace_id] = dict(metadata)
            return []

        mismatches: list[tuple[str, str | int | None, str | int | None]] = []
        for key, value in metadata.items():
            if existing.get(key) is None and value is not None:
                existing[key] = value
            elif value is not None and existing.get(key) not in (None, value):
                mismatches.append((key, existing.get(key), value))
        return mismatches


@dataclass
class TraceRecordingRegistry:
    """Bi-directional mapping between recordings and active traces."""

    _recording_by_trace: dict[str, str] = field(default_factory=dict)
    _traces_by_recording: dict[str, set[str]] = field(default_factory=dict)
    _unique_traces_by_recording: dict[str, set[str]] = field(default_factory=dict)

    def get_recording_id(self, trace_id: str) -> str | None:
        """Return the recording currently associated with one trace."""
        return self._recording_by_trace.get(trace_id)

    def register(self, recording_id: str, trace_id: str) -> str | None:
        """Associate a trace with a recording and return any previous mapping."""
        previous_recording_id = self._recording_by_trace.get(trace_id)
        if previous_recording_id == recording_id:
            self._traces_by_recording.setdefault(recording_id, set()).add(trace_id)
            self._unique_traces_by_recording.setdefault(recording_id, set()).add(
                trace_id
            )
            return None

        if previous_recording_id is not None:
            self._traces_by_recording.get(previous_recording_id, set()).discard(
                trace_id
            )
            previous_unique_traces = self._unique_traces_by_recording.get(
                previous_recording_id
            )
            if previous_unique_traces is not None:
                previous_unique_traces.discard(trace_id)
                if not previous_unique_traces:
                    self._unique_traces_by_recording.pop(previous_recording_id, None)

        self._recording_by_trace[trace_id] = recording_id
        self._traces_by_recording.setdefault(recording_id, set()).add(trace_id)
        self._unique_traces_by_recording.setdefault(recording_id, set()).add(trace_id)
        return previous_recording_id

    def remove_trace(self, recording_id: str, trace_id: str) -> None:
        """Remove one trace from the recording registries."""
        self._recording_by_trace.pop(trace_id, None)

        traces = self._traces_by_recording.get(recording_id)
        if traces is None:
            return

        traces.discard(trace_id)
        if not traces:
            self._traces_by_recording.pop(recording_id, None)

    def traces_for_recording(self, recording_id: str) -> set[str]:
        """Return the currently active traces for one recording."""
        return set(self._traces_by_recording.get(recording_id, set()))

    def unique_trace_count(self, recording_id: str) -> int:
        """Return the count of unique traces seen for one recording."""
        return len(self._unique_traces_by_recording.get(recording_id, set()))

    def clear_unique_traces(self, recording_id: str) -> None:
        """Forget the unique-trace history for one recording."""
        self._unique_traces_by_recording.pop(recording_id, None)

    def has_active_traces(self, recording_id: str) -> bool:
        """Return whether the recording still has active traces."""
        return bool(self._traces_by_recording.get(recording_id, set()))


@dataclass
class RecordingCloseRegistry:
    """Track recordings that are closing or fully closed."""

    _closing_by_recording: dict[str, RecordingClosingState] = field(
        default_factory=dict
    )
    _closed_recording_ids: set[str] = field(default_factory=set)

    def is_closed(self, recording_id: str) -> bool:
        """Return whether the recording has already been marked closed."""
        return recording_id in self._closed_recording_ids

    def mark_closing(
        self,
        recording_id: str,
        closing_state: RecordingClosingState,
    ) -> None:
        """Store closing state for a recording awaiting finalization."""
        self._closing_by_recording[recording_id] = closing_state

    def get_closing(self, recording_id: str) -> RecordingClosingState | None:
        """Return closing state for one recording, if present."""
        return self._closing_by_recording.get(recording_id)

    def items(self) -> Iterator[tuple[str, RecordingClosingState]]:
        """Iterate over recordings that are currently closing."""
        return iter(self._closing_by_recording.items())

    def close(self, recording_id: str) -> None:
        """Mark a recording as closed and clear any closing state."""
        self._closing_by_recording.pop(recording_id, None)
        self._closed_recording_ids.add(recording_id)
