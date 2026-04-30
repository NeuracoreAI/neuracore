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
        if index in self.chunks:
            return self.received_chunks == self.total_chunks

        self.chunks[index] = data
        self.received_chunks += 1
        return self.received_chunks == self.total_chunks

    def assemble(self) -> bytes:
        missing = [i for i in range(self.total_chunks) if i not in self.chunks]
        if missing:
            raise ValueError(f"Missing chunks: {missing}")
        return b"".join(self.chunks[i] for i in range(self.total_chunks))

    def register_metadata(
        self, trace_id: str, metadata: TraceTransportMetadata | None
    ) -> None:
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
        yield self.trace_id
        yield self.data_type
        yield self.payload

    def __getitem__(self, index: int) -> str | DataType | bytes:
        return (self.trace_id, self.data_type, self.payload)[index]

    def __len__(self) -> int:
        return 3

    def __eq__(self, other: object) -> bool:
        if isinstance(other, tuple):
            return (self.trace_id, self.data_type, self.payload) == other
        return super().__eq__(other)


class TransportMode(str, Enum):
    NONE = "none"
    SOCKET = "socket"
    SHARED_MEMORY = "shared_memory"


@dataclass
class SharedSlotTransportState:
    control_endpoint: str | None = None
    shm_name: str | None = None

    def reset(self) -> None:
        self.control_endpoint = None
        self.shm_name = None


@dataclass(frozen=True)
class CompletionChunkWork:
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
        if index in self.chunks:
            return self.received_chunks == self.total_chunks

        self.chunks[index] = (spool, ref)
        self.received_chunks += 1
        return self.received_chunks == self.total_chunks

    def ordered_refs(self) -> list[tuple[BridgeChunkSpool, ChunkSpoolRef]]:
        missing = [i for i in range(self.total_chunks) if i not in self.chunks]
        if missing:
            raise ValueError(f"Missing chunks: {missing}")
        return [self.chunks[i] for i in range(self.total_chunks)]

    def register_metadata(
        self, trace_id: str, metadata: TraceTransportMetadata | None
    ) -> None:
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
    producer_id: str
    trace_id: str
    recording_id: str
    data_type: DataType
    metadata: dict[str, str | int | None]


@dataclass(frozen=True)
class SpoolDescriptorWork:
    channel: ChannelState
    descriptor_payload: dict


@dataclass(frozen=True)
class RecordingDataDropRequest:
    channel: ChannelState
    recording_id: str
    trace_id: str
    sequence_number: int | None


@dataclass(frozen=True)
class TraceMetadataSnapshot:
    dataset_id: str | None = None
    dataset_name: str | None = None
    robot_name: str | None = None
    robot_id: str | None = None
    robot_instance: int | None = None
    data_type: str | None = None
    data_type_name: str | None = None

    def to_dict(self) -> dict[str, str | int | None]:
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
    trace_id: str
    metadata: TraceMetadataSnapshot


@dataclass(frozen=True)
class TraceRecordingLookupRequest:
    trace_id: str


@dataclass(frozen=True)
class SharedSlotSequenceProgressRequest:
    producer_id: str
    sequence_number: int


@dataclass
class ChannelRegistry:
    _channels: dict[str, "ChannelState"] = field(default_factory=dict)

    def get(self, producer_id: str) -> "ChannelState" | None:
        return self._channels.get(producer_id)

    def add(self, channel: "ChannelState") -> None:
        self._channels[channel.producer_id] = channel

    def remove(self, producer_id: str) -> "ChannelState" | None:
        return self._channels.pop(producer_id, None)

    def items(self) -> Iterator[tuple[str, "ChannelState"]]:
        return iter(self._channels.items())

    def values(self) -> Iterator["ChannelState"]:
        return iter(self._channels.values())


@dataclass
class ClosedProducerRegistry:
    _producer_ids: set[str] = field(default_factory=set)

    def add(self, producer_id: str) -> None:
        self._producer_ids.add(producer_id)

    def discard(self, producer_id: str) -> None:
        self._producer_ids.discard(producer_id)

    def contains(self, producer_id: str) -> bool:
        return producer_id in self._producer_ids


@dataclass
class ProducerSequenceRegistry:
    _last_sequence_numbers: dict[str, int] = field(default_factory=dict)

    def update(self, producer_id: str, sequence_number: int) -> None:
        self._last_sequence_numbers[producer_id] = sequence_number

    def get(self, producer_id: str) -> int | None:
        return self._last_sequence_numbers.get(producer_id)

    def set_max(self, producer_id: str, sequence_number: int) -> None:
        self._last_sequence_numbers[producer_id] = max(
            self._last_sequence_numbers.get(producer_id, 0),
            sequence_number,
        )


@dataclass
class PendingSharedSlotSequenceRegistry:
    _pending_by_producer: dict[str, set[int]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, producer_id: str, sequence_number: int) -> None:
        with self._lock:
            self._pending_by_producer.setdefault(producer_id, set()).add(
                sequence_number
            )

    def complete(self, producer_id: str, sequence_number: int) -> None:
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
        with self._lock:
            pending = self._pending_by_producer.get(producer_id, set())
            return any(
                sequence_number <= cutoff_sequence_number
                for sequence_number in pending
            )


@dataclass
class ChannelState:
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
        return self.transport_mode is not TransportMode.NONE

    def touch(self) -> None:
        self.last_heartbeat = datetime.now(timezone.utc)
        self.heartbeat_expired_at = None

    def is_open(self) -> bool:
        return self.transport_mode is not TransportMode.NONE

    def mark_socket_transport_open(self) -> None:
        self.transport_mode = TransportMode.SOCKET
        if self.opened_at is None:
            self.opened_at = datetime.now(timezone.utc)

    def mark_shared_slot_transport_open(
        self,
        *,
        control_endpoint: str,
        shm_name: str,
    ) -> None:
        self.transport_mode = TransportMode.SHARED_MEMORY
        self.shared_slot.control_endpoint = control_endpoint
        self.shared_slot.shm_name = shm_name
        self.opened_at = datetime.now(timezone.utc)

    def mark_shared_slot_descriptor_seen(self, *, shm_name: str) -> None:
        self.transport_mode = TransportMode.SHARED_MEMORY
        self.shared_slot.shm_name = shm_name
        if self.opened_at is None:
            self.opened_at = datetime.now(timezone.utc)

    def uses_shared_memory_transport(self) -> bool:
        return self.transport_mode is TransportMode.SHARED_MEMORY

    def clear_transport_state(self) -> None:
        self.opened_at = None
        self.transport_mode = TransportMode.NONE
        self.shared_slot.reset()
        self.socket_pending_messages.clear()

    def add_socket_data_chunk(
        self, data_chunk: DataChunkPayload
    ) -> CompletedChannelMessage | None:
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
        if heartbeat_timeout is None:
            heartbeat_timeout = timedelta(seconds=HEARTBEAT_TIMEOUT_SECS)
        return now - self.last_heartbeat > heartbeat_timeout

    def is_stale_unopened(
        self,
        now: datetime,
        never_opened_timeout: timedelta | None = None,
    ) -> bool:
        if never_opened_timeout is None:
            never_opened_timeout = timedelta(seconds=NEVER_OPENED_TIMEOUT_SECS)
        return (not self.is_open()) and (now - self.created_at > never_opened_timeout)

    def should_expire(self) -> bool:
        now = utc_now()
        return self.has_missed_heartbeat(now) or self.is_stale_unopened(now)

    def set_trace_id(self, trace_id: str) -> None:
        if trace_id != self.trace_id:
            self.trace_id = trace_id


@dataclass
class PendingTraceEnd:
    producer_id: str
    recording_id: str
    trace_id: str
    data_type: DataType
    sequence_number: int | None


@dataclass
class RecordingClosingState:
    producer_stop_sequence_numbers: dict[str, int]
    stop_requested_at: datetime


@dataclass
class PendingTraceEndRegistry:
    _pending_by_trace: dict[str, PendingTraceEnd] = field(default_factory=dict)

    def add(self, pending_trace_end: PendingTraceEnd) -> None:
        self._pending_by_trace[pending_trace_end.trace_id] = pending_trace_end

    def get(self, trace_id: str) -> PendingTraceEnd | None:
        return self._pending_by_trace.get(trace_id)

    def pop(self, trace_id: str) -> PendingTraceEnd | None:
        return self._pending_by_trace.pop(trace_id, None)


@dataclass
class FinalChunkRegistry:
    _trace_ids: set[str] = field(default_factory=set)

    def add(self, trace_id: str) -> None:
        self._trace_ids.add(trace_id)

    def discard(self, trace_id: str) -> None:
        self._trace_ids.discard(trace_id)

    def contains(self, trace_id: str) -> bool:
        return trace_id in self._trace_ids


@dataclass
class TraceMetadataRegistry:
    _metadata_by_trace: dict[str, dict[str, str | int | None]] = field(
        default_factory=dict
    )

    def get(self, trace_id: str) -> dict[str, str | int | None]:
        return self._metadata_by_trace.get(trace_id, {})

    def pop(self, trace_id: str) -> dict[str, str | int | None] | None:
        return self._metadata_by_trace.pop(trace_id, None)

    def register(
        self,
        request: TraceMetadataRegistrationRequest,
    ) -> list[tuple[str, str | int | None, str | int | None]]:
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
    _recording_by_trace: dict[str, str] = field(default_factory=dict)
    _traces_by_recording: dict[str, set[str]] = field(default_factory=dict)
    _unique_traces_by_recording: dict[str, set[str]] = field(default_factory=dict)

    def get_recording_id(self, trace_id: str) -> str | None:
        return self._recording_by_trace.get(trace_id)

    def register(self, recording_id: str, trace_id: str) -> str | None:
        previous_recording_id = self._recording_by_trace.get(trace_id)
        if previous_recording_id == recording_id:
            self._traces_by_recording.setdefault(recording_id, set()).add(trace_id)
            self._unique_traces_by_recording.setdefault(recording_id, set()).add(
                trace_id
            )
            return None

        if previous_recording_id is not None:
            self._traces_by_recording.get(previous_recording_id, set()).discard(trace_id)
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
        self._recording_by_trace.pop(trace_id, None)

        traces = self._traces_by_recording.get(recording_id)
        if traces is None:
            return

        traces.discard(trace_id)
        if not traces:
            self._traces_by_recording.pop(recording_id, None)

    def traces_for_recording(self, recording_id: str) -> set[str]:
        return set(self._traces_by_recording.get(recording_id, set()))

    def unique_trace_count(self, recording_id: str) -> int:
        return len(self._unique_traces_by_recording.get(recording_id, set()))

    def clear_unique_traces(self, recording_id: str) -> None:
        self._unique_traces_by_recording.pop(recording_id, None)

    def has_active_traces(self, recording_id: str) -> bool:
        return bool(self._traces_by_recording.get(recording_id, set()))


@dataclass
class RecordingCloseRegistry:
    _closing_by_recording: dict[str, RecordingClosingState] = field(default_factory=dict)
    _closed_recording_ids: set[str] = field(default_factory=set)

    def is_closed(self, recording_id: str) -> bool:
        return recording_id in self._closed_recording_ids

    def mark_closing(
        self,
        recording_id: str,
        closing_state: RecordingClosingState,
    ) -> None:
        self._closing_by_recording[recording_id] = closing_state

    def get_closing(self, recording_id: str) -> RecordingClosingState | None:
        return self._closing_by_recording.get(recording_id)

    def items(self) -> Iterator[tuple[str, RecordingClosingState]]:
        return iter(self._closing_by_recording.items())

    def close(self, recording_id: str) -> None:
        self._closing_by_recording.pop(recording_id, None)
        self._closed_recording_ids.add(recording_id)
