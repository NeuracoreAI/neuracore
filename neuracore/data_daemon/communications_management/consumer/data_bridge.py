"""Main neuracore data daemon."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

from neuracore_types import DataType

from neuracore.data_daemon.communications_management.bridge_chunk_spool import (
    BridgeChunkSpool,
    ChunkSpoolRef,
)
from neuracore.data_daemon.communications_management.channel_reader import (
    CompletedChannelMessage,
    PartialMessage,
)
from neuracore.data_daemon.communications_management.communications_manager import (
    CommunicationsManager,
)
from neuracore.data_daemon.communications_management.shared_slot_daemon_handler import (
    SharedSlotDaemonHandler,
)
from neuracore.data_daemon.const import (
    HEARTBEAT_TIMEOUT_SECS,
    NEVER_OPENED_TIMEOUT_SECS,
)
from neuracore.data_daemon.event_emitter import Emitter
from neuracore.data_daemon.helpers import get_daemon_recordings_root_path, utc_now
from neuracore.data_daemon.models import (
    CommandType,
    CompleteMessage,
    DataChunkPayload,
    MessageEnvelope,
    TraceTransportMetadata,
)
from neuracore.data_daemon.recording_encoding_disk_manager import (
    recording_disk_manager as rdm_module,
)

RecordingDiskManager = rdm_module.RecordingDiskManager

logger = logging.getLogger(__name__)


DEFAULT_COMPLETION_WORKER_SHARD_COUNT = 4
DEFAULT_MAX_SPOOLED_CHUNKS = int(os.getenv("NCD_MAX_SPOOLED_CHUNKS", "128"))

def _str_or_none(value: str | int | None) -> str | None:
    """Convert value to string or None.

    Used to safely convert metadata values that may be str, int, or None
    into the str | None type expected by CompleteMessage.from_bytes().
    """
    return None if value is None else str(value)


def _trace_metadata_dict(
    metadata: TraceTransportMetadata | None,
) -> dict[str, str | int | None]:
    """Return trace metadata as a plain dict for downstream registration."""
    return {} if metadata is None else metadata.to_dict()


class TransportMode(str, Enum):
    """Active transport mode for one producer channel."""

    NONE = "none"
    SOCKET = "socket"
    SHARED_MEMORY = "shared_memory"


@dataclass
class SharedSlotTransportState:
    """Daemon-side state for the shared-slot transport."""

    control_endpoint: str | None = None
    shm_name: str | None = None

    def reset(self) -> None:
        """Clear transport-specific state."""
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
    """One partially assembled trace backed by chunk spool references."""

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
        """Track one spooled chunk reference for the trace."""
        if index in self.chunks:
            return self.received_chunks == self.total_chunks

        self.chunks[index] = (spool, ref)
        self.received_chunks += 1
        return self.received_chunks == self.total_chunks

    def ordered_refs(self) -> list[tuple[BridgeChunkSpool, ChunkSpoolRef]]:
        """Return chunk refs in assembly order."""
        missing = [i for i in range(self.total_chunks) if i not in self.chunks]
        if missing:
            raise ValueError(f"Missing chunks: {missing}")
        return [self.chunks[i] for i in range(self.total_chunks)]

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


@dataclass(frozen=True)
class FinalTraceWork:
    """One deferred final-chunk marker to enqueue after chunk processing."""

    producer_id: str
    trace_id: str
    recording_id: str
    data_type: DataType
    metadata: dict[str, str | int | None]


@dataclass(frozen=True)
class SpoolDescriptorWork:
    """One shared-slot descriptor waiting to be copied out of shared memory."""

    channel: ChannelState
    descriptor_payload: dict


@dataclass(frozen=True)
class RecordingDataDropRequest:
    """Typed input for recording-state drop decisions."""

    channel: ChannelState
    recording_id: str
    trace_id: str
    sequence_number: int | None


@dataclass(frozen=True)
class TraceMetadataSnapshot:
    """Canonical trace metadata stored by the daemon."""

    dataset_id: str | None = None
    dataset_name: str | None = None
    robot_name: str | None = None
    robot_id: str | None = None
    robot_instance: int | None = None
    data_type: str | None = None
    data_type_name: str | None = None

    def to_dict(self) -> dict[str, str | int | None]:
        """Convert the typed snapshot to daemon storage shape."""
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
    """Typed input for trace metadata registration."""

    trace_id: str
    metadata: TraceMetadataSnapshot


@dataclass(frozen=True)
class TraceRecordingLookupRequest:
    """Typed input for trace-to-recording lookup."""

    trace_id: str


@dataclass(frozen=True)
class SharedSlotSequenceProgressRequest:
    """Typed input for shared-slot sequence progress tracking."""

    producer_id: str
    sequence_number: int


class _CompletionShard:
    """One completion shard that preserves ordering for its assigned traces."""

    def __init__(
        self,
        *,
        recording_disk_manager: RecordingDiskManager,
        shard_index: int,
        release_spool_admission: Callable[[], None],
    ) -> None:
        self._shard_index = shard_index
        self._recording_disk_manager = recording_disk_manager
        self._release_spool_admission = release_spool_admission
        self._queue: queue.Queue[CompletionChunkWork | FinalTraceWork | None] = (
            queue.Queue()
        )
        self._partials: dict[tuple[str, str], SpoolPartialMessage] = {}
        self._error: Exception | None = None
        self._error_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name=f"daemon-completion-shard-{shard_index}",
            daemon=True,
        )
        self._thread.start()

    def enqueue(self, work: CompletionChunkWork | FinalTraceWork) -> None:
        self._ensure_running()
        self._queue.put(work)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=10.0)

    def _ensure_running(self) -> None:
        with self._error_lock:
            if self._error is not None:
                raise RuntimeError(
                    f"Daemon completion shard {self._shard_index} failed"
                ) from self._error
        if not self._thread.is_alive():
            raise RuntimeError(
                f"Daemon completion shard {self._shard_index} is not running"
            )

    def _worker_loop(self) -> None:
        while True:
            work = self._queue.get()
            try:
                if work is None:
                    break
                if isinstance(work, CompletionChunkWork):
                    self._process_chunk_work(work)
                else:
                    self._process_final_trace_work(work)
            except Exception as exc:
                with self._error_lock:
                    self._error = exc
                logger.exception(
                    "Daemon completion shard failed shard_index=%d",
                    self._shard_index,
                )
                break
            finally:
                self._queue.task_done()

    def _process_chunk_work(self, work: CompletionChunkWork) -> None:
        key = (work.producer_id, work.trace_id)
        partial = self._partials.get(key)
        partial_released = False
        if partial is None:
            partial = SpoolPartialMessage(total_chunks=work.total_chunks)
            self._partials[key] = partial
        elif partial.total_chunks != work.total_chunks:
            logger.warning(
                "Inconsistent total_chunks for trace_id=%s producer_id=%s "
                "(existing=%d, new=%d)",
                work.trace_id,
                work.producer_id,
                partial.total_chunks,
                work.total_chunks,
            )

        partial.register_metadata(work.trace_id, work.trace_metadata)

        if work.chunk_index in partial.chunks:
            self._release_chunk_ref(work.chunk_spool, work.chunk_spool_ref)
            complete = partial.received_chunks == partial.total_chunks
        else:
            complete = partial.add_chunk(
                work.chunk_index,
                work.chunk_spool,
                work.chunk_spool_ref,
            )

        if not complete:
            return

        try:
            ordered_refs = partial.ordered_refs()
            if any(spool is not work.chunk_spool for spool, _ in ordered_refs):
                raise ValueError(
                    "Trace chunks were routed to multiple spools for "
                    f"trace_id={work.trace_id}."
                )
            payload = work.chunk_spool.materialize([ref for _, ref in ordered_refs])
            metadata_dict = _trace_metadata_dict(partial.metadata)

            if partial.metadata is not None:
                data_type = partial.metadata.data_type
            elif work.fallback_data_type is not None:
                data_type = work.fallback_data_type
            else:
                raise ValueError(
                    f"Missing data_type in metadata for trace_id={work.trace_id}."
                )

            self._partials.pop(key, None)
            self._release_partial_refs(partial)
            partial_released = True

            self._enqueue_complete_message(
                producer_id=work.producer_id,
                trace_id=work.trace_id,
                recording_id=work.recording_id,
                data_type=data_type,
                metadata=metadata_dict,
                data=payload,
            )
        finally:
            if not partial_released:
                self._partials.pop(key, None)
                self._release_partial_refs(partial)

    def _process_final_trace_work(self, work: FinalTraceWork) -> None:
        partial = self._partials.pop((work.producer_id, work.trace_id), None)
        if partial is not None:
            self._release_partial_refs(partial)

        self._enqueue_complete_message(
            producer_id=work.producer_id,
            trace_id=work.trace_id,
            recording_id=work.recording_id,
            data_type=work.data_type,
            metadata=work.metadata,
            data=b"",
            final_chunk=True,
        )

    def _enqueue_complete_message(
        self,
        *,
        producer_id: str,
        trace_id: str,
        recording_id: str,
        data_type: DataType,
        metadata: dict[str, str | int | None],
        data: bytes,
        final_chunk: bool = False,
    ) -> None:
        robot_instance = int(metadata.get("robot_instance") or 0)
        enqueue_start = time.monotonic()

        self._recording_disk_manager.enqueue(
            CompleteMessage.from_bytes(
                producer_id=producer_id,
                trace_id=trace_id,
                recording_id=recording_id,
                final_chunk=final_chunk,
                data_type=data_type,
                data_type_name=str(metadata.get("data_type_name") or ""),
                robot_instance=robot_instance,
                data=data,
                dataset_id=_str_or_none(metadata.get("dataset_id")),
                dataset_name=_str_or_none(metadata.get("dataset_name")),
                robot_name=_str_or_none(metadata.get("robot_name")),
                robot_id=_str_or_none(metadata.get("robot_id")),
            )
        )

        enqueue_elapsed = time.monotonic() - enqueue_start
        if enqueue_elapsed > 0.05:
            logger.warning(
                "RecordingDiskManager enqueue slow producer_id=%s trace_id=%s "
                "recording_id=%s bytes=%d elapsed=%.3fs",
                producer_id,
                trace_id,
                recording_id,
                len(data),
                enqueue_elapsed,
            )

    def _release_chunk_ref(
        self, chunk_spool: BridgeChunkSpool, ref: ChunkSpoolRef
    ) -> None:
        try:
            chunk_spool.release(ref)
        finally:
            self._release_spool_admission()

    def _release_partial_refs(self, partial: SpoolPartialMessage) -> None:
        for chunk_spool, ref in partial.chunks.values():
            self._release_chunk_ref(chunk_spool, ref)


class CompletionWorker:
    """Non-blocking sharded completion pipeline for shared-slot ingest."""

    def __init__(
        self,
        *,
        chunk_spool: BridgeChunkSpool | None = None,
        recording_disk_manager: RecordingDiskManager,
        release_spool_admission: Callable[[], None] = lambda: None,
        shard_count: int | None = None,
    ) -> None:
        self._owned_chunk_spools = [chunk_spool] if chunk_spool is not None else []
        resolved_shard_count = shard_count or min(
            8,
            max(1, os.cpu_count() or DEFAULT_COMPLETION_WORKER_SHARD_COUNT),
        )
        self._shards = [
            _CompletionShard(
                recording_disk_manager=recording_disk_manager,
                shard_index=index,
                release_spool_admission=release_spool_admission,
            )
            for index in range(resolved_shard_count)
        ]

    def enqueue_chunk(self, work: CompletionChunkWork) -> None:
        self._shard_for(work.producer_id, work.trace_id).enqueue(work)

    def enqueue_final_trace(self, work: FinalTraceWork) -> None:
        self._shard_for(work.producer_id, work.trace_id).enqueue(work)

    def close(self) -> None:
        for shard in self._shards:
            shard.close()
        for chunk_spool in self._owned_chunk_spools:
            chunk_spool.cleanup()

    def _shard_for(self, producer_id: str, trace_id: str) -> _CompletionShard:
        shard_key = f"{producer_id}:{trace_id}".encode("utf-8", errors="replace")
        shard_index = zlib.crc32(shard_key) % len(self._shards)
        return self._shards[shard_index]


class _SpoolShard:
    """One spool shard that copies shared-slot chunks before ACKing them."""

    def __init__(
        self,
        *,
        chunk_spool: BridgeChunkSpool,
        shared_slot_handler: SharedSlotDaemonHandler,
        completion_worker: CompletionWorker,
        acquire_spool_admission: Callable[[], None],
        release_spool_admission: Callable[[], None],
        should_drop_recording_data: Callable[[RecordingDataDropRequest], bool],
        mark_sequence_completed: Callable[[SharedSlotSequenceProgressRequest], None],
        register_trace: Callable[[str, str], None],
        register_trace_metadata: Callable[[TraceMetadataRegistrationRequest], None],
        get_trace_recording: Callable[[TraceRecordingLookupRequest], str | None],
        shard_index: int,
    ) -> None:
        self._chunk_spool = chunk_spool
        self._shared_slot_handler = shared_slot_handler
        self._completion_worker = completion_worker
        self._acquire_spool_admission = acquire_spool_admission
        self._release_spool_admission = release_spool_admission
        self._should_drop_recording_data = should_drop_recording_data
        self._mark_sequence_completed = mark_sequence_completed
        self._register_trace = register_trace
        self._register_trace_metadata = register_trace_metadata
        self._get_trace_recording = get_trace_recording
        self._queue: queue.Queue[SpoolDescriptorWork | None] = queue.Queue(maxsize=32)
        self._error: Exception | None = None
        self._error_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name=f"daemon-spool-shard-{shard_index}",
            daemon=True,
        )
        self._thread.start()

    def enqueue(self, channel: ChannelState, descriptor_payload: dict) -> None:
        self._ensure_running()
        self._queue.put(
            SpoolDescriptorWork(channel=channel, descriptor_payload=descriptor_payload)
        )

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=10.0)

    def cleanup(self) -> None:
        self._chunk_spool.cleanup()

    def _ensure_running(self) -> None:
        with self._error_lock:
            if self._error is not None:
                raise RuntimeError("Daemon spool shard failed") from self._error
        if not self._thread.is_alive():
            raise RuntimeError("Daemon spool shard is not running")

    def _worker_loop(self) -> None:
        while True:
            work = self._queue.get()
            try:
                if work is None:
                    break
                self._process(work)
            except Exception as exc:
                with self._error_lock:
                    self._error = exc
                logger.exception("Daemon spool shard failed")
                break
            finally:
                self._queue.task_done()

    def _process(self, work: SpoolDescriptorWork) -> None:
        self._acquire_spool_admission()
        chunk_spool_ref: ChunkSpoolRef | None = None
        try:
            transport_result = self._shared_slot_handler.handle_descriptor(
                work.channel,
                work.descriptor_payload,
                self._chunk_spool,
            )
            chunk_spool_ref = transport_result.chunk_spool_ref
        except Exception:
            self._release_spool_admission()
            raise

        try:
            descriptor = transport_result.descriptor
            chunk_metadata = transport_result.chunk_metadata
            trace_id = transport_result.trace_id
            trace_metadata = transport_result.trace_metadata

            recording_id = self._get_trace_recording(
                TraceRecordingLookupRequest(trace_id=trace_id)
            )
            if recording_id is None and trace_metadata is not None:
                recording_id = trace_metadata.recording_id

            if recording_id is None:
                self._release_chunk_ref(transport_result.chunk_spool_ref)
                chunk_spool_ref = None
                self._mark_sequence_completed(
                    SharedSlotSequenceProgressRequest(
                        producer_id=work.channel.producer_id,
                        sequence_number=descriptor.sequence_id,
                    )
                )
                logger.warning(
                    "Shared-slot packet missing recording metadata "
                    "trace_id=%s producer_id=%s sequence_id=%s",
                    trace_id,
                    work.channel.producer_id,
                    descriptor.sequence_id,
                )
                return

            if self._should_drop_recording_data(
                RecordingDataDropRequest(
                    channel=work.channel,
                    recording_id=recording_id,
                    trace_id=trace_id,
                    sequence_number=descriptor.sequence_id,
                )
            ):
                self._release_chunk_ref(transport_result.chunk_spool_ref)
                chunk_spool_ref = None
                self._mark_sequence_completed(
                    SharedSlotSequenceProgressRequest(
                        producer_id=work.channel.producer_id,
                        sequence_number=descriptor.sequence_id,
                    )
                )
                return

            work.channel.set_trace_id(trace_id)

            if trace_metadata is not None:
                self._register_trace(recording_id, trace_id)
                self._register_trace_metadata(
                    TraceMetadataRegistrationRequest(
                        trace_id=trace_id,
                        metadata=TraceMetadataSnapshot(
                            dataset_id=trace_metadata.dataset_id,
                            dataset_name=trace_metadata.dataset_name,
                            robot_name=trace_metadata.robot_name,
                            robot_id=trace_metadata.robot_id,
                            robot_instance=trace_metadata.robot_instance,
                            data_type=trace_metadata.data_type.value,
                            data_type_name=trace_metadata.data_type_name,
                        ),
                    )
                )

            self._completion_worker.enqueue_chunk(
                CompletionChunkWork(
                    producer_id=work.channel.producer_id,
                    trace_id=trace_id,
                    recording_id=str(recording_id),
                    chunk_index=chunk_metadata.chunk_index,
                    total_chunks=chunk_metadata.total_chunks,
                    chunk_spool=self._chunk_spool,
                    chunk_spool_ref=transport_result.chunk_spool_ref,
                    trace_metadata=trace_metadata,
                    fallback_data_type=(
                        trace_metadata.data_type if trace_metadata is not None else None
                    ),
                )
            )
            self._mark_sequence_completed(
                SharedSlotSequenceProgressRequest(
                    producer_id=work.channel.producer_id,
                    sequence_number=descriptor.sequence_id,
                )
            )
            chunk_spool_ref = None
        finally:
            if chunk_spool_ref is not None:
                self._release_chunk_ref(chunk_spool_ref)

    def _release_chunk_ref(self, ref: ChunkSpoolRef) -> None:
        try:
            self._chunk_spool.release(ref)
        finally:
            self._release_spool_admission()


class SpoolWorker:
    """Route shared-slot descriptors onto per-producer spool shards."""

    def __init__(
        self,
        *,
        root,
        shared_slot_handler: SharedSlotDaemonHandler,
        completion_worker: CompletionWorker,
        acquire_spool_admission: Callable[[], None],
        release_spool_admission: Callable[[], None],
        should_drop_recording_data: Callable[[RecordingDataDropRequest], bool],
        mark_sequence_completed: Callable[[SharedSlotSequenceProgressRequest], None],
        register_trace: Callable[[str, str], None],
        register_trace_metadata: Callable[[TraceMetadataRegistrationRequest], None],
        get_trace_recording: Callable[[TraceRecordingLookupRequest], str | None],
        shard_count: int = 4,
    ) -> None:
        self._shards = [
            _SpoolShard(
                chunk_spool=BridgeChunkSpool(root / f"shard-{index:02d}"),
                shared_slot_handler=shared_slot_handler,
                completion_worker=completion_worker,
                acquire_spool_admission=acquire_spool_admission,
                release_spool_admission=release_spool_admission,
                should_drop_recording_data=should_drop_recording_data,
                mark_sequence_completed=mark_sequence_completed,
                register_trace=register_trace,
                register_trace_metadata=register_trace_metadata,
                get_trace_recording=get_trace_recording,
                shard_index=index,
            )
            for index in range(shard_count)
        ]

    def enqueue(self, channel: ChannelState, descriptor_payload: dict) -> None:
        key = channel.producer_id.encode("utf-8", errors="replace")
        shard = self._shards[zlib.crc32(key) % len(self._shards)]
        shard.enqueue(channel, descriptor_payload)

    def close(self) -> None:
        for shard in self._shards:
            shard.close()

    def cleanup(self) -> None:
        for shard in self._shards:
            shard.cleanup()


@dataclass
class ChannelState:
    """Per-producer channel state owned by the daemon."""

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
        """Check if the channel has been opened with an active transport."""
        return self.transport_mode is not TransportMode.NONE

    def touch(self) -> None:
        """Update the last heartbeat time for the channel.

        This is called when a ManagementMessage is received from a producer.
        """
        self.last_heartbeat = datetime.now(timezone.utc)
        self.heartbeat_expired_at = None

    def is_open(self) -> bool:
        """Check if the channel has an initialized transport."""
        return self.transport_mode is not TransportMode.NONE

    def mark_socket_transport_open(self) -> None:
        """Mark this channel as active on direct socket transport."""
        self.transport_mode = TransportMode.SOCKET
        if self.opened_at is None:
            self.opened_at = datetime.now(timezone.utc)

    def mark_shared_slot_transport_open(
        self,
        *,
        control_endpoint: str,
        shm_name: str,
    ) -> None:
        """Mark this channel as active on the fixed shared-slot transport."""
        self.transport_mode = TransportMode.SHARED_MEMORY
        self.shared_slot.control_endpoint = control_endpoint
        self.shared_slot.shm_name = shm_name
        self.opened_at = datetime.now(timezone.utc)

    def mark_shared_slot_descriptor_seen(
        self,
        *,
        shm_name: str,
    ) -> None:
        """Record one shared-slot descriptor processed by the daemon."""
        self.transport_mode = TransportMode.SHARED_MEMORY
        self.shared_slot.shm_name = shm_name
        if self.opened_at is None:
            self.opened_at = datetime.now(timezone.utc)

    def uses_shared_memory_transport(self) -> bool:
        """Return True when the channel is active on shared-memory transport."""
        return self.transport_mode is TransportMode.SHARED_MEMORY

    def clear_transport_state(self) -> None:
        """Forget the current transport state for this channel."""
        self.opened_at = None
        self.transport_mode = TransportMode.NONE
        self.shared_slot.reset()
        self.socket_pending_messages.clear()

    def add_socket_data_chunk(
        self, data_chunk: DataChunkPayload
    ) -> CompletedChannelMessage | None:
        """Register one socket DATA_CHUNK and return a completed payload when ready."""
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
        """Register one transport chunk and return a completed payload when ready."""
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
        """Assemble one completed transport message into a payload."""
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
        """Return True when no heartbeat has been seen within timeout."""
        if heartbeat_timeout is None:
            heartbeat_timeout = timedelta(seconds=HEARTBEAT_TIMEOUT_SECS)
        return now - self.last_heartbeat > heartbeat_timeout

    def is_stale_unopened(
        self,
        now: datetime,
        never_opened_timeout: timedelta | None = None,
    ) -> bool:
        """Return True when a channel never opened within timeout."""
        if never_opened_timeout is None:
            never_opened_timeout = timedelta(seconds=NEVER_OPENED_TIMEOUT_SECS)
        return (not self.is_open()) and (now - self.created_at > never_opened_timeout)

    def should_expire(
        self,
    ) -> bool:
        """Return True if channel should be removed from daemon state."""
        now = utc_now()
        return self.has_missed_heartbeat(now) or (self.is_stale_unopened(now))

    def set_trace_id(self, trace_id: str) -> None:
        """Set the trace ID for the current channel.

        Args:
            trace_id: The trace ID to set for the current channel.
        """
        if trace_id != self.trace_id:
            self.trace_id = trace_id


CommandHandler = Callable[[ChannelState, MessageEnvelope], None]


@dataclass
class PendingTraceEnd:
    """Producer stop marker waiting for the final trace chunk to be enqueued."""

    producer_id: str
    recording_id: str
    trace_id: str
    data_type: DataType
    sequence_number: int | None


@dataclass
class RecordingClosingState:
    """Recording-level stop/drain state."""

    producer_stop_sequence_numbers: dict[str, int]
    stop_requested_at: datetime


class Daemon:
    """Main neuracore data daemon.

    - Owns per-producer channels + transport state.
    - Receives ManagementMessages from producers over ZMQ.
    - Handles heartbeats and channel lifetime cleanup.
    """

    def __init__(
        self,
        recording_disk_manager: RecordingDiskManager,
        emitter: Emitter,
        comm_manager: CommunicationsManager | None = None,
    ) -> None:
        """Initializes the daemon.

        Args:
            recording_disk_manager: The recording disk manager for persisting
                trace data to disk.
            emitter: Event emitter for cross-component signaling.
            comm_manager: The communications manager for ZMQ operations.
                If not provided, a new instance will be created.
        """
        self.comm = comm_manager or CommunicationsManager()
        self._pending_trace_ends: dict[str, PendingTraceEnd] = {}
        self._final_chunk_enqueued_traces: set[str] = set()
        self.recording_disk_manager = recording_disk_manager
        self.channels: dict[str, ChannelState] = {}
        self._closed_producers: set[str] = set()
        self._recording_traces: dict[str, set[str]] = {}
        self._recording_unique_traces: dict[str, set[str]] = {}
        self._trace_recordings: dict[str, str] = {}
        self._trace_metadata: dict[str, dict[str, str | int | None]] = {}
        self._closed_recordings: set[str] = set()
        self._closing_recordings: dict[str, RecordingClosingState] = {}
        self._producer_last_sequence_numbers: dict[str, int] = {}
        self._pending_shared_slot_sequences: dict[str, set[int]] = {}
        self._pending_shared_slot_sequences_lock = threading.Lock()
        self._shared_slot_handler = SharedSlotDaemonHandler(self.comm)
        self._spool_admission = threading.BoundedSemaphore(DEFAULT_MAX_SPOOLED_CHUNKS)
        self._completion_worker = CompletionWorker(
            recording_disk_manager=self.recording_disk_manager,
            release_spool_admission=self._spool_admission.release,
        )
        self._spool_worker = SpoolWorker(
            root=get_daemon_recordings_root_path() / ".bridge_chunk_spool",
            shared_slot_handler=self._shared_slot_handler,
            completion_worker=self._completion_worker,
            acquire_spool_admission=self._spool_admission.acquire,
            release_spool_admission=self._spool_admission.release,
            should_drop_recording_data=self._should_drop_recording_data,
            mark_sequence_completed=self._mark_shared_slot_sequence_completed,
            register_trace=self._register_trace,
            register_trace_metadata=self._register_trace_metadata,
            get_trace_recording=self._get_trace_recording,
            shard_count=4,
        )
        self._command_handlers: dict[CommandType, CommandHandler] = {
            CommandType.OPEN_FIXED_SHARED_SLOTS: self._handle_open_fixed_shared_slots,
            CommandType.SHARED_SLOT_DESCRIPTOR: self._handle_shared_slot_descriptor,
            CommandType.DATA_CHUNK: self._handle_write_data_chunk,
            CommandType.HEARTBEAT: self._handle_heartbeat,
            CommandType.TRACE_END: self._handle_end_trace,
        }

        self._emitter = emitter
        self._running = False
        self._emitter.on(Emitter.TRACE_WRITTEN, self.cleanup_channel_on_trace_written)

    def run(self) -> None:
        """Starts the daemon and begins accepting messages from producers.

        This function blocks until the daemon is shutdown via Ctrl-C.

        It is responsible for:

        - Starting the ZMQ consumer and publisher sockets.
        - Receiving and processing management messages from producers.
        - Periodically cleaning up expired channels.
        - Finalizing fully assembled transport messages.

        :return: None
        """
        if self._running:
            raise RuntimeError("Daemon is already running")

        self._running = True
        self.comm.start_consumer()

        logger.info("Daemon started and ready to receive messages...")
        try:
            last_receive_log_at = datetime.now(timezone.utc)
            datetime.now(timezone.utc)
            loop_count = 0

            while self._running:
                loop_count += 1
                self._finalize_closing_recordings()

                raw = self.comm.receive_raw()

                now = datetime.now(timezone.utc)

                if raw:
                    self.process_raw_message(raw)

                if (now - last_receive_log_at).total_seconds() >= 1.0:
                    last_receive_log_at = now

                self._cleanup_expired_channels()
        except KeyboardInterrupt:
            logger.info("Shutting down daemon...")
        finally:
            self._spool_worker.close()
            self._completion_worker.close()
            self._spool_worker.cleanup()
            self._shared_slot_handler.close()
            self.comm.cleanup_daemon()

    def stop(
        self,
    ) -> None:
        """Stop the daemon main loop.

        Sets the `_running` flag to False, which will cause the daemon main loop
        to exit on the next iteration.
        """
        self._running = False

    def process_raw_message(self, raw: bytes) -> None:
        """Process a raw message from a producer.

        This function will attempt to parse the raw bytes into a ManagementMessage.
        If the parsing fails, it will log an exception and return without handling
        the message.

        :param raw: The raw bytes of a message from a producer.
        :type raw: bytes
        :return: None
        """
        try:
            message = MessageEnvelope.from_bytes(raw)
        except Exception:
            logger.exception("Failed to parse incoming message bytes")
            return
        self.handle_message(message)

    # Producer
    def handle_message(self, message: MessageEnvelope) -> None:
        """Handles a ManagementMessage from a producer.

        This function is called when a ManagementMessage is received from a producer
        over ZMQ. It will handle the message by looking up the command handler
        associated with the message's command type, and then calling the handler
        with the producer's channel state and the message as arguments.

        If the command type is unknown, a warning will be logged.

        :param message: MessageEnvelope containing the ManagementMessage
        :return: None
        """
        producer_id = message.producer_id
        cmd = message.command

        if producer_id is None:
            # Stop recording commands are sent without a producer_id / channel
            if cmd != CommandType.RECORDING_STOPPED:
                logger.warning("Missing producer_id for command %s", cmd)
                return
            self._handle_recording_stopped(message)
            return

        if producer_id in self._closed_producers and cmd != CommandType.OPEN_FIXED_SHARED_SLOTS:
            return

        if cmd == CommandType.OPEN_FIXED_SHARED_SLOTS and producer_id in self._closed_producers:
            self._closed_producers.discard(producer_id)

        existing = self.channels.get(producer_id)
        if existing is None:
            existing = ChannelState(producer_id=producer_id)
            self.channels[producer_id] = existing
        channel = existing
        channel.touch()

        handler = self._command_handlers.get(cmd)
        if handler is None:
            logger.warning("Unknown command %s from producer_id=%s", cmd, producer_id)
            return

        if message.sequence_number is not None:
            if message.sequence_number > channel.last_sequence_number:
                channel.last_sequence_number = message.sequence_number
                self._producer_last_sequence_numbers[producer_id] = (
                    channel.last_sequence_number
                )
            else:
                logger.warning(
                    "Non-monotonic sequence_number=%s for producer_id=%s (last=%s)",
                    message.sequence_number,
                    producer_id,
                    channel.last_sequence_number,
                )
        try:
            handler(channel, message)
        except Exception:
            logger.exception(
                "Failed to handle command %s from producer_id=%s",
                cmd,
                producer_id,
            )

    def _handle_open_fixed_shared_slots(
        self, channel: ChannelState, message: MessageEnvelope
    ) -> None:
        """Handle an OPEN_FIXED_SHARED_SLOTS command from a producer."""
        payload = message.payload.get(message.command.value, {})
        self._shared_slot_handler.handle_open(channel, payload)

    def _handle_shared_slot_descriptor(
        self, channel: ChannelState, message: MessageEnvelope
    ) -> None:
        """Queue one shared-slot descriptor for sharded spool processing."""
        descriptor_payload = message.payload.get(message.command.value, {})
        sequence_number = message.sequence_number
        if sequence_number is None:
            raise ValueError("Shared-slot descriptor missing sequence_number")

        self._mark_shared_slot_sequence_pending(
            SharedSlotSequenceProgressRequest(
                producer_id=channel.producer_id,
                sequence_number=sequence_number,
            )
        )
        try:
            self._spool_worker.enqueue(channel, descriptor_payload)
        except Exception:
            self._mark_shared_slot_sequence_completed(
                SharedSlotSequenceProgressRequest(
                    producer_id=channel.producer_id,
                    sequence_number=sequence_number,
                )
            )
            raise

    def _ensure_result_trace_registered(
        self,
        *,
        channel: ChannelState,
        result: CompletedChannelMessage,
    ) -> str | None:
        """Ensure trace/recording metadata is registered for one completed result."""
        trace_id = result.trace_id
        recording_id = self._trace_recordings.get(trace_id)
        if recording_id is not None:
            return recording_id

        metadata = _trace_metadata_dict(result.metadata)
        recording_id = _str_or_none(metadata.get("recording_id"))
        if recording_id is None:
            return None

        channel.set_trace_id(trace_id)
        self._register_trace(recording_id, trace_id)
        self._register_trace_metadata(
            TraceMetadataRegistrationRequest(
                trace_id=trace_id,
                metadata=TraceMetadataSnapshot(
                    dataset_id=_str_or_none(metadata.get("dataset_id")),
                    dataset_name=_str_or_none(metadata.get("dataset_name")),
                    robot_name=_str_or_none(metadata.get("robot_name")),
                    robot_id=_str_or_none(metadata.get("robot_id")),
                    robot_instance=metadata.get("robot_instance"),
                    data_type=_str_or_none(metadata.get("data_type")),
                    data_type_name=_str_or_none(metadata.get("data_type_name")),
                ),
            ),
        )
        return recording_id

    def _channel_stop_cutoff_sequence_number(
        self, producer_id: str, channel: ChannelState
    ) -> int | None:
        """Return stop cutoff sequence for the channel's active trace, if known."""
        trace_id = channel.trace_id
        if trace_id is None:
            return None
        recording_id = self._trace_recordings.get(trace_id)
        if recording_id is None:
            return None
        closing_state = self._closing_recordings.get(recording_id)
        if closing_state is None:
            return None
        cutoffs = closing_state.producer_stop_sequence_numbers
        if not cutoffs:
            return None
        cutoff = cutoffs.get(producer_id)
        if cutoff is None:
            return None
        return int(cutoff)

    def _on_complete_message(
        self,
        channel: ChannelState,
        trace_id: str,
        data_type: DataType,
        data: bytes,
        recording_id: str,
        final_chunk: bool = False,
    ) -> None:
        """Handle a completed message from a channel.

        This function is called when a message is fully assembled from transport
        chunks. It is responsible for enqueueing the message in the recording disk
        manager.

        :param channel: The channel that the message was received on.
        :param trace_id: The trace ID that the message belongs to.
        :param data_type: The data type of the message payload.
        :param data: The message data.
        :param recording_id: The recording ID (from immutable _trace_recordings).
        :param final_chunk: Whether this is the final chunk for the trace.
        """
        metadata = self._trace_metadata.get(trace_id, {})
        robot_instance = int(metadata.get("robot_instance") or 0)
        try:
            enqueue_start = time.monotonic()
            self.recording_disk_manager.enqueue(
                CompleteMessage.from_bytes(
                    producer_id=channel.producer_id,
                    trace_id=trace_id,
                    recording_id=recording_id,
                    final_chunk=final_chunk,
                    data_type=data_type,
                    data_type_name=str(metadata.get("data_type_name") or ""),
                    robot_instance=robot_instance,
                    data=data,
                    dataset_id=_str_or_none(metadata.get("dataset_id")),
                    dataset_name=_str_or_none(metadata.get("dataset_name")),
                    robot_name=_str_or_none(metadata.get("robot_name")),
                    robot_id=_str_or_none(metadata.get("robot_id")),
                )
            )
            enqueue_elapsed = time.monotonic() - enqueue_start

            if enqueue_elapsed > 0.05:
                logger.warning(
                    "RecordingDiskManager enqueue slow producer_id=%s trace_id=%s "
                    "recording_id=%s bytes=%d elapsed=%.3fs",
                    channel.producer_id,
                    trace_id,
                    recording_id,
                    len(data),
                    enqueue_elapsed,
                )
        except Exception:
            logger.exception(
                "Failed to enqueue message for trace_id=%s producer_id=%s",
                trace_id,
                channel.producer_id,
            )

    def _handle_heartbeat(self, channel: ChannelState, _: MessageEnvelope) -> None:
        """Update the heartbeat timestamp for a producer.

        This does not perform any logic beyond updating the timestamp, so it is
        suitable for use in a high-throughput system.
        """
        channel.touch()
        if channel.transport_mode is TransportMode.NONE:
            channel.mark_socket_transport_open()

    def _register_trace(self, recording_id: str, trace_id: str) -> None:
        """Register a trace to a recording.

        This method registers a trace to a recording. If the trace is already
        registered to a different recording, it will be moved to the new
        recording. If the trace is moved, a log message will be emitted.

        :param recording_id: The recording ID to register the trace to.
        :param trace_id: The trace ID to register.
        """
        existing = self._trace_recordings.get(trace_id)
        if existing and existing != recording_id:
            logger.warning(
                "Trace %s moved from recording %s to %s",
                trace_id,
                existing,
                recording_id,
            )
            self._recording_traces.get(existing, set()).discard(trace_id)
            old_unique_traces = self._recording_unique_traces.get(existing)
            if old_unique_traces is not None:
                old_unique_traces.discard(trace_id)
                if not old_unique_traces:
                    self._recording_unique_traces.pop(existing, None)
        self._trace_recordings[trace_id] = recording_id
        self._recording_traces.setdefault(recording_id, set()).add(trace_id)
        self._recording_unique_traces.setdefault(recording_id, set()).add(trace_id)

    def _register_trace_metadata(
        self, request: TraceMetadataRegistrationRequest
    ) -> None:
        """Register metadata for a trace.

        This method registers metadata for a trace. If the trace already has
        metadata registered, it will update the existing metadata with the new
        values. If the new value is different from the existing value, a log
        message will be emitted.

        :param request: The typed metadata registration request.
        """
        trace_id = request.trace_id
        metadata = request.metadata.to_dict()
        existing = self._trace_metadata.get(trace_id)
        if existing is None:
            self._trace_metadata[trace_id] = dict(metadata)
            return
        for key, value in metadata.items():
            if existing.get(key) is None and value is not None:
                existing[key] = value
            elif value is not None and existing.get(key) not in (None, value):
                logger.warning(
                    "Trace %s metadata mismatch for %s (%s -> %s)",
                    trace_id,
                    key,
                    existing.get(key),
                    value,
                )

    def _remove_trace(self, recording_id: str, trace_id: str) -> None:
        self._trace_recordings.pop(trace_id, None)
        self._trace_metadata.pop(trace_id, None)
        self._pending_trace_ends.pop(trace_id, None)
        self._final_chunk_enqueued_traces.discard(trace_id)

        traces = self._recording_traces.get(recording_id)
        if traces is None:
            return

        traces.discard(trace_id)

        if not traces:
            self._recording_traces.pop(recording_id, None)

    def _get_trace_recording(self, request: TraceRecordingLookupRequest) -> str | None:
        """Return the recording currently associated with one trace, if any."""
        return self._trace_recordings.get(request.trace_id)

    def _mark_shared_slot_sequence_pending(
        self, request: SharedSlotSequenceProgressRequest
    ) -> None:
        """Record that one shared-slot descriptor still needs spool processing."""
        with self._pending_shared_slot_sequences_lock:
            self._pending_shared_slot_sequences.setdefault(request.producer_id, set()).add(
                request.sequence_number
            )

    def _mark_shared_slot_sequence_completed(
        self, request: SharedSlotSequenceProgressRequest
    ) -> None:
        """Record that one shared-slot descriptor reached completion handoff."""
        with self._pending_shared_slot_sequences_lock:
            pending = self._pending_shared_slot_sequences.get(request.producer_id)
            if pending is None:
                return
            pending.discard(request.sequence_number)
            if not pending:
                self._pending_shared_slot_sequences.pop(request.producer_id, None)

    def _should_drop_recording_data(self, request: RecordingDataDropRequest) -> bool:
        """Return True when recording state says this data should be dropped."""
        channel = request.channel
        recording_id = request.recording_id
        trace_id = request.trace_id
        sequence_number = request.sequence_number
        if recording_id in self._closed_recordings:
            logger.warning(
                "Dropping data for closed recording_id=%s trace_id=%s",
                recording_id,
                trace_id,
            )
            return True

        closing_state = self._closing_recordings.get(recording_id)
        if closing_state is None or not closing_state.producer_stop_sequence_numbers:
            return False
        cutoff_sequence_number = closing_state.producer_stop_sequence_numbers.get(
            channel.producer_id
        )
        if cutoff_sequence_number is None:
            logger.warning(
                "Dropping data from producer_id=%s while recording_id=%s is closing "
                "(missing stop sequence number)",
                channel.producer_id,
                recording_id,
            )
            return True
        if sequence_number is None:
            logger.warning(
                "Dropping data for producer_id=%s recording_id=%s without "
                "sequence_number while recording is closing",
                channel.producer_id,
                recording_id,
            )
            return True
        if sequence_number > cutoff_sequence_number:
            logger.warning(
                "Dropping post-stop data for producer_id=%s recording_id=%s "
                "(sequence_number=%s, cutoff_sequence_number=%s)",
                channel.producer_id,
                recording_id,
                sequence_number,
                cutoff_sequence_number,
            )
            return True
        return False

    def _handle_write_data_chunk(
        self, channel: ChannelState, message: MessageEnvelope
    ) -> None:
        """Handle a DATA_CHUNK message from a producer.

        This will assemble the data chunk into the channel's active transport
        message state. If the payload is incomplete, a warning will be logged
        and the message will be discarded.

        The message payload should contain the following fields:
        - data_chunk: DataChunkPayload

        If the payload is incomplete, a warning will be logged and the message
        will be discarded.

        The DATA_CHUNK message will be logged with the following format:
        DATA_CHUNK: producer_id=<producer_id> channel_id=<channel_id>
        trace_id=<trace_id> chunk_index=<chunk_index+1>/<total_chunks>
        size=<chunk_len>

        :param channel: channel state of the producer
        :param message: message envelope containing the data chunk payload
        """
        data_chunk_payload = message.payload.get("data_chunk")
        if data_chunk_payload is None:
            data_chunk_payload = message.payload

        data_chunk = DataChunkPayload.from_dict(data_chunk_payload)

        if not data_chunk:
            logger.warning("DATA_CHUNK received without payload …")
            return

        recording_id = data_chunk.recording_id
        if not recording_id:
            logger.warning(
                "DATA_CHUNK missing recording_id trace_id=%s producer_id=%s",
                data_chunk.trace_id,
                channel.producer_id,
            )
            return

        trace_id = data_chunk.trace_id
        if channel.trace_id != trace_id and channel.trace_id is not None:
            logger.warning(
                "DATA_CHUNK trace_id=%s does not match channel trace_id=%s",
                data_chunk.trace_id,
                channel.trace_id,
            )
        channel.set_trace_id(trace_id)

        if self._should_drop_recording_data(
            RecordingDataDropRequest(
                channel=channel,
                recording_id=recording_id,
                trace_id=trace_id,
                sequence_number=message.sequence_number,
            )
        ):
            return

        if recording_id:
            self._register_trace(recording_id, trace_id)
            self._register_trace_metadata(
                TraceMetadataRegistrationRequest(
                    trace_id=trace_id,
                    metadata=TraceMetadataSnapshot(
                        dataset_id=data_chunk.dataset_id,
                        dataset_name=data_chunk.dataset_name,
                        robot_name=data_chunk.robot_name,
                        robot_id=data_chunk.robot_id,
                        robot_instance=data_chunk.robot_instance,
                        data_type=data_chunk.data_type.value,
                        data_type_name=data_chunk.data_type_name,
                    ),
                )
            )
        completed = channel.add_socket_data_chunk(data_chunk)
        if completed is None:
            return
        self._on_complete_message(
            channel=channel,
            trace_id=completed.trace_id,
            data_type=completed.data_type,
            data=completed.payload,
            recording_id=recording_id,
        )

    def _handle_end_trace(
        self,
        channel: ChannelState,
        message: MessageEnvelope,
        *,
        reason: str = "producer_trace_end",
    ) -> None:
        """Handle an END_TRACE command from a producer.

        This command is sent by a producer to the daemon when it wants to end
        a trace. The daemon will remove the trace from its internal state
        and notify the RDM to end the trace.

        :param channel (ChannelState): the channel state of the producer
        :param message (MessageEnvelope): the message envelope containing
            the command and payload
        :param reason: source of the trace finalization request

        Returns:
            None
        """
        payload = message.payload.get("trace_end", {})
        trace_id = payload.get("trace_id")
        if not trace_id:
            return

        registered_recording_id = self._trace_recordings.get(str(trace_id))
        recording_id = registered_recording_id
        if not recording_id:
            recording_id = _str_or_none(payload.get("recording_id"))
        if not recording_id:
            logger.warning(
                "TRACE_END received without recording for producer_id=%s trace_id=%s "
                "sequence_number=%s",
                channel.producer_id,
                trace_id,
                message.sequence_number,
            )
            return

        # Get metadata before removing the trace
        metadata = self._trace_metadata.get(str(trace_id), {})
        data_type_str = metadata.get("data_type")
        if data_type_str:
            try:
                data_type = DataType(data_type_str)
            except ValueError:
                raise ValueError(
                    f"Unknown data_type '{data_type_str}' for trace_id={trace_id}."
                )
        else:
            if registered_recording_id is not None:
                raise ValueError(
                    f"Missing data_type in metadata for trace_id={trace_id}."
                )
            logger.warning(
                "TRACE_END received for trace_id=%s without registered metadata; "
                "ignoring finalization",
                trace_id,
            )
            return

        self._pending_trace_ends[str(trace_id)] = PendingTraceEnd(
            producer_id=channel.producer_id,
            recording_id=str(recording_id),
            trace_id=str(trace_id),
            data_type=data_type,
            sequence_number=message.sequence_number,
        )

        logger.info(
            "Deferred TRACE_END until recording cutoff is reached "
            "producer_id=%s recording_id=%s trace_id=%s sequence_number=%s",
            channel.producer_id,
            recording_id,
            trace_id,
            message.sequence_number,
        )

    def _handle_recording_stopped(self, message: MessageEnvelope) -> None:
        """Handle a RECORDING_STOPPED message from a producer.

        This function is called when a producer sends a RECORDING_STOPPED message to
        the daemon. It will mark the recording as pending-close and emit a
        STOP_RECORDING_REQUESTED event immediately.

        :param message: message envelope containing the recording_id of the stopped
            recording
        :return: None
        """
        payload = message.payload.get("recording_stopped", {})
        recording_id = payload.get("recording_id")
        if not recording_id:
            logger.warning(
                "RECORDING_STOPPED missing recording_id (producer_id=%s)",
                message.producer_id,
            )
            return
        producer_stop_sequence_numbers_raw = payload.get(
            "producer_stop_sequence_numbers", {}
        )
        producer_stop_sequence_numbers: dict[str, int] = {}
        if isinstance(producer_stop_sequence_numbers_raw, dict):
            for (
                producer_id,
                sequence_number,
            ) in producer_stop_sequence_numbers_raw.items():
                try:
                    producer_stop_sequence_numbers[str(producer_id)] = int(
                        sequence_number
                    )
                except (TypeError, ValueError):
                    logger.warning(
                        "Ignoring invalid stop sequence number for producer_id=%s: %r",
                        producer_id,
                        sequence_number,
                    )
        else:
            logger.warning(
                "recording_stopped.producer_stop_sequence_numbers must be a dict"
            )

        self._closing_recordings[recording_id] = RecordingClosingState(
            producer_stop_sequence_numbers=producer_stop_sequence_numbers,
            stop_requested_at=utc_now(),
        )

        self._emitter.emit(Emitter.STOP_RECORDING_REQUESTED, recording_id)

    def _finalize_closing_recordings(self) -> None:
        """Finalize recordings after stop cutoffs and final trace chunks are written."""
        to_close: list[str] = []

        for recording_id, closing_state in self._closing_recordings.items():
            if not self._has_reached_sequence_cutoffs(closing_state):
                continue

            traces = set(self._recording_traces.get(recording_id, set()))

            for trace_id in traces:
                if trace_id in self._final_chunk_enqueued_traces:
                    continue

                pending_trace_end = self._pending_trace_ends.get(trace_id)
                if pending_trace_end is None:
                    logger.info(
                        "Waiting for TRACE_END before finalizing "
                        "recording_id=%s trace_id=%s",
                        recording_id,
                        trace_id,
                    )
                    continue

                logger.info(
                    "Enqueuing deferred final chunk producer_id=%s recording_id=%s "
                    "trace_id=%s",
                    pending_trace_end.producer_id,
                    recording_id,
                    trace_id,
                )

                self._completion_worker.enqueue_final_trace(
                    FinalTraceWork(
                        producer_id=pending_trace_end.producer_id,
                        trace_id=trace_id,
                        recording_id=recording_id,
                        data_type=pending_trace_end.data_type,
                        metadata=dict(self._trace_metadata.get(trace_id, {})),
                    )
                )

                self._final_chunk_enqueued_traces.add(trace_id)

            # TRACE_WRITTEN will call cleanup_channel_on_trace_written(),
            # which removes traces from self._recording_traces.
            if not self._recording_traces.get(recording_id, set()):
                to_close.append(recording_id)

        for recording_id in to_close:
            expected_trace_count = len(
                self._recording_unique_traces.get(recording_id, set())
            )

            self._emitter.emit(
                Emitter.SET_EXPECTED_TRACE_COUNT,
                recording_id,
                expected_trace_count,
            )

            self._closing_recordings.pop(recording_id, None)
            self._closed_recordings.add(recording_id)
            self._recording_unique_traces.pop(recording_id, None)

            self._emitter.emit(Emitter.STOP_RECORDING, recording_id)

    def _has_reached_sequence_cutoffs(
        self, closing_state: RecordingClosingState
    ) -> bool:
        """Return True when all producer sequence cutoffs have been observed.

        If no cutoff map was provided (legacy producer behavior), this returns True.
        """
        stop_cutoffs = closing_state.producer_stop_sequence_numbers
        if not stop_cutoffs:
            return True

        for producer_id, cutoff_sequence_number in stop_cutoffs.items():
            last_sequence_number = self._producer_last_sequence_numbers.get(producer_id)
            if (
                last_sequence_number is None
                or last_sequence_number < cutoff_sequence_number
            ):
                return False
            with self._pending_shared_slot_sequences_lock:
                pending_sequences = self._pending_shared_slot_sequences.get(
                    producer_id, set()
                )
                if any(
                    sequence_number <= cutoff_sequence_number
                    for sequence_number in pending_sequences
                ):
                    return False
        return True

    def cleanup_channel_on_trace_written(
        self,
        trace_id: str,
        _: str | None = None,
        __: int | None = None,
    ) -> None:
        """Clean up a stopped channel.

        This function is called when a trace is marked as written/completed.
        It will remove the trace from the recording's trace list and reset the
        channel state.

        :param trace_id: ID of the trace to clean up
        :param bytes_written: total number of bytes written for the trace (unused)
        """
        recording_id = self._trace_recordings.get(trace_id)
        if recording_id:
            self._remove_trace(str(recording_id), str(trace_id))

        channel = next(
            (ch for ch in self.channels.values() if ch.trace_id == trace_id),
            None,
        )
        if channel is not None:
            if channel.uses_shared_memory_transport() and (
                channel.shared_slot.shm_name is not None
                or channel.socket_pending_messages
            ):
                logger.info(
                    "Cleaning up channel after TRACE_WRITTEN producer_id=%s "
                    "trace_id=%s shm_name=%s pending_partial_traces=%d",
                    channel.producer_id,
                    trace_id,
                    channel.shared_slot.shm_name,
                    len(channel.socket_pending_messages),
                )
            channel.trace_id = None
            if channel.uses_shared_memory_transport():
                self._shared_slot_handler.cleanup_channel_resources(channel)
            channel.clear_transport_state()

    def _cleanup_expired_channels(self) -> None:
        """Remove channels whose heartbeat has not been seen within the timeout."""
        now = utc_now()
        to_remove: list[str] = []

        for producer_id, state in self.channels.items():
            if state.is_stale_unopened(now):
                to_remove.append(producer_id)
                continue

            if not state.has_missed_heartbeat(now):
                state.heartbeat_expired_at = None
                continue

            if state.trace_id is None:
                to_remove.append(producer_id)
                continue

            cutoff_sequence_number = self._channel_stop_cutoff_sequence_number(
                producer_id, state
            )
            if (
                cutoff_sequence_number is not None
                and state.last_sequence_number < cutoff_sequence_number
            ):
                if state.heartbeat_expired_at is None:
                    state.heartbeat_expired_at = now
                continue

            to_remove.append(producer_id)

        for producer_id in to_remove:
            channel = self.channels.get(producer_id)
            if channel is None:
                continue
            if channel is not None and channel.trace_id is not None:
                recording_id = self._trace_recordings.get(channel.trace_id)
                self._handle_end_trace(
                    channel,
                    MessageEnvelope(
                        producer_id=producer_id,
                        command=CommandType.TRACE_END,
                        payload={
                            "trace_end": {
                                "trace_id": channel.trace_id,
                                "recording_id": recording_id,
                            }
                        },
                    ),
                    reason="heartbeat_expiry",
                )
                self._producer_last_sequence_numbers[channel.producer_id] = max(
                    self._producer_last_sequence_numbers.get(channel.producer_id, 0),
                    channel.last_sequence_number,
                )
            if channel.uses_shared_memory_transport():
                self._shared_slot_handler.cleanup_channel_resources(channel)
            channel.clear_transport_state()
            del self.channels[producer_id]
            self._closed_producers.add(producer_id)
