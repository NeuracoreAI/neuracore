"""Shared-slot spool worker that copies daemon payloads before ACKing them."""

from __future__ import annotations

import logging
import queue
import threading
import zlib
from collections.abc import Callable
from pathlib import Path

from neuracore.data_daemon.communications_management.shared_transport.models import (
    SharedSlotTransportResult,
)
from neuracore.data_daemon.models import SharedSlotDescriptor

from ..shared_transport.shared_slot_daemon_handler import (
    SharedSlotDaemonHandler,
    SharedSlotDescriptorAbandoned,
)
from .bridge_chunk_spool import BridgeChunkSpool, ChunkSpoolRef
from .completion_worker import CompletionWorker
from .models import (
    ChannelState,
    CompletionChunkWork,
    RecordingDataDropRequest,
    SharedSlotSequenceProgressRequest,
    SpoolDescriptorWork,
    TraceMetadataRegistrationRequest,
    TraceMetadataSnapshot,
    TraceRecordingLookupRequest,
)

logger = logging.getLogger(__name__)


class _SpoolShard:
    """One spool shard that copies shared-slot chunks before ACKing them."""

    def __init__(
        self,
        *,
        chunk_spool: BridgeChunkSpool,
        shared_slot_handler: SharedSlotDaemonHandler,
        completion_worker: CompletionWorker,
        acquire_spool_admission: Callable[[], object],
        release_spool_admission: Callable[[], None],
        should_drop_recording_data: Callable[[RecordingDataDropRequest], bool],
        mark_sequence_completed: Callable[[SharedSlotSequenceProgressRequest], None],
        register_trace: Callable[[str, str], None],
        register_trace_metadata: Callable[[TraceMetadataRegistrationRequest], None],
        get_trace_recording: Callable[[TraceRecordingLookupRequest], str | None],
        set_channel_trace_id: Callable[[ChannelState, str | None], None],
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
        self._set_channel_trace_id = set_channel_trace_id
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

    def _get_transport_result(
        self, work: SpoolDescriptorWork
    ) -> tuple[SharedSlotTransportResult, bool] | None:
        """Gets a transport result from the shared-slot handler."""
        descriptor: SharedSlotDescriptor | None = None
        release_admission = True

        try:
            transport_result = self._shared_slot_handler.handle_descriptor(
                work.channel,
                work.descriptor_payload,
                self._chunk_spool,
            )
            descriptor = transport_result.descriptor
            release_admission = False
        except SharedSlotDescriptorAbandoned:
            descriptor = self._descriptor_from_payload_or_none(work.descriptor_payload)
            logger.warning(
                "Skipping abandoned shared-slot descriptor "
                "producer_id=%s shm_name=%s sequence_id=%s",
                work.channel.producer_id,
                (
                    descriptor.shm_name
                    if descriptor is not None
                    else work.descriptor_payload.get("shm_name")
                ),
                (
                    descriptor.sequence_id
                    if descriptor is not None
                    else work.descriptor_payload.get("sequence_id")
                ),
            )
            return None
        finally:
            if descriptor is None:
                descriptor = self._descriptor_from_payload_or_none(
                    work.descriptor_payload
                )
            if descriptor is not None:
                self._shared_slot_handler.mark_descriptor_completed(
                    work.channel.producer_id,
                    descriptor,
                )
        return (transport_result, release_admission)

    def _process(self, work: SpoolDescriptorWork) -> None:
        self._acquire_spool_admission()
        chunk_spool_ref: ChunkSpoolRef | None = None
        release_admission = True

        try:
            transport_result_data = self._get_transport_result(work)
            if transport_result_data is None:
                return

            transport_result, release_admission = transport_result_data
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
                logger.debug(
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

            self._set_channel_trace_id(work.channel, trace_id)

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
                    sequence_number=descriptor.sequence_id,
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
            elif release_admission:
                self._release_spool_admission()

    @staticmethod
    def _descriptor_from_payload_or_none(
        descriptor_payload: dict,
    ) -> SharedSlotDescriptor | None:
        try:
            return SharedSlotDescriptor.from_dict(descriptor_payload)
        except Exception:
            return None

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
        root: Path,
        shared_slot_handler: SharedSlotDaemonHandler,
        completion_worker: CompletionWorker,
        acquire_spool_admission: Callable[[], object],
        release_spool_admission: Callable[[], None],
        should_drop_recording_data: Callable[[RecordingDataDropRequest], bool],
        mark_sequence_completed: Callable[[SharedSlotSequenceProgressRequest], None],
        register_trace: Callable[[str, str], None],
        register_trace_metadata: Callable[[TraceMetadataRegistrationRequest], None],
        get_trace_recording: Callable[[TraceRecordingLookupRequest], str | None],
        set_channel_trace_id: Callable[[ChannelState, str | None], None],
        shard_count: int = 4,
    ) -> None:
        """Initialize sharded spool workers rooted under the given spool path."""
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
                set_channel_trace_id=set_channel_trace_id,
                shard_index=index,
            )
            for index in range(shard_count)
        ]

    def enqueue(self, channel: ChannelState, descriptor_payload: dict) -> None:
        """Queue one shared-slot descriptor onto its owning shard."""
        key = channel.producer_id.encode("utf-8", errors="replace")
        shard = self._shards[zlib.crc32(key) % len(self._shards)]
        shard.enqueue(channel, descriptor_payload)

    def close(self) -> None:
        """Stop all spool shards."""
        for shard in self._shards:
            shard.close()

    def cleanup(self) -> None:
        """Remove spool files created by all owned shards."""
        for shard in self._shards:
            shard.cleanup()
