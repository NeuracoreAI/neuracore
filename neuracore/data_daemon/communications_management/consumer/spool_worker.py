from __future__ import annotations

import logging
import queue
import threading
import zlib
from collections.abc import Callable
from pathlib import Path

from neuracore.data_daemon.communications_management.shared_transport.shared_slot_daemon_handler import (
    SharedSlotDaemonHandler,
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
        root: Path,
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
