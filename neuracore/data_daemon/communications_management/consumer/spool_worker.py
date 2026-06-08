"""Spool worker that persists decoded video frames before completion.

Frames arrive already copied out of the iceoryx2 ring buffer by the daemon
drain, so this worker only spools the chunk to disk, resolves recording
metadata, applies drop policy, and hands the chunk to the completion worker.
Work is sharded by producer so ordering is preserved per channel.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import zlib
from collections.abc import Callable
from pathlib import Path

from neuracore.data_daemon.models import VideoTransportChunkMetadata

from .bridge_chunk_spool import BridgeChunkSpool
from .completion_worker import CompletionWorker
from .models import (
    ChannelState,
    CompletionChunkWork,
    DecodedFrameWork,
    RecordingDataDropRequest,
    TraceMetadataRegistrationRequest,
    TraceMetadataSnapshot,
    TraceRecordingLookupRequest,
    VideoFrameSequenceProgressRequest,
)

logger = logging.getLogger(__name__)

DEFAULT_SPOOL_SHARD_QUEUE_MAXSIZE = int(
    os.getenv("NCD_SPOOL_SHARD_QUEUE_MAXSIZE", "1024")
)


class _SpoolShard:
    """One spool shard that copies decoded frames before completing them."""

    def __init__(
        self,
        *,
        chunk_spool: BridgeChunkSpool,
        completion_worker: CompletionWorker,
        acquire_spool_admission: Callable[[], object],
        release_spool_admission: Callable[[], None],
        should_drop_recording_data: Callable[[RecordingDataDropRequest], bool],
        mark_sequence_completed: Callable[[VideoFrameSequenceProgressRequest], None],
        register_trace: Callable[[str, str], None],
        register_trace_metadata: Callable[[TraceMetadataRegistrationRequest], None],
        get_trace_recording: Callable[[TraceRecordingLookupRequest], str | None],
        set_channel_trace_id: Callable[[ChannelState, str | None], None],
        shard_index: int,
    ) -> None:
        self._chunk_spool = chunk_spool
        self._completion_worker = completion_worker
        self._acquire_spool_admission = acquire_spool_admission
        self._release_spool_admission = release_spool_admission
        self._should_drop_recording_data = should_drop_recording_data
        self._mark_sequence_completed = mark_sequence_completed
        self._register_trace = register_trace
        self._register_trace_metadata = register_trace_metadata
        self._get_trace_recording = get_trace_recording
        self._set_channel_trace_id = set_channel_trace_id
        self._queue: queue.Queue[DecodedFrameWork | None] = queue.Queue(
            maxsize=DEFAULT_SPOOL_SHARD_QUEUE_MAXSIZE
        )
        self._error: Exception | None = None
        self._error_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name=f"daemon-spool-shard-{shard_index}",
            daemon=True,
        )
        self._thread.start()

    def enqueue(self, work: DecodedFrameWork) -> None:
        """Queue one decoded frame for spool processing."""
        self._ensure_running()
        self._queue.put(work)

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

    def _process(self, work: DecodedFrameWork) -> None:
        """Spool one decoded frame and hand it to the completion worker.

        Admission and the chunk-spool reference are owned by this method until
        the chunk is handed to the completion worker, which then releases both
        once the chunk has been materialized. On drop/error paths this method
        releases them itself. The channel sequence is always marked completed so
        end-of-recording finalization never stalls on this frame.
        """
        self._acquire_spool_admission()
        chunk_spool_ref = None
        handed_off = False
        try:
            chunk_metadata = VideoTransportChunkMetadata.from_dict(work.metadata)
            chunk_spool_ref = self._chunk_spool.append(work.chunk)

            trace_id = chunk_metadata.trace_id
            trace_metadata = chunk_metadata.trace_metadata

            recording_id = self._get_trace_recording(
                TraceRecordingLookupRequest(trace_id=trace_id)
            )
            if recording_id is None and trace_metadata is not None:
                recording_id = trace_metadata.recording_id

            if recording_id is None:
                logger.debug(
                    "Decoded frame missing recording metadata trace_id=%s "
                    "producer_id=%s sequence_id=%s",
                    trace_id,
                    work.channel.producer_id,
                    work.sequence_id,
                )
                return

            if self._should_drop_recording_data(
                RecordingDataDropRequest(
                    channel=work.channel,
                    recording_id=recording_id,
                    trace_id=trace_id,
                    sequence_number=work.sequence_id,
                )
            ):
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
                    sequence_number=work.sequence_id,
                    chunk_spool=self._chunk_spool,
                    chunk_spool_ref=chunk_spool_ref,
                    trace_metadata=trace_metadata,
                    fallback_data_type=(
                        trace_metadata.data_type if trace_metadata is not None else None
                    ),
                )
            )
            handed_off = True
        finally:
            if not handed_off:
                if chunk_spool_ref is not None:
                    self._chunk_spool.release(chunk_spool_ref)
                self._release_spool_admission()
            # Mark the sequence completed only after the chunk has been enqueued
            # to the completion worker (when handed off), so finalization never
            # enqueues a final-trace marker ahead of this chunk.
            self._mark_sequence_completed(
                VideoFrameSequenceProgressRequest(
                    producer_id=work.channel.producer_id,
                    sequence_number=work.sequence_id,
                )
            )


class SpoolWorker:
    """Route decoded video frames onto per-producer spool shards."""

    def __init__(
        self,
        *,
        root: Path,
        completion_worker: CompletionWorker,
        acquire_spool_admission: Callable[[], object],
        release_spool_admission: Callable[[], None],
        should_drop_recording_data: Callable[[RecordingDataDropRequest], bool],
        mark_sequence_completed: Callable[[VideoFrameSequenceProgressRequest], None],
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

    def enqueue_frame(
        self,
        channel: ChannelState,
        sequence_id: int,
        metadata: dict,
        chunk: bytes,
    ) -> None:
        """Queue one decoded iceoryx2 frame onto its owning shard."""
        shard = self._shard_for(channel.producer_id)
        shard.enqueue(
            DecodedFrameWork(
                channel=channel,
                sequence_id=sequence_id,
                metadata=metadata,
                chunk=chunk,
            )
        )

    def close(self) -> None:
        """Stop all spool shards."""
        for shard in self._shards:
            shard.close()

    def cleanup(self) -> None:
        """Remove spool files created by all owned shards."""
        for shard in self._shards:
            shard.cleanup()

    def _shard_for(self, producer_id: str) -> _SpoolShard:
        key = producer_id.encode("utf-8", errors="replace")
        return self._shards[zlib.crc32(key) % len(self._shards)]
