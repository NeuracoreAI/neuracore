from __future__ import annotations

import logging
import os
import queue
import threading
import time
import zlib
from collections.abc import Callable

from neuracore_types import DataType

from neuracore.data_daemon.models import CompleteMessage
from neuracore.data_daemon.recording_encoding_disk_manager import (
    recording_disk_manager as rdm_module,
)

from .bridge_chunk_spool import BridgeChunkSpool, ChunkSpoolRef
from .helpers import str_or_none, trace_metadata_dict
from .models import CompletionChunkWork, FinalTraceWork, SpoolPartialMessage

RecordingDiskManager = rdm_module.RecordingDiskManager

logger = logging.getLogger(__name__)


DEFAULT_COMPLETION_WORKER_SHARD_COUNT = 4


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
            metadata_dict = trace_metadata_dict(partial.metadata)

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
                dataset_id=str_or_none(metadata.get("dataset_id")),
                dataset_name=str_or_none(metadata.get("dataset_name")),
                robot_name=str_or_none(metadata.get("robot_name")),
                robot_id=str_or_none(metadata.get("robot_id")),
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
