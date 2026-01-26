"""Handles the finalisation of trace outputs."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
from collections.abc import Callable
from typing import cast

from neuracore.data_daemon.event_emitter import Emitter, get_emitter
from neuracore.data_daemon.recording_encoding_disk_manager.core.storage_budget import (
    StorageBudget,
)
from neuracore.data_daemon.recording_encoding_disk_manager.encoding.json_trace import (
    JsonTrace,
)
from neuracore.data_daemon.recording_encoding_disk_manager.encoding.video_trace import (
    VideoTrace,
)

from ..core.trace_filesystem import _TraceFilesystem
from ..core.types import _BatchJob, _TraceKey
from ..lifecycle.encoder_manager import _EncoderManager

logger = logging.getLogger(__name__)


class _BatchEncoderWorker:
    """Encode-stage worker that consumes raw batch jobs and finalises trace outputs."""

    def __init__(
        self,
        *,
        raw_file_queue: asyncio.Queue[_BatchJob | object],
        filesystem: _TraceFilesystem,
        encoder_manager: _EncoderManager,
        storage_budget: StorageBudget,
        state_lock: threading.RLock,
        aborted_traces: set[_TraceKey],
        abort_trace: Callable[[_TraceKey], None],
        sentinel: object,
    ) -> None:
        """Initialise _BatchEncoderWorker.

        Args:
            raw_file_queue: Queue of raw batch jobs produced by the writer.
            filesystem: Filesystem helper for path resolution and sizing.
            encoder_manager: Encoder manager used to get/create per-trace encoders.
            storage_budget: Storage budget tracker used to enforce storage limits.
            state_lock: Shared lock protecting encoder state.
            aborted_traces: Shared set of traces that should be ignored.
            abort_trace: Callback used to abort traces on failure.
            sentinel: Sentinel object used to stop the worker.
        """
        self.raw_file_queue = raw_file_queue
        self._filesystem = filesystem
        self._encoder_manager = encoder_manager
        self._storage_budget = storage_budget

        self._state_lock = state_lock
        self._aborted_traces = aborted_traces
        self._abort_trace = abort_trace

        self.SENTINEL = sentinel

        self._emitter = get_emitter()

    async def worker(self) -> None:
        """Consumes raw batch jobs, decode them, and feed per-trace encoders.

        Args:
            None

        Returns:
            None
        """
        while True:
            job_item = await self.raw_file_queue.get()
            try:
                if job_item is self.SENTINEL:
                    self._finalise_remaining_encoders()
                    return

                batch_job = cast(_BatchJob, job_item)

                with self._state_lock:
                    if batch_job.trace_key in self._aborted_traces:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None, batch_job.batch_path.unlink, True
                        )
                        continue

                encoder = self._encoder_manager.safe_get_encoder(batch_job.trace_key)
                if encoder is None:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, batch_job.batch_path.unlink, True)
                    continue

                ok = await self._process_batch_into_encoder(batch_job, encoder)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, batch_job.batch_path.unlink, True)
                if not ok:
                    continue

                if batch_job.trace_done:
                    self._finalise_trace_encoder(batch_job.trace_key, encoder)

            finally:
                self.raw_file_queue.task_done()

    def _finalise_remaining_encoders(self) -> None:
        """Finalise and emit TRACE_WRITTEN for all encoders still active at shutdown.

        Args:
            None

        Returns:
            None
        """
        remaining = self._encoder_manager.clear_all_encoders()

        for trace_key, active_encoder in remaining:
            try:
                active_encoder.finish()
            except Exception:
                logger.exception(
                    "Encoder finish failed during shutdown for trace %s", trace_key
                )
                self._abort_trace(trace_key)
                continue

            if self._storage_budget.is_over_limit():
                self._abort_trace(trace_key)
                continue

            bytes_written = self._filesystem.trace_bytes_on_disk(trace_key)
            self._emitter.emit(
                Emitter.TRACE_WRITTEN,
                trace_key.trace_id,
                trace_key.recording_id,
                bytes_written,
            )

    async def _process_batch_into_encoder(
        self,
        batch_job: _BatchJob,
        encoder: JsonTrace | VideoTrace,
    ) -> bool:
        """Decode one raw batch file and feed its payloads into the provided encoder.

        Args:
            batch_job: The batch work item (trace_key, batch_path, trace_done).
            encoder: The encoder instance for the trace.

        Returns:
            True if the batch was successfully processed,
                otherwise False (trace aborted).
        """
        try:
            loop = asyncio.get_event_loop()
            raw_bytes = await loop.run_in_executor(
                None, batch_job.batch_path.read_bytes
            )
            for raw_line in raw_bytes.splitlines():
                if not raw_line:
                    continue

                envelope = json.loads(raw_line.decode("utf-8"))
                data_base64 = envelope.get("data")
                if not isinstance(data_base64, str):
                    continue

                payload = base64.b64decode(data_base64)
                if not payload:
                    continue

                if isinstance(encoder, VideoTrace):
                    encoder.add_payload(payload)
                else:
                    decoded = json.loads(payload.decode("utf-8"))
                    if isinstance(decoded, list):
                        for item in decoded:
                            if isinstance(item, dict):
                                encoder.add_frame(item)
                    elif isinstance(decoded, dict):
                        encoder.add_frame(decoded)

            return True

        except Exception:
            logger.exception(
                "Failed to process batch for trace %s", batch_job.trace_key
            )
            self._encoder_manager.pop_encoder(batch_job.trace_key)
            self._abort_trace(batch_job.trace_key)
            return False

    def _finalise_trace_encoder(
        self,
        trace_key: _TraceKey,
        encoder: JsonTrace | VideoTrace,
    ) -> None:
        """Finish a trace encoder, enforce storage limits, and emit TRACE_WRITTEN.

        Args:
            trace_key: Trace identifier tuple (recording_id, data_type, trace_id).
            encoder: The encoder instance to finalise.

        Returns:
            None
        """
        try:
            encoder.finish()
        except Exception:
            logger.exception("Encoder finish failed for trace %s", trace_key)
            self._encoder_manager.pop_encoder(trace_key)
            self._abort_trace(trace_key)
            return

        self._encoder_manager.pop_encoder(trace_key)

        if self._storage_budget.is_over_limit():
            self._abort_trace(trace_key)
            return

        bytes_written = self._filesystem.trace_bytes_on_disk(trace_key)
        self._emitter.emit(
            Emitter.TRACE_WRITTEN,
            trace_key.trace_id,
            trace_key.recording_id,
            bytes_written,
        )
