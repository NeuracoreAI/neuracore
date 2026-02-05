"""Handles the finalisation of trace outputs."""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Callable
from pathlib import Path

import aiofiles
import aiofiles.os

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
    """Encode-stage worker that processes raw batch jobs and finalises trace outputs.

    Uses event-driven architecture:
    - Listens for BATCH_READY events from RawBatchWriter
    - Listens for TRACE_ABORTED events from TraceController
    - Owns its own state (aborted_traces, in_flight_count)
    """

    def __init__(
        self,
        *,
        filesystem: _TraceFilesystem,
        encoder_manager: _EncoderManager,
        storage_budget: StorageBudget,
        abort_trace: Callable[[_TraceKey], None],
    ) -> None:
        """Initialise _BatchEncoderWorker.

        Args:
            filesystem: Filesystem helper for path resolution and sizing.
            encoder_manager: Encoder manager used to get/create per-trace encoders.
            storage_budget: Storage budget tracker used to enforce storage limits.
            abort_trace: Callback used to abort traces on failure.
        """
        self._filesystem = filesystem
        self._encoder_manager = encoder_manager
        self._storage_budget = storage_budget
        self._abort_trace = abort_trace

        self._emitter = get_emitter()

        self._aborted_traces: set[_TraceKey] = set()
        self._in_flight_count: int = 0

        self._emitter.on(Emitter.BATCH_READY, self._on_batch_ready)
        self._emitter.on(Emitter.TRACE_ABORTED, self._on_trace_aborted)

    @property
    def in_flight_count(self) -> int:
        """Return the number of batch jobs currently being processed."""
        return self._in_flight_count

    def _on_trace_aborted(self, trace_key: _TraceKey) -> None:
        """Handle TRACE_ABORTED event.

        Args:
            trace_key: Trace key that was aborted.
        """
        self._aborted_traces.add(trace_key)

    async def _on_batch_ready(self, batch_job: _BatchJob) -> None:
        """Handle BATCH_READY event.

        Args:
            batch_job: The batch work item (trace_key, batch_path, trace_done).
        """
        logger.info(
            "BATCH_READY received (trace_id=%s, recording_id=%s, trace_done=%s, "
            "path=%s)",
            batch_job.trace_key.trace_id,
            batch_job.trace_key.recording_id,
            batch_job.trace_done,
            batch_job.batch_path,
        )
        self._in_flight_count += 1
        try:
            if batch_job.trace_key in self._aborted_traces:
                await self._remove_file(batch_job.batch_path)
                return

            encoder = self._encoder_manager.safe_get_encoder(batch_job.trace_key)
            if encoder is None:
                await self._remove_file(batch_job.batch_path)
                return

            ok = await self._process_batch_into_encoder(batch_job, encoder)
            await self._remove_file(batch_job.batch_path)
            if not ok:
                return

            if batch_job.trace_done:
                self._finalise_trace_encoder(batch_job.trace_key, encoder)
        finally:
            self._in_flight_count -= 1

    @staticmethod
    async def _remove_file(path: Path) -> None:
        """Remove a file, ignoring if it doesn't exist."""
        try:
            await aiofiles.os.remove(path)
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("Failed to remove batch file: %s", path, exc_info=True)

    def shutdown(self) -> None:
        """Finalize remaining encoders and cleanup event listeners."""
        self._emitter.remove_listener(Emitter.BATCH_READY, self._on_batch_ready)
        self._emitter.remove_listener(Emitter.TRACE_ABORTED, self._on_trace_aborted)
        self._finalise_remaining_encoders()

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
            async with aiofiles.open(batch_job.batch_path, "rb") as f:
                raw_bytes = await f.read()
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
        logger.info(
            "Emitting TRACE_WRITTEN (trace_id=%s, recording_id=%s, bytes_written=%s)",
            trace_key.trace_id,
            trace_key.recording_id,
            bytes_written,
        )
        self._emitter.emit(
            Emitter.TRACE_WRITTEN,
            trace_key.trace_id,
            trace_key.recording_id,
            bytes_written,
        )
