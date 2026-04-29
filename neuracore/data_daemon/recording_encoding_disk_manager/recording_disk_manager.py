"""Handles the writing of traces to disk."""

from __future__ import annotations

import asyncio
import json
import struct
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from neuracore_types import DataType

from neuracore.data_daemon.config_manager.helpers import calculate_storage_limit
from neuracore.data_daemon.const import (
    DEFAULT_FLUSH_BYTES,
    DEFAULT_STORAGE_FREE_FRACTION,
    MIN_FREE_DISK_BYTES,
    SENTINEL,
    STORAGE_REFRESH_SECONDS,
)
from neuracore.data_daemon.event_emitter import Emitter
from neuracore.data_daemon.event_loop_manager import EventLoopManager
from neuracore.data_daemon.helpers import get_daemon_recordings_root_path
from neuracore.data_daemon.models import CompleteMessage, parse_data_type
from neuracore.data_daemon.recording_encoding_disk_manager.core.rgb_trace_spool import (
    RGBTraceSpool,
)
from neuracore.data_daemon.recording_encoding_disk_manager.core.storage_budget import (
    StorageBudget,
    StoragePolicy,
)
from neuracore.data_daemon.recording_encoding_disk_manager.core.types import (
    _RGBTraceMessage,
    _TraceKey,
)

from .core.trace_filesystem import _TraceFilesystem
from .lifecycle.encoder_manager import _EncoderManager
from .lifecycle.trace_controller import _TraceController
from .workers.batch_encoder_worker import _BatchEncoderWorker
from .workers.raw_batch_writer import _RawBatchWriter


class RecordingDiskManager:
    """Persist trace payloads to disk and emit lifecycle events.

    Uses event-driven architecture where workers own their own state and
    communicate via events (BATCH_READY, TRACE_ABORTED, RECORDING_STOPPED).
    """

    def __init__(
        self,
        *,
        loop_manager: EventLoopManager,
        emitter: Emitter,
        flush_bytes: int | None = None,
        storage_limit_bytes: int | None = None,
        recordings_root: str | None = None,
    ) -> None:
        """Initialise RecordingDiskManager.

        Args:
            loop_manager: EventLoopManager instance for scheduling workers.
            emitter: Event emitter for cross-component signaling.
            flush_bytes: Flush threshold for buffered raw writes.
            storage_limit_bytes: Max bytes allowed for on-disk trace storage.
            recordings_root: Root directory for per-recording trace folders.
        """
        self._emitter = emitter
        self.flush_bytes = flush_bytes or DEFAULT_FLUSH_BYTES
        self.storage_limit_bytes = storage_limit_bytes
        self._recordings_root = (
            Path(recordings_root)
            if recordings_root
            else get_daemon_recordings_root_path()
        )

        self.trace_message_queue: asyncio.Queue[
            CompleteMessage | _RGBTraceMessage | object
        ] = (
            asyncio.Queue(maxsize=1)
        )

        self.recording_traces: dict[str, dict[str, Any]] = {}

        self._stop_requested = False

        self._filesystem: _TraceFilesystem | None = None
        self._rgb_trace_spool = RGBTraceSpool()
        self._storage_budget: StorageBudget | None = None
        self._controller: _TraceController | None = None
        self._encoder_manager: _EncoderManager | None = None
        self._writer: _RawBatchWriter | None = None
        self._encoder_worker: _BatchEncoderWorker | None = None

        self._loop_manager = loop_manager
        self._writer_future: Future[Any] | None = None

        self._init_state()

        self.start()

    def _init_state(self) -> None:
        """Initialise state and dependencies without starting threads.

        Returns:
            None
        """
        self._recordings_root.mkdir(parents=True, exist_ok=True)

        if self.storage_limit_bytes is None:
            self.storage_limit_bytes = calculate_storage_limit(
                self._recordings_root, DEFAULT_STORAGE_FREE_FRACTION
            )

        self._storage_budget = StorageBudget(
            recordings_root=self._recordings_root,
            policy=StoragePolicy(
                storage_limit_bytes=self.storage_limit_bytes,
                min_free_disk_bytes=MIN_FREE_DISK_BYTES,
                refresh_seconds=STORAGE_REFRESH_SECONDS,
            ),
        )

        self._filesystem = _TraceFilesystem(self._recordings_root)

        self._controller = _TraceController(
            filesystem=self._filesystem,
            storage_budget=self._storage_budget,
            recording_traces=self.recording_traces,
            emitter=self._emitter,
        )

        self._encoder_manager = _EncoderManager(
            filesystem=self._filesystem,
            abort_trace=self._controller.abort_trace_due_to_storage,
            emitter=self._emitter,
        )

        self._writer = _RawBatchWriter(
            flush_bytes=self.flush_bytes,
            trace_message_queue=self.trace_message_queue,
            filesystem=self._filesystem,
            storage_budget=self._storage_budget,
            recording_traces=self.recording_traces,
            abort_trace=self._controller.abort_trace_due_to_storage,
            sentinel=SENTINEL,
            emitter=self._emitter,
        )
        assert (
            self._loop_manager.is_running() and self._loop_manager.general_loop
        ), "Event loop not running"
        self._encoder_worker = _BatchEncoderWorker(
            filesystem=self._filesystem,
            encoder_manager=self._encoder_manager,
            storage_budget=self._storage_budget,
            abort_trace=self._controller.abort_trace_due_to_storage,
            emitter=self._emitter,
            loop=self._loop_manager.general_loop,
        )

    def start(self) -> None:
        """Start worker tasks on event loops and register event handlers.

        Returns:
            None
        """
        if self._writer is None or self._encoder_worker is None:
            raise RuntimeError("RecordingDiskManager not initialised correctly")

        self._writer_future = self._loop_manager.schedule_on_general_loop(
            self._writer.worker()
        )

        self._emitter.on(
            Emitter.STOP_ALL_TRACES_FOR_RECORDING,
            self._on_stop_all_traces_for_recording,
        )
        self._emitter.on(Emitter.DELETE_TRACE, self._on_delete_trace)

    def enqueue(self, complete_message: CompleteMessage) -> None:
        if self._loop_manager is None:
            raise RuntimeError("RecordingDiskManager not started")

        queue_item = self._queue_item_for(complete_message)
        future = self._loop_manager.schedule_on_general_loop(
            self.trace_message_queue.put(queue_item)
        )
        future.result()

    def _queue_item_for(
        self, complete_message: CompleteMessage
    ) -> CompleteMessage | _RGBTraceMessage:
        """Convert one completed message into the queue item RDM should store."""
        if complete_message.data_type != DataType.RGB_IMAGES:
            return complete_message

        if self._filesystem is None:
            return complete_message

        trace_key = _TraceKey(
            recording_id=str(complete_message.recording_id),
            data_type=complete_message.data_type,
            trace_id=str(complete_message.trace_id),
        )
        frame_metadata: dict[str, Any] | None = None
        frame_ref = None

        if complete_message.data:
            parsed = self._parse_rgb_combined_payload(complete_message.data)
            if parsed is None:
                return complete_message
            frame_metadata, frame_bytes = parsed
            frame_ref = self._rgb_trace_spool.append_frame(
                trace_key=trace_key,
                trace_dir=self._filesystem.trace_dir_for(trace_key),
                frame_bytes=frame_bytes,
            )

        return _RGBTraceMessage(
            trace_key=trace_key,
            data_type_name=complete_message.data_type_name,
            robot_instance=complete_message.robot_instance,
            dataset_id=complete_message.dataset_id,
            dataset_name=complete_message.dataset_name,
            robot_name=complete_message.robot_name,
            robot_id=complete_message.robot_id,
            frame_metadata=frame_metadata,
            frame_ref=frame_ref,
            final_chunk=complete_message.final_chunk,
        )

    @staticmethod
    def _parse_rgb_combined_payload(
        payload: bytes,
    ) -> tuple[dict[str, Any], bytes] | None:
        """Parse the existing RGB combined packet into metadata and raw bytes."""
        if len(payload) < 4:
            return None

        metadata_len = struct.unpack("<I", payload[:4])[0]
        if metadata_len <= 0 or metadata_len > len(payload) - 4:
            return None

        json_end = 4 + metadata_len
        try:
            metadata = json.loads(payload[4:json_end].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(metadata, dict):
            return None

        frame_nbytes = metadata.get("frame_nbytes")
        if not isinstance(frame_nbytes, int) or frame_nbytes < 0:
            frame_nbytes = len(payload) - json_end

        frame_end = json_end + frame_nbytes
        if frame_end > len(payload):
            return None

        return metadata, payload[json_end:frame_end]

    async def request_stop(self) -> None:
        """Request a graceful stop of the worker tasks.

        Returns:
            None
        """
        if self._stop_requested:
            return
        self._stop_requested = True

        await self.trace_message_queue.put(SENTINEL)

    async def shutdown(self) -> None:
        """Stop worker tasks and wait for them to exit.

        Returns:
            None
        """
        await self.request_stop()

        if self._writer_future is not None:
            await asyncio.wrap_future(self._writer_future)

        if self._encoder_worker is not None:
            while self._encoder_worker.in_flight_count > 0:
                await asyncio.sleep(0.01)
            self._encoder_worker.shutdown()

        if self._encoder_manager is not None:
            self._encoder_manager.cleanup()

    def _on_stop_all_traces_for_recording(self, recording_id: str) -> None:
        """Handle STOP_ALL_TRACES_FOR_RECORDING(recording_id).

        Args:
            recording_id: Recording identifier to stop.

        Returns:
            None
        """
        if self._controller is None:
            raise RuntimeError("RecordingDiskManager not initialised correctly")

        self._controller.on_stop_all_traces_for_recording(recording_id)

    def _on_delete_trace(
        self,
        recording_id: str,
        trace_id: str,
        data_type: str,
    ) -> None:
        """Handle DELETE_TRACE event.

        Args:
            recording_id: Recording identifier.
            trace_id: Trace identifier.
            data_type: Data type of the trace.

        Returns:
            None
        """
        if self._controller is None:
            raise RuntimeError("RecordingDiskManager not initialised correctly")

        self._controller.delete_trace(
            recording_id=recording_id,
            trace_id=trace_id,
            data_type=parse_data_type(data_type),
        )

    def _on_delete_recording(self, recording_id: str) -> None:
        """Handle DELETE_RECORDING(recording_id).

        Args:
            recording_id: Recording identifier.

        Returns:
            None
        """
        if self._controller is None:
            raise RuntimeError("RecordingDiskManager not initialised correctly")

        self._controller.delete_recording(recording_id)
