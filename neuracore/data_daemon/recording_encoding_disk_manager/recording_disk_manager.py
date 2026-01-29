"""Handles the writing of traces to disk."""

from __future__ import annotations

import asyncio
import pathlib
from concurrent.futures import Future
from typing import Any

from neuracore.data_daemon.config_manager.helpers import calculate_storage_limit
from neuracore.data_daemon.const import (
    DEFAULT_FLUSH_BYTES,
    DEFAULT_RECORDING_ROOT_PATH,
    DEFAULT_STORAGE_FREE_FRACTION,
    MIN_FREE_DISK_BYTES,
    SENTINEL,
    STORAGE_REFRESH_SECONDS,
)
from neuracore.data_daemon.event_emitter import Emitter, get_emitter
from neuracore.data_daemon.event_loop_manager import EventLoopManager
from neuracore.data_daemon.models import CompleteMessage, parse_data_type
from neuracore.data_daemon.recording_encoding_disk_manager.core.storage_budget import (
    StorageBudget,
    StoragePolicy,
)

from .core.trace_filesystem import _TraceFilesystem
from .lifecycle.encoder_manager import _EncoderManager
from .lifecycle.trace_controller import _TraceController
from .workers.batch_encoder_worker import _BatchEncoderWorker
from .workers.raw_batch_writer import _RawBatchWriter


class RecordingDiskManager:
    """Persist trace payloads to disk and emit lifecycle events."""

    def __init__(
        self,
        *,
        loop_manager: EventLoopManager,
        flush_bytes: int | None = None,
        storage_limit_bytes: int | None = None,
        recordings_root: str | None = None,
    ) -> None:
        """Initialise RecordingDiskManager.

        Args:
            loop_manager: EventLoopManager instance for scheduling workers.
            flush_bytes: Flush threshold for buffered raw writes.
            storage_limit_bytes: Max bytes allowed for on-disk trace storage.
            recordings_root: Root directory for per-recording trace folders.
        """
        self.flush_bytes = flush_bytes or DEFAULT_FLUSH_BYTES
        self.storage_limit_bytes = storage_limit_bytes
        self._recordings_root_value = recordings_root
        self.recordings_root: pathlib.Path

        self.trace_message_queue: asyncio.Queue[CompleteMessage | object] = (
            asyncio.Queue()
        )

        self.recording_traces: dict[str, dict[str, Any]] = {}

        self._stop_requested = False

        self._filesystem: _TraceFilesystem | None = None
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
        root_value = self._recordings_root_value or str(DEFAULT_RECORDING_ROOT_PATH)
        self.recordings_root = pathlib.Path(root_value)
        self.recordings_root.mkdir(parents=True, exist_ok=True)

        if self.storage_limit_bytes is None:
            self.storage_limit_bytes = calculate_storage_limit(
                self.recordings_root, DEFAULT_STORAGE_FREE_FRACTION
            )

        self._storage_budget = StorageBudget(
            recordings_root=self.recordings_root,
            policy=StoragePolicy(
                storage_limit_bytes=self.storage_limit_bytes,
                min_free_disk_bytes=MIN_FREE_DISK_BYTES,
                refresh_seconds=STORAGE_REFRESH_SECONDS,
            ),
        )

        self._filesystem = _TraceFilesystem(self.recordings_root)

        self._controller = _TraceController(
            filesystem=self._filesystem,
            storage_budget=self._storage_budget,
            recording_traces=self.recording_traces,
        )

        self._encoder_manager = _EncoderManager(
            filesystem=self._filesystem,
            abort_trace=self._controller.abort_trace_due_to_storage,
        )

        self._writer = _RawBatchWriter(
            flush_bytes=self.flush_bytes,
            trace_message_queue=self.trace_message_queue,
            filesystem=self._filesystem,
            storage_budget=self._storage_budget,
            recording_traces=self.recording_traces,
            abort_trace=self._controller.abort_trace_due_to_storage,
            sentinel=SENTINEL,
        )

        self._encoder_worker = _BatchEncoderWorker(
            filesystem=self._filesystem,
            encoder_manager=self._encoder_manager,
            storage_budget=self._storage_budget,
            abort_trace=self._controller.abort_trace_due_to_storage,
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

        self._emitter = get_emitter()
        self._emitter.on(
            Emitter.STOP_ALL_TRACES_FOR_RECORDING,
            self._on_stop_all_traces_for_recording,
        )
        self._emitter.on(Emitter.DELETE_TRACE, self._on_delete_trace)

    def enqueue(self, complete_message: CompleteMessage) -> None:
        """Enqueue a completed message for persistence (thread-safe).

        Args:
            complete_message: Message to persist.

        Returns:
            None
        """
        if self._loop_manager is None:
            raise RuntimeError("RecordingDiskManager not started")
        self._loop_manager.schedule_on_general_loop(
            self.trace_message_queue.put(complete_message)
        )

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

        Returns:
            None
        """
        self.delete_trace(recording_id, trace_id, data_type)

    def _on_delete_recording(self, recording_id: str) -> None:
        """Handle DELETE_RECORDING(recording_id).

        Args:
            recording_id: Recording identifier.

        Returns:
            None
        """
        self.delete_recording(recording_id)

    def delete_trace(self, recording_id: str, trace_id: str, data_type: str) -> None:
        """Delete a trace and all its persisted files.

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

    def delete_recording(self, recording_id: str) -> None:
        """Delete a recording by deleting all known traces plus the recording directory.

        Args:
            recording_id: Recording identifier.

        Returns:
            None
        """
        if self._controller is None:
            raise RuntimeError("RecordingDiskManager not initialised correctly")

        self._controller.delete_recording(recording_id)
