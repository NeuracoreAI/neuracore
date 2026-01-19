"""Handles the writing of traces to disk."""

from __future__ import annotations

import pathlib
import queue
import threading
from typing import Any

from neuracore.data_daemon.config_manager.config import ConfigManager
from neuracore.data_daemon.event_emitter import Emitter, emitter
from neuracore.data_daemon.models import CompleteMessage, parse_data_type
from neuracore.data_daemon.recording_encoding_disk_manager.core.storage_budget import (
    StorageBudget,
    StoragePolicy,
)
from neuracore.data_daemon.recording_encoding_disk_manager.encoding.json_trace import (
    JsonTrace,
)
from neuracore.data_daemon.recording_encoding_disk_manager.encoding.video_trace import (
    VideoTrace,
)

from .core.trace_filesystem import _TraceFilesystem
from .core.types import _BatchJob, _TraceKey, _WriteState
from .lifecycle.encoder_manager import _EncoderManager
from .lifecycle.trace_controller import _TraceController
from .workers.batch_encoder_worker import _BatchEncoderWorker
from .workers.raw_batch_writer import _RawBatchWriter


class RecordingDiskManager:
    """Persist trace payloads to disk and emit lifecycle events."""

    SENTINEL = object()
    DEFAULT_FLUSH_BYTES = 4 * 1024 * 1024  # 4 MiB

    MIN_FREE_DISK_BYTES = 32 * 1024 * 1024  # 32 MiB safety margin
    STORAGE_REFRESH_SECONDS = 5.0

    def __init__(
        self,
        config_manager: ConfigManager,
        *,
        flush_bytes: int | None = None,
    ) -> None:
        """Initialise RecordingDiskManager.

        Args:
            config_manager: Config manager used to resolve effective config.
            flush_bytes: Flush threshold for buffered raw writes.
        """
        self.config_manager = config_manager
        self.flush_bytes = int(flush_bytes or self.DEFAULT_FLUSH_BYTES)

        self.trace_message_queue: queue.Queue[CompleteMessage | object] = queue.Queue()
        self.raw_file_queue: queue.Queue[_BatchJob | object] = queue.Queue()

        self.recording_traces: dict[str, dict[str, Any]] = {}

        self._writer_states: dict[_TraceKey, _WriteState] = {}
        self._encoders: dict[_TraceKey, JsonTrace | VideoTrace] = {}

        self._aborted_traces: set[_TraceKey] = set()
        self._stopped_recordings: set[str] = set()

        self._stop_lock = threading.Lock()
        self._stop_requested = False

        self._state_lock = threading.RLock()

        self.recordings_root: pathlib.Path | None = None
        self.storage_limit_bytes: int | None = None

        self._filesystem: _TraceFilesystem | None = None
        self._storage_budget: StorageBudget | None = None
        self._controller: _TraceController | None = None
        self._encoder_manager: _EncoderManager | None = None
        self._writer: _RawBatchWriter | None = None
        self._encoder_worker: _BatchEncoderWorker | None = None

        self.write_thread: threading.Thread | None = None
        self.encoder_thread: threading.Thread | None = None

        self._init_state()

        self.start()

    def _init_state(self) -> None:
        """Initialise state and dependencies without starting threads.

        Returns:
            None
        """
        effective_config = self.config_manager.resolve_effective_config()
        recordings_root_value = effective_config.path_to_store_record
        if recordings_root_value is None:
            recordings_root_value = str(
                pathlib.Path.home() / ".neuracore" / "data_daemon" / "recordings"
            )

        self.recordings_root = pathlib.Path(recordings_root_value)
        self.recordings_root.mkdir(parents=True, exist_ok=True)

        self.storage_limit_bytes = effective_config.storage_limit

        self._storage_budget = StorageBudget(
            recordings_root=self.recordings_root,
            policy=StoragePolicy(
                storage_limit_bytes=self.storage_limit_bytes,
                min_free_disk_bytes=self.MIN_FREE_DISK_BYTES,
                refresh_seconds=self.STORAGE_REFRESH_SECONDS,
            ),
        )

        self._filesystem = _TraceFilesystem(self.recordings_root)

        self._controller = _TraceController(
            filesystem=self._filesystem,
            storage_budget=self._storage_budget,
            state_lock=self._state_lock,
            writer_states=self._writer_states,
            encoders=self._encoders,
            recording_traces=self.recording_traces,
            aborted_traces=self._aborted_traces,
            stopped_recordings=self._stopped_recordings,
        )

        self._encoder_manager = _EncoderManager(
            filesystem=self._filesystem,
            state_lock=self._state_lock,
            encoders=self._encoders,
            abort_trace=self._controller.abort_trace_due_to_storage,
        )

        self._writer = _RawBatchWriter(
            flush_bytes=self.flush_bytes,
            trace_message_queue=self.trace_message_queue,
            raw_file_queue=self.raw_file_queue,
            filesystem=self._filesystem,
            storage_budget=self._storage_budget,
            state_lock=self._state_lock,
            writer_states=self._writer_states,
            aborted_traces=self._aborted_traces,
            stopped_recordings=self._stopped_recordings,
            recording_traces=self.recording_traces,
            abort_trace=self._controller.abort_trace_due_to_storage,
            sentinel=self.SENTINEL,
        )

        self._encoder_worker = _BatchEncoderWorker(
            raw_file_queue=self.raw_file_queue,
            filesystem=self._filesystem,
            encoder_manager=self._encoder_manager,
            storage_budget=self._storage_budget,
            state_lock=self._state_lock,
            aborted_traces=self._aborted_traces,
            abort_trace=self._controller.abort_trace_due_to_storage,
            sentinel=self.SENTINEL,
        )

    def start(self) -> None:
        """Start worker threads and register event handlers.

        Returns:
            None
        """
        if self._writer is None or self._encoder_worker is None:
            raise RuntimeError("RecordingDiskManager not initialised correctly")

        self.write_thread = threading.Thread(target=self._writer.worker, daemon=False)
        self.encoder_thread = threading.Thread(
            target=self._encoder_worker.worker, daemon=False
        )

        emitter.on(
            Emitter.STOP_ALL_TRACES_FOR_RECORDING,
            self._on_stop_all_traces_for_recording,
        )
        emitter.on(Emitter.DELETE_TRACE, self._on_delete_trace)

        self.write_thread.start()
        self.encoder_thread.start()

    def enqueue(self, complete_message: CompleteMessage) -> None:
        """Enqueue a completed message for persistence.

        Args:
            complete_message: Message to persist.

        Returns:
            None
        """
        self.trace_message_queue.put(complete_message)

    def request_stop(self) -> None:
        """Request a graceful stop of the worker threads.

        Returns:
            None
        """
        with self._stop_lock:
            if self._stop_requested:
                return
            self._stop_requested = True
        self.trace_message_queue.put(self.SENTINEL)

    def shutdown(self) -> None:
        """Stop worker threads and wait for them to exit.

        Returns:
            None
        """
        self.request_stop()
        if self.write_thread is not None:
            self.write_thread.join()
        if self.encoder_thread is not None:
            self.encoder_thread.join()

    def _on_stop_all_traces_for_recording(self, recording_id: str) -> None:
        """Handle STOP_ALL_TRACES_FOR_RECORDING(recording_id).

        Args:
            recording_id: Recording identifier to stop.

        Returns:
            None
        """
        if self._controller is None or self._writer is None:
            raise RuntimeError("RecordingDiskManager not initialised correctly")

        self._controller.on_stop_all_traces_for_recording(
            recording_id,
            flush_state=self._writer._flush_state,
        )

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
