"""Handles trace lifecycle operations."""

from __future__ import annotations

import logging
import shutil
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from neuracore_types import DataType

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
from ..core.types import _TraceKey, _WriteState

logger = logging.getLogger(__name__)


class _TraceController:
    """Coordinate trace lifecycle operations."""

    def __init__(
        self,
        *,
        filesystem: _TraceFilesystem,
        storage_budget: StorageBudget,
        state_lock: threading.RLock,
        writer_states: dict[_TraceKey, _WriteState],
        encoders: dict[_TraceKey, JsonTrace | VideoTrace],
        recording_traces: dict[str, dict[str, Any]],
        aborted_traces: set[_TraceKey],
        stopped_recordings: set[str],
        closed_traces: set[_TraceKey],
    ) -> None:
        """Initialise _TraceController.

        Args:
            filesystem: Filesystem helper for path resolution and sizing.
            storage_budget: Storage budget tracker.
            state_lock: Shared lock protecting writer/encoder state.
            writer_states: Shared write state registry.
            encoders: Shared encoder registry.
            recording_traces: Recording-to-traces bookkeeping map.
            aborted_traces: Shared set of aborted traces.
            stopped_recordings: Shared set of recordings that should be ignored.
            closed_traces: set of trace keys that are closed
        """
        self._filesystem = filesystem
        self._storage_budget = storage_budget

        self._state_lock = state_lock
        self._writer_states = writer_states
        self._encoders = encoders
        self.recording_traces = recording_traces
        self._aborted_traces = aborted_traces
        self._stopped_recordings = stopped_recordings
        self._closed_traces = closed_traces

        self._emitter = get_emitter()

    def abort_trace_due_to_storage(self, trace_key: _TraceKey) -> None:
        """Abort a trace due to storage constraints and emit TRACE_WRITTEN(trace_id, 0).

        Args:
            trace_key: Trace key to abort.

        Returns:
            None
        """
        with self._state_lock:
            self._aborted_traces.add(trace_key)
            self._closed_traces.add(trace_key)

            writer_state = self._writer_states.pop(trace_key, None)
            if writer_state is not None:
                writer_state.buffer.clear()

            active_encoder = self._encoders.pop(trace_key, None)

        if active_encoder is not None:
            try:
                active_encoder.finish()
            except Exception:
                logger.exception(
                    "Encoder finish failed during abort for trace %s",
                    trace_key,
                )

        trace_dir = self._filesystem.trace_dir_for(trace_key)
        try:
            reclaimed_bytes = self._filesystem.trace_bytes_on_disk(trace_key)
        except OSError:
            logger.warning("Failed to get bytes on disk for trace %s", trace_key)
            reclaimed_bytes = 0
        shutil.rmtree(trace_dir, ignore_errors=True)
        self._storage_budget.release(reclaimed_bytes)

        self._emitter.emit(
            Emitter.TRACE_WRITTEN, trace_key.trace_id, trace_key.recording_id, 0
        )

    async def on_stop_all_traces_for_recording(
        self,
        recording_id: str,
        *,
        flush_state: Callable[[_WriteState], Awaitable[None]],
    ) -> None:
        """Handle STOP_ALL_TRACES_FOR_RECORDING(recording_id).

        Args:
            recording_id: Recording identifier to stop.

        Returns:
            None
        """
        recording_id_value = str(recording_id)
        with self._state_lock:
            self._stopped_recordings.add(recording_id_value)
            writer_states_to_flush = [
                writer_state
                for writer_state in self._writer_states.values()
                if writer_state.trace_key.recording_id == recording_id_value
            ]

            for writer_state in writer_states_to_flush:
                self._closed_traces.add(writer_state.trace_key)

            for trace_key in list(self._encoders.keys()):
                if trace_key.recording_id == recording_id_value:
                    self._closed_traces.add(trace_key)

        for writer_state in writer_states_to_flush:
            with self._state_lock:
                writer_state.trace_done = True
            await flush_state(writer_state)
            with self._state_lock:
                self._writer_states.pop(writer_state.trace_key, None)

    def delete_trace(
        self, recording_id: str, trace_id: str, data_type: DataType
    ) -> None:
        """Delete a trace and all its persisted files.

        Args:
            recording_id: Recording identifier.
            trace_id: Trace identifier.
            data_type: Data type of the trace.

        Returns:
            None

        """
        trace_key = _TraceKey(
            recording_id=str(recording_id),
            trace_id=str(trace_id),
            data_type=data_type,
        )
        trace_dir_path = self._filesystem.trace_dir_for(trace_key)
        try:
            reclaimed_bytes = self._filesystem.trace_bytes_on_disk(trace_key)
        except OSError:
            logger.warning("Failed to get bytes on disk for trace %s", trace_key)
            reclaimed_bytes = 0

        with self._state_lock:
            recording_entry = self.recording_traces.get(trace_key.recording_id)
            if recording_entry is not None:
                recording_entry.pop(trace_key.trace_id, None)
                if not recording_entry:
                    self.recording_traces.pop(trace_key.recording_id, None)

            writer_state = self._writer_states.pop(trace_key, None)
            if writer_state is not None:
                writer_state.buffer.clear()

            active_encoder = self._encoders.pop(trace_key, None)
            self._aborted_traces.add(trace_key)
            self._closed_traces.add(trace_key)

        if active_encoder is not None:
            try:
                active_encoder.finish()
            except Exception:
                logger.exception(
                    "Encoder finish failed during delete_trace for trace %s",
                    trace_key,
                )

        shutil.rmtree(trace_dir_path, ignore_errors=True)
        self._storage_budget.release(reclaimed_bytes)

    def delete_recording(self, recording_id: str) -> None:
        """Delete a recording by deleting all known traces plus the recording directory.

        Args:
            recording_id: Recording identifier.

        Returns:
            None
        """
        recording_id_value = str(recording_id)
        with self._state_lock:
            self._stopped_recordings.add(recording_id_value)

            writer_trace_keys = [
                trace_key
                for trace_key in list(self._writer_states.keys())
                if trace_key.recording_id == recording_id_value
            ]
            encoder_trace_keys = [
                trace_key
                for trace_key in list(self._encoders.keys())
                if trace_key.recording_id == recording_id_value
            ]

        all_trace_keys = set(writer_trace_keys) | set(encoder_trace_keys)
        for trace_key in all_trace_keys:
            self.delete_trace(
                recording_id=trace_key.recording_id,
                trace_id=trace_key.trace_id,
                data_type=trace_key.data_type,
            )

        with self._state_lock:
            self.recording_traces.pop(recording_id_value, None)

        path = self._filesystem.recordings_root / recording_id_value
        shutil.rmtree(path, ignore_errors=True)
        self._storage_budget.refresh_if_stale()
