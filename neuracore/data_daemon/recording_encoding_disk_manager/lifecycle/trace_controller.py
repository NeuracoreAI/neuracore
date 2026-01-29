"""Handles trace lifecycle operations."""

from __future__ import annotations

import logging
import shutil
from typing import Any

from neuracore_types import DataType

from neuracore.data_daemon.event_emitter import Emitter, get_emitter
from neuracore.data_daemon.recording_encoding_disk_manager.core.storage_budget import (
    StorageBudget,
)

from ..core.trace_filesystem import _TraceFilesystem
from ..core.types import _TraceKey

logger = logging.getLogger(__name__)


class _TraceController:
    """Coordinate trace lifecycle operations."""

    def __init__(
        self,
        *,
        filesystem: _TraceFilesystem,
        storage_budget: StorageBudget,
        recording_traces: dict[str, dict[str, Any]],
    ) -> None:
        """Initialise _TraceController.

        Args:
            filesystem: Filesystem helper for path resolution and sizing.
            storage_budget: Storage budget tracker.
            recording_traces: Recording-to-traces bookkeeping map.
        """
        self._filesystem = filesystem
        self._storage_budget = storage_budget
        self.recording_traces = recording_traces

        self._emitter = get_emitter()

    def abort_trace_due_to_storage(self, trace_key: _TraceKey) -> None:
        """Abort a trace due to storage constraints and emit events.

        Args:
            trace_key: Trace key to abort.

        Returns:
            None
        """
        self._emitter.emit(Emitter.TRACE_ABORTED, trace_key)

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

    def on_stop_all_traces_for_recording(self, recording_id: str) -> None:
        """Handle STOP_ALL_TRACES_FOR_RECORDING(recording_id).

        Args:
            recording_id: Recording identifier to stop.

        Returns:
            None
        """
        self._emitter.emit(Emitter.RECORDING_STOPPED, str(recording_id))

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

        self._emitter.emit(Emitter.TRACE_ABORTED, trace_key)

        recording_entry = self.recording_traces.get(trace_key.recording_id)
        if recording_entry is not None:
            recording_entry.pop(trace_key.trace_id, None)
            if not recording_entry:
                self.recording_traces.pop(trace_key.recording_id, None)

        trace_dir_path = self._filesystem.trace_dir_for(trace_key)
        try:
            reclaimed_bytes = self._filesystem.trace_bytes_on_disk(trace_key)
        except OSError:
            logger.warning("Failed to get bytes on disk for trace %s", trace_key)
            reclaimed_bytes = 0

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

        self._emitter.emit(Emitter.RECORDING_STOPPED, recording_id_value)

        self.recording_traces.pop(recording_id_value, None)

        path = self._filesystem.recordings_root / recording_id_value
        shutil.rmtree(path, ignore_errors=True)
        self._storage_budget.refresh_if_stale()
