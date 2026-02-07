"""Handles resolving filesystem paths and disk usage for traces."""

from __future__ import annotations

import pathlib

from neuracore.data_daemon.recording_encoding_disk_manager.core.storage_budget import (
    scan_dir_bytes,
)

from .types import _TraceKey


class _TraceFilesystem:
    """Resolve filesystem paths and disk usage for traces."""

    def __init__(self, recordings_root: pathlib.Path) -> None:
        """Initialise _TraceFilesystem.

        Args:
            recordings_root: Root directory under which recordings are stored.
        """
        self.recordings_root = recordings_root

    def trace_dir_for(self, trace_key: _TraceKey) -> pathlib.Path:
        """Resolve the on-disk directory for a trace.

        Args:
            trace_key: Trace key.

        Returns:
            Directory path for the trace.
        """
        return (
            self.recordings_root
            / trace_key.recording_id
            / trace_key.data_type.value
            / trace_key.trace_id
        )

    def trace_bytes_on_disk(self, trace_key: _TraceKey) -> int:
        """Compute total bytes on disk for a single trace directory.

        Args:
            trace_key: Trace key.

        Returns:
            Total bytes used under the trace directory.
        """
        return scan_dir_bytes(self.trace_dir_for(trace_key))
