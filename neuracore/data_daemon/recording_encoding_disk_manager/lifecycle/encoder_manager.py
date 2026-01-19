"""Handles the encoding of raw batch files."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from neuracore.data_daemon.models import get_content_type
from neuracore.data_daemon.recording_encoding_disk_manager.encoding.json_trace import (
    JsonTrace,
)
from neuracore.data_daemon.recording_encoding_disk_manager.encoding.video_trace import (
    VideoTrace,
)

from ..core.trace_filesystem import _TraceFilesystem
from ..core.types import _TraceKey

logger = logging.getLogger(__name__)


class EncoderInitError(RuntimeError):
    """Raised when an encoder instance cannot be created."""


class _EncoderManager:
    """Create and manage encoder instances per trace with safe concurrency."""

    def __init__(
        self,
        *,
        filesystem: _TraceFilesystem,
        state_lock: threading.RLock,
        encoders: dict[_TraceKey, JsonTrace | VideoTrace],
        abort_trace: Callable[[_TraceKey], None],
    ) -> None:
        """Initialise _EncoderManager.

        Args:
            filesystem: Filesystem helper for path resolution.
            state_lock: Shared lock protecting encoder state.
            encoders: Shared encoder registry keyed by trace.
            abort_trace: Callback used to abort traces on failure.
        """
        self._filesystem = filesystem
        self._state_lock = state_lock
        self._encoders = encoders
        self._abort_trace = abort_trace

    def _get_encoder(self, trace_key: _TraceKey) -> JsonTrace | VideoTrace:
        """Get or create the encoder instance for a trace.

        Args:
            trace_key: Trace key.

        Returns:
            Encoder for the trace.
        """
        with self._state_lock:
            existing_encoder = self._encoders.get(trace_key)
            if existing_encoder is not None:
                return existing_encoder

            trace_dir = self._filesystem.trace_dir_for(trace_key)
            content_kind = get_content_type(trace_key.data_type)
            created_encoder: JsonTrace | VideoTrace

            try:
                if content_kind == "RGB":
                    created_encoder = VideoTrace(output_dir=trace_dir)
                else:
                    created_encoder = JsonTrace(output_dir=trace_dir)
            except Exception:
                self._abort_trace(trace_key)
                raise EncoderInitError(f"Failed to create encoder for {trace_key}")

            self._encoders[trace_key] = created_encoder
            return created_encoder

    def safe_get_encoder(self, trace_key: _TraceKey) -> JsonTrace | VideoTrace | None:
        """Get or create an encoder for a trace, converting failures into a trace abort.

        Args:
            trace_key: Trace identifier tuple.

        Returns:
            The encoder instance if available, otherwise None if the trace was aborted.
        """
        try:
            return self._get_encoder(trace_key)
        except Exception:
            self._abort_trace(trace_key)
            return None

    def pop_encoder(self, trace_key: _TraceKey) -> JsonTrace | VideoTrace | None:
        """Remove and return an encoder for a trace if present.

        Args:
            trace_key: Trace key.

        Returns:
            Encoder instance if present, otherwise None.
        """
        with self._state_lock:
            return self._encoders.pop(trace_key, None)

    def clear_all_encoders(self) -> list[tuple[_TraceKey, JsonTrace | VideoTrace]]:
        """Remove and return all active encoders.

        Returns:
            List of (trace_key, encoder) for all active encoders.
        """
        with self._state_lock:
            remaining = list(self._encoders.items())
            self._encoders.clear()
        return remaining
