"""Handles the encoding of raw batch files."""

from __future__ import annotations

import logging
from collections.abc import Callable

from neuracore_types import DataType

from neuracore.core.video_encoding import Libx264Options, codec_option_overrides
from neuracore.data_daemon.config_manager.config_watcher import ConfigWatcher
from neuracore.data_daemon.event_emitter import Emitter
from neuracore.data_daemon.models import get_content_type

from ..core.trace_filesystem import _TraceFilesystem
from ..core.types import TraceKey
from ..encoding.json_trace import JsonTrace
from ..encoding.point_cloud_trace import PointCloudTrace
from ..encoding.video_trace import VideoTrace

logger = logging.getLogger(__name__)


class EncoderInitError(RuntimeError):
    """Raised when an encoder instance cannot be created."""


class _EncoderManager:
    """Create and manage encoder instances per trace.

    Listens for TRACE_ABORTED events to cleanup encoders.
    Owns its own encoder registry.
    """

    def __init__(
        self,
        *,
        filesystem: _TraceFilesystem,
        abort_trace: Callable[[TraceKey], None],
        emitter: Emitter,
        config_watcher: ConfigWatcher,
    ) -> None:
        """Initialise _EncoderManager.

        Args:
            filesystem: Filesystem helper for path resolution.
            abort_trace: Callback used to abort traces on failure.
            emitter: Event emitter for cross-component signaling.
            config_watcher: In-memory daemon config, read for the video codec for new
                encoders.
        """
        self._filesystem = filesystem
        self._abort_trace = abort_trace
        self._config_watcher = config_watcher

        self._encoders: dict[TraceKey, JsonTrace | VideoTrace | PointCloudTrace] = {}

        self._emitter = emitter
        self._emitter.on(Emitter.TRACE_ABORTED, self._on_trace_aborted)

    def _on_trace_aborted(self, trace_key: TraceKey) -> None:
        """Handle TRACE_ABORTED event.

        Args:
            trace_key: Trace key that was aborted.
        """
        encoder = self._encoders.pop(trace_key, None)
        if encoder is not None:
            try:
                encoder.finish()
            except Exception:
                logger.exception(
                    "Encoder finish failed during abort for trace %s", trace_key
                )

    def _get_encoder(
        self, trace_key: TraceKey
    ) -> JsonTrace | VideoTrace | PointCloudTrace:
        """Get or create the encoder instance for a trace.

        Args:
            trace_key: Trace key.

        Returns:
            Encoder for the trace.
        """
        existing_encoder = self._encoders.get(trace_key)
        if existing_encoder is not None:
            return existing_encoder

        trace_dir = self._filesystem.trace_dir_for(trace_key)
        content_kind = get_content_type(trace_key.data_type)
        created_encoder: JsonTrace | VideoTrace | PointCloudTrace

        try:
            if content_kind == "RGB":
                created_encoder = VideoTrace(
                    output_dir=trace_dir,
                    codec_option_overrides=self._resolve_lossy_only_options(
                        trace_key.data_type
                    ),
                )
            elif content_kind == "POINT_CLOUD":
                created_encoder = PointCloudTrace(output_dir=trace_dir)
            else:
                created_encoder = JsonTrace(output_dir=trace_dir)
        except Exception:
            self._abort_trace(trace_key)
            raise EncoderInitError(f"Failed to create encoder for {trace_key}")

        self._encoders[trace_key] = created_encoder
        return created_encoder

    def _resolve_lossy_only_options(self, data_type: DataType) -> Libx264Options | None:
        """Resolve lossy-only encoder options for a trace, or None for default.

        Returns:
            Lossy codec-context options (dropping the lossless archive), or None
            to keep the default lossless+lossy encoders.
        """
        if data_type != DataType.RGB_IMAGES:
            return None
        self._config_watcher.refresh_now()
        return codec_option_overrides(self._config_watcher.video_codec())

    def safe_get_encoder(
        self, trace_key: TraceKey
    ) -> JsonTrace | VideoTrace | PointCloudTrace | None:
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

    def pop_encoder(
        self, trace_key: TraceKey
    ) -> JsonTrace | VideoTrace | PointCloudTrace | None:
        """Remove and return an encoder for a trace if present.

        Args:
            trace_key: Trace key.

        Returns:
            Encoder instance if present, otherwise None.
        """
        return self._encoders.pop(trace_key, None)

    def clear_all_encoders(
        self,
    ) -> list[tuple[TraceKey, JsonTrace | VideoTrace | PointCloudTrace]]:
        """Remove and return all active encoders.

        Returns:
            List of (trace_key, encoder) for all active encoders.
        """
        remaining = list(self._encoders.items())
        self._encoders.clear()
        return remaining

    def cleanup(self) -> None:
        """Remove event listeners during shutdown."""
        self._emitter.remove_listener(Emitter.TRACE_ABORTED, self._on_trace_aborted)
