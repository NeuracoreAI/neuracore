"""Shared event emitter for cross-component signaling."""

import asyncio
import logging
from typing import Any

from pyee.asyncio import AsyncIOEventEmitter

logger = logging.getLogger(__name__)


class Emitter(AsyncIOEventEmitter):
    """Shared event emitter for cross-component signaling."""

    # Comms manager to State manager
    STOP_RECORDING = "STOP_RECORDING"

    # State manager -> RDM
    STOP_ALL_TRACES_FOR_RECORDING = "STOP_ALL_TRACES_FOR_RECORDING"
    # (recording_id, trace_id)

    # RDM -> State manager
    TRACE_WRITTEN = "TRACE_WRITTEN"
    # (trace_id, recording_id, bytes_written)

    # RDM -> State manager
    START_TRACE = "START_TRACE"

    # State manager -> Uploader
    READY_FOR_UPLOAD = "READY_FOR_UPLOAD"
    # (trace_id, recording_id, path, data_type, data_type_name, bytes_uploaded)

    # Connection manager -> Uploader
    IS_CONNECTED = "IS_CONNECTED"
    # (Event will trigger a state change in the consumers,
    #  this will enable tasks with
    # internet requirements to operate)

    # Upload manager -> State manager
    UPLOADED_BYTES = "UPLOADED_BYTES"
    # (trace_id, bytes_uploaded(total))

    # State manager -> Progress reporter
    PROGRESS_REPORT = "PROGRESS_REPORT"
    # (start_time:float, end_time:float, traces:List[TraceRecord])

    # Progress reporter -> State manager
    PROGRESS_REPORTED = "PROGRESS_REPORTED"
    # (recording_id:str)

    # Progress reporter -> State manager
    PROGRESS_REPORT_FAILED = "PROGRESS_REPORT_FAILED"
    # (recording_id:str, error_message:str)

    # Uploader -> State manager
    UPLOAD_COMPLETE = "UPLOAD_COMPLETE"
    # (trace_id)

    # Uploader -> state manager
    UPLOAD_FAILED = "UPLOAD_FAILED"
    # (Trace_id, bytes_uploaded, status, error_code, error_message)

    # State manager -> RDM
    DELETE_TRACE = "DELETE_TRACE"
    # (recording_id, trace_id, data_type)

    def __init__(self, *, loop: asyncio.AbstractEventLoop) -> None:
        """Initialize the event emitter.

        Args:
            loop: The event loop to use for async event handlers.
        """
        super().__init__(loop=loop)

    def emit(self, event: str, *args: Any, **kwargs: Any) -> bool:
        """Emit an event with logging.

        Args:
            event: The event name to emit.
            *args: Positional arguments to pass to handlers.
            **kwargs: Keyword arguments to pass to handlers.

        Returns:
            True if the event had listeners, False otherwise.
        """
        formatted_args = []
        for arg in args:
            if isinstance(arg, str) and len(arg) > 50:
                formatted_args.append(f"{arg[:50]}...")
            elif isinstance(arg, bytes) and len(arg) > 20:
                formatted_args.append(f"<{len(arg)} bytes>")
            elif isinstance(arg, list):
                formatted_args.append(f"<list of {len(arg)} items>")
            else:
                r = repr(arg)
                if len(r) > 100:
                    formatted_args.append(f"{r[:100]}...")
                else:
                    formatted_args.append(r)
        args_str = ", ".join(formatted_args) if formatted_args else ""
        logger.info("EVENT %s: %s", event, args_str)
        return super().emit(event, *args, **kwargs)


_emitter: Emitter | None = None


def init_emitter(*, loop: asyncio.AbstractEventLoop) -> Emitter:
    """Initialize the global emitter once the General loop is running.

    Args:
        loop: The event loop to use for async event handlers.

    """
    global _emitter
    if _emitter is not None:
        raise RuntimeError("Emitter already initialized")
    _emitter = Emitter(loop=loop)
    return _emitter


def get_emitter() -> Emitter:
    """Return the initialized emitter."""
    if _emitter is None:
        raise RuntimeError("Emitter not initialized.")
    return _emitter
