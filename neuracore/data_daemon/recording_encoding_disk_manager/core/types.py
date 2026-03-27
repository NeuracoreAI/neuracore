"""The types associated to storing and batching messages."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

from neuracore_types import DataType


@dataclass(frozen=True)
class _TraceKey:
    """Unique key for a trace within a recording.

    Args:
        recording_id: Recording identifier.
        data_type: Trace data type.
        trace_id: Trace identifier.
    """

    recording_id: str
    data_type: DataType
    trace_id: str


@dataclass
class _WriteState:
    """In-memory write state for a trace.

    Args:
        trace_key: Key identifying the trace.
        trace_dir: Directory where trace files are written.
        batch_index: Current batch index for raw batch files.
        buffer: Buffered newline-delimited message envelopes for non-RGB traces.
        trace_done: Whether the trace has received its final chunk.
        rgb_batch_bytes: Bytes already appended to the active RGB batch.
        rgb_batch_started: Whether the active RGB batch file exists on disk.
    """

    trace_key: _TraceKey
    trace_dir: pathlib.Path
    batch_index: int
    buffer: bytearray
    trace_done: bool
    rgb_batch_bytes: int = 0
    rgb_batch_started: bool = False
    rgb_batch_file: Any | None = None
    rgb_metadata_file: Any | None = None


@dataclass
class _BatchJob:
    """Work item for the encoder thread.

    Args:
        trace_key: Key identifying the trace.
        batch_path: Path to the raw batch file to decode.
        trace_done: Whether this batch ends the trace.
    """

    trace_key: _TraceKey
    batch_path: pathlib.Path
    trace_done: bool
