"""The types associated to storing and batching messages."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

from neuracore_types import DataType


@dataclass(frozen=True, slots=True, repr=True)
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


@dataclass(slots=True, repr=True)
class _WriteState:
    """In-memory write state for a trace.

    Args:
        trace_key: Key identifying the trace.
        trace_dir: Directory where trace files are written.
        batch_index: Current batch index for raw batch files.
        buffer: Buffered newline-delimited message envelopes.
        trace_done: Whether the trace has received its final chunk.
    """

    trace_key: _TraceKey
    trace_dir: pathlib.Path
    batch_index: int
    buffer: bytearray
    trace_done: bool


@dataclass(slots=True, repr=True)
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


@dataclass(frozen=True, slots=True, repr=True)
class _RGBFrameRef:
    """Reference to one raw RGB frame stored in a per-trace spool file."""

    spool_path: pathlib.Path
    offset: int
    length: int


@dataclass(frozen=True, slots=True, repr=True)
class _RGBIndexedFrame:
    """One RGB frame's metadata plus its raw-byte reference."""

    metadata: dict[str, Any]
    frame_ref: _RGBFrameRef


@dataclass(slots=True, repr=True)
class _RGBWriteState:
    """In-memory write state for one RGB trace."""

    trace_key: _TraceKey
    trace_dir: pathlib.Path
    frames: list[_RGBIndexedFrame]
    trace_done: bool
    data_type_name: str
    robot_instance: int
    dataset_id: str | None
    dataset_name: str | None
    robot_name: str | None
    robot_id: str | None


@dataclass(frozen=True, slots=True, repr=True)
class _RGBTraceMessage:
    """RGB trace ingress item that stores frame metadata and a spool ref."""

    trace_key: _TraceKey
    data_type_name: str
    robot_instance: int
    dataset_id: str | None
    dataset_name: str | None
    robot_name: str | None
    robot_id: str | None
    frame_metadata: dict[str, Any] | None
    frame_ref: _RGBFrameRef | None
    final_chunk: bool


@dataclass(frozen=True, slots=True, repr=True)
class _RGBSpoolJob:
    """Encoder work item for one RGB trace backed by raw frame refs."""

    trace_key: _TraceKey
    frames: list[_RGBIndexedFrame]
    trace_done: bool
