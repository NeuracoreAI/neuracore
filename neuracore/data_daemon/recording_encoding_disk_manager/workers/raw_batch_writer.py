"""Handles buffered CompleteMessage envelopes into raw batch files."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import aiofiles
from neuracore_types import DataType

from neuracore.data_daemon.event_emitter import Emitter
from neuracore.data_daemon.models import CompleteMessage
from neuracore.data_daemon.recording_encoding_disk_manager.core.storage_budget import (
    StorageBudget,
)

from ..core.trace_filesystem import _TraceFilesystem
from ..core.types import (
    _BatchJob,
    _RGBIndexedFrame,
    _RGBSpoolJob,
    _RGBTraceMessage,
    _RGBWriteState,
    _TraceKey,
    _WriteState,
)


@dataclass(frozen=True)
class _StopRecording:
    recording_id: str


class _RawBatchWriter:
    """Write worker that buffers CompleteMessage envelopes into raw batch files.

    Uses event-driven architecture:
    - Listens for TRACE_ABORTED and RECORDING_STOPPED events
    - Emits BATCH_READY events when batches are written to disk
    - Owns its own state (writer_states, aborted_traces, etc.)
    """

    def __init__(
        self,
        *,
        flush_bytes: int,
        trace_message_queue: asyncio.Queue[CompleteMessage | object],
        filesystem: _TraceFilesystem,
        storage_budget: StorageBudget,
        recording_traces: dict[str, dict[str, Any]],
        abort_trace: Callable[[_TraceKey], None],
        sentinel: object,
        emitter: Emitter,
    ) -> None:
        """Initialise _RawBatchWriter.

        Args:
            flush_bytes: Flush threshold for buffered raw writes.
            trace_message_queue: Queue that receives CompleteMessage items.
            filesystem: Filesystem helper for path resolution.
            storage_budget: Storage budget tracker.
            recording_traces: Recording-to-traces bookkeeping map.
            abort_trace: Callback used to abort traces on failure.
            sentinel: Sentinel object used to stop the worker.
            emitter: Event emitter for cross-component signaling.
        """
        self.flush_bytes = flush_bytes
        self.trace_message_queue = trace_message_queue

        self._filesystem = filesystem
        self._storage_budget = storage_budget

        self.recording_traces = recording_traces
        self._abort_trace = abort_trace

        self.SENTINEL = sentinel

        self._emitter = emitter

        self._writer_states: dict[_TraceKey, _WriteState] = {}
        self._rgb_writer_states: dict[_TraceKey, _RGBWriteState] = {}
        self._aborted_traces: set[_TraceKey] = set()
        self._stopped_recordings: set[str] = set()
        self._closed_traces: set[_TraceKey] = set()

        self._emitter.on(Emitter.TRACE_ABORTED, self._on_trace_aborted)
        self._emitter.on(Emitter.RECORDING_STOPPED, self._on_recording_stopped)

    def _on_trace_aborted(self, trace_key: _TraceKey) -> None:
        """Handle TRACE_ABORTED event.

        Args:
            trace_key: Trace key that was aborted.
        """
        self._aborted_traces.add(trace_key)
        self._closed_traces.add(trace_key)
        self._writer_states.pop(trace_key, None)
        rgb_state = self._rgb_writer_states.pop(trace_key, None)
        if rgb_state is not None and rgb_state.frames:
            rgb_state.frames[0].frame_ref.spool_path.unlink(missing_ok=True)

    async def _on_recording_stopped(self, recording_id: str) -> None:
        """Handle RECORDING_STOPPED event.

        Marks the recording as stopped and flushes all pending writer states
        for traces belonging to this recording.

        Args:
            recording_id: Recording that was stopped.
        """
        await self.trace_message_queue.put(_StopRecording(recording_id))

    async def _flush_state(self, writer_state: _WriteState) -> None:
        """Flush buffered data to a raw batch file and emit BATCH_READY.

        Args:
            writer_state: Write state to flush.

        Returns:
            None
        """
        trace_key = writer_state.trace_key
        buffered_bytes = len(writer_state.buffer)
        trace_done = writer_state.trace_done
        trace_dir = writer_state.trace_dir
        batch_index = writer_state.batch_index

        if buffered_bytes == 0 and not trace_done:
            return

        writer_state.batch_index += 1
        payload_bytes = bytes(writer_state.buffer)
        writer_state.buffer.clear()

        if not self._storage_budget.has_free_disk_for_write(buffered_bytes):
            self._abort_trace(trace_key)
            return

        if not self._storage_budget.reserve(buffered_bytes):
            self._abort_trace(trace_key)
            return

        batch_file_name = f"batch_{batch_index:06d}.raw"
        batch_path = trace_dir / batch_file_name
        try:
            async with aiofiles.open(batch_path, "wb") as f:
                await f.write(payload_bytes)
        except Exception:
            self._storage_budget.release(buffered_bytes)
            self._abort_trace(trace_key)
            return

        self._emitter.emit(
            Emitter.BATCH_READY,
            _BatchJob(
                trace_key=trace_key,
                batch_path=batch_path,
                trace_done=trace_done,
            ),
        )

    async def worker(self) -> None:
        """Worker loop: buffer messages and write raw batch files to disk.

        Returns:
            None
        """
        while True:
            queue_item = await self.trace_message_queue.get()

            if queue_item is self.SENTINEL:
                writer_states_remaining = list(self._writer_states.values())
                rgb_writer_states_remaining = list(self._rgb_writer_states.values())

                for state_to_flush in writer_states_remaining:
                    state_to_flush.trace_done = True
                    await self._flush_state(state_to_flush)

                for rgb_state in rgb_writer_states_remaining:
                    rgb_state.trace_done = True
                    await self._flush_rgb_state(rgb_state)

                self._writer_states.clear()
                self._rgb_writer_states.clear()

                self._emitter.remove_listener(
                    Emitter.TRACE_ABORTED, self._on_trace_aborted
                )
                self._emitter.remove_listener(
                    Emitter.RECORDING_STOPPED, self._on_recording_stopped
                )

                self.trace_message_queue.task_done()
                break

            if isinstance(queue_item, _StopRecording):
                recording_id = queue_item.recording_id
                self._stopped_recordings.add(recording_id)

                writer_states_to_flush = [
                    ws
                    for ws in self._writer_states.values()
                    if ws.trace_key.recording_id == recording_id
                ]
                rgb_states_to_flush = [
                    ws
                    for ws in self._rgb_writer_states.values()
                    if ws.trace_key.recording_id == recording_id
                ]

                for ws in writer_states_to_flush:
                    self._closed_traces.add(ws.trace_key)
                    ws.trace_done = True
                    await self._flush_state(ws)
                    self._writer_states.pop(ws.trace_key, None)

                for ws in rgb_states_to_flush:
                    self._closed_traces.add(ws.trace_key)
                    ws.trace_done = True
                    await self._flush_rgb_state(ws)
                    self._rgb_writer_states.pop(ws.trace_key, None)

                self.trace_message_queue.task_done()
                continue

            if isinstance(queue_item, _RGBTraceMessage):
                await self._handle_rgb_trace_message(queue_item)
                self.trace_message_queue.task_done()
                continue

            raw_message = cast(CompleteMessage, queue_item)
            recording_id_value = str(raw_message.recording_id)

            if recording_id_value in self._stopped_recordings:
                self.trace_message_queue.task_done()
                continue

            trace_key = _TraceKey(
                recording_id=recording_id_value,
                data_type=raw_message.data_type,
                trace_id=str(raw_message.trace_id),
            )

            if trace_key in self._aborted_traces or trace_key in self._closed_traces:
                self.trace_message_queue.task_done()
                continue

            recording_entry = self.recording_traces.setdefault(recording_id_value, {})
            is_new_trace = trace_key.trace_id not in recording_entry
            if is_new_trace:
                recording_entry[trace_key.trace_id] = {}

            trace_dir = self._filesystem.trace_dir_for(trace_key)

            writer_state: _WriteState | None = self._writer_states.get(trace_key)
            if writer_state is None:
                try:
                    trace_dir.mkdir(parents=True, exist_ok=True)
                except OSError:
                    self._abort_trace(trace_key)
                    self.trace_message_queue.task_done()
                    continue
                writer_state = _WriteState(
                    trace_key=trace_key,
                    trace_dir=trace_dir,
                    batch_index=0,
                    buffer=bytearray(),
                    trace_done=False,
                )
                self._writer_states[trace_key] = writer_state

            if is_new_trace:
                self._emitter.emit(
                    Emitter.START_TRACE,
                    trace_key.trace_id,
                    trace_key.recording_id,
                    trace_key.data_type,
                    raw_message.data_type_name,
                    raw_message.robot_instance,
                    raw_message.dataset_id,
                    raw_message.dataset_name,
                    raw_message.robot_name,
                    raw_message.robot_id,
                    str(trace_dir),
                )

            if writer_state is None:
                raise RuntimeError("Writer state unexpectedly None")

            writer_state.buffer.extend(raw_message.to_batch_record())

            if raw_message.final_chunk:
                writer_state.trace_done = True
            should_flush = (
                len(writer_state.buffer) >= self.flush_bytes or raw_message.final_chunk
            )

            if should_flush:
                await self._flush_state(writer_state)
                if writer_state.trace_done:
                    self._writer_states.pop(trace_key, None)
                    if raw_message.final_chunk:
                        self._closed_traces.add(trace_key)

            self.trace_message_queue.task_done()

    async def _handle_rgb_trace_message(self, raw_message: _RGBTraceMessage) -> None:
        """Track RGB frame refs until the trace can be handed to the encoder."""
        trace_key = raw_message.trace_key
        recording_id_value = str(trace_key.recording_id)

        if recording_id_value in self._stopped_recordings:
            return

        if trace_key in self._aborted_traces or trace_key in self._closed_traces:
            if raw_message.frame_ref is not None:
                raw_message.frame_ref.spool_path.unlink(missing_ok=True)
            return

        recording_entry = self.recording_traces.setdefault(recording_id_value, {})
        is_new_trace = trace_key.trace_id not in recording_entry
        if is_new_trace:
            recording_entry[trace_key.trace_id] = {}

        trace_dir = self._filesystem.trace_dir_for(trace_key)
        writer_state = self._rgb_writer_states.get(trace_key)
        if writer_state is None:
            trace_dir.mkdir(parents=True, exist_ok=True)
            writer_state = _RGBWriteState(
                trace_key=trace_key,
                trace_dir=trace_dir,
                frames=[],
                trace_done=False,
                data_type_name=raw_message.data_type_name,
                robot_instance=raw_message.robot_instance,
                dataset_id=raw_message.dataset_id,
                dataset_name=raw_message.dataset_name,
                robot_name=raw_message.robot_name,
                robot_id=raw_message.robot_id,
            )
            self._rgb_writer_states[trace_key] = writer_state

        if is_new_trace:
            self._emitter.emit(
                Emitter.START_TRACE,
                trace_key.trace_id,
                trace_key.recording_id,
                DataType.RGB_IMAGES,
                writer_state.data_type_name,
                writer_state.robot_instance,
                writer_state.dataset_id,
                writer_state.dataset_name,
                writer_state.robot_name,
                writer_state.robot_id,
                str(trace_dir),
            )

        if raw_message.frame_metadata is not None and raw_message.frame_ref is not None:
            writer_state.frames.append(
                _RGBIndexedFrame(
                    metadata=raw_message.frame_metadata,
                    frame_ref=raw_message.frame_ref,
                )
            )

        if raw_message.final_chunk:
            writer_state.trace_done = True
            await self._flush_rgb_state(writer_state)
            self._rgb_writer_states.pop(trace_key, None)

    async def _flush_rgb_state(self, writer_state: _RGBWriteState) -> None:
        """Emit one RGB spool job for a trace backed by frame refs."""
        self._emitter.emit(
            Emitter.BATCH_READY,
            _RGBSpoolJob(
                trace_key=writer_state.trace_key,
                frames=list(writer_state.frames),
                trace_done=writer_state.trace_done,
            ),
        )
