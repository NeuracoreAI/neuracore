"""Handles buffered CompleteMessage envelopes into raw batch files."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable
from typing import Any, cast

import aiofiles

from neuracore.data_daemon.event_emitter import Emitter, get_emitter
from neuracore.data_daemon.models import CompleteMessage
from neuracore.data_daemon.recording_encoding_disk_manager.core.storage_budget import (
    StorageBudget,
)

from ..core.trace_filesystem import _TraceFilesystem
from ..core.types import _BatchJob, _TraceKey, _WriteState


class _RawBatchWriter:
    """Write worker that buffers CompleteMessage envelopes into raw batch files."""

    def __init__(
        self,
        *,
        flush_bytes: int,
        trace_message_queue: asyncio.Queue[CompleteMessage | object],
        raw_file_queue: asyncio.Queue[_BatchJob | object],
        filesystem: _TraceFilesystem,
        storage_budget: StorageBudget,
        state_lock: threading.RLock,
        writer_states: dict[_TraceKey, _WriteState],
        aborted_traces: set[_TraceKey],
        stopped_recordings: set[str],
        recording_traces: dict[str, dict[str, Any]],
        abort_trace: Callable[[_TraceKey], None],
        closed_traces: set[_TraceKey],
        sentinel: object,
    ) -> None:
        """Initialise _RawBatchWriter.

        Args:
            flush_bytes: Flush threshold for buffered raw writes.
            trace_message_queue: Queue that receives CompleteMessage items.
            raw_file_queue: Queue to publish raw batch jobs for encoding.
            filesystem: Filesystem helper for path resolution.
            storage_budget: Storage budget tracker.
            state_lock: Shared lock protecting writer state.
            writer_states: Shared write-state registry keyed by trace.
            aborted_traces: Shared set of traces that should be ignored.
            stopped_recordings: Shared set of recordings that should be ignored.
            recording_traces: Recording-to-traces bookkeeping map.
            abort_trace: Callback used to abort traces on failure.
            closed_traces: Set of trace keys that are closed.
            sentinel: Sentinel object used to stop the worker.
        """
        self.flush_bytes = flush_bytes
        self.trace_message_queue = trace_message_queue
        self.raw_file_queue = raw_file_queue

        self._filesystem = filesystem
        self._storage_budget = storage_budget

        self._state_lock = state_lock
        self._writer_states = writer_states
        self._aborted_traces = aborted_traces
        self._stopped_recordings = stopped_recordings
        self.recording_traces = recording_traces
        self._abort_trace = abort_trace
        self._closed_traces = closed_traces

        self.SENTINEL = sentinel

        self._emitter = get_emitter()

    async def _flush_state(self, writer_state: _WriteState) -> None:
        """Flush buffered data to a raw batch file and enqueue an encoder job.

        Args:
            writer_state: Write state to flush.

        Returns:
            None
        """
        with self._state_lock:
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

        await self.raw_file_queue.put(
            _BatchJob(
                trace_key=trace_key,
                batch_path=batch_path,
                trace_done=trace_done,
            )
        )

    async def worker(self) -> None:
        """Worker loop: buffer messages and write raw batch files to disk.

        Returns:
            None
        """
        while True:
            queue_item = await self.trace_message_queue.get()

            if queue_item is self.SENTINEL:
                with self._state_lock:
                    writer_states_remaining = list(self._writer_states.values())

                for state_to_flush in writer_states_remaining:
                    with self._state_lock:
                        state_to_flush.trace_done = True
                    await self._flush_state(state_to_flush)

                with self._state_lock:
                    self._writer_states.clear()

                await self.raw_file_queue.put(self.SENTINEL)
                self.trace_message_queue.task_done()
                break

            raw_message = cast(CompleteMessage, queue_item)
            recording_id_value = str(raw_message.recording_id)

            with self._state_lock:
                if recording_id_value in self._stopped_recordings:
                    self.trace_message_queue.task_done()
                    continue

            trace_key = _TraceKey(
                recording_id=recording_id_value,
                data_type=raw_message.data_type,
                trace_id=str(raw_message.trace_id),
            )

            with self._state_lock:
                if (
                    trace_key in self._aborted_traces
                    or trace_key in self._closed_traces
                ):
                    self.trace_message_queue.task_done()
                    continue

                recording_entry = self.recording_traces.setdefault(
                    recording_id_value, {}
                )
                is_new_trace = trace_key.trace_id not in recording_entry
                if is_new_trace:
                    recording_entry[trace_key.trace_id] = {}

            trace_dir = self._filesystem.trace_dir_for(trace_key)

            with self._state_lock:
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
                    path=str(trace_dir),
                )

            if writer_state is None:
                raise RuntimeError(
                    "Writer state unexpectedly None after lock acquisition"
                )

            json_line = json.dumps(
                raw_message.to_dict(),
                separators=(",", ":"),
                ensure_ascii=False,
            )
            encoded_line = (json_line + "\n").encode("utf-8")

            with self._state_lock:
                writer_state.buffer.extend(encoded_line)
                if raw_message.final_chunk:
                    writer_state.trace_done = True
                should_flush = (
                    len(writer_state.buffer) >= self.flush_bytes
                    or raw_message.final_chunk
                )

            if should_flush:
                await self._flush_state(writer_state)
                with self._state_lock:
                    if writer_state.trace_done:
                        self._writer_states.pop(trace_key, None)
                        if raw_message.final_chunk:
                            self._closed_traces.add(trace_key)

            self.trace_message_queue.task_done()
