"""Handles buffered CompleteMessage envelopes into raw batch files."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import aiofiles

from neuracore.data_daemon.event_emitter import Emitter
from neuracore.data_daemon.models import CompleteMessage
from neuracore.data_daemon.recording_encoding_disk_manager.core.storage_budget import (
    StorageBudget,
)
from neuracore.data_daemon.recording_encoding_disk_manager.encoding.video_trace import (
    rgb_batch_metadata_path,
    split_video_payload_for_rgb_batch,
)

from ..core.trace_filesystem import _TraceFilesystem
from ..core.types import _BatchJob, _TraceKey, _WriteState


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
        rgb_flush_bytes: int,
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
            rgb_flush_bytes: Flush threshold for buffered RGB batch files.
            trace_message_queue: Queue that receives CompleteMessage items.
            filesystem: Filesystem helper for path resolution.
            storage_budget: Storage budget tracker.
            recording_traces: Recording-to-traces bookkeeping map.
            abort_trace: Callback used to abort traces on failure.
            sentinel: Sentinel object used to stop the worker.
            emitter: Event emitter for cross-component signaling.
        """
        self.flush_bytes = flush_bytes
        self.rgb_flush_bytes = rgb_flush_bytes
        self.trace_message_queue = trace_message_queue

        self._filesystem = filesystem
        self._storage_budget = storage_budget

        self.recording_traces = recording_traces
        self._abort_trace = abort_trace

        self.SENTINEL = sentinel

        self._emitter = emitter

        self._writer_states: dict[_TraceKey, _WriteState] = {}
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
        writer_state = self._writer_states.pop(trace_key, None)
        if writer_state is not None:
            self._schedule_rgb_close(writer_state)

    async def _on_recording_stopped(self, recording_id: str) -> None:
        """Handle RECORDING_STOPPED event.

        Marks the recording as stopped and flushes all pending writer states
        for traces belonging to this recording.

        Args:
            recording_id: Recording that was stopped.
        """
        await self.trace_message_queue.put(_StopRecording(recording_id))

    @staticmethod
    def _is_rgb_trace(trace_key: _TraceKey) -> bool:
        return trace_key.data_type.value == "RGB_IMAGES"

    @staticmethod
    def _current_rgb_batch_paths(writer_state: _WriteState) -> tuple[Path, Path]:
        batch_path = writer_state.trace_dir / f"batch_{writer_state.batch_index:06d}.rgb"
        return batch_path, rgb_batch_metadata_path(batch_path)

    def _schedule_rgb_close(self, writer_state: _WriteState) -> None:
        if writer_state.rgb_batch_file is None and writer_state.rgb_metadata_file is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._close_rgb_batch_files(writer_state))

    async def _close_rgb_batch_files(self, writer_state: _WriteState) -> None:
        for attr_name in ("rgb_batch_file", "rgb_metadata_file"):
            handle = getattr(writer_state, attr_name)
            if handle is None:
                continue
            try:
                await handle.close()
            except Exception:
                pass
            finally:
                setattr(writer_state, attr_name, None)

    async def _ensure_rgb_batch_files_open(self, writer_state: _WriteState) -> None:
        if writer_state.rgb_batch_file is not None and writer_state.rgb_metadata_file is not None:
            return

        batch_path, metadata_path = self._current_rgb_batch_paths(writer_state)
        batch_file = None
        metadata_file = None
        try:
            batch_file = await aiofiles.open(batch_path, "ab")
            metadata_file = await aiofiles.open(metadata_path, "ab")
        except Exception:
            if batch_file is not None:
                await batch_file.close()
            if metadata_file is not None:
                await metadata_file.close()
            raise

        writer_state.rgb_batch_file = batch_file
        writer_state.rgb_metadata_file = metadata_file

    def _emit_batch_ready(
        self,
        *,
        trace_key: _TraceKey,
        batch_path: Path,
        trace_done: bool,
    ) -> None:
        self._emitter.emit(
            Emitter.BATCH_READY,
            _BatchJob(
                trace_key=trace_key,
                batch_path=batch_path,
                trace_done=trace_done,
            ),
        )

    async def _flush_non_rgb_state(self, writer_state: _WriteState) -> None:
        """Flush buffered non-RGB data to a batch file and emit BATCH_READY."""
        trace_key = writer_state.trace_key
        trace_done = writer_state.trace_done
        trace_dir = writer_state.trace_dir
        batch_index = writer_state.batch_index

        if len(writer_state.buffer) == 0 and not trace_done:
            return

        writer_state.batch_index += 1
        payload_bytes = bytes(writer_state.buffer)
        writer_state.buffer.clear()

        batch_path = trace_dir / f"batch_{batch_index:06d}.raw"
        bytes_to_write = len(payload_bytes)

        if not self._storage_budget.has_free_disk_for_write(bytes_to_write):
            self._abort_trace(trace_key)
            return

        if not self._storage_budget.reserve(bytes_to_write):
            self._abort_trace(trace_key)
            return

        try:
            async with aiofiles.open(batch_path, "wb") as f:
                await f.write(payload_bytes)
        except Exception:
            self._storage_budget.release(bytes_to_write)
            self._abort_trace(trace_key)
            return

        self._emit_batch_ready(
            trace_key=trace_key,
            batch_path=batch_path,
            trace_done=trace_done,
        )

    async def _append_rgb_payload(
        self,
        writer_state: _WriteState,
        payload_bytes: bytes,
    ) -> bool:
        """Append one RGB payload directly into the active on-disk batch."""
        trace_key = writer_state.trace_key

        try:
            metadata_bytes, frame_bytes = split_video_payload_for_rgb_batch(payload_bytes)
        except Exception:
            self._abort_trace(trace_key)
            return False

        metadata_line = b""
        if metadata_bytes is not None:
            metadata_line = metadata_bytes + b"\n"
        frame_bytes_to_write = frame_bytes or b""
        bytes_to_write = len(frame_bytes_to_write) + len(metadata_line)

        if (
            writer_state.rgb_batch_started
            and writer_state.rgb_batch_bytes > 0
            and writer_state.rgb_batch_bytes + bytes_to_write > self.rgb_flush_bytes
        ):
            await self._finalize_rgb_batch(writer_state, trace_done=False)
            if trace_key in self._aborted_traces:
                return False

        if not self._storage_budget.has_free_disk_for_write(bytes_to_write):
            self._abort_trace(trace_key)
            return False

        if not self._storage_budget.reserve(bytes_to_write):
            self._abort_trace(trace_key)
            return False

        try:
            await self._ensure_rgb_batch_files_open(writer_state)
            batch_f = writer_state.rgb_batch_file
            meta_f = writer_state.rgb_metadata_file
            if batch_f is None or meta_f is None:
                raise RuntimeError("RGB batch files failed to open")
            await batch_f.write(frame_bytes_to_write)
            await meta_f.write(metadata_line)
        except Exception:
            self._storage_budget.release(bytes_to_write)
            self._abort_trace(trace_key)
            return False

        writer_state.rgb_batch_bytes += bytes_to_write
        writer_state.rgb_batch_started = True
        return True

    async def _finalize_rgb_batch(
        self,
        writer_state: _WriteState,
        *,
        trace_done: bool,
    ) -> None:
        """Close the active RGB batch and emit BATCH_READY.

        When trace_done is True and no payload has been spooled yet for the active
        batch index, create empty batch files so downstream finalisation still sees
        a terminal BATCH_READY signal.
        """
        trace_key = writer_state.trace_key
        if not writer_state.rgb_batch_started and not trace_done:
            return

        batch_path, metadata_path = self._current_rgb_batch_paths(writer_state)
        if not writer_state.rgb_batch_started:
            try:
                async with aiofiles.open(batch_path, "ab") as batch_f:
                    await batch_f.write(b"")
                async with aiofiles.open(metadata_path, "ab") as meta_f:
                    await meta_f.write(b"")
            except Exception:
                self._abort_trace(trace_key)
                return
        else:
            await self._close_rgb_batch_files(writer_state)

        self._emit_batch_ready(
            trace_key=trace_key,
            batch_path=batch_path,
            trace_done=trace_done,
        )
        writer_state.batch_index += 1
        writer_state.rgb_batch_bytes = 0
        writer_state.rgb_batch_started = False

    async def worker(self) -> None:
        """Worker loop: buffer messages and write raw batch files to disk.

        Returns:
            None
        """
        while True:
            queue_item = await self.trace_message_queue.get()

            if queue_item is self.SENTINEL:
                writer_states_remaining = list(self._writer_states.values())

                for state_to_flush in writer_states_remaining:
                    state_to_flush.trace_done = True
                    if self._is_rgb_trace(state_to_flush.trace_key):
                        await self._finalize_rgb_batch(state_to_flush, trace_done=True)
                    else:
                        await self._flush_non_rgb_state(state_to_flush)

                self._writer_states.clear()

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

                for ws in writer_states_to_flush:
                    self._closed_traces.add(ws.trace_key)
                    ws.trace_done = True
                    if self._is_rgb_trace(ws.trace_key):
                        await self._finalize_rgb_batch(ws, trace_done=True)
                    else:
                        await self._flush_non_rgb_state(ws)
                    self._writer_states.pop(ws.trace_key, None)

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

            if self._is_rgb_trace(trace_key):
                if raw_message.data and not await self._append_rgb_payload(
                    writer_state, raw_message.data
                ):
                    self.trace_message_queue.task_done()
                    continue

                if raw_message.final_chunk:
                    writer_state.trace_done = True

                should_finalize_rgb = (
                    writer_state.rgb_batch_started
                    and writer_state.rgb_batch_bytes >= self.rgb_flush_bytes
                ) or raw_message.final_chunk
                if should_finalize_rgb:
                    await self._finalize_rgb_batch(
                        writer_state,
                        trace_done=writer_state.trace_done,
                    )
                    if trace_key in self._aborted_traces:
                        self.trace_message_queue.task_done()
                        continue
                    if writer_state.trace_done:
                        self._writer_states.pop(trace_key, None)
                        if raw_message.final_chunk:
                            self._closed_traces.add(trace_key)

                self.trace_message_queue.task_done()
                continue

            json_line = json.dumps(
                raw_message.to_dict(),
                separators=(",", ":"),
                ensure_ascii=False,
            )
            encoded_line = (json_line + "\n").encode("utf-8")

            writer_state.buffer.extend(encoded_line)
            if raw_message.final_chunk:
                writer_state.trace_done = True
            should_flush = (
                len(writer_state.buffer) >= self.flush_bytes or raw_message.final_chunk
            )

            if should_flush:
                await self._flush_non_rgb_state(writer_state)
                if trace_key in self._aborted_traces:
                    self.trace_message_queue.task_done()
                    continue
                if writer_state.trace_done:
                    self._writer_states.pop(trace_key, None)
                    if raw_message.final_chunk:
                        self._closed_traces.add(trace_key)

            self.trace_message_queue.task_done()
