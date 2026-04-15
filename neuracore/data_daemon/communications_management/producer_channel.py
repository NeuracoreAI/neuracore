"""High-level wrapper for a producer channel to the data daemon."""

from __future__ import annotations

import json
import logging
import math
import queue
import struct
import threading
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from neuracore.data_daemon.communications_management.communications_manager import (
    CommunicationsManager,
    MessageEnvelope,
)
from neuracore.data_daemon.communications_management.ring_buffer import RingBuffer
from neuracore.data_daemon.const import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_RING_BUFFER_SIZE,
    SHARED_RING_RECORD_HEADER_FORMAT,
    SHARED_RING_RECORD_MAGIC,
)
from neuracore.data_daemon.models import CommandType, DataChunkPayload, DataType

logger = logging.getLogger(__name__)

BytePart = bytes | bytearray | memoryview


@dataclass
class QueuedSharedRingWrite:
    """A shared-ring data write to be processed by the sender thread."""

    metadata: dict[str, str | int | None]
    chunk: bytes | bytearray | memoryview


class ProducerChannel:
    """High-level wrapper for a producer channel to the data daemon."""

    def __init__(
        self,
        id: str | None = None,
        comm_manager: CommunicationsManager | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        send_queue_maxsize: int = 0,
        recording_id: str | None = None,
    ) -> None:
        """Initialize the producer channel."""
        self.channel_id = id or str(uuid.uuid4())
        self._comm = comm_manager or CommunicationsManager()
        self._comm.create_producer_socket()
        self._shared_ring_transport_enabled = isinstance(
            self._comm,
            CommunicationsManager,
        )
        self.chunk_size = chunk_size
        self.send_queue_maxsize = max(0, int(send_queue_maxsize))
        self.trace_id: str | None = None
        self._stop_event = threading.Event()
        self.recording_id: str | None = recording_id
        self._heartbeat_interval = 1.0
        self._heartbeat_thread: threading.Thread | None = None
        self._send_queue: queue.Queue[
            MessageEnvelope | QueuedSharedRingWrite | None
        ] = queue.Queue(maxsize=self.send_queue_maxsize)
        self._sender_thread: threading.Thread | None = None
        self._next_sequence_number = 1
        self._last_enqueued_sequence_number = 0
        self._last_socket_sent_sequence_number = 0
        self._sequence_cv = threading.Condition()
        self._enqueue_lock = threading.Lock()
        self._shared_ring_buffer: RingBuffer | None = None
        self._sender_thread = threading.Thread(
            target=self._sender_loop,
            name="producer-channel-sender",
            daemon=True,
        )
        self._sender_thread.start()

    def _close_shared_ring_buffer(self) -> None:
        """Close the producer process handle to the shared ring buffer."""
        shared_ring_buffer = self._shared_ring_buffer
        self._shared_ring_buffer = None
        if shared_ring_buffer is None:
            return
        shared_ring_buffer.close()

    def _write_shared_ring_record(
        self,
        *,
        metadata: dict[str, str | int | None],
        chunk: bytes | bytearray | memoryview,
    ) -> None:
        """Write a self-describing record into the shared ring buffer."""
        if self._shared_ring_buffer is None:
            raise RuntimeError("Shared ring buffer not initialized")

        metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        chunk_bytes = chunk if isinstance(chunk, bytes) else bytes(chunk)
        record = b"".join(
            (
                struct.pack(
                    SHARED_RING_RECORD_HEADER_FORMAT,
                    SHARED_RING_RECORD_MAGIC,
                    len(metadata_bytes),
                    len(chunk_bytes),
                ),
                metadata_bytes,
                chunk_bytes,
            )
        )
        self._shared_ring_buffer.write(record)

    def _sender_loop(self) -> None:
        """Single thread that owns the transport ordering.

        This thread serializes shared-ring writes and ZMQ control messages so
        TRACE_END is only sent after prior ring writes have been committed.
        """
        while True:
            item = self._send_queue.get()
            try:
                if item is None:
                    break

                try:
                    if isinstance(item, QueuedSharedRingWrite):
                        self._write_shared_ring_record(
                            metadata=item.metadata,
                            chunk=item.chunk,
                        )
                    else:
                        envelope = item
                        self._comm.send_message(envelope)
                        if envelope.sequence_number is not None:
                            with self._sequence_cv:
                                if (
                                    envelope.sequence_number
                                    > self._last_socket_sent_sequence_number
                                ):
                                    self._last_socket_sent_sequence_number = (
                                        envelope.sequence_number
                                    )
                                self._sequence_cv.notify_all()
                except Exception as exc:
                    logger.warning("Send failed: %s", exc)
            finally:
                self._send_queue.task_done()

        with self._sequence_cv:
            self._sequence_cv.notify_all()

    def start_producer_channel(self) -> None:
        """Starts the producer channel's heartbeat loop."""
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return

        self._stop_event.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="producer-channel-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        """Heartbeat loop for producer."""
        self.heartbeat()

        while not self._stop_event.wait(self._heartbeat_interval):
            try:
                self.heartbeat()
            except Exception as exc:
                logger.warning("Heartbeat failed: %s", exc)

    def heartbeat(self) -> None:
        """Send a heartbeat message to the daemon."""
        self._send(CommandType.HEARTBEAT, {})

    def set_recording_id(self, recording_id: str | None) -> None:
        """Set the recording ID for the producer."""
        self.recording_id = recording_id

    def start_new_trace(self) -> None:
        """Start a new trace for the given recording."""
        if not self.recording_id:
            raise ValueError("recording_id is required; set on ProducerChannel init.")
        self.trace_id = str(uuid.uuid4())

    def end_trace(self) -> None:
        """End the active trace and notify the daemon."""
        trace_id = self.trace_id
        recording_id = self.recording_id
        if trace_id is None or recording_id is None:
            logger.warning("Cannot end trace; missing trace_id or recording_id.")
            return
        self._send(
            CommandType.TRACE_END,
            {
                "trace_end": {
                    "trace_id": trace_id,
                    "recording_id": recording_id,
                }
            },
        )
        self.trace_id = None
        self.recording_id = None

    def stop_producer_channel(self) -> None:
        """Stops the producer channel and cleans up any associated resources."""
        self._stop_event.set()
        self._send_queue.put(None)
        if self._sender_thread is not None:
            self._sender_thread.join(timeout=2)
            self._sender_thread = None
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1)
            self._heartbeat_thread = None
        self._close_shared_ring_buffer()
        self._comm.cleanup_producer()

    def _send(self, command: CommandType, payload: dict | None = None) -> int:
        """Enqueue a control message for the daemon."""
        with self._enqueue_lock:
            with self._sequence_cv:
                sequence_number = self._next_sequence_number
                self._next_sequence_number += 1
                self._last_enqueued_sequence_number = sequence_number
            envelope = MessageEnvelope(
                producer_id=self.channel_id,
                command=command,
                payload=payload or {},
                sequence_number=sequence_number,
            )
            self._send_queue.put(envelope)
            return sequence_number

    def get_last_sent_sequence_number(self) -> int:
        """Return the most recent sequence number successfully sent on the socket."""
        with self._sequence_cv:
            return self._last_socket_sent_sequence_number

    def get_last_enqueued_sequence_number(self) -> int:
        """Return the most recent sequence number enqueued for the sender thread."""
        with self._sequence_cv:
            return self._last_enqueued_sequence_number

    def get_transport_stats(self) -> dict[str, object]:
        """Return a lightweight snapshot of producer transport state."""
        with self._sequence_cv:
            last_enqueued_sequence_number = self._last_enqueued_sequence_number
            last_socket_sent_sequence_number = self._last_socket_sent_sequence_number

        pending_sequence_count = max(
            0,
            last_enqueued_sequence_number - last_socket_sent_sequence_number,
        )

        sender_thread = self._sender_thread
        heartbeat_thread = self._heartbeat_thread

        return {
            "channel_id": self.channel_id,
            "recording_id": self.recording_id,
            "trace_id": self.trace_id,
            "chunk_size": self.chunk_size,
            "shared_ring_buffer_name": (
                self._shared_ring_buffer.shared_name
                if self._shared_ring_buffer is not None
                else None
            ),
            "send_queue_qsize": self._send_queue.qsize(),
            "send_queue_maxsize": self.send_queue_maxsize,
            "last_enqueued_sequence_number": last_enqueued_sequence_number,
            "last_socket_sent_sequence_number": last_socket_sent_sequence_number,
            "pending_sequence_count": pending_sequence_count,
            "sender_thread_alive": (
                sender_thread.is_alive() if sender_thread is not None else False
            ),
            "heartbeat_thread_alive": (
                heartbeat_thread.is_alive() if heartbeat_thread is not None else False
            ),
        }

    def wait_until_sequence_sent(self, sequence_number: int) -> bool:
        """Block until the sender thread has sent up to `sequence_number`."""
        if sequence_number <= 0:
            return True
        with self._sequence_cv:
            while self._last_socket_sent_sequence_number < sequence_number:
                sender_thread = self._sender_thread
                if sender_thread is None or not sender_thread.is_alive():
                    logger.warning(
                        "ProducerChannel %s sender stopped before sequence %s "
                        "was sent (last_sent=%s last_enqueued=%s)",
                        self.channel_id,
                        sequence_number,
                        self._last_socket_sent_sequence_number,
                        self._last_enqueued_sequence_number,
                    )
                    return False
                self._sequence_cv.wait()
            return True

    def open_ring_buffer(self, size: int = DEFAULT_RING_BUFFER_SIZE) -> None:
        """Open the daemon-side ring buffer transport for this producer."""
        if not self._shared_ring_transport_enabled:
            self._send(
                CommandType.OPEN_RING_BUFFER,
                {"open_ring_buffer": {"size": size}},
            )
            return

        effective_size = max(int(size), int(self.chunk_size) * 256)
        self._close_shared_ring_buffer()
        self._shared_ring_buffer = RingBuffer.create_shared(effective_size)
        self._send(
            CommandType.OPEN_RING_BUFFER,
            {
                "open_ring_buffer": {
                    "size": effective_size,
                    "shared_memory_name": self._shared_ring_buffer.shared_name,
                }
            },
        )

    def send_data(
        self,
        data: bytes,
        data_type: DataType,
        robot_instance: int,
        data_type_name: str,
        robot_id: str | None = None,
        robot_name: str | None = None,
        dataset_id: str | None = None,
        dataset_name: str | None = None,
    ) -> None:
        """Send data to the daemon."""
        if not data:
            return

        self.send_data_parts(
            (memoryview(data),),
            total_bytes=len(data),
            data_type=data_type,
            robot_instance=robot_instance,
            data_type_name=data_type_name,
            robot_id=robot_id,
            robot_name=robot_name,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
        )

    @staticmethod
    def _normalise_parts(parts: Sequence[BytePart]) -> list[memoryview]:
        views: list[memoryview] = []
        for part in parts:
            view = part if isinstance(part, memoryview) else memoryview(part)
            if view.ndim != 1 or view.itemsize != 1 or view.format != "B":
                view = view.cast("B")
            if len(view) > 0:
                views.append(view)
        return views

    def _iter_chunk_views(
        self,
        parts: Sequence[memoryview],
    ) -> Iterator[bytes | memoryview]:
        if not parts:
            return

        chunk_parts: list[memoryview] = []
        remaining = self.chunk_size

        for part in parts:
            start = 0
            part_len = len(part)

            while start < part_len:
                take = min(remaining, part_len - start)
                chunk_parts.append(part[start : start + take])
                start += take
                remaining -= take

                if remaining == 0:
                    yield (
                        chunk_parts[0]
                        if len(chunk_parts) == 1
                        else b"".join(chunk_parts)
                    )
                    chunk_parts = []
                    remaining = self.chunk_size

        if chunk_parts:
            yield chunk_parts[0] if len(chunk_parts) == 1 else b"".join(chunk_parts)

    def send_data_parts(
        self,
        parts: Sequence[BytePart],
        *,
        total_bytes: int | None = None,
        data_type: DataType,
        robot_instance: int,
        data_type_name: str,
        robot_id: str | None = None,
        robot_name: str | None = None,
        dataset_id: str | None = None,
        dataset_name: str | None = None,
    ) -> None:
        """Send a logical payload assembled from multiple byte-like parts."""
        normalised_parts = self._normalise_parts(parts)
        if total_bytes is None:
            total_bytes = sum(len(view) for view in normalised_parts)
        if total_bytes <= 0:
            return

        trace_id = self.trace_id
        recording_id = self.recording_id
        if not trace_id or not recording_id:
            raise ValueError(
                "Trace ID required; call start_new_trace() before send_data()."
            )

        if not robot_id and not robot_name:
            raise ValueError("Robot ID or name required")

        if not dataset_id and not dataset_name:
            raise ValueError("Dataset ID or name required")

        total_chunks = math.ceil(total_bytes / self.chunk_size)
        produced_chunks = 0

        for idx, chunk in enumerate(self._iter_chunk_views(normalised_parts)):
            produced_chunks += 1
            if self._shared_ring_buffer is not None:
                self._send_queue.put(
                    QueuedSharedRingWrite(
                        metadata={
                            "channel_id": self.channel_id,
                            "recording_id": recording_id,
                            "trace_id": trace_id,
                            "chunk_index": idx,
                            "total_chunks": total_chunks,
                            "data_type": data_type.value,
                            "data_type_name": data_type_name,
                            "dataset_name": dataset_name,
                            "dataset_id": dataset_id,
                            "robot_name": robot_name,
                            "robot_id": robot_id,
                            "robot_instance": robot_instance,
                        },
                        chunk=chunk,
                    )
                )
            else:
                payload = DataChunkPayload(
                    channel_id=self.channel_id,
                    recording_id=recording_id,
                    trace_id=trace_id,
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    data_type=data_type,
                    data_type_name=data_type_name,
                    dataset_name=dataset_name,
                    dataset_id=dataset_id,
                    robot_name=robot_name,
                    robot_id=robot_id,
                    robot_instance=robot_instance,
                    data=chunk,
                )
                self._send(CommandType.DATA_CHUNK, {"data_chunk": payload.to_dict()})

        if produced_chunks != total_chunks:
            raise RuntimeError(
                "Chunk count mismatch while serializing payload for transport"
            )

    def initialize_new_producer_channel(
        self,
        ring_buffer_size: int = DEFAULT_RING_BUFFER_SIZE,
    ) -> None:
        """Initialize a new producer channel."""
        if not self.trace_id:
            self.start_new_trace()

        self.start_producer_channel()
        self.open_ring_buffer(size=ring_buffer_size)

    def cleanup_producer_channel(self) -> None:
        """Clean up the producer channel."""
        self.end_trace()
