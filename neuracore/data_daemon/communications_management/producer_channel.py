"""High-level wrapper for a producer channel to the data daemon."""

from __future__ import annotations

import logging
import math
import queue
import uuid
from collections.abc import Iterator, Sequence

import zmq

from neuracore.data_daemon.communications_management.communications_manager import (
    CommunicationsManager,
    MessageEnvelope,
)
from neuracore.data_daemon.const import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_RING_BUFFER_SIZE,
    DEFAULT_VIDEO_CHUNK_SIZE,
    DEFAULT_VIDEO_RING_BUFFER_SIZE,
    DEFAULT_VIDEO_SEND_QUEUE_MAXSIZE,
)
from neuracore.data_daemon.models import CommandType, DataType

from .producer_channel_message_sender import (
    ProducerChannelMessageSender,
    QueuedSharedRingWrite,
)
from .producer_heartbeat_service import (
    ProducerHeartbeatService,
    get_producer_heartbeat_service,
)
from .producer_shared_ring_buffer_transport import ProducerSharedRingBufferTransport
from .producer_transport_debug_models import ProducerTransportDebugStats
from .ring_buffer import RingBuffer

logger = logging.getLogger(__name__)

BytePart = bytes | bytearray | memoryview

__all__ = ["ProducerChannel", "RingBuffer", "producer_transport_args_for_data_type"]


def producer_transport_args_for_data_type(
    data_type: DataType,
) -> tuple[int, int, int]:
    """Return producer transport arguments for the given data type."""
    if data_type in (DataType.RGB_IMAGES, DataType.DEPTH_IMAGES):
        return (
            DEFAULT_VIDEO_CHUNK_SIZE,
            DEFAULT_VIDEO_RING_BUFFER_SIZE,
            DEFAULT_VIDEO_SEND_QUEUE_MAXSIZE,
        )

    return (
        DEFAULT_CHUNK_SIZE,
        DEFAULT_RING_BUFFER_SIZE,
        0,
    )


class ProducerChannel:
    """High-level wrapper for a producer channel to the data daemon."""

    def __init__(
        self,
        *,
        data_type: DataType,
        id: str | None = None,
        context: zmq.Context | None = None,
        chunk_size: int | None = None,
        send_queue_maxsize: int | None = None,
        recording_id: str | None = None,
        ring_buffer_size: int | None = None,
        heartbeat_service: ProducerHeartbeatService | None = None,
    ) -> None:
        """Initialize the producer channel."""
        if data_type is None:
            raise ValueError("data_type is required")

        self._heartbeat_service = heartbeat_service or get_producer_heartbeat_service()

        self._heartbeat_service.register_heartbeat_listener(self.heartbeat)

        (
            default_chunk_size,
            default_ring_buffer_size,
            default_send_queue_maxsize,
        ) = producer_transport_args_for_data_type(data_type)

        self.channel_id = id or str(uuid.uuid4())
        self._comm = CommunicationsManager(context=context)
        self._comm.create_producer_socket()
        self.chunk_size = int(default_chunk_size if chunk_size is None else chunk_size)
        self.send_queue_maxsize = max(
            0,
            int(
                default_send_queue_maxsize
                if send_queue_maxsize is None
                else send_queue_maxsize
            ),
        )
        self.trace_id: str | None = None
        self.recording_id: str | None = recording_id

        self._shared_ring_transport = ProducerSharedRingBufferTransport(
            int(
                default_ring_buffer_size
                if ring_buffer_size is None
                else ring_buffer_size
            )
        )
        self._message_sender = ProducerChannelMessageSender(
            producer_id=self.channel_id,
            comm=self._comm,
            send_queue_maxsize=self.send_queue_maxsize,
            write_shared_ring_record=self._shared_ring_transport.write_record,
        )

    @property
    def _send_queue(
        self,
    ) -> queue.Queue[MessageEnvelope | QueuedSharedRingWrite | None]:
        """Expose the sender queue for compatibility with existing tests."""
        return self._message_sender.queue

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
            logger.warning("Cannot end trace without trace_id and recording_id")
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

    # TODO: cleanup producer channels after inactivity
    def stop_producer_channel(self) -> None:
        """Stops the producer channel and cleans up any associated resources."""
        self._heartbeat_service.unregister_heartbeat_listener(self.heartbeat)
        self._stop_message_sender()
        self._close_shared_ring_transport()
        self._comm.cleanup_producer()

    def _send(self, command: CommandType, payload: dict | None = None) -> int:
        """Send a message to the daemon."""
        return self._message_sender.send(command, payload)

    def get_last_sent_sequence_number(self) -> int:
        """Return the most recent sequence number successfully sent on the socket."""
        return self._message_sender.get_last_sent_sequence_number()

    def get_last_enqueued_sequence_number(self) -> int:
        """Return the most recent sequence number enqueued for the sender thread."""
        return self._message_sender.get_last_enqueued_sequence_number()

    def wait_until_sequence_sent(self, sequence_number: int) -> bool:
        """Block until the sender thread has sent up to `sequence_number`."""
        return self._message_sender.wait_until_sequence_sent(sequence_number)

    def get_transport_stats(self) -> ProducerTransportDebugStats:
        """Return a typed snapshot of producer transport debug state."""
        return ProducerTransportDebugStats(
            channel_id=self.channel_id,
            recording_id=self.recording_id,
            trace_id=self.trace_id,
            chunk_size=self.chunk_size,
            heartbeat_thread_alive=self._heartbeat_service.get_stats()[
                "heartbeat_thread_alive"
            ],
            shared_ring=self._shared_ring_transport.get_stats(),
            message_sender=self._message_sender.get_stats(),
        )

    def open_ring_buffer(self, size: int | None = None) -> None:
        """Open the daemon-side ring buffer transport for this producer."""
        payload = self._shared_ring_transport.open(size)
        self._send(
            CommandType.OPEN_RING_BUFFER,
            {"open_ring_buffer": payload},
        )

    def _ensure_shared_ring_buffer(self) -> None:
        """Create and announce the shared ring buffer on first data send."""
        if self._shared_ring_transport.is_open():
            return

        self.open_ring_buffer()
        self._shared_ring_transport.ensure_open()

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

        self._ensure_shared_ring_buffer()

        total_chunks = math.ceil(total_bytes / self.chunk_size)
        produced_chunks = 0

        for idx, chunk in enumerate(self._iter_chunk_views(normalised_parts)):
            produced_chunks += 1
            self._message_sender.enqueue_shared_ring_write(
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

        if produced_chunks != total_chunks:
            raise RuntimeError(
                "Chunk count mismatch while serializing payload for transport"
            )

    def initialize_new_producer_channel(
        self,
        ring_buffer_size: int | None = None,
    ) -> None:
        """Initialize a new producer channel."""
        if not self.trace_id:
            self.start_new_trace()

        self.open_ring_buffer(size=ring_buffer_size)

    def cleanup_producer_channel(self) -> None:
        """Clean up the producer channel."""
        self.end_trace()

    def _stop_message_sender(self) -> None:
        self._message_sender.close(join_timeout_s=2.0)

    def _close_shared_ring_transport(self) -> None:
        self._shared_ring_transport.close()
