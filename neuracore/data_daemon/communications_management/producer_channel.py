"""High-level wrapper for a producer channel to the data daemon."""

from __future__ import annotations

import logging
import math
import queue
import threading
import uuid
from collections.abc import Sequence

from neuracore.data_daemon.communications_management.communications_manager import (
    CommunicationsManager,
    MessageEnvelope,
)
from neuracore.data_daemon.const import DEFAULT_CHUNK_SIZE, DEFAULT_RING_BUFFER_SIZE
from neuracore.data_daemon.models import CommandType, DataChunkPayload, DataType

logger = logging.getLogger(__name__)

BytePart = bytes | bytearray | memoryview


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
        self._comm = comm_manager or CommunicationsManager()
        self._comm.create_producer_socket()
        self.channel_id = id or str(uuid.uuid4())
        self.chunk_size = chunk_size
        self.send_queue_maxsize = max(0, int(send_queue_maxsize))
        self.trace_id: str | None = None
        self._stop_event = threading.Event()
        self.recording_id: str | None = recording_id
        self._heartbeat_interval = 1.0
        self._heartbeat_thread: threading.Thread | None = None
        self._send_queue: queue.Queue[MessageEnvelope | None] = queue.Queue(
            maxsize=self.send_queue_maxsize
        )
        self._sender_thread: threading.Thread | None = None
        self._next_sequence_number = 1
        self._last_enqueued_sequence_number = 0
        self._last_socket_sent_sequence_number = 0
        self._sequence_cv = threading.Condition()
        self._enqueue_lock = threading.Lock()

        self._sender_thread = threading.Thread(
            target=self._sender_loop, name="producer-channel-sender", daemon=True
        )
        self._sender_thread.start()

    def _sender_loop(self) -> None:
        """Single thread that owns the ZMQ socket; drains queue and sends.

        Only this thread may call send_message/close on the socket (ZMQ is not
        thread-safe). Other threads enqueue MessageEnvelopes.
        """
        while True:
            envelope = self._send_queue.get()
            try:
                if envelope is None:
                    break

                try:
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
        """Starts the producer channel's heartbeat loop.

        This function starts a separate thread which is responsible for sending
        periodic heartbeats to the daemon. If a heartbeat fails, it will log
        a warning message but continue running.

        """
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return

        self._stop_event.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, name="producer-channel-heartbeat", daemon=True
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        """Heartbeat loop for producer.

        This function runs in a separate thread and is responsible for sending
        periodic heartbeats to the daemon. If a heartbeat fails, it will log
        a warning message but continue running.

        """
        self.heartbeat()

        while not self._stop_event.wait(self._heartbeat_interval):
            try:
                self.heartbeat()
            except Exception as exc:
                logger.warning("Heartbeat failed: %s", exc)

    def heartbeat(self) -> None:
        """Send a heartbeat message to the daemon.

        This message is used by the daemon to detect whether
        a producer is still alive.
        If the daemon does not receive a heartbeat message
        from a producer within a certain
        timeout period, it will assume that the producer has stopped and will clean up
        any associated resources (e.g. the ring buffer).

        """
        self._send(CommandType.HEARTBEAT, {})

    def set_recording_id(self, recording_id: str | None) -> None:
        """Set the recording ID for the producer.

        Args:
            recording_id (str): The unique identifier for the recording session.

        Returns:
            None
        """
        self.recording_id = recording_id

    def _stop_recording(self) -> None:
        self.recording_id = None

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
        self.trace_id = None
        self.recording_id = None
        self._send(
            CommandType.TRACE_END,
            {
                "trace_end": {
                    "trace_id": trace_id,
                    "recording_id": recording_id,
                }
            },
        )

    def stop_producer_channel(self) -> None:
        """Stops the producer channel and cleans up any associated resources."""
        self._stop_event.set()
        self._send_queue.put(None)  # poison pill: sender thread closes socket and exits
        if self._sender_thread is not None:
            self._sender_thread.join(timeout=2)
            self._sender_thread = None
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1)
            self._heartbeat_thread = None
        self._comm.cleanup_producer()

    def _send(self, command: CommandType, payload: dict | None = None) -> int:
        """Send a message to the daemon.

        This method serializes the message into a JSON bytes
        object and then sends it over
        the ZeroMQ socket to the daemon.

        Args:
            command: The command to send to the daemon. This is used by the daemon to
                determine how to handle the message.
            payload: A dictionary containing any additional data required by the daemon
                to process the message.

        Returns:
            The sequence number assigned to the enqueued message.
        """
        # Serialize sequence assignment with queue insertion so concurrent
        # heartbeat/data senders cannot enqueue messages out of sequence order.
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

    def wait_until_sequence_sent(self, sequence_number: int) -> bool:
        """Block until the sender thread has sent up to `sequence_number`.

        Returns False if the sender thread exits before reaching the target.
        """
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
        """Open a ring buffer for sending data chunks to the daemon.

        This method sends an OPEN_RING_BUFFER command to the daemon, which
        creates a new RingBuffer instance of the specified size and associates it
        with the producer's channel.

        :param  size (int): The size of the ring buffer in bytes.
        """
        self._send(
            CommandType.OPEN_RING_BUFFER,
            {"open_ring_buffer": {"size": size}},
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
        """Send data to the daemon.

        This method sends the data to the daemon in chunks, using the
        DATA_CHUNK command. Requires start_new_trace() to be called first.

        :param data (bytes): The data to send.
        :param robot_instance (int): The robot instance identifier.

        Returns:
            None
        """
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
    ) -> list[bytes | memoryview]:
        if not parts:
            return []

        chunks: list[bytes | memoryview] = []
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
                    chunks.append(
                        chunk_parts[0]
                        if len(chunk_parts) == 1
                        else b"".join(chunk_parts)
                    )
                    chunk_parts = []
                    remaining = self.chunk_size

        if chunk_parts:
            chunks.append(
                chunk_parts[0] if len(chunk_parts) == 1 else b"".join(chunk_parts)
            )

        return chunks

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
        """Send a logical payload assembled from multiple byte-like parts.

        This avoids materializing a large contiguous payload in memory before
        chunking it for transport.
        """
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
        chunks = self._iter_chunk_views(normalised_parts)
        if len(chunks) != total_chunks:
            raise RuntimeError(
                "Chunk count mismatch while serializing payload for transport"
            )

        for idx, chunk in enumerate(chunks):

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

    def initialize_new_producer_channel(
        self,
        ring_buffer_size: int = DEFAULT_RING_BUFFER_SIZE,
    ) -> None:
        """Initialize a new producer channel.

        This method starts a new trace and opens a ring buffer.
        """
        if not self.trace_id:
            self.start_new_trace()

        self.start_producer_channel()
        self.open_ring_buffer(size=ring_buffer_size)

    def cleanup_producer_channel(self) -> None:
        """Clean up the producer channel.

        This method stops the trace and closes the ring buffer, releasing any
        associated resources.
        """
        self.end_trace()
        self._stop_recording()
