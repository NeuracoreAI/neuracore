"""High-level wrapper exposing NC / NDD contexts."""

from __future__ import annotations

import logging
import math
import queue
import threading
import uuid

from neuracore.data_daemon.communications_management.communications_manager import (
    CommunicationsManager,
    MessageEnvelope,
)
from neuracore.data_daemon.const import DEFAULT_CHUNK_SIZE, DEFAULT_RING_BUFFER_SIZE
from neuracore.data_daemon.models import CommandType, DataChunkPayload, DataType

logger = logging.getLogger(__name__)


class RecordingContext:
    """Recording-scoped context for sending recording control messages."""

    def __init__(
        self,
        recording_id: str | None = None,
        comm_manager: CommunicationsManager | None = None,
    ) -> None:
        """Initialize the recording context."""
        self.recording_id = recording_id
        self._comm = comm_manager or CommunicationsManager()
        self.socket = self._comm.create_producer_socket()

        if self.socket is None:
            raise RuntimeError(
                "RecordingContext could not connect to daemon. "
                "Start the daemon with `nc-data-daemon launch` before recording. "
                "Data cannot be captured without a running daemon."
            )

    def set_recording_id(self, recording_id: str | None) -> None:
        """Set or clear the recording identifier for this context."""
        self.recording_id = recording_id

    def stop_recording(
        self,
        recording_id: str | None = None,
        producer_stop_sequence_numbers: dict[str, int] | None = None,
    ) -> None:
        """Send a recording-stopped control message."""
        effective_recording_id = recording_id or self.recording_id
        if not effective_recording_id:
            raise ValueError("recording_id is required to stop a recording.")

        recording_stopped_payload: dict[str, object] = {
            "recording_id": effective_recording_id
        }
        if producer_stop_sequence_numbers:
            recording_stopped_payload["producer_stop_sequence_numbers"] = (
                producer_stop_sequence_numbers
            )
        self._send(
            CommandType.RECORDING_STOPPED,
            {"recording_stopped": recording_stopped_payload},
        )
        self.recording_id = effective_recording_id

    def close(self) -> None:
        """Close sockets and cleanup context resources owned by this instance."""
        if self.socket is not None:
            self.socket.close(0)
            self.socket = None
        self._comm.cleanup_producer()

    def _send(self, command: CommandType, payload: dict | None = None) -> None:
        """Send a management message to the daemon.

        Args:
            command: The CommandType to send to the daemon.
            payload: A dictionary containing any additional data required by the daemon
                to process the message.

        Returns:
            None
        """
        envelope = MessageEnvelope(
            producer_id=None,
            command=command,
            payload=payload or {},
        )
        self._comm.send_message(self.socket, envelope)


class Producer:
    """High-level wrapper exposing NC / NDD contexts."""

    def __init__(
        self,
        id: str | None = None,
        comm_manager: CommunicationsManager | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        recording_id: str | None = None,
    ) -> None:
        """Initialize the producer."""
        self._comm = comm_manager or CommunicationsManager()
        self.socket = self._comm.create_producer_socket()
        self.producer_id = id or str(uuid.uuid4())
        self.chunk_size = chunk_size
        self.trace_id: str | None = None
        self._stop_event = threading.Event()
        self.recording_id: str | None = recording_id
        self.id = id or str(uuid.uuid4())
        self._heartbeat_interval = 1.0
        self._heartbeat_thread: threading.Thread | None = None
        self._send_queue: queue.Queue[MessageEnvelope | None] = queue.Queue()
        self._sender_thread: threading.Thread | None = None
        self._next_sequence_number = 1
        self._last_sent_sequence_number = 0

        if self.socket is None:
            raise RuntimeError(
                "Producer could not connect to daemon. "
                "Start the daemon with `nc-data-daemon launch` before logging data. "
                "Data cannot be captured without a running daemon."
            )

        self._sender_thread = threading.Thread(
            target=self._sender_loop, name="producer-sender", daemon=True
        )
        self._sender_thread.start()

    def _sender_loop(self) -> None:
        """Single thread that owns the ZMQ socket; drains queue and sends.

        Only this thread may call send_message/close on the socket (ZMQ is not
        thread-safe). Other threads enqueue MessageEnvelopes.
        """
        while True:
            envelope = self._send_queue.get()
            if envelope is None:
                break
            if self.socket is not None:
                try:
                    self._comm.send_message(self.socket, envelope)
                except Exception as exc:
                    logger.warning("Send failed: %s", exc)
        if self.socket is not None:
            self.socket.close(0)
            self.socket = None

    def start_producer(self) -> None:
        """Starts the producer's heartbeat loop.

        This function starts a separate thread which is responsible for sending
        periodic heartbeats to the daemon. If a heartbeat fails, it will log
        a warning message but continue running.

        """
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return

        self._stop_event.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, name="producer-heartbeat", daemon=True
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
            raise ValueError("recording_id is required; set on Producer init.")
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

    def stop_producer(self) -> None:
        """Stops the producer and cleans up any associated resources."""
        self._stop_event.set()
        self._send_queue.put(None)  # poison pill: sender thread closes socket and exits
        if self._sender_thread is not None:
            self._sender_thread.join(timeout=2)
            self._sender_thread = None
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1)
            self._heartbeat_thread = None
        self._comm.cleanup_producer()

    def _send(self, command: CommandType, payload: dict | None = None) -> None:
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
            None
        """
        sequence_number = self._next_sequence_number
        self._next_sequence_number += 1
        self._last_sent_sequence_number = sequence_number
        envelope = MessageEnvelope(
            producer_id=self.producer_id,
            command=command,
            payload=payload or {},
            sequence_number=sequence_number,
        )
        self._send_queue.put(envelope)

    def get_last_sent_sequence_number(self) -> int:
        """Return the most recent message sequence number sent by this producer."""
        return self._last_sent_sequence_number

    def has_consumer(self) -> bool:
        """Check if the producer has a consumer.

        This method checks if the producer has a consumer associated with it.
        If the producer has a consumer, this method returns `True`. Otherwise, it
        returns `False`.

        Returns:
            bool: Whether the producer has a consumer.
        """
        return self.socket is not None

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

        if not self.trace_id or not self.recording_id:
            raise ValueError(
                "Trace ID required; call start_new_trace() before send_data()."
            )

        if not robot_id and not robot_name:
            raise ValueError("Robot ID or name required")

        if not dataset_id and not dataset_name:
            raise ValueError("Dataset ID or name required")

        total_chunks = math.ceil(len(data) / self.chunk_size)
        for idx in range(total_chunks):
            start = idx * self.chunk_size
            end = min(start + self.chunk_size, len(data))
            chunk = data[start:end]

            payload = DataChunkPayload(
                channel_id=self.producer_id,
                recording_id=self.recording_id,
                trace_id=self.trace_id,
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

    def initialize_new_producer(self) -> None:
        """Initialize a new producer.

        This method starts a new trace and opens a ring buffer.
        """
        if not self.trace_id:
            self.start_new_trace()

        self.start_producer()
        self.open_ring_buffer()

    def cleanup_producer(self) -> None:
        """Clean up the producer.

        This method stops the trace and closes the ring buffer, releasing any
        associated resources.
        """
        self.end_trace()
        self._stop_recording()
