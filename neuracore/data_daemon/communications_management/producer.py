"""High-level wrapper exposing NC / NDD contexts."""

from __future__ import annotations

import logging
import math
import threading
import uuid

from neuracore.data_daemon.communications_management.communications_manager import (
    CommunicationsManager,
    MessageEnvelope,
)
from neuracore.data_daemon.const import DEFAULT_CHUNK_SIZE
from neuracore.data_daemon.models import CommandType, DataChunkPayload, DataType

logger = logging.getLogger(__name__)


class RecordingContext:
    """Recording-scoped context for sending recording control messages."""

    def __init__(
        self,
        recording_id: str,
        comm_manager: CommunicationsManager | None = None,
    ) -> None:
        """Initialize the recording context."""
        self.recording_id = recording_id
        self._comm = comm_manager or CommunicationsManager()
        self.socket = self._comm.create_producer_socket()

        if self.socket is None:
            raise RuntimeError(
                "RecordingContext could not connect to daemon. "
                "Start the daemon with `nc-daemon start` before recording. "
                "Data cannot be captured without a running daemon."
            )

    def stop_recording(self) -> None:
        """Send a recording-stopped control message."""
        self._send(
            CommandType.RECORDING_STOPPED,
            {"recording_stopped": {"recording_id": self.recording_id}},
        )

    def _send(self, command: CommandType, payload: dict | None = None) -> None:
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

        if self.socket is None:
            raise RuntimeError(
                "Producer could not connect to daemon. "
                "Start the daemon with `nc-daemon start` before logging data. "
                "Data cannot be captured without a running daemon."
            )

    def start_new_trace(self, recording_id: str | None = None) -> None:
        """Start a new trace for the given recording."""
        if not recording_id:
            raise ValueError("recording_id is required to start a new recording.")
        self.recording_id = recording_id
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
        if self.socket is not None:
            self.socket.close(0)
            self.socket = None
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
        envelope = MessageEnvelope(
            producer_id=self.producer_id,
            command=command,
            payload=payload or {},
        )
        self._comm.send_message(self.socket, envelope)

    def has_consumer(self) -> bool:
        """Check if the producer has a consumer.

        This method checks if the producer has a consumer associated with it.
        If the producer has a consumer, this method returns `True`. Otherwise, it
        returns `False`.

        Returns:
            bool: Whether the producer has a consumer.
        """
        return self.socket is not None

    def open_ring_buffer(self, size: int = 1024) -> None:
        """Open a ring buffer for sending data chunks to the daemon.

        This method sends an OPEN_RING_BUFFER command to the daemon, which
        creates a new RingBuffer instance of the specified size and associates it
        with the producer's channel.

        :param  size (int): The size of the ring buffer in bytes. Defaults to 1024.
        """
        if not self.trace_id:
            self.trace_id = str(uuid.uuid4())
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
        trace_id: str | None = None,
        recording_id: str | None = None,
        robot_id: str | None = None,
        robot_name: str | None = None,
        dataset_id: str | None = None,
        dataset_name: str | None = None,
    ) -> None:
        """Send data to the daemon.

        This method sends the data to the daemon in chunks, using the
        DATA_CHUNK command. If no trace ID is provided, a random one will
        be generated.

        :param data (bytes): The data to send.
        :param robot_instance (int): The robot instance identifier.
        :param trace_id (str | None): The trace ID of the data. If None, a random
            trace ID will be generated.

        Returns:
            None
        """
        if not data:
            return

        if trace_id:
            self.trace_id = trace_id

        elif not self.trace_id:
            self.trace_id = str(uuid.uuid4().hex)

        if recording_id:
            self.recording_id = recording_id

        elif not self.recording_id:
            raise ValueError(
                "Recording ID required; set one "
                "on init or pass recording_id to send_data()."
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
