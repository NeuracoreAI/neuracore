"""Main neuracore data daemon."""

from __future__ import annotations
import time

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

from neuracore_types import DataType

from neuracore.data_daemon.communications_management.channel_reader import (
    ChannelMessageReader,
    CompletedChannelMessage,
    PartialMessage,
)
from neuracore.data_daemon.communications_management.communications_manager import (
    CommunicationsManager,
)
from neuracore.data_daemon.communications_management.ring_buffer import RingBuffer
from neuracore.data_daemon.communications_management.shared_slot_daemon_handler import (
    SharedSlotDaemonHandler,
)
from neuracore.data_daemon.const import (
    DEFAULT_RING_BUFFER_SIZE,
    HEARTBEAT_TIMEOUT_SECS,
    NEVER_OPENED_TIMEOUT_SECS,
)
from neuracore.data_daemon.event_emitter import Emitter
from neuracore.data_daemon.helpers import utc_now
from neuracore.data_daemon.models import (
    CommandType,
    CompleteMessage,
    DataChunkPayload,
    MessageEnvelope,
    TraceTransportMetadata,
)
from neuracore.data_daemon.recording_encoding_disk_manager import (
    recording_disk_manager as rdm_module,
)

RecordingDiskManager = rdm_module.RecordingDiskManager

logger = logging.getLogger(__name__)


def _str_or_none(value: str | int | None) -> str | None:
    """Convert value to string or None.

    Used to safely convert metadata values that may be str, int, or None
    into the str | None type expected by CompleteMessage.from_bytes().
    """
    return None if value is None else str(value)


def _trace_metadata_dict(
    metadata: TraceTransportMetadata | None,
) -> dict[str, str | int | None]:
    """Return trace metadata as a plain dict for downstream registration."""
    return {} if metadata is None else metadata.to_dict()


class TransportMode(str, Enum):
    """Active transport mode for one producer channel."""

    NONE = "none"
    SOCKET = "socket"
    RING_BUFFER = "ring_buffer"
    SHARED_SLOT = "shared_slot"


@dataclass
class SharedSlotTransportState:
    """Daemon-side state for the shared-slot transport."""

    control_endpoint: str | None = None
    shm_name: str | None = None
    descriptors_received: int = 0
    completed_messages: int = 0
    copied_bytes: int = 0

    def reset(self) -> None:
        """Clear transport-specific state."""
        self.control_endpoint = None
        self.shm_name = None
        self.descriptors_received = 0
        self.completed_messages = 0
        self.copied_bytes = 0


@dataclass
class ChannelState:
    """Per-producer channel state owned by the daemon."""

    producer_id: str
    ring_buffer: RingBuffer | None = None
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reader: ChannelMessageReader | None = None
    trace_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_sequence_number: int = 0
    opened_at: datetime | None = None
    heartbeat_expired_at: datetime | None = None
    transport_mode: TransportMode = TransportMode.NONE
    socket_pending_messages: dict[str, PartialMessage] = field(default_factory=dict)
    shared_slot: SharedSlotTransportState = field(
        default_factory=SharedSlotTransportState
    )

    def is_opened(self) -> bool:
        """Check if the channel has been opened with a ring buffer."""
        return self.transport_mode is not TransportMode.NONE

    def touch(self) -> None:
        """Update the last heartbeat time for the channel.

        This is called when a ManagementMessage is received from a producer.
        """
        self.last_heartbeat = datetime.now(timezone.utc)
        self.heartbeat_expired_at = None

    def set_ring_buffer(self, ring_buffer: RingBuffer) -> None:
        """Set the ring buffer for the channel.

        This method is called when a new channel is opened from a producer.
        It takes a RingBuffer instance as an argument and sets the channel's ring buffer
        to it. If the argument is not an instance of RingBuffer, it raises a TypeError.

        The method also creates a ChannelMessageReader
        instance to read from the ring buffer
        and sets the opened_at timestamp to the current time in UTC.
        """
        if not isinstance(ring_buffer, RingBuffer):
            raise TypeError("Invalid ring buffer instance provided for new channel.")
        if self.ring_buffer is not None and self.ring_buffer is not ring_buffer:
            try:
                self.ring_buffer.close()
            finally:
                self.ring_buffer.unlink()
        self.transport_mode = TransportMode.RING_BUFFER
        self.shared_slot.reset()
        self.ring_buffer = ring_buffer
        self.reader = ChannelMessageReader(ring_buffer)
        self.opened_at = datetime.now(timezone.utc)

    def is_open(self) -> bool:
        """Check if the channel is open (i.e. has an initialized ring buffer)."""
        return self.transport_mode is not TransportMode.NONE

    def mark_socket_transport_open(self) -> None:
        """Mark this channel as active on direct socket transport."""
        self.transport_mode = TransportMode.SOCKET
        if self.opened_at is None:
            self.opened_at = datetime.now(timezone.utc)

    def mark_shared_slot_transport_open(
        self,
        *,
        control_endpoint: str,
        shm_name: str,
    ) -> None:
        """Mark this channel as active on the fixed shared-slot transport."""
        self.transport_mode = TransportMode.SHARED_SLOT
        self.shared_slot.control_endpoint = control_endpoint
        self.shared_slot.shm_name = shm_name
        self.opened_at = datetime.now(timezone.utc)

    def mark_shared_slot_descriptor_seen(
        self,
        *,
        shm_name: str,
        copied_bytes: int,
    ) -> None:
        """Record one shared-slot descriptor processed by the daemon."""
        self.transport_mode = TransportMode.SHARED_SLOT
        self.shared_slot.shm_name = shm_name
        self.shared_slot.descriptors_received += 1
        self.shared_slot.copied_bytes += copied_bytes
        if self.opened_at is None:
            self.opened_at = datetime.now(timezone.utc)

    def mark_shared_slot_message_completed(self) -> None:
        """Record one fully assembled shared-slot message."""
        self.shared_slot.completed_messages += 1

    def uses_ring_buffer_transport(self) -> bool:
        """Return True when the channel is active on ring-buffer transport."""
        return self.transport_mode is TransportMode.RING_BUFFER

    def uses_shared_slot_transport(self) -> bool:
        """Return True when the channel is active on shared-slot transport."""
        return self.transport_mode is TransportMode.SHARED_SLOT

    def has_local_buffered_data(self) -> bool:
        """Return True when the daemon still has unread local ring-buffer data."""
        if not self.uses_ring_buffer_transport() or self.ring_buffer is None:
            return False
        available = getattr(self.ring_buffer, "available", None)
        return bool(callable(available) and available() > 0)

    def clear_ring_buffer(self) -> None:
        """Close and forget the current transport state for this channel."""
        ring_buffer = self.ring_buffer
        self.ring_buffer = None
        self.reader = None
        self.opened_at = None
        self.transport_mode = TransportMode.NONE
        self.shared_slot.reset()
        self.socket_pending_messages.clear()
        if ring_buffer is None:
            return
        try:
            ring_buffer.close()
        finally:
            ring_buffer.unlink()

    def add_socket_data_chunk(
        self, data_chunk: DataChunkPayload
    ) -> CompletedChannelMessage | None:
        """Register one socket DATA_CHUNK and return a completed payload when ready."""
        self.mark_socket_transport_open()
        return self.add_transport_chunk(
            trace_id=data_chunk.trace_id,
            chunk_index=data_chunk.chunk_index,
            total_chunks=data_chunk.total_chunks,
            chunk_data=data_chunk.data,
            trace_metadata=data_chunk.trace_metadata,
            fallback_data_type=data_chunk.data_type,
        )

    def add_transport_chunk(
        self,
        *,
        trace_id: str,
        chunk_index: int,
        total_chunks: int,
        chunk_data: bytes,
        trace_metadata: TraceTransportMetadata | None,
        fallback_data_type: DataType | None = None,
    ) -> CompletedChannelMessage | None:
        """Register one transport chunk and return a completed payload when ready."""
        partial_message = self.socket_pending_messages.get(trace_id)
        if partial_message is None:
            partial_message = PartialMessage(total_chunks=total_chunks)
            self.socket_pending_messages[trace_id] = partial_message
        elif partial_message.total_chunks != total_chunks:
            logger.warning(
                "Inconsistent total_chunks for trace_id=%s (existing=%d, new=%d)",
                trace_id,
                partial_message.total_chunks,
                total_chunks,
            )

        partial_message.register_metadata(trace_id, trace_metadata)
        complete = partial_message.add_chunk(chunk_index, chunk_data)
        if not complete:
            return None

        try:
            payload = partial_message.assemble()
        except ValueError as exc:
            logger.error("Failed to assemble trace_id=%s: %s", trace_id, exc)
            self.socket_pending_messages.pop(trace_id, None)
            return None

        metadata = partial_message.metadata
        if metadata is not None:
            data_type = metadata.data_type
        elif fallback_data_type is not None:
            data_type = fallback_data_type
        else:
            raise ValueError(f"Missing data_type in metadata for trace_id={trace_id}.")

        self.socket_pending_messages.pop(trace_id, None)
        return CompletedChannelMessage(
            trace_id=trace_id,
            data_type=data_type,
            payload=payload,
            metadata=metadata,
        )

    def has_missed_heartbeat(
        self,
        now: datetime,
        heartbeat_timeout: timedelta | None = None,
    ) -> bool:
        """Return True when no heartbeat has been seen within timeout."""
        if heartbeat_timeout is None:
            heartbeat_timeout = timedelta(seconds=HEARTBEAT_TIMEOUT_SECS)
        return now - self.last_heartbeat > heartbeat_timeout

    def is_stale_unopened(
        self,
        now: datetime,
        never_opened_timeout: timedelta | None = None,
    ) -> bool:
        """Return True when a channel never opened within timeout."""
        if never_opened_timeout is None:
            never_opened_timeout = timedelta(seconds=NEVER_OPENED_TIMEOUT_SECS)
        return (not self.is_open()) and (now - self.created_at > never_opened_timeout)

    def should_expire(
        self,
    ) -> bool:
        """Return True if channel should be removed from daemon state."""
        now = utc_now()
        return self.has_missed_heartbeat(now) or (self.is_stale_unopened(now))

    def set_trace_id(self, trace_id: str) -> None:
        """Set the trace ID for the current channel.

        Args:
            trace_id: The trace ID to set for the current channel.
        """
        if trace_id != self.trace_id:
            self.trace_id = trace_id


CommandHandler = Callable[[ChannelState, MessageEnvelope], None]

@dataclass
class PendingTraceEnd:
    producer_id: str
    recording_id: str
    trace_id: str
    data_type: DataType
    sequence_number: int | None

@dataclass
class RecordingClosingState:
    """Recording-level stop/drain state."""

    producer_stop_sequence_numbers: dict[str, int]
    stop_requested_at: datetime


class Daemon:
    """Main neuracore data daemon.

    - Owns per-producer channels + ring buffers.
    - Receives ManagementMessages from producers over ZMQ.
    - Handles heartbeats and channel lifetime cleanup.
    """

    def __init__(
        self,
        recording_disk_manager: RecordingDiskManager,
        emitter: Emitter,
        comm_manager: CommunicationsManager | None = None,
    ) -> None:
        """Initializes the daemon.

        Args:
            recording_disk_manager: The recording disk manager for persisting
                trace data to disk.
            emitter: Event emitter for cross-component signaling.
            comm_manager: The communications manager for ZMQ operations.
                If not provided, a new instance will be created.
        """
        self.comm = comm_manager or CommunicationsManager()
        self._pending_trace_ends: dict[str, PendingTraceEnd] = {}
        self._final_chunk_enqueued_traces: set[str] = set()
        self.recording_disk_manager = recording_disk_manager
        self.channels: dict[str, ChannelState] = {}
        self._closed_producers: set[str] = set()
        self._recording_traces: dict[str, set[str]] = {}
        self._recording_unique_traces: dict[str, set[str]] = {}
        self._trace_recordings: dict[str, str] = {}
        self._trace_metadata: dict[str, dict[str, str | int | None]] = {}
        self._closed_recordings: set[str] = set()
        self._closing_recordings: dict[str, RecordingClosingState] = {}
        self._producer_last_sequence_numbers: dict[str, int] = {}
        self._shared_slot_handler = SharedSlotDaemonHandler(self.comm)
        self._command_handlers: dict[CommandType, CommandHandler] = {
            CommandType.OPEN_RING_BUFFER: self._handle_open_ring_buffer,
            CommandType.OPEN_FIXED_SHARED_SLOTS: self._handle_open_fixed_shared_slots,
            CommandType.SHARED_SLOT_DESCRIPTOR: self._handle_shared_slot_descriptor,
            CommandType.DATA_CHUNK: self._handle_write_data_chunk,
            CommandType.HEARTBEAT: self._handle_heartbeat,
            CommandType.TRACE_END: self._handle_end_trace,
        }

        self._emitter = emitter
        self._running = False
        self._emitter.on(Emitter.TRACE_WRITTEN, self.cleanup_channel_on_trace_written)

    def run(self) -> None:
        """Starts the daemon and begins accepting messages from producers.

        This function blocks until the daemon is shutdown via Ctrl-C.

        It is responsible for:

        - Starting the ZMQ consumer and publisher sockets.
        - Receiving and processing management messages from producers.
        - Periodically cleaning up expired channels.
        - Draining full messages from the ring buffer.

        :return: None
        """
        if self._running:
            raise RuntimeError("Daemon is already running")

        self._running = True
        self.comm.start_consumer()

        logger.info("Daemon started and ready to receive messages...")
        try:
            last_receive_log_at = datetime.now(timezone.utc)
            last_raw_at = datetime.now(timezone.utc)
            loop_count = 0

            while self._running:
                loop_count += 1
                self._finalize_closing_recordings()

                raw = self.comm.receive_raw()

                now = datetime.now(timezone.utc)

                if raw:
                    self.process_raw_message(raw)

                if (now - last_receive_log_at).total_seconds() >= 1.0:
                    last_receive_log_at = now

                self._cleanup_expired_channels()
                self._drain_channel_messages()
        except KeyboardInterrupt:
            logger.info("Shutting down daemon...")
        finally:
            self._shared_slot_handler.close()
            self.comm.cleanup_daemon()

    def stop(
        self,
    ) -> None:
        """Stop the daemon main loop.

        Sets the `_running` flag to False, which will cause the daemon main loop
        to exit on the next iteration.
        """
        self._running = False

    def process_raw_message(self, raw: bytes) -> None:
        """Process a raw message from a producer.

        This function will attempt to parse the raw bytes into a ManagementMessage.
        If the parsing fails, it will log an exception and return without handling
        the message.

        :param raw: The raw bytes of a message from a producer.
        :type raw: bytes
        :return: None
        """
        try:
            message = MessageEnvelope.from_bytes(raw)
        except Exception:
            logger.exception("Failed to parse incoming message bytes")
            return
        self.handle_message(message)

    # Producer
    def handle_message(self, message: MessageEnvelope) -> None:
        """Handles a ManagementMessage from a producer.

        This function is called when a ManagementMessage is received from a producer
        over ZMQ. It will handle the message by looking up the command handler
        associated with the message's command type, and then calling the handler
        with the producer's channel state and the message as arguments.

        If the command type is unknown, a warning will be logged.

        :param message: MessageEnvelope containing the ManagementMessage
        :return: None
        """
        producer_id = message.producer_id
        cmd = message.command

        if producer_id is None:
            # Stop recording commands are sent without a producer_id / channel
            if cmd != CommandType.RECORDING_STOPPED:
                logger.warning("Missing producer_id for command %s", cmd)
                return
            self._handle_recording_stopped(message)
            return

        if (
            producer_id in self._closed_producers
            and cmd
            not in (
                CommandType.OPEN_RING_BUFFER,
                CommandType.OPEN_FIXED_SHARED_SLOTS,
            )
        ):
            return

        if (
            cmd in (
                CommandType.OPEN_RING_BUFFER,
                CommandType.OPEN_FIXED_SHARED_SLOTS,
            )
            and producer_id in self._closed_producers
        ):
            self._closed_producers.discard(producer_id)

        existing = self.channels.get(producer_id)
        if existing is None:
            existing = ChannelState(producer_id=producer_id)
            self.channels[producer_id] = existing
        channel = existing
        channel.touch()

        handler = self._command_handlers.get(cmd)
        if handler is None:
            logger.warning("Unknown command %s from producer_id=%s", cmd, producer_id)
            return

        if message.sequence_number is not None:
            if message.sequence_number > channel.last_sequence_number:
                channel.last_sequence_number = message.sequence_number
                self._producer_last_sequence_numbers[producer_id] = (
                    channel.last_sequence_number
                )
            else:
                logger.warning(
                    "Non-monotonic sequence_number=%s for producer_id=%s (last=%s)",
                    message.sequence_number,
                    producer_id,
                    channel.last_sequence_number,
                )
        try:
            handler(channel, message)
        except Exception:
            logger.exception(
                "Failed to handle command %s from producer_id=%s",
                cmd,
                producer_id,
            )

    def _handle_open_ring_buffer(
        self, channel: ChannelState, message: MessageEnvelope
    ) -> None:
        """Handle an OPEN_RING_BUFFER command from a producer.

        This command is sent by a producer to the daemon when it wants to open
        a ring buffer for writing data chunks. The daemon will create a new
        RingBuffer instance of the specified size, and associate it with the
        producer's channel.

        The daemon will also create a ChannelMessageReader instance to read from
        the ring buffer.

        The daemon will log a message at INFO level when a ring buffer is opened,
        containing the size of the ring buffer and the producer_id of the producer
        that opened the ring buffer.

        :param  channel (ChannelState): the channel state of the producer
        :param  message (MessageEnvelope): the message envelope containing
        the command and payload

        Returns:
            None
        """
        payload = message.payload.get(message.command.value, {})
        size = payload.get("size", DEFAULT_RING_BUFFER_SIZE)
        shared_memory_name = payload.get("shared_memory_name")
        if not shared_memory_name:
            raise ValueError(
                "OPEN_RING_BUFFER requires shared_memory_name for shared transport"
            )
        channel.set_ring_buffer(
            RingBuffer.create_shared(
                int(size),
                name=str(shared_memory_name),
            )
        )

    def _handle_open_fixed_shared_slots(
        self, channel: ChannelState, message: MessageEnvelope
    ) -> None:
        """Handle an OPEN_FIXED_SHARED_SLOTS command from a producer."""
        payload = message.payload.get(message.command.value, {})
        self._shared_slot_handler.handle_open(channel, payload)

    def _handle_shared_slot_descriptor(
        self, channel: ChannelState, message: MessageEnvelope
    ) -> None:
        """Copy one producer-owned shared-memory packet and ACK immediately."""
        descriptor_payload = message.payload.get(message.command.value, {})
        transport_result = self._shared_slot_handler.handle_descriptor(
            channel,
            descriptor_payload,
        )
        descriptor = transport_result.descriptor
        chunk_metadata = transport_result.chunk_metadata
        chunk_data = transport_result.chunk_data
        trace_id = transport_result.trace_id
        trace_metadata = transport_result.trace_metadata

        recording_id = self._trace_recordings.get(trace_id)
        if recording_id is None and trace_metadata is not None:
            recording_id = trace_metadata.recording_id

        if recording_id is None:
            logger.warning(
                "Shared-slot packet missing recording metadata "
                "trace_id=%s producer_id=%s sequence_id=%s",
                trace_id,
                channel.producer_id,
                descriptor.sequence_id,
            )
            return

        if self._should_drop_recording_data(
            channel=channel,
            recording_id=recording_id,
            trace_id=trace_id,
            sequence_number=descriptor.sequence_id,
        ):
            return

        channel.set_trace_id(trace_id)

        if trace_metadata is not None:
            self._register_trace(recording_id, trace_id)
            self._register_trace_metadata(
                trace_id,
                {
                    "dataset_id": trace_metadata.dataset_id,
                    "dataset_name": trace_metadata.dataset_name,
                    "robot_name": trace_metadata.robot_name,
                    "robot_id": trace_metadata.robot_id,
                    "robot_instance": trace_metadata.robot_instance,
                    "data_type": trace_metadata.data_type.value,
                    "data_type_name": trace_metadata.data_type_name,
                },
            )

        completed = channel.add_transport_chunk(
            trace_id=trace_id,
            chunk_index=chunk_metadata.chunk_index,
            total_chunks=chunk_metadata.total_chunks,
            chunk_data=chunk_data,
            trace_metadata=trace_metadata,
        )

        if completed is None:
            return

        channel.mark_shared_slot_message_completed()

        if (
            channel.shared_slot.completed_messages == 1
            or channel.shared_slot.completed_messages % 60 == 0
        ):
            logger.info(
                "Shared-slot assembled message "
                "producer_id=%s completed_messages=%d "
                "descriptors_received=%d copied_mib=%.2f trace_id=%s",
                channel.producer_id,
                channel.shared_slot.completed_messages,
                channel.shared_slot.descriptors_received,
                channel.shared_slot.copied_bytes / (1024 * 1024),
                trace_id,
            )

        resolved_recording_id = self._ensure_result_trace_registered(
            channel=channel,
            result=completed,
        )

        if not resolved_recording_id:
            logger.warning(
                "No recording_id found for shared-slot "
                "trace_id=%s producer_id=%s sequence_id=%s",
                trace_id,
                channel.producer_id,
                descriptor.sequence_id,
            )
            return

        on_complete_start = time.monotonic()

        self._on_complete_message(
            channel=channel,
            trace_id=completed.trace_id,
            data_type=completed.data_type,
            data=completed.payload,
            recording_id=resolved_recording_id,
        )

        on_complete_elapsed = time.monotonic() - on_complete_start

        logger.info(
            "Shared-slot complete handled producer_id=%s sequence_id=%s "
            "trace_id=%s bytes=%d elapsed=%.3fs",
            channel.producer_id,
            descriptor.sequence_id,
            completed.trace_id,
            len(completed.payload),
            on_complete_elapsed,
        )

    def _ensure_result_trace_registered(
        self,
        *,
        channel: ChannelState,
        result: CompletedChannelMessage,
    ) -> str | None:
        """Ensure trace/recording metadata is registered for a drained ring result."""
        trace_id = result.trace_id
        recording_id = self._trace_recordings.get(trace_id)
        if recording_id is not None:
            return recording_id

        metadata = _trace_metadata_dict(result.metadata)
        recording_id = _str_or_none(metadata.get("recording_id"))
        if recording_id is None:
            return None

        channel.set_trace_id(trace_id)
        self._register_trace(recording_id, trace_id)
        self._register_trace_metadata(
            trace_id,
            {
                "dataset_id": _str_or_none(metadata.get("dataset_id")),
                "dataset_name": _str_or_none(metadata.get("dataset_name")),
                "robot_name": _str_or_none(metadata.get("robot_name")),
                "robot_id": _str_or_none(metadata.get("robot_id")),
                "robot_instance": metadata.get("robot_instance"),
                "data_type": _str_or_none(metadata.get("data_type")),
                "data_type_name": _str_or_none(metadata.get("data_type_name")),
            },
        )
        return recording_id

    # Consumer
    def _drain_channel_messages(self) -> None:
        """Poll all channels for completed messages and handle them."""
        for channel in self.channels.values():
            self._drain_single_channel_messages(channel)

    def _drain_single_channel_messages(self, channel: ChannelState) -> None:
        """Drain all currently-complete messages for a single channel."""
        if not channel.uses_ring_buffer_transport():
            return

        if channel.reader is None or channel.ring_buffer is None:
            return

        while True:
            result = channel.reader.poll_one()
            if result is None:
                break

            trace_id = result.trace_id
            data_type = result.data_type
            payload = result.payload
            recording_id = self._ensure_result_trace_registered(
                channel=channel,
                result=result,
            )
            if not recording_id:
                logger.warning(
                    "No recording_id found for trace_id=%s, dropping message",
                    trace_id,
                )
                continue
            self._on_complete_message(
                channel, trace_id, data_type, payload, recording_id
            )

    def _channel_stop_cutoff_sequence_number(
        self, producer_id: str, channel: ChannelState
    ) -> int | None:
        """Return stop cutoff sequence for the channel's active trace, if known."""
        trace_id = channel.trace_id
        if trace_id is None:
            return None
        recording_id = self._trace_recordings.get(trace_id)
        if recording_id is None:
            return None
        closing_state = self._closing_recordings.get(recording_id)
        if closing_state is None:
            return None
        cutoffs = closing_state.producer_stop_sequence_numbers
        if not cutoffs:
            return None
        cutoff = cutoffs.get(producer_id)
        if cutoff is None:
            return None
        return int(cutoff)

    def _on_complete_message(
        self,
        channel: ChannelState,
        trace_id: str,
        data_type: DataType,
        data: bytes,
        recording_id: str,
        final_chunk: bool = False,
    ) -> None:
        """Handle a completed message from a channel.

        This function is called when a message is fully assembled from a channel's ring
        buffer. It is responsible for enqueueing the message in the recording
        disk manager.

        :param channel: The channel that the message was received on.
        :param trace_id: The trace ID that the message belongs to.
        :param data_type: The data type of the message payload.
        :param data: The message data.
        :param recording_id: The recording ID (from immutable _trace_recordings).
        :param final_chunk: Whether this is the final chunk for the trace.
        """
        metadata = self._trace_metadata.get(trace_id, {})
        robot_instance = int(metadata.get("robot_instance") or 0)
        try:
            enqueue_start = time.monotonic()
            self.recording_disk_manager.enqueue(
                CompleteMessage.from_bytes(
                    producer_id=channel.producer_id,
                    trace_id=trace_id,
                    recording_id=recording_id,
                    final_chunk=final_chunk,
                    data_type=data_type,
                    data_type_name=str(metadata.get("data_type_name") or ""),
                    robot_instance=robot_instance,
                    data=data,
                    dataset_id=_str_or_none(metadata.get("dataset_id")),
                    dataset_name=_str_or_none(metadata.get("dataset_name")),
                    robot_name=_str_or_none(metadata.get("robot_name")),
                    robot_id=_str_or_none(metadata.get("robot_id")),
                )
            )
            enqueue_elapsed = time.monotonic() - enqueue_start

            if enqueue_elapsed > 0.05:
                logger.warning(
                    "RecordingDiskManager enqueue slow producer_id=%s trace_id=%s "
                    "recording_id=%s bytes=%d elapsed=%.3fs",
                    channel.producer_id,
                    trace_id,
                    recording_id,
                    len(data),
                    enqueue_elapsed,
                )
        except Exception:
            logger.exception(
                "Failed to enqueue message for trace_id=%s producer_id=%s",
                trace_id,
                channel.producer_id,
            )

    def _handle_heartbeat(self, channel: ChannelState, _: MessageEnvelope) -> None:
        """Update the heartbeat timestamp for a producer.

        This does not perform any logic beyond updating the timestamp, so it is
        suitable for use in a high-throughput system.
        """
        channel.touch()
        if channel.transport_mode is TransportMode.NONE:
            channel.mark_socket_transport_open()

    def _register_trace(self, recording_id: str, trace_id: str) -> None:
        """Register a trace to a recording.

        This method registers a trace to a recording. If the trace is already
        registered to a different recording, it will be moved to the new
        recording. If the trace is moved, a log message will be emitted.

        :param recording_id: The recording ID to register the trace to.
        :param trace_id: The trace ID to register.
        """
        existing = self._trace_recordings.get(trace_id)
        if existing and existing != recording_id:
            logger.warning(
                "Trace %s moved from recording %s to %s",
                trace_id,
                existing,
                recording_id,
            )
            self._recording_traces.get(existing, set()).discard(trace_id)
            old_unique_traces = self._recording_unique_traces.get(existing)
            if old_unique_traces is not None:
                old_unique_traces.discard(trace_id)
                if not old_unique_traces:
                    self._recording_unique_traces.pop(existing, None)
        self._trace_recordings[trace_id] = recording_id
        self._recording_traces.setdefault(recording_id, set()).add(trace_id)
        self._recording_unique_traces.setdefault(recording_id, set()).add(trace_id)

    def _register_trace_metadata(
        self, trace_id: str, metadata: dict[str, str | int | None]
    ) -> None:
        """Register metadata for a trace.

        This method registers metadata for a trace. If the trace already has
        metadata registered, it will update the existing metadata with the new
        values. If the new value is different from the existing value, a log
        message will be emitted.

        :param trace_id: The trace ID to register metadata for.
        :param metadata: A dictionary of metadata to register.
        """
        existing = self._trace_metadata.get(trace_id)
        if existing is None:
            self._trace_metadata[trace_id] = dict(metadata)
            return
        for key, value in metadata.items():
            if existing.get(key) is None and value is not None:
                existing[key] = value
            elif value is not None and existing.get(key) not in (None, value):
                logger.warning(
                    "Trace %s metadata mismatch for %s (%s -> %s)",
                    trace_id,
                    key,
                    existing.get(key),
                    value,
                )

    def _remove_trace(self, recording_id: str, trace_id: str) -> None:
        self._trace_recordings.pop(trace_id, None)
        self._trace_metadata.pop(trace_id, None)
        self._pending_trace_ends.pop(trace_id, None)
        self._final_chunk_enqueued_traces.discard(trace_id)

        traces = self._recording_traces.get(recording_id)
        if traces is None:
            return

        traces.discard(trace_id)

        if not traces:
            self._recording_traces.pop(recording_id, None)

    def _should_drop_recording_data(
        self,
        *,
        channel: ChannelState,
        recording_id: str,
        trace_id: str,
        sequence_number: int | None,
    ) -> bool:
        """Return True when recording state says this data should be dropped."""
        if recording_id in self._closed_recordings:
            logger.warning(
                "Dropping data for closed recording_id=%s trace_id=%s",
                recording_id,
                trace_id,
            )
            return True

        closing_state = self._closing_recordings.get(recording_id)
        if closing_state is None or not closing_state.producer_stop_sequence_numbers:
            return False
        cutoff_sequence_number = closing_state.producer_stop_sequence_numbers.get(
            channel.producer_id
        )
        if cutoff_sequence_number is None:
            logger.warning(
                "Dropping data from producer_id=%s while recording_id=%s is closing "
                "(missing stop sequence number)",
                channel.producer_id,
                recording_id,
            )
            return True
        if sequence_number is None:
            logger.warning(
                "Dropping data for producer_id=%s recording_id=%s without "
                "sequence_number while recording is closing",
                channel.producer_id,
                recording_id,
            )
            return True
        if sequence_number > cutoff_sequence_number:
            logger.warning(
                "Dropping post-stop data for producer_id=%s recording_id=%s "
                "(sequence_number=%s, cutoff_sequence_number=%s)",
                channel.producer_id,
                recording_id,
                sequence_number,
                cutoff_sequence_number,
            )
            return True
        return False

    def _handle_write_data_chunk(
        self, channel: ChannelState, message: MessageEnvelope
    ) -> None:
        """Handle a DATA_CHUNK message from a producer.

        This will write the data chunk into the appropriate ring buffer. If the ring
        buffer is not initialized, a warning will be logged and the message
        will be discarded.

        The message payload should contain the following fields:
        - data_chunk: DataChunkPayload

        If the payload is incomplete, a warning will be logged and the message
        will be discarded.

        The DATA_CHUNK message will be logged with the following format:
        DATA_CHUNK: producer_id=<producer_id> channel_id=<channel_id>
        trace_id=<trace_id> chunk_index=<chunk_index+1>/<total_chunks>
        size=<chunk_len>

        :param channel: channel state of the producer
        :param message: message envelope containing the data chunk payload
        """
        data_chunk_payload = message.payload.get("data_chunk")
        if data_chunk_payload is None:
            data_chunk_payload = message.payload

        data_chunk = DataChunkPayload.from_dict(data_chunk_payload)

        if not data_chunk:
            logger.warning("DATA_CHUNK received without payload …")
            return

        recording_id = data_chunk.recording_id
        if not recording_id:
            logger.warning(
                "DATA_CHUNK missing recording_id trace_id=%s producer_id=%s",
                data_chunk.trace_id,
                channel.producer_id,
            )
            return

        trace_id = data_chunk.trace_id
        if channel.trace_id != trace_id and channel.trace_id is not None:
            logger.warning(
                "DATA_CHUNK trace_id=%s does not match channel trace_id=%s",
                data_chunk.trace_id,
                channel.trace_id,
            )
        channel.set_trace_id(trace_id)

        if self._should_drop_recording_data(
            channel=channel,
            recording_id=recording_id,
            trace_id=trace_id,
            sequence_number=message.sequence_number,
        ):
            return

        if recording_id:
            self._register_trace(recording_id, trace_id)
            self._register_trace_metadata(
                trace_id,
                {
                    "dataset_id": data_chunk.dataset_id,
                    "dataset_name": data_chunk.dataset_name,
                    "robot_name": data_chunk.robot_name,
                    "robot_id": data_chunk.robot_id,
                    "robot_instance": data_chunk.robot_instance,
                    "data_type": data_chunk.data_type.value,
                    "data_type_name": data_chunk.data_type_name,
                },
            )
        completed = channel.add_socket_data_chunk(data_chunk)
        if completed is None:
            return
        self._on_complete_message(
            channel=channel,
            trace_id=completed.trace_id,
            data_type=completed.data_type,
            data=completed.payload,
            recording_id=recording_id,
        )

    def _handle_end_trace(
        self,
        channel: ChannelState,
        message: MessageEnvelope,
        *,
        reason: str = "producer_trace_end",
    ) -> None:
        """Handle an END_TRACE command from a producer.

        This command is sent by a producer to the daemon when it wants to end
        a trace. The daemon will remove the trace from its internal state
        and notify the RDM to end the trace.

        :param channel (ChannelState): the channel state of the producer
        :param message (MessageEnvelope): the message envelope containing
            the command and payload
        :param reason: source of the trace finalization request

        Returns:
            None
        """
        payload = message.payload.get("trace_end", {})
        trace_id = payload.get("trace_id")
        if not trace_id:
            return

        self._drain_single_channel_messages(channel)

        registered_recording_id = self._trace_recordings.get(str(trace_id))
        recording_id = registered_recording_id
        if not recording_id:
            recording_id = _str_or_none(payload.get("recording_id"))
        if not recording_id:
            logger.warning(
                "TRACE_END received without recording for producer_id=%s trace_id=%s "
                "sequence_number=%s",
                channel.producer_id,
                trace_id,
                message.sequence_number,
            )
            return

        # Get metadata before removing the trace
        metadata = self._trace_metadata.get(str(trace_id), {})
        data_type_str = metadata.get("data_type")
        if data_type_str:
            try:
                data_type = DataType(data_type_str)
            except ValueError:
                raise ValueError(
                    f"Unknown data_type '{data_type_str}' for trace_id={trace_id}."
                )
        else:
            if registered_recording_id is not None:
                raise ValueError(
                    f"Missing data_type in metadata for trace_id={trace_id}."
                )
            logger.warning(
                "TRACE_END received for trace_id=%s without registered metadata; "
                "ignoring finalization",
                trace_id,
            )
            return

        self._pending_trace_ends[str(trace_id)] = PendingTraceEnd(
        producer_id=channel.producer_id,
        recording_id=str(recording_id),
        trace_id=str(trace_id),
        data_type=data_type,
        sequence_number=message.sequence_number,
        )

        logger.info(
            "Deferred TRACE_END until recording cutoff is reached "
            "producer_id=%s recording_id=%s trace_id=%s sequence_number=%s",
            channel.producer_id,
            recording_id,
            trace_id,
            message.sequence_number,
        )

    def _handle_recording_stopped(self, message: MessageEnvelope) -> None:
        """Handle a RECORDING_STOPPED message from a producer.

        This function is called when a producer sends a RECORDING_STOPPED message to
        the daemon. It will mark the recording as pending-close and emit a
        STOP_RECORDING_REQUESTED event immediately.

        :param message: message envelope containing the recording_id of the stopped
            recording
        :return: None
        """
        payload = message.payload.get("recording_stopped", {})
        recording_id = payload.get("recording_id")
        if not recording_id:
            logger.warning(
                "RECORDING_STOPPED missing recording_id (producer_id=%s)",
                message.producer_id,
            )
            return
        producer_stop_sequence_numbers_raw = payload.get(
            "producer_stop_sequence_numbers", {}
        )
        producer_stop_sequence_numbers: dict[str, int] = {}
        if isinstance(producer_stop_sequence_numbers_raw, dict):
            for (
                producer_id,
                sequence_number,
            ) in producer_stop_sequence_numbers_raw.items():
                try:
                    producer_stop_sequence_numbers[str(producer_id)] = int(
                        sequence_number
                    )
                except (TypeError, ValueError):
                    logger.warning(
                        "Ignoring invalid stop sequence number for producer_id=%s: %r",
                        producer_id,
                        sequence_number,
                    )
        else:
            logger.warning(
                "recording_stopped.producer_stop_sequence_numbers must be a dict"
            )

        self._closing_recordings[recording_id] = RecordingClosingState(
            producer_stop_sequence_numbers=producer_stop_sequence_numbers,
            stop_requested_at=utc_now(),
        )

        self._emitter.emit(Emitter.STOP_RECORDING_REQUESTED, recording_id)

    def _finalize_closing_recordings(self) -> None:
        """Finalize recordings only after stop cutoffs and trace final chunks are written."""
        to_close: list[str] = []

        for recording_id, closing_state in self._closing_recordings.items():
            if not self._has_reached_sequence_cutoffs(closing_state):
                continue

            traces = set(self._recording_traces.get(recording_id, set()))

            for trace_id in traces:
                if trace_id in self._final_chunk_enqueued_traces:
                    continue

                pending_trace_end = self._pending_trace_ends.get(trace_id)
                if pending_trace_end is None:
                    logger.info(
                        "Waiting for TRACE_END before finalizing recording_id=%s trace_id=%s",
                        recording_id,
                        trace_id,
                    )
                    continue

                channel = self.channels.get(pending_trace_end.producer_id)
                if channel is None:
                    logger.warning(
                        "Cannot enqueue final chunk; missing channel producer_id=%s "
                        "recording_id=%s trace_id=%s",
                        pending_trace_end.producer_id,
                        recording_id,
                        trace_id,
                    )
                    continue

                logger.info(
                    "Enqueuing deferred final chunk producer_id=%s recording_id=%s "
                    "trace_id=%s",
                    pending_trace_end.producer_id,
                    recording_id,
                    trace_id,
                )

                self._on_complete_message(
                    channel=channel,
                    trace_id=trace_id,
                    data_type=pending_trace_end.data_type,
                    data=b"",
                    recording_id=recording_id,
                    final_chunk=True,
                )

                self._final_chunk_enqueued_traces.add(trace_id)

            # TRACE_WRITTEN will call cleanup_channel_on_trace_written(),
            # which removes traces from self._recording_traces.
            if not self._recording_traces.get(recording_id, set()):
                to_close.append(recording_id)

        for recording_id in to_close:
            expected_trace_count = len(
                self._recording_unique_traces.get(recording_id, set())
            )

            self._emitter.emit(
                Emitter.SET_EXPECTED_TRACE_COUNT,
                recording_id,
                expected_trace_count,
            )

            self._closing_recordings.pop(recording_id, None)
            self._closed_recordings.add(recording_id)
            self._recording_unique_traces.pop(recording_id, None)

            self._emitter.emit(Emitter.STOP_RECORDING, recording_id)

    def _has_reached_sequence_cutoffs(
        self, closing_state: RecordingClosingState
    ) -> bool:
        """Return True when all producer sequence cutoffs have been observed.

        If no cutoff map was provided (legacy producer behavior), this returns True.
        """
        stop_cutoffs = closing_state.producer_stop_sequence_numbers
        if not stop_cutoffs:
            return True

        for producer_id, cutoff_sequence_number in stop_cutoffs.items():
            last_sequence_number = self._producer_last_sequence_numbers.get(producer_id)
            if (
                last_sequence_number is None
                or last_sequence_number < cutoff_sequence_number
            ):
                return False
        return True

    def cleanup_channel_on_trace_written(
        self,
        trace_id: str,
        _: str | None = None,
        __: int | None = None,
    ) -> None:
        """Clean up a stopped channel.

        This function is called when a trace is marked as written/completed.
        It will remove the trace from the recording's trace list and reset the
        channel state.

        :param trace_id: ID of the trace to clean up
        :param bytes_written: total number of bytes written for the trace (unused)
        """
        recording_id = self._trace_recordings.get(trace_id)
        if recording_id:
            self._remove_trace(str(recording_id), str(trace_id))

        channel = next(
            (ch for ch in self.channels.values() if ch.trace_id == trace_id),
            None,
        )
        if channel is not None:
            channel.trace_id = None
            if channel.uses_shared_slot_transport():
                self._shared_slot_handler.cleanup_channel_resources(channel)
            channel.clear_ring_buffer()

    def _cleanup_expired_channels(self) -> None:
        """Remove channels whose heartbeat has not been seen within the timeout."""
        now = utc_now()
        to_remove: list[str] = []

        for producer_id, state in self.channels.items():
            if state.is_stale_unopened(now):
                to_remove.append(producer_id)
                continue

            if not state.has_missed_heartbeat(now):
                state.heartbeat_expired_at = None
                continue

            if state.trace_id is None:
                to_remove.append(producer_id)
                continue

            cutoff_sequence_number = self._channel_stop_cutoff_sequence_number(
                producer_id, state
            )
            if (
                cutoff_sequence_number is not None
                and state.last_sequence_number < cutoff_sequence_number
            ):
                if state.heartbeat_expired_at is None:
                    state.heartbeat_expired_at = now
                continue

            # If the daemon already has unread bytes queued locally, let the normal
            # drain step run before deciding to synthesize TRACE_END.
            if state.has_local_buffered_data():
                continue

            to_remove.append(producer_id)

        for producer_id in to_remove:
            channel = self.channels.get(producer_id)
            if channel is None:
                continue
            if channel is not None and channel.trace_id is not None:
                recording_id = self._trace_recordings.get(channel.trace_id)
                self._handle_end_trace(
                    channel,
                    MessageEnvelope(
                        producer_id=producer_id,
                        command=CommandType.TRACE_END,
                        payload={
                            "trace_end": {
                                "trace_id": channel.trace_id,
                                "recording_id": recording_id,
                            }
                        },
                    ),
                    reason="heartbeat_expiry",
                )
                self._producer_last_sequence_numbers[channel.producer_id] = max(
                    self._producer_last_sequence_numbers.get(channel.producer_id, 0),
                    channel.last_sequence_number,
                )
            if channel.uses_shared_slot_transport():
                self._shared_slot_handler.cleanup_channel_resources(channel)
            channel.clear_ring_buffer()
            del self.channels[producer_id]
            self._closed_producers.add(producer_id)
