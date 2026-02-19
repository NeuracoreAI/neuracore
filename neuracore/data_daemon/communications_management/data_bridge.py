"""Main neuracore data daemon."""

from __future__ import annotations

import logging
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from neuracore_types import DataType

from neuracore.data_daemon.communications_management.channel_reader import (
    CHUNK_HEADER_FORMAT,
    ChannelMessageReader,
)
from neuracore.data_daemon.communications_management.communications_manager import (
    CommunicationsManager,
)
from neuracore.data_daemon.communications_management.ring_buffer import RingBuffer
from neuracore.data_daemon.const import (
    DATA_TYPE_FIELD_SIZE,
    DEFAULT_RING_BUFFER_SIZE,
    HEARTBEAT_TIMEOUT_SECS,
    NEVER_OPENED_TIMEOUT_SECS,
    TRACE_ID_FIELD_SIZE,
)
from neuracore.data_daemon.event_emitter import Emitter, get_emitter
from neuracore.data_daemon.helpers import utc_now
from neuracore.data_daemon.models import (
    CommandType,
    CompleteMessage,
    DataChunkPayload,
    MessageEnvelope,
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

    def is_opened(self) -> bool:
        """Check if the channel has been opened with a ring buffer."""
        return self.ring_buffer is not None

    def touch(self) -> None:
        """Update the last heartbeat time for the channel.

        This is called when a ManagementMessage is received from a producer.
        """
        self.last_heartbeat = datetime.now(timezone.utc)

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
        self.ring_buffer = ring_buffer
        self.reader = ChannelMessageReader(ring_buffer)
        self.opened_at = datetime.now(timezone.utc)

    def is_open(self) -> bool:
        """Check if the channel is open (i.e. has an initialized ring buffer)."""
        return self.ring_buffer is not None

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
        comm_manager: CommunicationsManager | None = None,
    ) -> None:
        """Initializes the daemon.

        Args:
            recording_disk_manager: The recording disk manager for persisting
                trace data to disk.
            comm_manager: The communications manager for ZMQ operations.
                If not provided, a new instance will be created.
        """
        self.comm = comm_manager or CommunicationsManager()
        self.recording_disk_manager = recording_disk_manager
        self.channels: dict[str, ChannelState] = {}
        self._closed_producers: set[str] = set()
        self._recording_traces: dict[str, set[str]] = {}
        self._trace_recordings: dict[str, str] = {}
        self._trace_metadata: dict[str, dict[str, str | int | None]] = {}
        self._closed_recordings: set[str] = set()
        self._pending_close_recordings: dict[str, dict[str, int]] = {}
        self._closing_recordings: dict[str, RecordingClosingState] = {}
        self._producer_last_sequence_numbers: dict[str, int] = {}
        self._command_handlers: dict[CommandType, CommandHandler] = {
            CommandType.OPEN_RING_BUFFER: self._handle_open_ring_buffer,
            CommandType.DATA_CHUNK: self._handle_write_data_chunk,
            CommandType.HEARTBEAT: self._handle_heartbeat,
            CommandType.TRACE_END: self._handle_end_trace,
        }

        self._emitter = get_emitter()
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
        self.comm.start_publisher()
        logger.info("Daemon started and ready to receive messages...")
        try:
            while self._running:
                self._finalize_pending_closes()
                raw = self.comm.receive_raw()
                if raw:
                    self.process_raw_message(raw)
                self._cleanup_expired_channels()
                self._drain_channel_messages()
        except KeyboardInterrupt:
            logger.info("Shutting down daemon...")
        finally:
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
            and cmd != CommandType.OPEN_RING_BUFFER
        ):
            logger.warning(
                "Ignoring command %s from closed producer_id=%s",
                cmd,
                producer_id,
            )
            return

        if (
            cmd == CommandType.OPEN_RING_BUFFER
            and producer_id in self._closed_producers
        ):
            self._closed_producers.discard(producer_id)

        existing = self.channels.get(producer_id)
        if existing is None:
            existing = ChannelState(producer_id=producer_id)
            self.channels[producer_id] = existing
            logger.info("Created new channel for producer_id=%s", producer_id)
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
                logger.debug(
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

        channel.set_ring_buffer(RingBuffer(size))
        logger.info(
            "Opened ring buffer (size=%d) for producer_id=%s",
            size,
            channel.producer_id,
        )

    # Consumer
    def _drain_channel_messages(self) -> None:
        """Poll all channels for completed messages and handle them."""
        for channel in self.channels.values():
            # guard against uninitialised channels
            if channel.reader is None or channel.ring_buffer is None:
                continue
            # Loop to receive full message
            while True:
                result = channel.reader.poll_one()
                if result is None:
                    break

                trace_id, data_type, payload = result
                recording_id = self._trace_recordings.get(trace_id)
                if not recording_id:
                    logger.warning(
                        "No recording_id found for trace_id=%s, dropping message",
                        trace_id,
                    )
                    continue
                self._on_complete_message(
                    channel, trace_id, data_type, payload, recording_id
                )

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
        self._trace_recordings[trace_id] = recording_id
        self._recording_traces.setdefault(recording_id, set()).add(trace_id)

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
        """Remove a trace from the recording-trace mapping.

        This method removes a trace from the recording-trace mapping and
        updates the internal state accordingly. It does not perform any
        checks or validation beyond ensuring the trace is removed from
        the mapping.

        :param recording_id: The recording ID to remove the trace from.
        :param trace_id: The trace ID to remove.
        """
        self._trace_recordings.pop(trace_id, None)
        self._trace_metadata.pop(trace_id, None)
        traces = self._recording_traces.get(recording_id)
        if traces is None:
            return
        traces.discard(trace_id)
        if not traces:
            self._recording_traces.pop(recording_id, None)

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
        if channel.ring_buffer is None:
            logger.warning(
                "DATA_CHUNK received but no ring buffer initialized for producer_id=%s",
                channel.producer_id,
            )
            return
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

        if recording_id in self._closed_recordings:
            logger.warning(
                "Dropping data for closed recording_id=%s trace_id=%s",
                recording_id,
                data_chunk.trace_id,
            )
            return

        closing_state = self._closing_recordings.get(recording_id)
        if closing_state is not None and closing_state.producer_stop_sequence_numbers:
            cutoff_sequence_number = closing_state.producer_stop_sequence_numbers.get(
                channel.producer_id
            )
            if cutoff_sequence_number is None:
                logger.warning(
                    "Dropping data from producer_id=%s while "
                    "recording_id=%s is closing "
                    "(missing stop sequence number)",
                    channel.producer_id,
                    recording_id,
                )
                return
            if message.sequence_number is None:
                logger.warning(
                    "Dropping data for producer_id=%s recording_id=%s "
                    "without sequence_number "
                    "while recording is closing",
                    channel.producer_id,
                    recording_id,
                )
                return
            if message.sequence_number > cutoff_sequence_number:
                logger.warning(
                    "Dropping post-stop data for producer_id=%s recording_id=%s "
                    "(sequence_number=%s, cutoff_sequence_number=%s)",
                    channel.producer_id,
                    recording_id,
                    message.sequence_number,
                    cutoff_sequence_number,
                )
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
        chunk_index = data_chunk.chunk_index
        total_chunks = data_chunk.total_chunks
        data = data_chunk.data
        chunk_len = len(data)

        trace_id_bytes = trace_id.encode("utf-8")
        if len(trace_id_bytes) > TRACE_ID_FIELD_SIZE:
            logger.warning(
                "Trace ID '%s' truncated to %d bytes",
                trace_id,
                TRACE_ID_FIELD_SIZE,
            )
        trace_id_field = trace_id_bytes[:TRACE_ID_FIELD_SIZE].ljust(
            TRACE_ID_FIELD_SIZE, b"\x00"
        )
        data_type_bytes = data_chunk.data_type.value.encode("utf-8")
        if len(data_type_bytes) > DATA_TYPE_FIELD_SIZE:
            logger.warning(
                "Data type '%s' truncated to %d bytes",
                data_chunk.data_type.value,
                DATA_TYPE_FIELD_SIZE,
            )
        data_type_field = data_type_bytes[:DATA_TYPE_FIELD_SIZE].ljust(
            DATA_TYPE_FIELD_SIZE, b"\x00"
        )

        header = struct.pack(
            CHUNK_HEADER_FORMAT,
            trace_id_field,
            data_type_field,
            chunk_index,
            total_chunks,
            chunk_len,
        )

        channel.ring_buffer.write(header + data)

    def _handle_end_trace(
        self, channel: ChannelState, message: MessageEnvelope
    ) -> None:
        """Handle an END_TRACE command from a producer.

        This command is sent by a producer to the daemon when it wants to end
        a trace. The daemon will remove the trace from its internal state
        and notify the RDM to end the trace.

        :param channel (ChannelState): the channel state of the producer
        :param message (MessageEnvelope): the message envelope containing
            the command and payload

        Returns:
            None
        """
        payload = message.payload.get("trace_end", {})
        trace_id = payload.get("trace_id")
        if not trace_id:
            logger.warning(
                "TRACE_END missing trace_id (producer_id=%s)",
                channel.producer_id,
            )
            return

        recording_id = self._trace_recordings.get(str(trace_id))
        if not recording_id:
            logger.warning(
                "TRACE_END: trace_id=%s not in _trace_recordings producer_id=%s",
                trace_id,
                channel.producer_id,
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
            raise ValueError(f"Missing data_type in metadata for trace_id={trace_id}.")

        # Send final_chunk=True message to RDM to signal trace end
        self._on_complete_message(
            channel=channel,
            trace_id=str(trace_id),
            data_type=data_type,
            data=b"",
            recording_id=str(recording_id),
            final_chunk=True,
        )

        self._remove_trace(str(recording_id), str(trace_id))

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

        self._pending_close_recordings[str(recording_id)] = (
            producer_stop_sequence_numbers
        )

        self._emitter.emit(Emitter.STOP_RECORDING_REQUESTED, recording_id)

    def _finalize_pending_closes(self) -> None:
        """Transition pending close recordings into closing state.

        Also finalize recordings that are done draining.
        """
        now = utc_now()
        if self._pending_close_recordings:
            for (
                recording_id,
                producer_stop_sequence_numbers,
            ) in self._pending_close_recordings.items():
                self._closing_recordings[recording_id] = RecordingClosingState(
                    producer_stop_sequence_numbers=producer_stop_sequence_numbers,
                    stop_requested_at=now,
                )
            self._pending_close_recordings.clear()

        to_close: list[str] = []
        for recording_id, closing_state in self._closing_recordings.items():
            if not self._has_reached_sequence_cutoffs(closing_state):
                continue

            traces = self._recording_traces.get(recording_id, set())
            if not traces:
                to_close.append(recording_id)

        for recording_id in to_close:
            self._closing_recordings.pop(recording_id, None)
            self._closed_recordings.add(recording_id)
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

    def _cleanup_expired_channels(self) -> None:
        """Remove channels whose heartbeat has not been seen within the timeout."""
        to_remove = [
            producer_id
            for producer_id, state in self.channels.items()
            if state.should_expire()
        ]

        for producer_id in to_remove:
            logger.info(
                "Channel for producer_id=%s expired due to heartbeat timeout; "
                "cleaning up ring buffer and state",
                producer_id,
            )
            channel = self.channels.get(producer_id)
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
                )
                self._producer_last_sequence_numbers[channel.producer_id] = max(
                    self._producer_last_sequence_numbers.get(channel.producer_id, 0),
                    channel.last_sequence_number,
                )
            del self.channels[producer_id]
            self._closed_producers.add(producer_id)
