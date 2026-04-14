"""Main neuracore data daemon."""

from __future__ import annotations

import logging
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from neuracore_types import DataType

from neuracore.data_daemon.communications_management.channel_reader import (
    CHUNK_HEADER_FORMAT,
    ChannelMessageReader,
)
from neuracore.data_daemon.communications_management.bridge_spool import BridgeSpool
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
from neuracore.data_daemon.event_emitter import Emitter
from neuracore.data_daemon.helpers import (
    get_daemon_recordings_root_path,
    utc_now,
)
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
_BRIDGE_DRAIN_IDLE_SLEEP_S = 0.001
_BRIDGE_FORWARD_WAIT_TIMEOUT_S = 0.05
_CLOSING_DEBUG_LOG_INTERVAL_S = 5.0


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
    pending_close: "PendingTraceClose | None" = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_sequence_number: int = 0
    opened_at: datetime | None = None
    heartbeat_expired_at: datetime | None = None
    drain_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def is_opened(self) -> bool:
        """Check if the channel has been opened with a ring buffer."""
        return self.ring_buffer is not None

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
class PendingTraceEnd:
    """Deferred trace-end state until the channel fully drains."""

    trace_id: str
    recording_id: str
    data_type: DataType


PendingTraceClose = PendingTraceEnd


@dataclass
class RecordingClosingState:
    """Recording-level stop/drain state."""

    producer_stop_sequence_numbers: dict[str, int]
    stop_requested_at: datetime
    cutoff_observed_producers: set[str] = field(default_factory=set)
    last_blocked_log_at: datetime | None = None


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
        self.recording_disk_manager = recording_disk_manager
        self.channels: dict[str, ChannelState] = {}
        self._closed_producers: set[str] = set()
        self._recording_traces: dict[str, set[str]] = {}
        self._trace_recordings: dict[str, str] = {}
        self._trace_metadata: dict[str, dict[str, str | int | None]] = {}
        self._closed_recordings: set[str] = set()
        self._closing_recordings: dict[str, RecordingClosingState] = {}
        self._closed_recording_cutoff_states: dict[str, RecordingClosingState] = {}
        self._producer_last_sequence_numbers: dict[str, int] = {}
        self._state_lock = threading.RLock()
        self._worker_stop_event = threading.Event()
        self._drain_thread: threading.Thread | None = None
        self._forward_thread: threading.Thread | None = None
        self._bridge_spool = BridgeSpool(
            get_daemon_recordings_root_path() / ".bridge_spool"
        )
        self._command_handlers: dict[CommandType, CommandHandler] = {
            CommandType.OPEN_RING_BUFFER: self._handle_open_ring_buffer,
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
        self._worker_stop_event.clear()
        self.comm.start_consumer()
        self._start_background_workers()

        logger.info("Daemon started and ready to receive messages...")
        try:
            while self._running:
                self._finalize_closing_recordings()
                self._drain_bridge_cutoff_queries()
                raw = self.comm.receive_raw()
                if raw:
                    self.process_raw_message(raw)
                self._cleanup_expired_channels()
        except KeyboardInterrupt:
            logger.info("Shutting down daemon...")
        finally:
            self._stop_background_workers()
            self._bridge_spool.cleanup()
            self.comm.cleanup_daemon()

    def stop(
        self,
    ) -> None:
        """Stop the daemon main loop.

        Sets the `_running` flag to False, which will cause the daemon main loop
        to exit on the next iteration.
        """
        self._running = False
        self._worker_stop_event.set()

    def _start_background_workers(self) -> None:
        """Start daemon-owned drain and forward worker loops."""
        if self._drain_thread is None or not self._drain_thread.is_alive():
            self._drain_thread = threading.Thread(
                target=self._drain_loop,
                name="daemon-bridge-drain",
                daemon=True,
            )
            self._drain_thread.start()

        if self._forward_thread is None or not self._forward_thread.is_alive():
            self._forward_thread = threading.Thread(
                target=self._forward_loop,
                name="daemon-bridge-forward",
                daemon=True,
            )
            self._forward_thread.start()

    def _stop_background_workers(self) -> None:
        """Stop daemon-owned worker loops and wait for them to exit."""
        self._worker_stop_event.set()
        self._bridge_spool.wait_for_item(timeout_s=0.0)

        if self._drain_thread is not None:
            self._drain_thread.join(timeout=2.0)
            self._drain_thread = None

        if self._forward_thread is not None:
            self._forward_thread.join(timeout=2.0)
            self._forward_thread = None

    def _drain_loop(self) -> None:
        """Drain complete channel messages into the bridge spool."""
        while not self._worker_stop_event.is_set():
            drained_any = self._drain_channel_messages()
            if not drained_any:
                time.sleep(_BRIDGE_DRAIN_IDLE_SLEEP_S)

    def _forward_loop(self) -> None:
        """Forward spooled messages into the recording disk manager."""
        while not self._worker_stop_event.is_set():
            if not self._bridge_spool.wait_for_item(
                timeout_s=_BRIDGE_FORWARD_WAIT_TIMEOUT_S
            ):
                continue
            self._forward_spool_once()

    def _forward_spool_once(self) -> bool:
        """Forward a single spooled message to the recording disk manager."""
        message = self._bridge_spool.peek()
        if message is None:
            return False

        try:
            self.recording_disk_manager.enqueue(message)
        except Exception:
            logger.exception(
                "Failed to forward spooled message for trace_id=%s producer_id=%s",
                message.trace_id,
                message.producer_id,
            )
            time.sleep(_BRIDGE_DRAIN_IDLE_SLEEP_S)
            return False

        self._bridge_spool.ack()
        return True

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

    def _emit_bridge_cutoff_observed(self) -> None:
        """Emit events for producer cutoffs already observed by the bridge."""
        with self._state_lock:
            for recording_id, closing_state in self._closing_recordings.items():
                if not closing_state.producer_stop_sequence_numbers:
                    continue
                for producer_id, cutoff_sequence_number in (
                    closing_state.producer_stop_sequence_numbers.items()
                ):
                    if producer_id in closing_state.cutoff_observed_producers:
                        continue
                    last_sequence_number = self._producer_last_sequence_numbers.get(
                        producer_id
                    )
                    if (
                        last_sequence_number is None
                        or last_sequence_number < cutoff_sequence_number
                    ):
                        continue
                    closing_state.cutoff_observed_producers.add(producer_id)
                    logger.info(
                        "Bridge observed cutoff recording_id=%s producer_id=%s observed_sequence_number=%s cutoff_sequence_number=%s",
                        recording_id,
                        producer_id,
                        last_sequence_number,
                        cutoff_sequence_number,
                    )

    def _drain_bridge_cutoff_queries(self) -> None:
        """Reply to any pending producer bridge cutoff queries."""
        while True:
            message = self.comm.receive_query_message(timeout_ms=0)
            if message is None:
                return
            self.comm.send_query_response(self._handle_bridge_cutoff_query(message))

    def _handle_bridge_cutoff_query(self, message: MessageEnvelope) -> MessageEnvelope:
        """Return whether the bridge has observed the queried cutoff."""
        payload = message.payload.get("bridge_cutoff_query", {})
        recording_id = payload.get("recording_id")
        requested_cutoffs_raw = payload.get("producer_stop_sequence_numbers", {})
        observed_producer_sequence_numbers: dict[str, int] = {}

        if isinstance(requested_cutoffs_raw, dict) and recording_id is not None:
            with self._state_lock:
                closing_state = self._closing_recordings.get(str(recording_id))
                state_source = "closing"
                if closing_state is None:
                    closing_state = self._closed_recording_cutoff_states.get(
                        str(recording_id)
                    )
                    state_source = "closed"
                observed_producers = (
                    closing_state.cutoff_observed_producers
                    if closing_state is not None
                    else set()
                )
                for producer_id, cutoff_sequence_number_raw in (
                    requested_cutoffs_raw.items()
                ):
                    try:
                        cutoff_sequence_number = int(cutoff_sequence_number_raw)
                    except (TypeError, ValueError):
                        continue
                    if cutoff_sequence_number <= 0:
                        continue
                    observed_sequence_number = int(
                        self._producer_last_sequence_numbers.get(str(producer_id), 0)
                    )
                    if (
                        str(producer_id) in observed_producers
                        and observed_sequence_number >= cutoff_sequence_number
                    ):
                        observed_producer_sequence_numbers[str(producer_id)] = (
                            observed_sequence_number
                        )

                if closing_state is not None:
                    missing_cutoffs = {
                        str(producer_id): int(cutoff_sequence_number_raw)
                        for producer_id, cutoff_sequence_number_raw in requested_cutoffs_raw.items()
                        if str(producer_id)
                        not in observed_producer_sequence_numbers
                    }
                    now = utc_now()
                    should_log_blocked = (
                        missing_cutoffs
                        and (
                            closing_state.last_blocked_log_at is None
                            or (
                                now - closing_state.last_blocked_log_at
                            ).total_seconds()
                            >= _CLOSING_DEBUG_LOG_INTERVAL_S
                        )
                    )
                    if should_log_blocked:
                        closing_state.last_blocked_log_at = now
                        current_sequences = {
                            producer_id: int(
                                self._producer_last_sequence_numbers.get(
                                    producer_id, 0
                                )
                            )
                            for producer_id in missing_cutoffs
                        }
                        logger.info(
                            "Bridge cutoff query blocked recording_id=%s state_source=%s missing_cutoffs=%s current_sequences=%s observed_producers=%s",
                            recording_id,
                            state_source,
                            missing_cutoffs,
                            current_sequences,
                            sorted(observed_producers),
                        )

                if (
                    state_source == "closed"
                    and closing_state is not None
                    and requested_cutoffs_raw
                    and len(observed_producer_sequence_numbers)
                    == len(requested_cutoffs_raw)
                ):
                    self._closed_recording_cutoff_states.pop(str(recording_id), None)
                    logger.info(
                        "Bridge cutoff closed-state ack delivered recording_id=%s observed_producer_sequence_numbers=%s",
                        recording_id,
                        observed_producer_sequence_numbers,
                    )

        return MessageEnvelope(
            producer_id=None,
            command=CommandType.BRIDGE_CUTOFF_QUERY_RESPONSE,
            payload={
                "bridge_cutoff_query_response": {
                    "recording_id": recording_id,
                    "observed_producer_sequence_numbers": (
                        observed_producer_sequence_numbers
                    ),
                }
            },
        )

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
            return

        if (
            cmd == CommandType.OPEN_RING_BUFFER
            and producer_id in self._closed_producers
        ):
            self._closed_producers.discard(producer_id)

        with self._state_lock:
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
            with self._state_lock:
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
        self._emit_bridge_cutoff_observed()

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
        with channel.drain_lock:
            channel.set_ring_buffer(RingBuffer(size))

    # Consumer
    def _drain_channel_messages(self) -> bool:
        """Poll all channels for completed messages and handle them."""
        with self._state_lock:
            channels = list(self.channels.values())

        drained_any = False
        for channel in channels:
            if self._drain_single_channel_messages(channel):
                drained_any = True
        return drained_any

    def _drain_single_channel_messages(self, channel: ChannelState) -> bool:
        """Drain all currently-complete messages for a single channel."""
        drained_any = False
        with channel.drain_lock:
            reader = channel.reader
            if reader is None or channel.ring_buffer is None:
                return False
            while True:
                result = reader.poll_one()
                if result is None:
                    break

                trace_id, data_type, payload = result
                with self._state_lock:
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
                drained_any = True

            if self._finalize_pending_close(channel):
                drained_any = True

        return drained_any

    def _channel_stop_cutoff_sequence_number(
        self, producer_id: str, channel: ChannelState
    ) -> int | None:
        """Return stop cutoff sequence for the channel's active trace, if known."""
        with self._state_lock:
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
        complete_message = self._build_complete_message(
            channel=channel,
            trace_id=trace_id,
            recording_id=recording_id,
            data_type=data_type,
            data=data,
            final_chunk=final_chunk,
        )
        try:
            self._bridge_spool.append(complete_message)
        except Exception:
            logger.exception(
                "Failed to append spooled message for trace_id=%s producer_id=%s",
                trace_id,
                channel.producer_id,
            )

    def _build_complete_message(
        self,
        *,
        channel: ChannelState,
        trace_id: str,
        recording_id: str,
        data_type: DataType,
        data: bytes,
        final_chunk: bool,
    ) -> CompleteMessage:
        """Construct a complete message using the latest registered metadata."""
        with self._state_lock:
            metadata = dict(self._trace_metadata.get(trace_id, {}))

        return CompleteMessage.from_bytes(
            producer_id=channel.producer_id,
            trace_id=trace_id,
            recording_id=recording_id,
            final_chunk=final_chunk,
            data_type=data_type,
            data_type_name=str(metadata.get("data_type_name") or ""),
            robot_instance=int(metadata.get("robot_instance") or 0),
            data=data,
            dataset_id=_str_or_none(metadata.get("dataset_id")),
            dataset_name=_str_or_none(metadata.get("dataset_name")),
            robot_name=_str_or_none(metadata.get("robot_name")),
            robot_id=_str_or_none(metadata.get("robot_id")),
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
        with self._state_lock:
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
        with self._state_lock:
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
        with self._state_lock:
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
        with self._state_lock:
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
                    trace_id,
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

        packet = header + data
        channel.ring_buffer.write(packet)

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

        Returns:
            None
        """
        payload = message.payload.get("trace_end", {})
        trace_id = payload.get("trace_id")
        if not trace_id:
            return

        with self._state_lock:
            recording_id = self._trace_recordings.get(str(trace_id))
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
        with self._state_lock:
            metadata = dict(self._trace_metadata.get(str(trace_id), {}))
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

        with channel.drain_lock:
            reader = channel.reader

            # Flush any fully assembled payloads already in the channel ring buffer
            # before sending the trace-end marker into the RDM queue.
            # Otherwise the RDM can close the trace and drop trailing
            # frames that were buffered but not yet drained.
            if reader is not None and channel.ring_buffer is not None:
                while True:
                    result = reader.poll_one()
                    if result is None:
                        break

                    drained_trace_id, drained_data_type, drained_payload = result
                    with self._state_lock:
                        drained_recording_id = self._trace_recordings.get(
                            drained_trace_id
                        )
                    if not drained_recording_id:
                        logger.warning(
                            "No recording_id found for trace_id=%s, dropping message",
                            drained_trace_id,
                        )
                        continue
                    self._on_complete_message(
                        channel,
                        drained_trace_id,
                        drained_data_type,
                        drained_payload,
                        drained_recording_id,
                    )

            channel.pending_close = PendingTraceClose(
                trace_id=str(trace_id),
                recording_id=str(recording_id),
                data_type=data_type,
            )
            self._finalize_pending_close(channel)

    def _trace_has_buffered_data(self, channel: ChannelState, trace_id: str) -> bool:
        """Return True when a trace still has unread or partially assembled data."""
        reader = channel.reader
        if reader is not None and reader.has_pending_trace(trace_id):
            return True
        if channel.ring_buffer is not None and channel.ring_buffer.available() > 0:
            return channel.trace_id == trace_id
        return False

    def _finalize_pending_close(self, channel: ChannelState) -> bool:
        """Finalize a pending channel close once buffered data has drained."""
        pending_close = channel.pending_close
        if pending_close is None:
            return False
        if self._trace_has_buffered_data(channel, pending_close.trace_id):
            return False

        self._on_complete_message(
            channel=channel,
            trace_id=pending_close.trace_id,
            data_type=pending_close.data_type,
            data=b"",
            recording_id=pending_close.recording_id,
            final_chunk=True,
        )
        self._remove_trace(
            pending_close.recording_id,
            pending_close.trace_id,
        )
        if channel.trace_id == pending_close.trace_id:
            channel.trace_id = None
        channel.pending_close = None
        return True

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

        with self._state_lock:
            self._closing_recordings[recording_id] = RecordingClosingState(
                producer_stop_sequence_numbers=producer_stop_sequence_numbers,
                stop_requested_at=utc_now(),
            )
            current_sequences = {
                producer_id: int(self._producer_last_sequence_numbers.get(producer_id, 0))
                for producer_id in producer_stop_sequence_numbers
            }

        logger.info(
            "Recording stopped received recording_id=%s producer_stop_sequence_numbers=%s current_sequences=%s",
            recording_id,
            producer_stop_sequence_numbers,
            current_sequences,
        )

        self._emit_bridge_cutoff_observed()

        self._emitter.emit(Emitter.STOP_RECORDING_REQUESTED, recording_id)

    def _finalize_closing_recordings(self) -> None:
        """Finalize recordings that have reached the stop sequence number.

        This function is called periodically by the daemon to finalize recordings
        that have reached the stop sequence number. It will emit a STOP_RECORDING
        event for each finalized recording.

        :return: None
        """
        to_close: list[str] = []
        with self._state_lock:
            closing_items = list(self._closing_recordings.items())

        for recording_id, closing_state in closing_items:
            with self._state_lock:
                traces = set(self._recording_traces.get(recording_id, set()))
            spool_pending_count = self._bridge_spool.pending_count_for_recording(
                recording_id
            )
            cutoffs_reached = self._has_reached_sequence_cutoffs(
                recording_id, closing_state
            )

            # Not processed all chunks up to stop yet
            if not cutoffs_reached:
                continue

            if not traces and spool_pending_count == 0:
                to_close.append(recording_id)
                continue

        # Only stop recording (signal flush RDM) when all traces have been processed
        for recording_id in to_close:
            with self._state_lock:
                closing_state = self._closing_recordings.pop(recording_id, None)
                self._closed_recordings.add(recording_id)
                if closing_state is not None:
                    self._closed_recording_cutoff_states[recording_id] = closing_state
            logger.info(
                "Closing recording finalized recording_id=%s",
                recording_id,
            )
            self._emitter.emit(Emitter.STOP_RECORDING, recording_id)

    def _has_reached_sequence_cutoffs(
        self, recording_id: str, closing_state: RecordingClosingState
    ) -> bool:
        """Return True when all producer sequence cutoffs have been observed.

        If no cutoff map was provided (legacy producer behavior), this returns True.
        """
        stop_cutoffs = closing_state.producer_stop_sequence_numbers
        if not stop_cutoffs:
            return True

        with self._state_lock:
            for producer_id, cutoff_sequence_number in stop_cutoffs.items():
                last_sequence_number = self._producer_last_sequence_numbers.get(
                    producer_id
                )
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
        with self._state_lock:
            channel = next(
                (ch for ch in self.channels.values() if ch.trace_id == trace_id),
                None,
            )
        if channel is not None:
            channel.trace_id = None

    def _cleanup_expired_channels(self) -> None:
        """Remove channels whose heartbeat has not been seen within the timeout."""
        now = utc_now()
        to_remove: list[str] = []
        with self._state_lock:
            channel_items = list(self.channels.items())

        for producer_id, state in channel_items:
            if state.is_stale_unopened(now):
                to_remove.append(producer_id)
                continue

            if not state.has_missed_heartbeat(now):
                state.heartbeat_expired_at = None
                continue

            if state.pending_close is not None:
                if state.heartbeat_expired_at is None:
                    state.heartbeat_expired_at = now
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
            if state.ring_buffer is not None and state.ring_buffer.available() > 0:
                continue

            to_remove.append(producer_id)

        for producer_id in to_remove:
            with self._state_lock:
                channel = self.channels.get(producer_id)
            if channel is None:
                continue
            if channel is not None and channel.trace_id is not None:
                with self._state_lock:
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
                with self._state_lock:
                    self._producer_last_sequence_numbers[channel.producer_id] = max(
                        self._producer_last_sequence_numbers.get(
                            channel.producer_id, 0
                        ),
                        channel.last_sequence_number,
                    )
            with self._state_lock:
                self.channels.pop(producer_id, None)
                self._closed_producers.add(producer_id)
