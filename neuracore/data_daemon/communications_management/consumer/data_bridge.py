"""Daemon bridge and recording coordination."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
import json

from neuracore_types import DataType

from neuracore.data_daemon.event_emitter import Emitter
from neuracore.data_daemon.helpers import get_daemon_recordings_root_path, utc_now
from neuracore.data_daemon.models import (
    BatchedJointDataPayload,
    CommandType,
    CompleteMessage,
    DataChunkPayload,
    MessageEnvelope,
)
from neuracore.data_daemon.recording_encoding_disk_manager import (
    recording_disk_manager as rdm_module,
)

from ..shared_transport.communications_manager import CommunicationsManager
from ..shared_transport.shared_slot_daemon_handler import SharedSlotDaemonHandler
from .completion_worker import CompletionWorker
from .helpers import str_or_none
from .models import (
    ChannelRegistry,
    ChannelState,
    ClosedProducerRegistry,
    RecordingDataDropRequest,
    SharedSlotSequenceProgressRequest,
    TraceMetadataRegistrationRequest,
    TraceMetadataSnapshot,
    TraceRecordingLookupRequest,
    TransportMode,
)
from .spool_worker import SpoolWorker
from .trace_lifecycle_coordinator import TraceLifecycleCoordinator

RecordingDiskManager = rdm_module.RecordingDiskManager

logger = logging.getLogger(__name__)


DEFAULT_MAX_SPOOLED_CHUNKS = int(os.getenv("NCD_MAX_SPOOLED_CHUNKS", "128"))
CommandHandler = Callable[[ChannelState, MessageEnvelope], None]


class DataBridge:
    """Main neuracore data daemon bridge.

    - Owns per-producer channels + transport state.
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
        self.channels = ChannelRegistry()
        self._closed_producers = ClosedProducerRegistry()
        self._shared_slot_handler = SharedSlotDaemonHandler(self.comm)
        self._spool_admission = threading.BoundedSemaphore(DEFAULT_MAX_SPOOLED_CHUNKS)
        self._completion_worker = CompletionWorker(
            recording_disk_manager=self.recording_disk_manager,
            release_spool_admission=self._spool_admission.release,
        )
        self._trace_lifecycle = TraceLifecycleCoordinator(
            emitter=emitter,
            enqueue_final_trace=self._completion_worker.enqueue_final_trace,
        )
        self._spool_worker = SpoolWorker(
            root=get_daemon_recordings_root_path() / ".bridge_chunk_spool",
            shared_slot_handler=self._shared_slot_handler,
            completion_worker=self._completion_worker,
            acquire_spool_admission=self._spool_admission.acquire,
            release_spool_admission=self._spool_admission.release,
            should_drop_recording_data=self._trace_lifecycle.should_drop_recording_data,
            mark_sequence_completed=(
                self._trace_lifecycle.mark_shared_slot_sequence_completed
            ),
            register_trace=self._trace_lifecycle.register_trace,
            register_trace_metadata=self._trace_lifecycle.register_trace_metadata,
            get_trace_recording=self._trace_lifecycle.get_trace_recording,
            shard_count=4,
        )
        self._command_handlers: dict[CommandType, CommandHandler] = {
            CommandType.OPEN_FIXED_SHARED_SLOTS: self._handle_open_fixed_shared_slots,
            CommandType.SHARED_SLOT_DESCRIPTOR: self._handle_shared_slot_descriptor,
            CommandType.DATA_CHUNK: self._handle_write_data_chunk,
            CommandType.BATCHED_JOINT_DATA: self._handle_batched_joint_data,
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
        - Finalizing fully assembled transport messages.

        :return: None
        """
        if self._running:
            raise RuntimeError("Daemon is already running")

        self._running = True
        self.comm.start_consumer()

        logger.info("Daemon started and ready to receive messages...")
        try:
            last_receive_log_at = datetime.now(timezone.utc)
            datetime.now(timezone.utc)
            loop_count = 0

            while self._running:
                loop_count += 1
                self._trace_lifecycle.finalize_closing_recordings()

                raw = self.comm.receive_raw()

                now = datetime.now(timezone.utc)

                if raw:
                    self.process_raw_message(raw)

                if (now - last_receive_log_at).total_seconds() >= 1.0:
                    last_receive_log_at = now

                self._cleanup_expired_channels()
        except KeyboardInterrupt:
            logger.info("Shutting down daemon...")
        finally:
            self._spool_worker.close()
            self._completion_worker.close()
            self._spool_worker.cleanup()
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
            self._trace_lifecycle.handle_recording_stopped(message)
            return

        if (
            self._closed_producers.contains(producer_id)
            and cmd != CommandType.OPEN_FIXED_SHARED_SLOTS
        ):
            return

        if (
            cmd == CommandType.OPEN_FIXED_SHARED_SLOTS
            and self._closed_producers.contains(producer_id)
        ):
            self._closed_producers.discard(producer_id)

        existing = self.channels.get(producer_id)
        if existing is None:
            existing = ChannelState(producer_id=producer_id)
            self.channels.add(existing)
        channel = existing
        channel.touch()

        handler = self._command_handlers.get(cmd)
        if handler is None:
            logger.warning("Unknown command %s from producer_id=%s", cmd, producer_id)
            return

        if message.sequence_number is not None:
            if message.sequence_number > channel.last_sequence_number:
                channel.last_sequence_number = message.sequence_number
                self._trace_lifecycle.note_producer_sequence(
                    producer_id, channel.last_sequence_number
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

    def _handle_open_fixed_shared_slots(
        self, channel: ChannelState, message: MessageEnvelope
    ) -> None:
        """Handle an OPEN_FIXED_SHARED_SLOTS command from a producer."""
        payload = message.payload.get(message.command.value, {})
        self._shared_slot_handler.handle_open(channel, payload)

    def _handle_shared_slot_descriptor(
        self, channel: ChannelState, message: MessageEnvelope
    ) -> None:
        """Queue one shared-slot descriptor for sharded spool processing."""
        descriptor_payload = message.payload.get(message.command.value, {})
        sequence_number = message.sequence_number
        if sequence_number is None:
            raise ValueError("Shared-slot descriptor missing sequence_number")

        self._mark_shared_slot_sequence_pending(
            SharedSlotSequenceProgressRequest(
                producer_id=channel.producer_id,
                sequence_number=sequence_number,
            )
        )
        try:
            self._spool_worker.enqueue(channel, descriptor_payload)
        except Exception:
            self._mark_shared_slot_sequence_completed(
                SharedSlotSequenceProgressRequest(
                    producer_id=channel.producer_id,
                    sequence_number=sequence_number,
                )
            )
            raise

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

        This function is called when a message is fully assembled from transport
        chunks. It is responsible for enqueueing the message in the recording disk
        manager.

        :param channel: The channel that the message was received on.
        :param trace_id: The trace ID that the message belongs to.
        :param data_type: The data type of the message payload.
        :param data: The message data.
        :param recording_id: The recording ID (from immutable _trace_recordings).
        :param final_chunk: Whether this is the final chunk for the trace.
        """
        metadata = self._trace_lifecycle.get_trace_metadata(trace_id)
        robot_instance = int(metadata.get("robot_instance") or 0)
        try:
            time.monotonic()
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
                    dataset_id=str_or_none(metadata.get("dataset_id")),
                    dataset_name=str_or_none(metadata.get("dataset_name")),
                    robot_name=str_or_none(metadata.get("robot_name")),
                    robot_id=str_or_none(metadata.get("robot_id")),
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
        if channel.transport_mode is TransportMode.NONE:
            channel.mark_socket_transport_open()

    def _mark_shared_slot_sequence_pending(
        self, request: SharedSlotSequenceProgressRequest
    ) -> None:
        """Record that one shared-slot descriptor still needs spool processing."""
        self._trace_lifecycle.mark_shared_slot_sequence_pending(request)

    def _mark_shared_slot_sequence_completed(
        self, request: SharedSlotSequenceProgressRequest
    ) -> None:
        """Record that one shared-slot descriptor reached completion handoff."""
        self._trace_lifecycle.mark_shared_slot_sequence_completed(request)

    def _should_drop_recording_data(self, request: RecordingDataDropRequest) -> bool:
        """Return True when recording state says this data should be dropped."""
        return self._trace_lifecycle.should_drop_recording_data(request)

    def _handle_write_data_chunk(
        self, channel: ChannelState, message: MessageEnvelope
    ) -> None:
        """Handle a DATA_CHUNK message from a producer.

        This will assemble the data chunk into the channel's active transport
        message state. If the payload is incomplete, a warning will be logged
        and the message will be discarded.

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
            RecordingDataDropRequest(
                channel=channel,
                recording_id=recording_id,
                trace_id=trace_id,
                sequence_number=message.sequence_number,
            )
        ):
            return

        if recording_id:
            self._trace_lifecycle.register_trace(recording_id, trace_id)
            self._trace_lifecycle.register_trace_metadata(
                TraceMetadataRegistrationRequest(
                    trace_id=trace_id,
                    metadata=TraceMetadataSnapshot(
                        dataset_id=data_chunk.dataset_id,
                        dataset_name=data_chunk.dataset_name,
                        robot_name=data_chunk.robot_name,
                        robot_id=data_chunk.robot_id,
                        robot_instance=data_chunk.robot_instance,
                        data_type=data_chunk.data_type.value,
                        data_type_name=data_chunk.data_type_name,
                    ),
                )
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

    def _handle_batched_joint_data(
        self, channel: ChannelState, message: MessageEnvelope
    ) -> None:
        """Handle one batched joint transport message from a producer."""
        batch_payload_dict = message.payload.get(CommandType.BATCHED_JOINT_DATA.value)
        if batch_payload_dict is None:
            batch_payload_dict = message.payload

        batch_payload = BatchedJointDataPayload.from_dict(batch_payload_dict)
        if not batch_payload.items:
            logger.warning("BATCHED_JOINT_DATA received without items")
            return

        recording_id = batch_payload.recording_id
        if not recording_id:
            logger.warning(
                "BATCHED_JOINT_DATA missing recording_id producer_id=%s",
                channel.producer_id,
            )
            return

        first_trace_id = batch_payload.items[0].trace_id
        if self._should_drop_recording_data(
            RecordingDataDropRequest(
                channel=channel,
                recording_id=recording_id,
                trace_id=first_trace_id,
                sequence_number=message.sequence_number,
            )
        ):
            return

        for item in batch_payload.items:
            self._trace_lifecycle.register_trace(recording_id, item.trace_id)
            self._trace_lifecycle.register_trace_metadata(
                TraceMetadataRegistrationRequest(
                    trace_id=item.trace_id,
                    metadata=TraceMetadataSnapshot(
                        dataset_id=batch_payload.dataset_id,
                        dataset_name=batch_payload.dataset_name,
                        robot_name=batch_payload.robot_name,
                        robot_id=batch_payload.robot_id,
                        robot_instance=batch_payload.robot_instance,
                        data_type=batch_payload.data_type.value,
                        data_type_name=item.data_type_name,
                    ),
                )
            )
            joint_bytes = json.dumps({
                "timestamp": batch_payload.timestamp,
                "value": item.value,
            }).encode("utf-8")
            self._on_complete_message(
                channel=channel,
                trace_id=item.trace_id,
                data_type=batch_payload.data_type,
                data=joint_bytes,
                recording_id=recording_id,
            )

    def _handle_end_trace(
        self,
        channel: ChannelState,
        message: MessageEnvelope,
        *,
        reason: str = "producer_trace_end",
    ) -> None:
        """Handle an END_TRACE command from a producer."""
        self._trace_lifecycle.handle_trace_end(channel, message)

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
        self._trace_lifecycle.cleanup_trace_written(trace_id)

        channel = next(
            (ch for ch in self.channels.values() if ch.trace_id == trace_id),
            None,
        )
        if channel is not None:
            if channel.uses_shared_memory_transport() and (
                channel.shared_slot.shm_name is not None
                or channel.socket_pending_messages
            ):
                logger.debug(
                    "Cleaning up channel after TRACE_WRITTEN producer_id=%s "
                    "trace_id=%s shm_name=%s pending_partial_traces=%d",
                    channel.producer_id,
                    trace_id,
                    channel.shared_slot.shm_name,
                    len(channel.socket_pending_messages),
                )
            channel.trace_id = None
            if channel.uses_shared_memory_transport():
                self._shared_slot_handler.cleanup_channel_resources(channel)
            channel.clear_transport_state()

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

            cutoff_sequence_number = (
                self._trace_lifecycle.channel_stop_cutoff_sequence_number(
                    producer_id, state
                )
            )
            if (
                cutoff_sequence_number is not None
                and state.last_sequence_number < cutoff_sequence_number
            ):
                if state.heartbeat_expired_at is None:
                    state.heartbeat_expired_at = now
                continue

            if cutoff_sequence_number is not None and (
                self._trace_lifecycle.has_pending_shared_slot_sequences_at_or_before(
                    producer_id,
                    cutoff_sequence_number,
                )
            ):
                if state.heartbeat_expired_at is None:
                    state.heartbeat_expired_at = now
                continue

            to_remove.append(producer_id)

        for producer_id in to_remove:
            channel = self.channels.get(producer_id)
            if channel is None:
                continue
            if channel is not None and channel.trace_id is not None:
                recording_id = self._trace_lifecycle.get_trace_recording(
                    TraceRecordingLookupRequest(trace_id=channel.trace_id)
                )
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
                self._trace_lifecycle.set_max_producer_sequence(
                    channel.producer_id, channel.last_sequence_number
                )
            if channel.uses_shared_memory_transport():
                self._shared_slot_handler.cleanup_channel_resources(channel)
            channel.clear_transport_state()
            self.channels.remove(producer_id)
            self._closed_producers.add(producer_id)


Daemon = DataBridge
