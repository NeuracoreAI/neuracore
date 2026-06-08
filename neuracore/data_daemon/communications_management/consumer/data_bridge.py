"""Daemon bridge and recording coordination."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable

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
from ..shared_transport.iox2_daemon_drain import Iox2DaemonDrain
from .completion_worker import CompletionWorker
from .helpers import str_or_none
from .models import (
    ChannelRegistry,
    ChannelState,
    ClosedProducerRegistry,
    RecordingDataDropRequest,
    TraceMetadataRegistrationRequest,
    TraceMetadataSnapshot,
    TraceRecordingLookupRequest,
    VideoFrameSequenceProgressRequest,
)
from .spool_worker import SpoolWorker
from .trace_lifecycle_coordinator import TraceLifecycleCoordinator

RecordingDiskManager = rdm_module.RecordingDiskManager

logger = logging.getLogger(__name__)


DEFAULT_MAX_SPOOLED_CHUNKS = int(os.getenv("NCD_MAX_SPOOLED_CHUNKS", "128"))
IOX2_DRAIN_POLL_INTERVAL_S = float(os.getenv("NCD_IOX2_DRAIN_POLL_INTERVAL_S", "0.001"))
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
        self._iox2_drain = Iox2DaemonDrain()
        self._spool_admission = threading.BoundedSemaphore(DEFAULT_MAX_SPOOLED_CHUNKS)
        self._completion_worker = CompletionWorker(
            recording_disk_manager=self.recording_disk_manager,
            release_spool_admission=self._spool_admission.release,
        )
        self._trace_lifecycle = TraceLifecycleCoordinator(
            emitter=emitter,
            enqueue_final_trace=self._completion_worker.enqueue_final_trace,
            set_channel_trace_id=self.channels.set_trace_id,
        )
        self._spool_worker = SpoolWorker(
            root=get_daemon_recordings_root_path() / ".bridge_chunk_spool",
            completion_worker=self._completion_worker,
            acquire_spool_admission=self._spool_admission.acquire,
            release_spool_admission=self._spool_admission.release,
            should_drop_recording_data=self._trace_lifecycle.should_drop_recording_data,
            mark_sequence_completed=(
                self._trace_lifecycle.mark_video_frame_sequence_completed
            ),
            register_trace=self._trace_lifecycle.register_trace,
            register_trace_metadata=self._trace_lifecycle.register_trace_metadata,
            get_trace_recording=self._trace_lifecycle.get_trace_recording,
            set_channel_trace_id=self.channels.set_trace_id,
            shard_count=4,
        )
        self._command_handlers: dict[CommandType, CommandHandler] = {
            CommandType.DATA_CHUNK: self._handle_write_data_chunk,
            CommandType.BATCHED_JOINT_DATA: self._handle_batched_joint_data,
            CommandType.HEARTBEAT: self._handle_heartbeat,
            CommandType.TRACE_END: self._handle_end_trace,
        }

        self._emitter = emitter
        self._running = False
        self._iox2_drain_stop = threading.Event()
        self._iox2_drain_thread: threading.Thread | None = None
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
        self._start_iox2_drain_thread()
        try:
            while self._running:
                raw = self.comm.receive_raw()

                if raw:
                    self.process_raw_message(raw)

                self._cleanup_expired_channels()
        except KeyboardInterrupt:
            logger.info("Shutting down daemon...")
        finally:
            self._stop_iox2_drain_thread()
            self._iox2_drain.drain_all(self._on_iox2_frame)
            self._iox2_drain.close()
            self._spool_worker.close()
            self._completion_worker.close()
            self._spool_worker.cleanup()
            self.comm.cleanup_daemon()

    def stop(
        self,
    ) -> None:
        """Stop the daemon main loop.

        Sets the `_running` flag to False, which will cause the daemon main loop
        to exit on the next iteration.
        """
        self._running = False

    def _start_iox2_drain_thread(self) -> None:
        """Continuously drain iceoryx video frames off the main ZMQ loop."""
        if self._iox2_drain_thread is not None:
            return
        self._iox2_drain_stop.clear()
        self._iox2_drain_thread = threading.Thread(
            target=self._iox2_drain_loop,
            name="daemon-iox2-drain",
            daemon=True,
        )
        self._iox2_drain_thread.start()

    def _stop_iox2_drain_thread(self) -> None:
        """Stop the background iceoryx drain thread."""
        self._iox2_drain_stop.set()
        thread = self._iox2_drain_thread
        if thread is not None:
            thread.join(timeout=2.0)
            self._iox2_drain_thread = None

    def _iox2_drain_loop(self) -> None:
        """Drain video subscribers frequently enough to avoid ring overflow."""
        while not self._iox2_drain_stop.is_set():
            try:
                drained = self._iox2_drain.drain_all(self._on_iox2_frame)
            except Exception:
                logger.exception("Iox2 drain loop failed")
                time.sleep(IOX2_DRAIN_POLL_INTERVAL_S)
                continue
            if drained == 0:
                self._iox2_drain_stop.wait(IOX2_DRAIN_POLL_INTERVAL_S)

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

        if self._closed_producers.contains(producer_id):
            # A heartbeat from a previously closed producer means it has come
            # back for a new recording session; revive the channel. Any other
            # late command from a closed producer is ignored.
            if cmd != CommandType.HEARTBEAT:
                return
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

        if message.sequence_number is not None and cmd != CommandType.HEARTBEAT:
            sequence_number = int(message.sequence_number)
            if sequence_number > channel.last_sequence_number:
                channel.last_sequence_number = sequence_number
                self._trace_lifecycle.note_producer_sequence(
                    producer_id, channel.last_sequence_number
                )
            if sequence_number > channel.last_socket_sequence_number:
                channel.last_socket_sequence_number = sequence_number
            else:
                logger.warning(
                    "Non-monotonic socket sequence_number=%s command=%s "
                    "for producer_id=%s (last_socket=%s, last_any=%s)",
                    sequence_number,
                    cmd,
                    producer_id,
                    channel.last_socket_sequence_number,
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

    def _on_iox2_frame(
        self,
        producer_id: str,
        sequence_id: int,
        metadata: dict,
        chunk: bytes,
    ) -> None:
        """Handle one video frame drained from an iceoryx2 subscriber.

        The frame carries the owning robot coordinator ``producer_id``. This
        advances the coordinator sequence (so end-of-recording cutoffs account for
        video frames), marks the sequence pending, and hands the decoded chunk to
        the spool worker. The spool worker marks the sequence completed once the
        chunk has been enqueued to the completion worker, which preserves the
        ordering guarantee that a trace is never finalized before its frames are
        spooled.
        """
        channel = self.channels.get(producer_id)
        if channel is None:
            logger.debug(
                "Iox2 frame for unknown coordinator producer_id=%s sequence_id=%s",
                producer_id,
                sequence_id,
            )
            return

        if sequence_id > channel.last_sequence_number:
            channel.last_sequence_number = sequence_id
        self._trace_lifecycle.set_max_producer_sequence(producer_id, sequence_id)

        request = VideoFrameSequenceProgressRequest(
            producer_id=producer_id,
            sequence_number=sequence_id,
        )
        self._mark_video_frame_sequence_pending(request)
        try:
            self._spool_worker.enqueue_frame(channel, sequence_id, metadata, chunk)
        except Exception:
            self._mark_video_frame_sequence_completed(request)
            raise

    def _on_complete_message(
        self,
        channel: ChannelState,
        trace_id: str,
        data_type: DataType,
        data: bytes,
        recording_id: str,
        sequence_number: int | None = None,
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
        :param sequence_number: The producer sequence number for this message.
        :param final_chunk: Whether this is the final chunk for the trace.
        """
        metadata = self._trace_lifecycle.get_trace_metadata(trace_id)
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
                    sequence_number=sequence_number,
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

    def _handle_heartbeat(
        self, channel: ChannelState, message: MessageEnvelope
    ) -> None:
        """Update liveness and register video subscribers advertised in heartbeats."""
        channel.touch()

        video_service_id = self._heartbeat_video_service_id(message)
        if video_service_id is None:
            return

        raw = message.payload.get("data_type") if message.payload else None
        try:
            data_type = DataType(raw)
        except ValueError:
            logger.warning(
                "HEARTBEAT from producer_id=%s carried invalid data_type=%r "
                "for video_service_id=%s",
                channel.producer_id,
                raw,
                video_service_id,
            )
            return

        channel.data_type = data_type
        channel.mark_video_transport_open()
        channel.video_service_ids.add(video_service_id)
        self._iox2_drain.register_channel(video_service_id)

    @staticmethod
    def _heartbeat_video_service_id(message: MessageEnvelope) -> str | None:
        """Parse the optional iceoryx2 video service id from a heartbeat payload."""
        raw = message.payload.get("video_service_id") if message.payload else None
        return str(raw) if raw else None

    def _mark_video_frame_sequence_pending(
        self, request: VideoFrameSequenceProgressRequest
    ) -> None:
        """Record that one video frame still needs spool processing."""
        self._trace_lifecycle.mark_video_frame_sequence_pending(request)

    def _mark_video_frame_sequence_completed(
        self, request: VideoFrameSequenceProgressRequest
    ) -> None:
        """Record that one video frame reached completion handoff."""
        self._trace_lifecycle.mark_video_frame_sequence_completed(request)

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
        self.channels.set_trace_id(channel, trace_id)

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
        completed = channel.add_socket_data_chunk(
            data_chunk,
            sequence_number=message.sequence_number,
        )
        if completed is None:
            return
        self._on_complete_message(
            channel=channel,
            trace_id=completed.trace_id,
            data_type=completed.data_type,
            data=completed.payload,
            recording_id=recording_id,
            sequence_number=completed.sequence_number,
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
            self.channels.set_trace_id(channel, item.trace_id)
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
                sequence_number=message.sequence_number,
            )

    def _handle_end_trace(
        self,
        channel: ChannelState,
        message: MessageEnvelope,
        *,
        reason: str = "producer_trace_end",
    ) -> None:
        """Handle an END_TRACE command from a producer.

        TRACE_END is sent over ZMQ after all video frames have been published to
        iceoryx2, so any frame still buffered in the ring was produced before
        this trace ended. Drain them now (marking their sequences pending) before
        finalizing, so finalization defers until those frames are spooled. The
        iceoryx2 subscriber itself is left in place; it persists for the lifetime
        of the channel across recording sessions.
        """
        self._iox2_drain.drain_all(self._on_iox2_frame)
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

        channel = self.channels.get_by_trace_id(trace_id)
        if channel is not None:
            # Drop only this trace from the coordinator; its other traces and the
            # iceoryx2 subscribers stay registered. Transport state is reset only
            # once the coordinator has no remaining active traces.
            self.channels.remove_trace_id(channel, trace_id)
            if not channel.has_active_traces():
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

            if not state.has_active_traces():
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
                self._trace_lifecycle.has_pending_video_frame_sequences_at_or_before(
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
            if channel.has_active_traces():
                for trace_id in list(channel.active_trace_ids):
                    recording_id = self._trace_lifecycle.get_trace_recording(
                        TraceRecordingLookupRequest(trace_id=trace_id)
                    )
                    self._handle_end_trace(
                        channel,
                        MessageEnvelope(
                            producer_id=producer_id,
                            command=CommandType.TRACE_END,
                            payload={
                                "trace_end": {
                                    "trace_id": trace_id,
                                    "recording_id": recording_id,
                                }
                            },
                        ),
                        reason="heartbeat_expiry",
                    )
                self._trace_lifecycle.set_max_producer_sequence(
                    channel.producer_id, channel.last_sequence_number
                )
            for service_id in list(channel.video_service_ids):
                self._iox2_drain.unregister_channel(service_id)
            channel.clear_transport_state()
            self.channels.remove(producer_id)
            self._closed_producers.add(producer_id)


Daemon = DataBridge
