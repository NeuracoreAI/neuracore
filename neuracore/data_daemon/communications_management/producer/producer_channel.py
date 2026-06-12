"""High-level wrapper for a producer channel to the data daemon."""

from __future__ import annotations

import logging
import math
import queue
import threading
import uuid
from collections.abc import Iterator, Sequence

import zmq

from neuracore.data_daemon.communications_management.producer.models import (
    QueuedEnvelope,
)
from neuracore.data_daemon.communications_management.sequence_allocator import (
    ChannelSequenceAllocator,
)
from neuracore.data_daemon.const import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TRANSPORT_BUFFER_SIZE,
    DEFAULT_VIDEO_CHUNK_SIZE,
    DEFAULT_VIDEO_SEND_QUEUE_MAXSIZE,
    DEFAULT_VIDEO_SLOT_SIZE,
)
from neuracore.data_daemon.models import (
    BatchedJointDataPayload,
    CommandType,
    DataChunkPayload,
    DataType,
    TraceTransportMetadata,
    VideoTransportChunkMetadata,
)

from ..shared_transport.communications_manager import CommunicationsManager
from ..shared_transport.iox2_video_transport import Iox2VideoTransport
from .producer_channel_message_sender import ProducerChannelMessageSender
from .producer_heartbeat_service import ProducerHeartbeatService

logger = logging.getLogger(__name__)

BytePart = bytes | bytearray | memoryview

__all__ = [
    "ProducerChannel",
    "data_type_uses_video_transport",
    "producer_transport_args_for_data_type",
]


def data_type_uses_video_transport(data_type: DataType) -> bool:
    """Return True when the data type should use the iceoryx2 video transport."""
    return data_type in (
        DataType.RGB_IMAGES,
        DataType.DEPTH_IMAGES,
        DataType.POINT_CLOUDS,
    )


def producer_transport_args_for_data_type(
    data_type: DataType,
) -> tuple[int, int, int]:
    """Return producer transport arguments for the given data type."""
    if data_type in (
        DataType.RGB_IMAGES,
        DataType.DEPTH_IMAGES,
        DataType.POINT_CLOUDS,
    ):
        return (
            DEFAULT_VIDEO_CHUNK_SIZE,
            DEFAULT_VIDEO_SLOT_SIZE,
            DEFAULT_VIDEO_SEND_QUEUE_MAXSIZE,
        )

    return (
        DEFAULT_CHUNK_SIZE,
        DEFAULT_TRANSPORT_BUFFER_SIZE,
        512,
    )


class ProducerChannel:
    """High-level wrapper for a producer channel to the data daemon."""

    def __init__(
        self,
        data_type: DataType,
        id: str | None = None,
        context: zmq.Context | None = None,
        chunk_size: int | None = None,
        send_queue_maxsize: int | None = None,
        recording_id: str | None = None,
        max_frame_bytes: int | None = None,
    ) -> None:
        """Initialize the producer channel."""
        if data_type is None:
            raise ValueError("data_type is required")

        (
            default_chunk_size,
            default_max_frame_bytes,
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
        self._heartbeat_interval = 1.0
        self._data_type = data_type
        self._use_video_transport = data_type_uses_video_transport(data_type)
        self._sequence_allocator = ChannelSequenceAllocator()
        self._iox2_transport: Iox2VideoTransport | None = (
            Iox2VideoTransport(
                channel_id=self.channel_id,
                sequence_allocator=self._sequence_allocator,
                max_frame_bytes=int(
                    default_max_frame_bytes
                    if max_frame_bytes is None
                    else max_frame_bytes
                ),
            )
            if self._use_video_transport
            else None
        )
        self._message_sender = ProducerChannelMessageSender(
            producer_id=self.channel_id,
            comm=self._comm,
            send_queue_maxsize=self.send_queue_maxsize,
            sequence_allocator=self._sequence_allocator,
        )
        self._heartbeat_service = ProducerHeartbeatService(
            interval_s=self._heartbeat_interval,
            send_heartbeat=self.heartbeat,
        )

        self._recording_send_lock = threading.RLock()
        self._stop_cutoff_sequence_number: int | None = None

    @property
    def _send_queue(self) -> queue.Queue[QueuedEnvelope | None]:
        """Expose the sender queue for compatibility with existing tests."""
        return self._message_sender.queue

    @property
    def _stop_event(self) -> threading.Event:
        """Expose the heartbeat stop event for compatibility with existing tests."""
        return self._heartbeat_service.stop_event

    def start_producer_channel(self) -> None:
        """Starts the producer channel's heartbeat loop."""
        self._heartbeat_service.start()

    def heartbeat(self) -> None:
        """Send a heartbeat message to the daemon.

        The heartbeat carries the channel data type so the daemon can create the
        matching iceoryx2 subscriber for video channels on first contact. It also
        refreshes the publisher connections so a daemon subscriber that just
        registered receives buffered history frames even while idle.
        """
        if self._iox2_transport is not None:
            self._iox2_transport.update_connections()
        self._send(CommandType.HEARTBEAT, {"data_type": self._data_type.value})

    def set_recording_id(self, recording_id: str | None) -> None:
        """Set the recording ID for the producer."""
        self.recording_id = recording_id

    def get_last_accepted_sequence_number(self) -> int:
        """Return the latest sequence accepted by the sender or video transport."""
        last_enqueued = self.get_last_enqueued_sequence_number()

        if self._iox2_transport is None:
            return last_enqueued

        return max(
            last_enqueued,
            self._iox2_transport.get_last_reserved_sequence_number(),
        )

    def mark_recording_stop_requested(self) -> int:
        """Freeze recording data sends and return the last accepted sequence number."""
        with self._recording_send_lock:
            if self._stop_cutoff_sequence_number is None:
                self._stop_cutoff_sequence_number = (
                    self.get_last_accepted_sequence_number()
                )
            return self._stop_cutoff_sequence_number

    def _recording_data_stopped(self) -> bool:
        return self._stop_cutoff_sequence_number is not None

    def start_recording_session(
        self,
        recording_id: str | None = None,
        max_frame_bytes: int | None = None,
    ) -> None:
        """Start a fresh recording session for this producer channel."""
        with self._recording_send_lock:
            self._stop_cutoff_sequence_number = None

            if recording_id is not None:
                self.set_recording_id(recording_id)
            if not self.recording_id:
                raise ValueError(
                    "recording_id is required; set on ProducerChannel init."
                )
            if self.trace_id is not None:
                raise RuntimeError(
                    "Cannot start a new recording session while a trace is active."
                )

            self.start_producer_channel()
            self.start_new_trace()

        if self._use_video_transport:
            self._announce_video_channel()

    def _announce_video_channel(self) -> None:
        """Prompt the daemon to register its iceoryx2 subscriber for this channel.

        Sends a heartbeat (which carries the data type) and waits for it to be
        flushed so the daemon learns about the channel before video frames flow.
        Combined with the iceoryx2 service history, this avoids losing the first
        frames of a recording to the subscriber-registration race.
        """
        sequence_number = self._send(
            CommandType.HEARTBEAT, {"data_type": self._data_type.value}
        )
        self.wait_until_sequence_sent(sequence_number)

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
        sequence_number = self._send(
            CommandType.TRACE_END,
            {
                "trace_end": {
                    "trace_id": trace_id,
                    "recording_id": recording_id,
                }
            },
        )
        if not self.wait_until_sequence_sent(sequence_number):
            raise RuntimeError("Failed to send TRACE_END before ending trace")
        self.trace_id = None
        self.recording_id = None

    def stop_producer_channel(
        self,
        wait_for_transport_drain: bool = True,
    ) -> None:
        """Stop the producer channel and release local resources."""
        self._stop_heartbeat_service()

        final_flush_sequence = self.get_last_enqueued_sequence_number()
        stop_failure: RuntimeError | None = None
        if not self.wait_until_sequence_sent(final_flush_sequence):
            sender_error = self._get_message_sender_error()
            if sender_error is not None:
                logger.warning(
                    "Producer channel stopping after sender failure without "
                    "flushing final sequence_number=%s error=%r",
                    final_flush_sequence,
                    sender_error,
                )
            else:
                logger.error(
                    "Producer channel sender stopped before flushing final "
                    "sequence_number=%s",
                    final_flush_sequence,
                )
                stop_failure = RuntimeError(
                    "Failed to send all enqueued messages "
                    "before stopping producer channel"
                )

        # Video frames are published synchronously into the iceoryx2 ring as they
        # are produced, so there is no producer-side queue to drain before
        # shutdown. The daemon's sequence tracking guarantees all frames up to the
        # stop cutoff are spooled before finalization.
        self._close_iox2_transport()
        self._stop_message_sender()
        self._comm.cleanup_producer()

        if stop_failure is not None:
            raise stop_failure

    def _send(self, command: CommandType, payload: dict | None = None) -> int:
        """Send a message to the daemon."""
        sequence_number = self._sequence_allocator.reserve()
        return self._message_sender.send(
            command,
            payload,
            sequence_number=sequence_number,
        )

    def get_last_sent_sequence_number(self) -> int:
        """Return the most recent sequence number successfully sent on the socket."""
        return self._message_sender.get_last_sent_sequence_number()

    def get_last_enqueued_sequence_number(self) -> int:
        """Return the most recent sequence number enqueued for the sender thread."""
        return self._message_sender.get_last_enqueued_sequence_number()

    def wait_until_sequence_sent(
        self,
        sequence_number: int,
        timeout_s: float | None = None,
    ) -> bool:
        """Block until the sender thread has sent up to `sequence_number`."""
        return self._message_sender.wait_until_sequence_sent(
            sequence_number,
            timeout_s=timeout_s,
        )

    def _send_socket_data_chunk(self, payload: DataChunkPayload) -> None:
        """Send one DATA_CHUNK payload directly over the producer socket."""
        with self._recording_send_lock:
            if self._recording_data_stopped():
                return

            self._send(
                CommandType.DATA_CHUNK,
                {"data_chunk": payload.to_dict()},
            )

    def send_batched_joint_data(self, payload: BatchedJointDataPayload) -> None:
        """Send one explicit batched joint payload over the producer socket."""
        with self._recording_send_lock:
            if self._recording_data_stopped():
                return

            self._send(
                CommandType.BATCHED_JOINT_DATA,
                {CommandType.BATCHED_JOINT_DATA.value: payload.to_dict()},
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

    def _stop_video_logging_after_failure(self) -> None:
        """Stop accepting more recording data after an unrecoverable video error."""
        with self._recording_send_lock:
            if self._stop_cutoff_sequence_number is None:
                self._stop_cutoff_sequence_number = (
                    self.get_last_accepted_sequence_number()
                )
        self._close_iox2_transport()

    def _send_data_parts_iox2(
        self,
        normalised_parts: Sequence[memoryview],
        total_chunks: int,
        trace_metadata: TraceTransportMetadata,
    ) -> None:
        """Publish one logical payload over the iceoryx2 video transport."""
        transport = self._iox2_transport
        if transport is None:
            raise RuntimeError("iceoryx2 video transport is not available")

        trace_id = self.trace_id
        if trace_id is None:
            raise RuntimeError("Trace ID required for video transport")

        for idx, chunk in enumerate(self._iter_chunk_views(normalised_parts)):
            metadata = VideoTransportChunkMetadata(
                trace_id=trace_id,
                chunk_index=idx,
                total_chunks=total_chunks,
                trace_metadata=trace_metadata if idx == 0 else None,
            ).to_dict()

            with self._recording_send_lock:
                if self._recording_data_stopped():
                    return

                sequence_number = transport.send_frame(
                    metadata=metadata,
                    chunk=chunk,
                    stop_cutoff_sequence_number=self._stop_cutoff_sequence_number,
                )

            if sequence_number is None:
                # send_frame returns None either because the frame was rejected
                # by the stop cutoff (transport still healthy) or because the
                # publisher errored (transport unhealthy).
                if not transport.is_healthy():
                    self._stop_video_logging_after_failure()
                    raise RuntimeError("iceoryx2 video transport became unhealthy")
                return

    def send_data_parts(
        self,
        parts: Sequence[BytePart],
        data_type: DataType,
        robot_instance: int,
        data_type_name: str,
        total_bytes: int | None = None,
        robot_id: str | None = None,
        robot_name: str | None = None,
        dataset_id: str | None = None,
        dataset_name: str | None = None,
    ) -> None:
        """Send a logical payload assembled from multiple byte-like parts."""
        if self._recording_data_stopped():
            return

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
        trace_metadata = TraceTransportMetadata(
            recording_id=recording_id,
            data_type=data_type,
            data_type_name=data_type_name,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            robot_name=robot_name,
            robot_id=robot_id,
            robot_instance=robot_instance,
        )

        if not self._use_video_transport:
            if not normalised_parts:
                return
            payload_bytes = (
                bytes(normalised_parts[0])
                if len(normalised_parts) == 1
                else b"".join(bytes(part) for part in normalised_parts)
            )
            payload = DataChunkPayload(
                channel_id=self.channel_id,
                recording_id=recording_id,
                trace_id=trace_id,
                chunk_index=0,
                total_chunks=1,
                data_type_name=data_type_name,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                robot_name=robot_name,
                robot_id=robot_id,
                robot_instance=robot_instance,
                data=payload_bytes,
                data_type=data_type,
            )

            self._send_socket_data_chunk(payload)
            return

        self._send_data_parts_iox2(
            normalised_parts,
            total_chunks,
            trace_metadata,
        )

    def initialize_new_producer_channel(
        self,
        max_frame_bytes: int | None = None,
    ) -> None:
        """Initialize a new producer channel for recording."""
        self.start_recording_session(max_frame_bytes=max_frame_bytes)

    def cleanup_producer_channel(
        self,
        stop_cutoff_sequence_number: int,
        wait_for_transport_drain: bool = True,
    ) -> None:
        """Finish one trace after recording data up to the stop cutoff is sent.

        Video frames are published synchronously into the iceoryx2 ring as they
        are produced, so there is nothing to drain here. The TRACE_END sent below
        carries the channel's last sequence number, and the daemon defers
        finalization until every frame up to the stop cutoff has been spooled.
        """
        if stop_cutoff_sequence_number < 0:
            raise ValueError("stop_cutoff_sequence_number must be non-negative")

        # The stop cutoff spans the channel sequence space, so for video channels it
        # is typically an iceoryx2 frame sequence that never travels over ZMQ — the
        # ZMQ sender would never report it as "sent". Video frames are published
        # synchronously into the iceoryx2 ring as they are produced, so only the
        # ZMQ control/data messages need flushing here. Wait on the sender's own
        # last-enqueued sequence instead of the global cutoff.
        flush_sequence_number = self.get_last_enqueued_sequence_number()
        if not self.wait_until_sequence_sent(flush_sequence_number):
            raise RuntimeError("Failed to send queued recording data before cleanup")

        self.end_trace()

        if self._iox2_transport is not None:
            self._iox2_transport.finish_recording_session()

    def _stop_heartbeat_service(self) -> None:
        self._heartbeat_service.stop(join_timeout_s=1.0)

    def _stop_message_sender(self) -> None:
        self._message_sender.close(join_timeout_s=2.0)

    def _close_iox2_transport(self) -> None:
        if self._iox2_transport is not None:
            self._iox2_transport.close()
            self._iox2_transport = None

    def _get_message_sender_error(self) -> Exception | None:
        sender = getattr(self, "_message_sender", None)
        if sender is None:
            return None
        get_error = getattr(sender, "get_error", None)
        if get_error is None:
            return None
        return get_error()
