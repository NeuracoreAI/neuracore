"""High-level wrapper for a producer channel to the data daemon."""

from __future__ import annotations

import logging
import math
import queue
import threading
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import zmq

from neuracore.data_daemon.const import (
    CONTROL_SOCKET_PATH,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SHARED_MEMORY_SIZE,
    DEFAULT_VIDEO_CHUNK_SIZE,
    DEFAULT_VIDEO_SEND_QUEUE_MAXSIZE,
    DEFAULT_VIDEO_SLOT_SIZE,
    SOCKET_PATH,
    TRANSPORT_GROUP_SOCKET_PATHS,
    VIDEO_SOCKET_PATH,
)
from neuracore.data_daemon.models import (
    BatchedJointDataPayload,
    CommandType,
    DataChunkPayload,
    DataType,
    SharedMemoryChunkMetadata,
    TraceTransportMetadata,
)

from ..shared_transport.communications_manager import CommunicationsManager
from ..shared_transport.shared_slot_transport import SharedSlotVideoTransport
from .producer_channel_message_sender import (
    ProducerChannelMessageSender,
    QueuedEnvelope,
)
from .producer_heartbeat_service import ProducerHeartbeatService
from .models import (
    ProducerSharedMemoryDebugStats,
    ProducerTransportDebugStats,
    ProducerTransportTimingStats,
)
from .shared_sender_registry import SharedSenderHandle, SharedSenderRegistry

logger = logging.getLogger(__name__)

BytePart = bytes | bytearray | memoryview

__all__ = ["ProducerChannel", "producer_transport_args_for_data_type"]


@dataclass
class _OwnedSenderHandle:
    """One sender/comm pair plus its release callback."""

    comm: CommunicationsManager
    sender: ProducerChannelMessageSender
    release: Callable[[], None]


def data_type_uses_shared_slot_transport(data_type: DataType) -> bool:
    """Return True when the data type should use shared-slot transport."""
    return data_type in (DataType.RGB_IMAGES, DataType.DEPTH_IMAGES)


def producer_transport_args_for_data_type(
    data_type: DataType,
) -> tuple[int, int, int]:
    """Return producer transport arguments for the given data type."""
    if data_type in (DataType.RGB_IMAGES, DataType.DEPTH_IMAGES):
        return (
            DEFAULT_VIDEO_CHUNK_SIZE,
            DEFAULT_VIDEO_SLOT_SIZE,
            DEFAULT_VIDEO_SEND_QUEUE_MAXSIZE,
        )

    return (
        DEFAULT_CHUNK_SIZE,
        DEFAULT_SHARED_MEMORY_SIZE,
        512,
    )


def producer_socket_path_for_data_type(data_type: DataType) -> str:
    """Return the daemon socket path for the given producer data type."""
    if data_type_uses_shared_slot_transport(data_type):
        return str(VIDEO_SOCKET_PATH)
    return str(SOCKET_PATH)


def producer_socket_path_for_transport_group(
    data_type: DataType,
    transport_group: str | None = None,
) -> str:
    """Return the daemon data socket path for the logical transport group."""
    if data_type_uses_shared_slot_transport(data_type):
        return str(VIDEO_SOCKET_PATH)
    if transport_group is not None:
        path = TRANSPORT_GROUP_SOCKET_PATHS.get(transport_group)
        if path is not None:
            return str(path)
    return str(producer_socket_path_for_data_type(data_type))


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
        shared_memory_size: int | None = None,
        transport_group: str | None = None,
        share_transport: bool | None = None,
    ) -> None:
        """Initialize the producer channel."""
        if data_type is None:
            raise ValueError("data_type is required")

        (
            default_chunk_size,
            default_shared_memory_size,
            default_send_queue_maxsize,
        ) = producer_transport_args_for_data_type(data_type)
        socket_path = producer_socket_path_for_transport_group(
            data_type,
            transport_group,
        )

        self.channel_id = id or str(uuid.uuid4())
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
        self._transport_group = transport_group
        self._use_shared_slot_transport = data_type_uses_shared_slot_transport(
            data_type
        )
        self._share_transport = bool(
            (not self._use_shared_slot_transport) if share_transport is None else share_transport
        )
        self._data_runtime = self._acquire_sender_runtime(
            context=context,
            socket_path=socket_path,
            send_queue_maxsize=self.send_queue_maxsize,
            share_runtime=self._share_transport,
        )
        self._control_runtime = self._acquire_sender_runtime(
            context=context,
            socket_path=str(CONTROL_SOCKET_PATH),
            send_queue_maxsize=512,
            share_runtime=True,
        )
        self._shared_slot_transport = (
            SharedSlotVideoTransport(
                slot_size=int(
                    default_shared_memory_size
                    if shared_memory_size is None
                    else shared_memory_size
                )
            )
            if self._use_shared_slot_transport
            else None
        )
        self._heartbeat_service = ProducerHeartbeatService(
            interval_s=self._heartbeat_interval,
            send_heartbeat=self.heartbeat,
        )
        self._last_data_sequence_number = 0
        self._last_control_sequence_number = 0

    @property
    def _send_queue(
        self,
    ) -> queue.Queue[QueuedEnvelope | None]:
        """Expose the control sender queue for compatibility with existing tests."""
        return self._control_runtime.sender.queue

    @property
    def _stop_event(self) -> threading.Event:
        """Expose the heartbeat stop event for compatibility with existing tests."""
        return self._heartbeat_service.stop_event

    @staticmethod
    def _acquire_sender_runtime(
        *,
        context: zmq.Context | None,
        socket_path: str,
        send_queue_maxsize: int,
        share_runtime: bool,
    ) -> SharedSenderHandle | _OwnedSenderHandle:
        if context is None and share_runtime:
            return SharedSenderRegistry.acquire(
                socket_path=socket_path,
                send_queue_maxsize=send_queue_maxsize,
            )

        comm = CommunicationsManager(context=context, socket_path=socket_path)
        comm.create_producer_socket()
        sender = ProducerChannelMessageSender(
            producer_id=None,
            comm=comm,
            send_queue_maxsize=send_queue_maxsize,
        )
        return _OwnedSenderHandle(
            comm=comm,
            sender=sender,
            release=lambda: (
                sender.close(join_timeout_s=2.0),
                comm.cleanup_producer(),
            ),
        )

    def start_producer_channel(self) -> None:
        """Starts the producer channel's heartbeat loop."""
        self._heartbeat_service.start()

    def heartbeat(self) -> None:
        """Send a heartbeat message to the daemon."""
        self._send(CommandType.HEARTBEAT, {}, use_control_sender=True)

    def set_recording_id(self, recording_id: str | None) -> None:
        """Set the recording ID for the producer."""
        self.recording_id = recording_id

    def start_recording_session(
        self,
        recording_id: str | None = None,
        shared_memory_size: int | None = None,
        *,
        trace_id: str | None = None,
        create_trace: bool = True,
    ) -> None:
        """Start a fresh recording session for this producer channel."""
        if recording_id is not None:
            self.set_recording_id(recording_id)
        if not self.recording_id:
            raise ValueError("recording_id is required; set on ProducerChannel init.")
        if create_trace and self.trace_id is not None:
            raise RuntimeError(
                "Cannot start a new recording session while a trace is active."
            )

        self.start_producer_channel()
        if create_trace:
            self.start_new_trace(trace_id=trace_id)
        if self._use_shared_slot_transport:
            self.open_fixed_shared_slots(slot_size=shared_memory_size)

    def start_new_trace(self, trace_id: str | None = None) -> None:
        """Start a new trace for the given recording."""
        if not self.recording_id:
            raise ValueError("recording_id is required; set on ProducerChannel init.")
        self.trace_id = trace_id or str(uuid.uuid4())

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
            use_control_sender=True,
        )
        self.trace_id = None
        self.recording_id = None

    def stop_producer_channel(self) -> None:
        """Stops the producer channel and cleans up any associated resources."""
        self._stop_heartbeat_service()

        # Ensure all descriptors were actually sent
        cutoff_sequence = self.get_last_enqueued_sequence_number()
        self.wait_until_sequence_sent(cutoff_sequence)

        if self._shared_slot_transport is not None:
            self._shared_slot_transport.wait_until_drained(timeout_s=30.0)

        self._close_shared_slot_transport()
        self._release_sender_runtimes()

    def _send(
        self,
        command: CommandType,
        payload: dict | None = None,
        *,
        use_control_sender: bool = False,
    ) -> int:
        """Send a message to the daemon."""
        sender = self._control_runtime.sender if use_control_sender else self._data_runtime.sender
        sequence_number = None
        if (
            not use_control_sender
            and self._use_shared_slot_transport
            and self._shared_slot_transport is not None
        ):
            sequence_number = self._shared_slot_transport.next_sequence_number()
        sent_sequence_number = sender.send(
            command,
            payload,
            producer_id=self.channel_id,
            sequence_number=sequence_number,
            on_failed_send=(
                self._shared_slot_transport.notify_sender_failure
                if not use_control_sender
                and self._use_shared_slot_transport
                and self._shared_slot_transport is not None
                else None
            ),
        )
        if use_control_sender:
            self._last_control_sequence_number = sent_sequence_number
        else:
            self._last_data_sequence_number = sent_sequence_number
        return sent_sequence_number

    def get_last_sent_sequence_number(self) -> int:
        """Return the most recent sequence number successfully sent on the socket."""
        return max(
            self._data_runtime.sender.get_last_sent_sequence_number(),
            self._control_runtime.sender.get_last_sent_sequence_number(),
        )

    def get_last_enqueued_sequence_number(self) -> int:
        """Return the most recent sequence number enqueued for the sender thread."""
        return max(
            self._data_runtime.sender.get_last_enqueued_sequence_number(),
            self._control_runtime.sender.get_last_enqueued_sequence_number(),
        )

    def wait_until_sequence_sent(self, sequence_number: int) -> bool:
        """Block until the sender thread has sent up to `sequence_number`."""
        if sequence_number <= 0:
            return True
        control_ok = True
        data_ok = True
        if sequence_number == self._last_control_sequence_number:
            control_ok = self._control_runtime.sender.wait_until_sequence_sent(
                sequence_number
            )
        if sequence_number == self._last_data_sequence_number:
            data_ok = self._data_runtime.sender.wait_until_sequence_sent(sequence_number)
        if (
            sequence_number != self._last_control_sequence_number
            and sequence_number != self._last_data_sequence_number
        ):
            return (
                self._data_runtime.sender.wait_until_sequence_sent(sequence_number)
                or self._control_runtime.sender.wait_until_sequence_sent(sequence_number)
            )
        return control_ok and data_ok

    def get_transport_stats(self) -> ProducerTransportDebugStats:
        """Return a best-effort debug snapshot for this producer channel."""
        data_sender_stats = self._data_runtime.sender.get_stats()
        control_sender_stats = self._control_runtime.sender.get_stats()
        message_sender_stats = type(data_sender_stats)(
            last_enqueued_sequence_number=max(
                data_sender_stats.last_enqueued_sequence_number,
                control_sender_stats.last_enqueued_sequence_number,
            ),
            last_socket_sent_sequence_number=max(
                data_sender_stats.last_socket_sent_sequence_number,
                control_sender_stats.last_socket_sent_sequence_number,
            ),
            send_queue_qsize=data_sender_stats.send_queue_qsize,
            send_queue_maxsize=data_sender_stats.send_queue_maxsize,
            pending_sequence_count=0,
            sender_thread_alive=(
                data_sender_stats.sender_thread_alive
                or control_sender_stats.sender_thread_alive
            ),
            queue_put=ProducerTransportTimingStats(
                count=(
                    data_sender_stats.queue_put.count
                    + control_sender_stats.queue_put.count
                ),
                total_s=(
                    data_sender_stats.queue_put.total_s
                    + control_sender_stats.queue_put.total_s
                ),
                max_s=max(
                    data_sender_stats.queue_put.max_s,
                    control_sender_stats.queue_put.max_s,
                ),
            ),
            socket_send=ProducerTransportTimingStats(
                count=(
                    data_sender_stats.socket_send.count
                    + control_sender_stats.socket_send.count
                ),
                total_s=(
                    data_sender_stats.socket_send.total_s
                    + control_sender_stats.socket_send.total_s
                ),
                max_s=max(
                    data_sender_stats.socket_send.max_s,
                    control_sender_stats.socket_send.max_s,
                ),
            ),
            shared_memory_dispatch=data_sender_stats.shared_memory_dispatch,
            send_error_count=(
                data_sender_stats.send_error_count
                + control_sender_stats.send_error_count
            ),
            last_send_error=(
                data_sender_stats.last_send_error or control_sender_stats.last_send_error
            ),
        )
        message_sender_stats = type(message_sender_stats)(
            send_queue_qsize=message_sender_stats.send_queue_qsize,
            send_queue_maxsize=message_sender_stats.send_queue_maxsize,
            last_enqueued_sequence_number=message_sender_stats.last_enqueued_sequence_number,
            last_socket_sent_sequence_number=message_sender_stats.last_socket_sent_sequence_number,
            pending_sequence_count=max(
                0,
                message_sender_stats.last_enqueued_sequence_number
                - message_sender_stats.last_socket_sent_sequence_number,
            ),
            sender_thread_alive=message_sender_stats.sender_thread_alive,
            queue_put=message_sender_stats.queue_put,
            socket_send=message_sender_stats.socket_send,
            shared_memory_dispatch=message_sender_stats.shared_memory_dispatch,
            send_error_count=message_sender_stats.send_error_count,
            last_send_error=message_sender_stats.last_send_error,
        )
        shared_memory_stats = (
            self._shared_slot_transport.get_stats()
            if self._shared_slot_transport is not None
            else ProducerSharedMemoryDebugStats(
                shared_memory_name=None,
                shared_memory_size=0,
                shared_memory_open=ProducerTransportTimingStats(),
                shared_memory_write=ProducerTransportTimingStats(),
                shared_memory_write_bytes=0,
            )
        )
        return ProducerTransportDebugStats(
            channel_id=self.channel_id,
            recording_id=self.recording_id,
            trace_id=self.trace_id,
            chunk_size=self.chunk_size,
            heartbeat_thread_alive=self._heartbeat_service.get_stats()[
                "heartbeat_thread_alive"
            ],
            shared_memory=shared_memory_stats,
            message_sender=message_sender_stats,
        )

    def open_fixed_shared_slots(self, slot_size: int | None = None) -> None:
        """Announce the fixed shared-slot transport for this producer."""
        if not self._use_shared_slot_transport or self._shared_slot_transport is None:
            return
        if (
            slot_size is not None
            and not self._shared_slot_transport.is_announced()
            and int(slot_size) != self._shared_slot_transport.slot_size
        ):
            self._shared_slot_transport.close()
            self._shared_slot_transport = SharedSlotVideoTransport(
                slot_size=int(slot_size)
            )
        if self._shared_slot_transport.is_announced():
            return
        payload = self._shared_slot_transport.open_payload()
        sequence_number = self._send(
            CommandType.OPEN_FIXED_SHARED_SLOTS,
            {
                "open_fixed_shared_slots": payload.model_dump(exclude_none=True),
            },
            use_control_sender=True,
        )
        if not self._control_runtime.sender.wait_until_sequence_sent(sequence_number):
            raise RuntimeError(
                "Failed to send OPEN_FIXED_SHARED_SLOTS before video transport use"
            )

    def _send_socket_data_chunk(self, payload: DataChunkPayload) -> None:
        """Send one DATA_CHUNK payload directly over the producer socket."""
        self._send(
            CommandType.DATA_CHUNK,
            {"data_chunk": payload.to_dict()},
        )

    def send_batched_joint_data(self, payload: BatchedJointDataPayload) -> None:
        """Send one explicit batched joint payload over the producer socket."""
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

        total_chunks = math.ceil(total_bytes / self.chunk_size)
        produced_chunks = 0
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

        if not self._use_shared_slot_transport:
            if not normalised_parts:
                return
            payload_bytes = (
                bytes(normalised_parts[0])
                if len(normalised_parts) == 1
                else b"".join(bytes(part) for part in normalised_parts)
            )
            self._send_socket_data_chunk(
                DataChunkPayload(
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
            )
            return

        self.open_fixed_shared_slots()
        shared_slot_transport = self._shared_slot_transport
        if shared_slot_transport is None:
            raise RuntimeError("Shared-slot transport is not available")

        for idx, chunk in enumerate(self._iter_chunk_views(normalised_parts)):
            produced_chunks += 1
            shared_slot_transport.enqueue_packet(
                producer_id=self.channel_id,
                sender=self._data_runtime.sender,
                metadata=SharedMemoryChunkMetadata(
                    trace_id=trace_id,
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    trace_metadata=trace_metadata if idx == 0 else None,
                ).to_dict(),
                chunk=chunk,
            )

        if produced_chunks != total_chunks:
            raise RuntimeError(
                "Chunk count mismatch while serializing payload for transport"
            )

    def initialize_new_producer_channel(
        self,
        shared_memory_size: int | None = None,
    ) -> None:
        """Initialize a new producer channel for recording."""
        self.start_recording_session(shared_memory_size=shared_memory_size)

    def cleanup_producer_channel(
        self,
        *,
        wait_for_slot_drain: bool = True,
    ) -> None:
        """Finish one trace after all queued payload descriptors are sent.

        When ``wait_for_slot_drain`` is False, this returns after the producer has
        pushed all currently queued shared-slot descriptors and the TRACE_END
        control message onto the socket. Slot credits may still return
        asynchronously while the channel remains alive for later cleanup.
        """
        if self._shared_slot_transport is not None:
            self._shared_slot_transport.wait_until_payload_handed_off(timeout_s=30.0)

        cutoff_sequence = self.get_last_enqueued_sequence_number()
        self.wait_until_sequence_sent(cutoff_sequence)

        if wait_for_slot_drain and self._shared_slot_transport is not None:
            self._shared_slot_transport.wait_until_drained(timeout_s=30.0)

        self.end_trace()

        trace_end_sequence = self.get_last_enqueued_sequence_number()
        self.wait_until_sequence_sent(trace_end_sequence)

        if wait_for_slot_drain and self._shared_slot_transport is not None:
            self._shared_slot_transport.finish_recording_session()

    def _stop_heartbeat_service(self) -> None:
        self._heartbeat_service.stop(join_timeout_s=1.0)

    def _release_sender_runtimes(self) -> None:
        self._data_runtime.release()
        self._control_runtime.release()

    def _close_shared_slot_transport(self) -> None:
        if self._shared_slot_transport is not None:
            self._shared_slot_transport.close()
            self._shared_slot_transport = None
