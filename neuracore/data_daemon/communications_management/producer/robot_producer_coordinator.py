"""Robot-scoped producer coordinator.

One :class:`RobotProducerCoordinator` is owned per robot instance and replaces
per-stream producer channels. It owns the single ZMQ control/data socket, the
single ordered socket-sender thread, the single heartbeat thread, the shared
sequence space, and the per-stream iceoryx2 video transports for the whole
robot. Lightweight :class:`StreamSession` objects describe each recording stream
and enqueue :class:`StreamPayload` work into the coordinator.

Traffic model:

- High-volume video-like payloads (``RGB_IMAGES``/``DEPTH_IMAGES``/
  ``POINT_CLOUDS``) are published over iceoryx2, one service per stream.
- Everything else (control lifecycle + semantic JSON/batched-joint data) travels
  over the one socket. Control/lifecycle heartbeats use a high-priority lane so
  they are never starved behind a burst of semantic data; semantic data stays in
  a strictly ordered lane so it is lossless and in order.

All socket data and all video frames draw from one shared sequence allocator, so
a single end-of-recording stop cutoff spans both planes for the robot.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import IntEnum

import zmq

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
    MessageEnvelope,
    TraceTransportMetadata,
    VideoTransportChunkMetadata,
)

from ..shared_transport.communications_manager import CommunicationsManager
from ..shared_transport.iox2_video_transport import Iox2VideoTransport
from .producer_heartbeat_service import ProducerHeartbeatService

logger = logging.getLogger(__name__)

BytePart = bytes | bytearray | memoryview

_VIDEO_DATA_TYPES = (
    DataType.RGB_IMAGES,
    DataType.DEPTH_IMAGES,
    DataType.POINT_CLOUDS,
)

__all__ = [
    "RobotProducerCoordinator",
    "StreamSession",
    "StreamPayload",
    "TransportPriority",
    "data_type_uses_video_transport",
    "producer_transport_args_for_data_type",
]


def data_type_uses_video_transport(data_type: DataType) -> bool:
    """Return True when the data type should use the iceoryx2 video transport."""
    return data_type in _VIDEO_DATA_TYPES


def producer_transport_args_for_data_type(
    data_type: DataType,
) -> tuple[int, int, int]:
    """Return ``(chunk_size, max_frame_bytes, send_queue_maxsize)`` for a type."""
    if data_type in _VIDEO_DATA_TYPES:
        return (
            DEFAULT_VIDEO_CHUNK_SIZE,
            DEFAULT_VIDEO_SLOT_SIZE,
            DEFAULT_VIDEO_SEND_QUEUE_MAXSIZE,
        )
    return (DEFAULT_CHUNK_SIZE, DEFAULT_TRANSPORT_BUFFER_SIZE, 512)


class TransportPriority(IntEnum):
    """Relative priority of a payload on the coordinator's socket lanes."""

    CONTROL = 0  # heartbeats / liveness; jump the queue, carry no sequence number
    SEMANTIC = 1  # ordered, lossless data (DATA_CHUNK, BATCHED_JOINT_DATA, ...)


@dataclass
class StreamSession:
    """Lightweight per-stream recording session registered with a coordinator."""

    session_id: str
    data_type: DataType
    data_type_name: str
    recording_id: str
    robot_instance: int
    robot_id: str | None
    robot_name: str | None
    dataset_id: str | None
    dataset_name: str | None
    trace_id: str | None = None
    video_service_id: str | None = None

    def uses_video_transport(self) -> bool:
        """Return whether this stream's payloads route over iceoryx2."""
        return data_type_uses_video_transport(self.data_type)

    def trace_metadata(self) -> TraceTransportMetadata:
        """Build the trace-level metadata carried with this stream's payloads."""
        return TraceTransportMetadata(
            recording_id=self.recording_id,
            data_type=self.data_type,
            data_type_name=self.data_type_name,
            dataset_id=self.dataset_id,
            dataset_name=self.dataset_name,
            robot_name=self.robot_name,
            robot_id=self.robot_id,
            robot_instance=self.robot_instance,
        )


@dataclass
class StreamPayload:
    """One logical payload from a stream, assembled from byte-like parts."""

    session: StreamSession
    parts: tuple[BytePart, ...]
    total_bytes: int


# Token used to wake the socket-sender loop when high-priority control work
# arrives while the loop is blocked waiting for the next semantic data item.
_WAKE = object()


@dataclass
class _ControlItem:
    """A control-lane socket message plus an optional sent-notification event."""

    envelope: MessageEnvelope
    on_sent: threading.Event | None = None


class _SocketSender:
    """Ordered single-socket dispatcher with a control lane and a data lane.

    The data lane carries sequence-numbered, strictly ordered semantic payloads.
    The control lane carries sequence-less heartbeats that may overtake the data
    lane (they are exempt from the daemon's monotonic-sequence checks). Sequence
    progress is tracked only for data, so end-of-recording flush waits are never
    falsely satisfied by an out-of-order control send.
    """

    def __init__(
        self,
        producer_id: str,
        comm: CommunicationsManager,
        send_queue_maxsize: int,
    ) -> None:
        """Start the background sender thread for one coordinator socket."""
        self._producer_id = producer_id
        self._comm = comm
        self._data_queue: queue.Queue[MessageEnvelope | object | None] = queue.Queue(
            maxsize=max(0, send_queue_maxsize)
        )
        self._control_queue: queue.Queue[_ControlItem] = queue.Queue()
        self._cv = threading.Condition()
        self._inflight_data_sequences: set[int] = set()
        self._last_enqueued_sequence_number = 0
        self._error: Exception | None = None
        self._thread: threading.Thread | None = threading.Thread(
            target=self._loop,
            name="coordinator-socket-sender",
            daemon=True,
        )
        self._thread.start()

    def send_control(
        self,
        command: CommandType,
        payload: dict | None = None,
        *,
        on_sent: threading.Event | None = None,
    ) -> None:
        """Enqueue a sequence-less control message on the high-priority lane."""
        envelope = MessageEnvelope(
            producer_id=self._producer_id,
            command=command,
            payload=payload or {},
            sequence_number=None,
        )
        self._control_queue.put(_ControlItem(envelope=envelope, on_sent=on_sent))
        self._data_queue.put(_WAKE)

    def send_data(
        self,
        command: CommandType,
        payload: dict,
        sequence_number: int,
    ) -> int:
        """Enqueue a sequence-numbered data message on the ordered lane."""
        envelope = MessageEnvelope(
            producer_id=self._producer_id,
            command=command,
            payload=payload,
            sequence_number=sequence_number,
        )
        with self._cv:
            if self._error is not None:
                raise RuntimeError("Coordinator socket sender failed") from self._error
            self._inflight_data_sequences.add(sequence_number)
            self._last_enqueued_sequence_number = max(
                self._last_enqueued_sequence_number, sequence_number
            )
            self._cv.notify_all()
        self._data_queue.put(envelope)
        return sequence_number

    def get_last_enqueued_sequence_number(self) -> int:
        """Return the most recent data sequence number enqueued."""
        with self._cv:
            return self._last_enqueued_sequence_number

    def get_error(self) -> Exception | None:
        """Return the sender thread error, if the background loop failed."""
        with self._cv:
            return self._error

    def wait_until_sequence_sent(
        self, sequence_number: int, timeout_s: float | None = None
    ) -> bool:
        """Block until every data message up to ``sequence_number`` has been sent."""
        if sequence_number <= 0:
            return True
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._cv:
            while any(seq <= sequence_number for seq in self._inflight_data_sequences):
                if self._error is not None:
                    return False
                thread = self._thread
                if thread is None or not thread.is_alive():
                    return False
                if deadline is None:
                    self._cv.wait(timeout=0.1)
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(timeout=min(0.05, remaining))
            return True

    def close(self, *, join_timeout_s: float = 2.0) -> None:
        """Stop the sender thread after flushing already-enqueued work."""
        self._data_queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=join_timeout_s)
            self._thread = None
        with self._cv:
            self._cv.notify_all()

    def _loop(self) -> None:
        while True:
            item = self._data_queue.get()
            try:
                self._drain_control()
                if item is _WAKE:
                    continue
                if item is None:
                    break
                if not self._send_data_envelope(item):  # type: ignore[arg-type]
                    break
            finally:
                self._data_queue.task_done()
        with self._cv:
            self._cv.notify_all()

    def _drain_control(self) -> None:
        while True:
            try:
                control_item = self._control_queue.get_nowait()
            except queue.Empty:
                return
            try:
                self._comm.send_message(control_item.envelope)
            except Exception:
                logger.warning("Coordinator control send failed", exc_info=True)
            finally:
                if control_item.on_sent is not None:
                    control_item.on_sent.set()

    def _send_data_envelope(self, envelope: MessageEnvelope) -> bool:
        sequence_number = envelope.sequence_number
        try:
            self._comm.send_message(envelope)
        except Exception as exc:
            with self._cv:
                self._error = exc
                if sequence_number is not None:
                    self._inflight_data_sequences.discard(sequence_number)
                self._cv.notify_all()
            logger.exception("Coordinator data send failed")
            return False
        if sequence_number is not None:
            with self._cv:
                self._inflight_data_sequences.discard(sequence_number)
                self._cv.notify_all()
        return True


class RobotProducerCoordinator:
    """Owns one socket/sender/heartbeat and all video transports for a robot."""

    def __init__(
        self,
        producer_id: str,
        context: zmq.Context | None = None,
        heartbeat_interval_s: float = 1.0,
    ) -> None:
        """Create the coordinator's socket, sender, and heartbeat machinery."""
        self._producer_id = producer_id
        self._comm = CommunicationsManager(context=context)
        self._comm.create_producer_socket()
        self._sequence_allocator = ChannelSequenceAllocator()
        self._sender = _SocketSender(
            producer_id=producer_id,
            comm=self._comm,
            send_queue_maxsize=512,
        )
        self._heartbeat_service = ProducerHeartbeatService(
            interval_s=heartbeat_interval_s,
            send_heartbeat=self.heartbeat,
        )
        self._video_chunk_size = DEFAULT_VIDEO_CHUNK_SIZE

        self._lock = threading.RLock()
        self._sessions: dict[str, StreamSession] = {}
        self._video_transports: dict[str, Iox2VideoTransport] = {}
        self._stop_cutoff_sequence_number: int | None = None
        self._closed = False

    @property
    def producer_id(self) -> str:
        """Return the coordinator's producer identifier."""
        return self._producer_id

    # -- session lifecycle ------------------------------------------------

    def register_stream_session(
        self,
        *,
        stream_name: str,
        data_type: DataType,
        recording_id: str,
        robot_instance: int,
        robot_id: str | None,
        robot_name: str | None,
        dataset_id: str | None,
        dataset_name: str | None,
    ) -> StreamSession:
        """Register (or refresh) a stream session and start a new trace for it."""
        session_id = f"{data_type.value}:{stream_name}"
        with self._lock:
            # Registering a stream is the start of (or continuation within) a
            # recording, so clear any stop freeze left from a prior recording.
            self._stop_cutoff_sequence_number = None

            session = self._sessions.get(session_id)
            if session is None:
                session = StreamSession(
                    session_id=session_id,
                    data_type=data_type,
                    data_type_name=stream_name,
                    recording_id=recording_id,
                    robot_instance=robot_instance,
                    robot_id=robot_id,
                    robot_name=robot_name,
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                )
                self._sessions[session_id] = session
            else:
                session.recording_id = recording_id
                session.robot_instance = robot_instance
                session.robot_id = robot_id
                session.robot_name = robot_name
                session.dataset_id = dataset_id
                session.dataset_name = dataset_name

            session.trace_id = str(uuid.uuid4())

            if session.uses_video_transport():
                session.video_service_id = f"{self._producer_id}:{session_id}"
                self._ensure_video_transport(session)

            self._heartbeat_service.start()

        if session.uses_video_transport():
            self._announce_video_service(session)
        return session

    def get_stream_session(self, session_id: str) -> StreamSession | None:
        """Return a registered stream session by id, if present."""
        with self._lock:
            return self._sessions.get(session_id)

    def _ensure_video_transport(self, session: StreamSession) -> Iox2VideoTransport:
        service_id = session.video_service_id
        assert service_id is not None
        transport = self._video_transports.get(session.session_id)
        if transport is None:
            transport = Iox2VideoTransport(
                service_id=service_id,
                producer_id=self._producer_id,
                sequence_allocator=self._sequence_allocator,
                max_frame_bytes=DEFAULT_VIDEO_SLOT_SIZE,
            )
            self._video_transports[session.session_id] = transport
        return transport

    def _announce_video_service(self, session: StreamSession) -> None:
        """Ask the daemon to register the iceoryx2 subscriber before frames flow.

        Sends a heartbeat carrying the stream's service id + data type and waits
        for it to leave the socket. Combined with the iceoryx2 service history,
        this avoids losing the first frames to the subscriber-registration race.
        """
        sent = threading.Event()
        self._sender.send_control(
            CommandType.HEARTBEAT,
            self._video_heartbeat_payload(session),
            on_sent=sent,
        )
        sent.wait(timeout=1.0)

    # -- data path --------------------------------------------------------

    def enqueue_stream_payload(self, payload: StreamPayload) -> None:
        """Route one stream payload to iceoryx2 (video) or the socket (semantic)."""
        session = payload.session
        with self._lock:
            if self._is_frozen():
                return
            if session.trace_id is None:
                return
            normalised_parts = _normalise_parts(payload.parts)
            total_bytes = payload.total_bytes or sum(
                len(view) for view in normalised_parts
            )
            if total_bytes <= 0 or not normalised_parts:
                return

            if session.uses_video_transport():
                self._publish_video_locked(session, normalised_parts, total_bytes)
            else:
                self._send_socket_data_chunk_locked(session, normalised_parts)

    def enqueue_batched_joint(self, payload: BatchedJointDataPayload) -> None:
        """Enqueue one batched joint payload directly on the ordered data lane."""
        with self._lock:
            if self._is_frozen():
                return
            sequence_number = self._sequence_allocator.reserve()
            self._sender.send_data(
                CommandType.BATCHED_JOINT_DATA,
                {CommandType.BATCHED_JOINT_DATA.value: payload.to_dict()},
                sequence_number,
            )

    def _send_socket_data_chunk_locked(
        self, session: StreamSession, parts: Sequence[memoryview]
    ) -> None:
        payload_bytes = (
            bytes(parts[0])
            if len(parts) == 1
            else b"".join(bytes(part) for part in parts)
        )
        assert session.trace_id is not None
        chunk = DataChunkPayload(
            channel_id=self._producer_id,
            recording_id=session.recording_id,
            trace_id=session.trace_id,
            chunk_index=0,
            total_chunks=1,
            data_type_name=session.data_type_name,
            dataset_id=session.dataset_id,
            dataset_name=session.dataset_name,
            robot_name=session.robot_name,
            robot_id=session.robot_id,
            robot_instance=session.robot_instance,
            data=payload_bytes,
            data_type=session.data_type,
        )
        sequence_number = self._sequence_allocator.reserve()
        self._sender.send_data(
            CommandType.DATA_CHUNK,
            {"data_chunk": chunk.to_dict()},
            sequence_number,
        )

    def _publish_video_locked(
        self,
        session: StreamSession,
        parts: Sequence[memoryview],
        total_bytes: int,
    ) -> None:
        transport = self._video_transports.get(session.session_id)
        if transport is None:
            raise RuntimeError("iceoryx2 video transport is not available")
        assert session.trace_id is not None

        total_chunks = math.ceil(total_bytes / self._video_chunk_size)
        trace_metadata = session.trace_metadata()
        for idx, chunk in enumerate(_iter_chunk_views(parts, self._video_chunk_size)):
            metadata = VideoTransportChunkMetadata(
                trace_id=session.trace_id,
                chunk_index=idx,
                total_chunks=total_chunks,
                trace_metadata=trace_metadata if idx == 0 else None,
            ).to_dict()
            sequence_number = transport.send_frame(
                metadata=metadata,
                chunk=chunk,
                stop_cutoff_sequence_number=self._stop_cutoff_sequence_number,
            )
            if sequence_number is None:
                if not transport.is_healthy():
                    self._stop_video_after_failure(session)
                    raise RuntimeError("iceoryx2 video transport became unhealthy")
                return

    def _stop_video_after_failure(self, session: StreamSession) -> None:
        if self._stop_cutoff_sequence_number is None:
            self._stop_cutoff_sequence_number = (
                self._sequence_allocator.get_last_reserved_sequence_number()
            )
        transport = self._video_transports.pop(session.session_id, None)
        if transport is not None:
            transport.close()

    # -- stop / teardown --------------------------------------------------

    def mark_recording_stopped(self) -> int:
        """Freeze new recording payloads and return the coordinator stop cutoff."""
        with self._lock:
            if self._stop_cutoff_sequence_number is None:
                self._stop_cutoff_sequence_number = (
                    self._sequence_allocator.get_last_reserved_sequence_number()
                )
            return self._stop_cutoff_sequence_number

    def stop_recording(self, *, wait_for_drain: bool = True) -> int:
        """Freeze intake, drain accepted work, then end all active stream traces.

        Returns the coordinator stop cutoff sequence number. Sockets, the sender
        thread, and video transports are kept alive for reuse across recordings;
        the heartbeat is stopped until a new stream session starts so stale
        producer channels do not reattach to later daemon runs.
        """
        cutoff = self.mark_recording_stopped()

        flush_sequence_number = self._sender.get_last_enqueued_sequence_number()
        if not self._sender.wait_until_sequence_sent(
            flush_sequence_number,
            timeout_s=None if wait_for_drain else 0.0,
        ):
            if self._sender.get_error() is not None:
                logger.warning(
                    "Coordinator stopping after sender failure (cutoff=%s)", cutoff
                )

        with self._lock:
            sessions = list(self._sessions.values())

        for session in sessions:
            self._end_session_trace(session, wait_for_drain=wait_for_drain)
        self._heartbeat_service.stop(join_timeout_s=1.0)
        return cutoff

    def _end_session_trace(
        self, session: StreamSession, *, wait_for_drain: bool
    ) -> None:
        trace_id = session.trace_id
        if trace_id is None:
            return
        sequence_number = self._sequence_allocator.reserve()
        self._sender.send_data(
            CommandType.TRACE_END,
            {
                "trace_end": {
                    "trace_id": trace_id,
                    "recording_id": session.recording_id,
                }
            },
            sequence_number,
        )
        self._sender.wait_until_sequence_sent(
            sequence_number, timeout_s=None if wait_for_drain else 1.0
        )
        session.trace_id = None
        transport = self._video_transports.get(session.session_id)
        if transport is not None:
            transport.finish_recording_session()

    def heartbeat(self) -> None:
        """Send liveness + per-video-stream registration heartbeats."""
        with self._lock:
            video_sessions = [
                session
                for session in self._sessions.values()
                if session.uses_video_transport()
            ]
        self._sender.send_control(CommandType.HEARTBEAT, {})
        for session in video_sessions:
            transport = self._video_transports.get(session.session_id)
            if transport is not None:
                transport.update_connections()
            self._sender.send_control(
                CommandType.HEARTBEAT, self._video_heartbeat_payload(session)
            )

    @staticmethod
    def _video_heartbeat_payload(session: StreamSession) -> dict:
        return {
            "data_type": session.data_type.value,
            "video_service_id": session.video_service_id,
        }

    def _is_frozen(self) -> bool:
        return self._stop_cutoff_sequence_number is not None

    def close(self) -> None:
        """Tear down the coordinator's threads, sockets, and video transports."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._heartbeat_service.stop(join_timeout_s=1.0)
        self._sender.wait_until_sequence_sent(
            self._sender.get_last_enqueued_sequence_number(), timeout_s=2.0
        )
        self._sender.close(join_timeout_s=2.0)
        with self._lock:
            transports = list(self._video_transports.values())
            self._video_transports.clear()
        for transport in transports:
            transport.close()
        self._comm.cleanup_producer()


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
    parts: Sequence[memoryview], chunk_size: int
) -> Iterator[bytes | memoryview]:
    if not parts:
        return

    chunk_parts: list[memoryview] = []
    remaining = chunk_size
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
                    chunk_parts[0] if len(chunk_parts) == 1 else b"".join(chunk_parts)
                )
                chunk_parts = []
                remaining = chunk_size
    if chunk_parts:
        yield chunk_parts[0] if len(chunk_parts) == 1 else b"".join(chunk_parts)
