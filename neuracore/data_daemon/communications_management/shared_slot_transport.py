"""Fixed shared-slot transport for producer-side video packets."""

from __future__ import annotations

import json
import logging
import os
import queue
import struct
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory

import zmq

from neuracore.data_daemon.const import (
    ACK_BASE_DIR,
    DEFAULT_VIDEO_ACK_TIMEOUT_SECONDS,
    DEFAULT_VIDEO_SLOT_ALLOCATE_TIMEOUT_SECONDS,
    DEFAULT_VIDEO_SLOT_COUNT,
    DEFAULT_VIDEO_SLOT_SIZE,
    SHARED_RING_RECORD_HEADER_FORMAT,
    SHARED_RING_RECORD_HEADER_SIZE,
    SHARED_RING_RECORD_MAGIC,
)
from neuracore.data_daemon.models import (
    CommandType,
    MessageEnvelope,
    OpenFixedSharedSlotsModel,
    SharedSlotCreditReturn,
    SharedSlotDescriptor,
    SharedSlotReadyModel,
)

from .producer_channel_message_sender import ProducerChannelMessageSender
from .producer_transport_debug_models import (
    ProducerSharedRingBufferDebugStats,
    ProducerTransportTimingStats,
)

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    """Read a float override from the environment, falling back safely."""
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return float(default)


class PacketTooLarge(ValueError):
    """Raised when a packet cannot fit in a single shared slot."""


class SharedSlotTimeout(TimeoutError):
    """Raised when shared-slot setup, allocation, or credit return times out."""


@dataclass(frozen=True)
class InFlightSlot:
    """Tracked producer-side state for one descriptor awaiting slot credit."""

    shm_name: str
    slot_id: int
    sequence_id: int
    reserved_at: float
    socket_sent_at: float | None = None


@dataclass
class QueuedSharedSlotPacket:
    """One packet awaiting shared-memory copy and descriptor enqueue."""

    producer_id: str
    sender: ProducerChannelMessageSender
    metadata_bytes: bytes
    chunk: bytes | bytearray | memoryview
    packet_length: int


def build_shared_frame_packet(
    metadata: dict[str, str | int | None],
    chunk: bytes | bytearray | memoryview,
) -> bytes:
    """Build the self-describing packet stored in one shared slot."""
    metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    payload = bytes(chunk)
    return (
        struct.pack(
            SHARED_RING_RECORD_HEADER_FORMAT,
            SHARED_RING_RECORD_MAGIC,
            len(metadata_bytes),
            len(payload),
        )
        + metadata_bytes
        + payload
    )


def build_shared_frame_packet_metadata(
    metadata: dict[str, str | int | None],
    chunk: bytes | bytearray | memoryview,
) -> tuple[bytes, int]:
    """Return serialized metadata plus total packet length without copying the chunk."""
    metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    chunk_len = len(chunk)
    packet_length = SHARED_RING_RECORD_HEADER_SIZE + len(metadata_bytes) + chunk_len
    return metadata_bytes, packet_length


def parse_shared_frame_packet(packet: bytes) -> tuple[dict[str, object], bytes]:
    """Parse one self-describing packet copied out of a shared slot."""
    if len(packet) < SHARED_RING_RECORD_HEADER_SIZE:
        raise ValueError("Shared-slot packet shorter than record header")
    magic, metadata_len, chunk_len = struct.unpack(
        SHARED_RING_RECORD_HEADER_FORMAT,
        packet[:SHARED_RING_RECORD_HEADER_SIZE],
    )
    if magic != SHARED_RING_RECORD_MAGIC:
        raise ValueError("Shared-slot packet missing shared record magic")
    expected = SHARED_RING_RECORD_HEADER_SIZE + metadata_len + chunk_len
    if len(packet) < expected:
        raise ValueError("Shared-slot packet shorter than declared lengths")
    if len(packet) > expected:
        raise ValueError("Shared-slot packet contains trailing bytes")
    metadata_start = SHARED_RING_RECORD_HEADER_SIZE
    metadata_end = metadata_start + metadata_len
    metadata = json.loads(packet[metadata_start:metadata_end].decode("utf-8"))
    return metadata, packet[metadata_end:]


class SharedSlotRegistry:
    """Producer-side session state for one daemon-owned shared-slot transport."""

    _instance: SharedSlotRegistry | None = None
    _refcount = 0
    _instance_lock = threading.Lock()

    def __init__(
        self,
        *,
        slot_size: int,
        slot_count: int,
        ack_timeout_s: float,
        allocate_timeout_s: float,
    ) -> None:
        self.slot_size = int(slot_size)
        self.slot_count = int(slot_count)
        self.ack_timeout_s = float(ack_timeout_s)
        self.allocate_timeout_s = float(allocate_timeout_s)
        self.shm_name: str | None = None
        self._shm: SharedMemory | None = None

        self._free_slots: deque[int] = deque()
        self._condition = threading.Condition()
        self._sequence_id = 1
        self._healthy = True
        self._ready = False
        self._in_flight: dict[int, InFlightSlot] = {}
        self._closed = False
        self._acked_sequence_count = 0
        self._ack_timeout_count = 0
        self._last_acked_sequence_id: int | None = None
        self._last_ack_latency_s: float | None = None
        self._max_ack_latency_s = 0.0
        self._max_in_flight_slot_count = 0
        self._unhealthy_reason: str | None = None

        ACK_BASE_DIR.mkdir(parents=True, exist_ok=True)
        socket_path = ACK_BASE_DIR / f"slot_control_{os.getpid()}_{uuid.uuid4().hex}.ipc"
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass
        self._control_socket_path = socket_path
        self.control_endpoint = f"ipc://{socket_path}"

        self._context = zmq.Context()
        self._control_socket = self._context.socket(zmq.PULL)
        self._control_socket.setsockopt(zmq.LINGER, 0)
        self._control_socket.bind(self.control_endpoint)

        self._stop_event = threading.Event()
        self._control_thread = threading.Thread(
            target=self._control_listener_loop,
            name="shared-slot-control-listener",
            daemon=True,
        )
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="shared-slot-watchdog",
            daemon=True,
        )
        self._control_thread.start()
        self._watchdog_thread.start()

    @classmethod
    def acquire(
        cls,
        *,
        slot_size: int = DEFAULT_VIDEO_SLOT_SIZE,
        slot_count: int = DEFAULT_VIDEO_SLOT_COUNT,
        ack_timeout_s: float | None = None,
        allocate_timeout_s: float | None = None,
    ) -> "SharedSlotRegistry":
        """Acquire a singleton registry instance for isolated unit tests."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(
                    slot_size=slot_size,
                    slot_count=slot_count,
                    ack_timeout_s=_env_float(
                        "NCD_VIDEO_ACK_TIMEOUT_SECONDS",
                        DEFAULT_VIDEO_ACK_TIMEOUT_SECONDS
                        if ack_timeout_s is None
                        else ack_timeout_s,
                    ),
                    allocate_timeout_s=_env_float(
                        "NCD_VIDEO_SLOT_ALLOCATE_TIMEOUT_SECONDS",
                        DEFAULT_VIDEO_SLOT_ALLOCATE_TIMEOUT_SECONDS
                        if allocate_timeout_s is None
                        else allocate_timeout_s,
                    ),
                )
            cls._refcount += 1
            return cls._instance

    @classmethod
    def release_shared_instance(cls) -> None:
        """Release one singleton test registry reference."""
        with cls._instance_lock:
            if cls._instance is None:
                return
            cls._refcount = max(0, cls._refcount - 1)
            if cls._refcount > 0:
                return
            instance = cls._instance
            cls._instance = None
        instance.close()

    @classmethod
    def reset_shared_instance_for_tests(cls) -> None:
        """Tear down the singleton test registry, if any."""
        with cls._instance_lock:
            instance = cls._instance
            cls._instance = None
            cls._refcount = 0
        if instance is not None:
            instance.close()

    @property
    def total_shared_memory_bytes(self) -> int:
        """Return the total bytes reserved in shared memory."""
        return self.slot_size * self.slot_count

    def request_payload(self) -> OpenFixedSharedSlotsModel:
        """Return the setup request payload for daemon-owned fixed shared slots."""
        return OpenFixedSharedSlotsModel(
            control_endpoint=self.control_endpoint,
            slot_size=self.slot_size,
            slot_count=self.slot_count,
        )

    def is_ready(self) -> bool:
        """Return True when the daemon has opened the shared memory session."""
        with self._condition:
            return self._ready

    def wait_until_ready(self) -> bool:
        """Block until the daemon has opened the shared-slot session."""
        deadline = time.monotonic() + self.allocate_timeout_s
        with self._condition:
            while not self._ready:
                if not self._is_healthy_locked():
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._mark_unhealthy_locked("open_timeout")
                    return False
                self._condition.wait(timeout=min(0.1, remaining))
            return True

    def is_healthy(self) -> bool:
        """Return True while the shared-slot transport is still healthy."""
        with self._condition:
            return self._is_healthy_locked()

    def ensure_healthy(self) -> None:
        """Raise when the shared-slot transport is unhealthy."""
        if self.is_healthy():
            return
        raise RuntimeError("Shared-slot transport is unhealthy")

    def allocate_slot(self) -> tuple[int, int]:
        """Reserve one free slot or fail when backpressure persists."""
        deadline = time.monotonic() + self.allocate_timeout_s
        with self._condition:
            while True:
                self._check_for_timeouts_locked()
                if not self._is_healthy_locked():
                    raise RuntimeError("Shared-slot transport is unhealthy")
                if self._ready and self._free_slots:
                    slot_id = self._free_slots.popleft()
                    return int(slot_id), int(slot_id) * self.slot_size
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if not self._ready:
                        self._mark_unhealthy_locked("open_timeout")
                        raise SharedSlotTimeout(
                            "Timed out waiting for daemon-owned shared slots to open"
                        )
                    raise SharedSlotTimeout("Timed out waiting for a free shared slot")
                self._condition.wait(timeout=min(0.1, remaining))

    def mark_in_flight(self, slot_id: int) -> int:
        """Record that the slot now backs one sent descriptor."""
        with self._condition:
            self._check_for_timeouts_locked()
            if not self._is_healthy_locked():
                raise RuntimeError("Shared-slot transport is unhealthy")
            return self._reserve_slot_for_descriptor_locked(slot_id)

    def mark_sent(self, sequence_id: int) -> None:
        """Start the credit-return timeout clock once the descriptor is on the socket."""
        with self._condition:
            self._mark_descriptor_sent_locked(sequence_id)

    def release_slot(self, shm_name: str, slot_id: int, sequence_id: int) -> bool:
        """Release one in-flight slot after a matching credit return arrives."""
        with self._condition:
            self._check_for_timeouts_locked()
            return self._apply_slot_credit_locked(
                shm_name=shm_name,
                slot_id=slot_id,
                sequence_id=sequence_id,
            )

    def rollback_enqueued_slot(self, sequence_id: int) -> None:
        """Return a slot immediately when descriptor enqueue fails."""
        with self._condition:
            self._release_sequence_locked(sequence_id)

    def notify_sender_failure(self) -> None:
        """Fail fast when a descriptor could not be written to ZMQ."""
        with self._condition:
            if not self._is_healthy_locked():
                return
            self._mark_unhealthy_locked("sender_failure")

    def get_debug_stats(self) -> dict[str, object]:
        """Return a snapshot of shared-slot occupancy and credit health."""
        with self._condition:
            return self._debug_stats_locked()

    def get_in_flight_count(self) -> int:
        """Return the number of descriptors still awaiting slot credit."""
        with self._condition:
            return len(self._in_flight)

    def get_unhealthy_reason(self) -> str | None:
        """Return the current unhealthy reason, if any."""
        with self._condition:
            return self._unhealthy_reason

    def shared_memory_view(self, offset: int, length: int) -> memoryview:
        """Return a writable view into one slot-sized shared-memory span."""
        if self._shm is None:
            raise RuntimeError("Shared-slot transport is not ready")
        return self._shm.buf[offset : offset + length]

    def close(self) -> None:
        """Stop threads and close local handles."""
        with self._condition:
            if self._closed:
                return
            self._mark_closed_locked()

        self._stop_event.set()
        self._control_thread.join(timeout=1.0)
        self._watchdog_thread.join(timeout=1.0)
        self._close_control_resources()
        self._close_shared_memory()
        self._remove_control_socket_path()

    def _should_trace_credit(self, sequence_id: int | None) -> bool:
        return (
            sequence_id is None
            or sequence_id < 10
            or sequence_id % 100 == 0
            or 1090 <= sequence_id <= 1150
        )

    def _control_listener_loop(self) -> None:
        logger.info(
            "Shared-slot control receiver started endpoint=%s",
            self.control_endpoint,
        )
        poller = zmq.Poller()
        poller.register(self._control_socket, zmq.POLLIN)

        try:
            while not self._stop_event.is_set():
                try:
                    events = dict(poller.poll(100))
                except zmq.ZMQError:
                    logger.exception(
                        "Shared-slot control receiver poll failed endpoint=%s",
                        self.control_endpoint,
                    )
                    break

                if self._control_socket not in events:
                    continue

                try:
                    message = MessageEnvelope.from_bytes(self._control_socket.recv())
                    self._process_control_message(message)
                except Exception:
                    logger.exception("Failed to process shared-slot control message")
        finally:
            log_fn = logger.info if self._stop_event.is_set() and self._closed else logger.warning
            log_fn(
                "Shared-slot control receiver exiting endpoint=%s stop_event=%s closed=%s healthy=%s",
                self.control_endpoint,
                self._stop_event.is_set(),
                self._closed,
                self._healthy,
            )

    def _process_control_message(self, message: MessageEnvelope) -> None:
        if message.command == CommandType.SHARED_SLOT_READY:
            ready = SharedSlotReadyModel(**message.payload[message.command.value])
            self._apply_ready_message(ready)
            return
        if message.command == CommandType.SHARED_SLOT_CREDIT_RETURN:
            credit = SharedSlotCreditReturn.from_dict(message.payload[message.command.value])
            self._process_slot_credit_return(credit)
            return
        logger.warning("Ignoring unexpected shared-slot control command %s", message.command)

    def _apply_ready_message(self, ready: SharedSlotReadyModel) -> None:
        try:
            shm = SharedMemory(name=ready.shm_name, create=False)
            try:
                resource_tracker.unregister(shm._name, "shared_memory")
            except Exception:
                logger.debug(
                    "Failed to unregister daemon-owned shared-memory handle %s",
                    ready.shm_name,
                    exc_info=True,
                )
        except Exception:
            logger.exception("Failed to attach daemon-owned shared memory %s", ready.shm_name)
            with self._condition:
                self._mark_unhealthy_locked("attach_failed")
            return

        with self._condition:
            if self._closed:
                shm.close()
                return
            if self._shm is not None:
                self._shm.close()
            self._shm = shm
            self.shm_name = ready.shm_name
            self.slot_size = int(ready.slot_size)
            self.slot_count = int(ready.slot_count)
            self._free_slots = deque(range(self.slot_count))
            self._ready = True
            self._condition.notify_all()

    def _watchdog_loop(self) -> None:
        while not self._stop_event.wait(0.1):
            with self._condition:
                self._check_for_timeouts_locked()

    def _check_for_timeouts_locked(self) -> None:
        if not self._is_healthy_locked():
            return
        now = time.monotonic()
        timed_out = [
            sequence_id
            for sequence_id, entry in self._in_flight.items()
            if entry.socket_sent_at is not None
            and now - entry.socket_sent_at >= self.ack_timeout_s
        ]
        if not timed_out:
            return
        self._ack_timeout_count += len(timed_out)
        last_reason: str | None = None
        for sequence_id in timed_out:
            entry = self._in_flight.get(sequence_id)
            if entry is not None:
                last_reason = (
                    f"credit_timeout(sequence_id={entry.sequence_id},slot_id={entry.slot_id})"
                )
                logger.warning(
                    "Shared-slot credit timed out shm_name=%s slot_id=%s sequence_id=%s",
                    entry.shm_name,
                    entry.slot_id,
                    entry.sequence_id,
                )
        self._mark_unhealthy_locked(last_reason or "credit_timeout", sequence_ids=timed_out)

    def _release_sequence_locked(self, sequence_id: int) -> None:
        entry = self._in_flight.pop(sequence_id, None)
        if entry is None:
            return
        self._free_slots.append(entry.slot_id)
        self._condition.notify_all()

    def _reserve_slot_for_descriptor_locked(self, slot_id: int) -> int:
        """Create in-flight tracking for a reserved slot."""
        if self.shm_name is None:
            raise RuntimeError("Shared-slot transport is not ready")
        sequence_id = self._sequence_id
        self._sequence_id += 1
        self._in_flight[sequence_id] = InFlightSlot(
            shm_name=self.shm_name,
            slot_id=int(slot_id),
            sequence_id=sequence_id,
            reserved_at=time.monotonic(),
        )
        self._max_in_flight_slot_count = max(
            self._max_in_flight_slot_count,
            len(self._in_flight),
        )
        return sequence_id

    def _mark_descriptor_sent_locked(self, sequence_id: int) -> None:
        """Start the credit timeout clock for one in-flight descriptor."""
        entry = self._in_flight.get(sequence_id)
        if entry is None or entry.socket_sent_at is not None:
            return
        self._in_flight[sequence_id] = InFlightSlot(
            shm_name=entry.shm_name,
            slot_id=entry.slot_id,
            sequence_id=entry.sequence_id,
            reserved_at=entry.reserved_at,
            socket_sent_at=time.monotonic(),
        )

    def _apply_slot_credit_locked(
        self,
        *,
        shm_name: str,
        slot_id: int,
        sequence_id: int,
    ) -> bool:
        """Apply one returned slot credit to the in-flight state."""
        entry = self._in_flight.get(sequence_id)
        if entry is None or entry.shm_name != shm_name or entry.slot_id != slot_id:
            logger.warning(
                "Ignoring stale or unknown slot credit shm_name=%s slot_id=%s sequence_id=%s",
                shm_name,
                slot_id,
                sequence_id,
            )
            return False

        self._record_credit_stats_locked(entry)
        self._release_sequence_locked(sequence_id)
        return True

    def _record_credit_stats_locked(self, entry: InFlightSlot) -> None:
        """Update credit-return latency and release metrics for one completed slot."""
        started_at = (
            entry.socket_sent_at
            if entry.socket_sent_at is not None
            else entry.reserved_at
        )
        latency_s = max(0.0, time.monotonic() - started_at)
        self._acked_sequence_count += 1
        self._last_acked_sequence_id = entry.sequence_id
        self._last_ack_latency_s = latency_s
        self._max_ack_latency_s = max(self._max_ack_latency_s, latency_s)

    def _process_slot_credit_return(self, credit: SharedSlotCreditReturn) -> None:
        """Apply one returned slot credit and emit trace logging when enabled."""
        if self._should_trace_credit(credit.sequence_id):
            logger.info(
                "Shared-slot credit received sequence_id=%s slot_id=%s shm_name=%s",
                credit.sequence_id,
                credit.slot_id,
                credit.shm_name,
            )

        applied = self.release_slot(
            credit.shm_name,
            credit.slot_id,
            credit.sequence_id,
        )

        if self._should_trace_credit(credit.sequence_id):
            stats = self.get_debug_stats()
            logger.info(
                "Shared-slot credit applied=%s sequence_id=%s slot_id=%s "
                "acked=%s inflight=%s free=%s last_ack_latency=%.3fs "
                "max_ack_latency=%.3fs healthy=%s unhealthy_reason=%s",
                applied,
                credit.sequence_id,
                credit.slot_id,
                stats["acked_sequence_count"],
                stats["in_flight_slot_count"],
                stats["free_slot_count"],
                float(stats["last_ack_latency_s"] or 0.0),
                float(stats["max_ack_latency_s"] or 0.0),
                stats["healthy"],
                stats["unhealthy_reason"],
            )

    def _is_healthy_locked(self) -> bool:
        """Return True when the registry can still accept work."""
        return self._healthy and not self._closed

    def _mark_closed_locked(self) -> None:
        """Mark the registry closed and wake any waiters."""
        self._closed = True
        self._healthy = False
        self._condition.notify_all()

    def _mark_unhealthy_locked(
        self,
        reason: str,
        *,
        sequence_ids: list[int] | None = None,
    ) -> None:
        """Transition to unhealthy state and release affected slots."""
        self._healthy = False
        self._unhealthy_reason = reason
        ids_to_release = list(self._in_flight) if sequence_ids is None else sequence_ids
        for sequence_id in ids_to_release:
            self._release_sequence_locked(sequence_id)
        self._condition.notify_all()

    def _debug_stats_locked(self) -> dict[str, object]:
        """Return a debug snapshot while the condition is already held."""
        return {
            "slot_count": self.slot_count,
            "free_slot_count": len(self._free_slots),
            "in_flight_slot_count": len(self._in_flight),
            "max_in_flight_slot_count": self._max_in_flight_slot_count,
            "acked_sequence_count": self._acked_sequence_count,
            "ack_timeout_count": self._ack_timeout_count,
            "last_acked_sequence_id": self._last_acked_sequence_id,
            "last_ack_latency_s": self._last_ack_latency_s,
            "max_ack_latency_s": self._max_ack_latency_s,
            "healthy": self._is_healthy_locked(),
            "unhealthy_reason": self._unhealthy_reason,
        }

    def _close_control_resources(self) -> None:
        """Close the producer-side control socket and its ZMQ context."""
        try:
            self._control_socket.close(0)
        except Exception:
            logger.warning("Failed to close shared-slot control socket", exc_info=True)
        try:
            self._context.term()
        except Exception:
            logger.warning("Failed to terminate shared-slot control context", exc_info=True)

    def _close_shared_memory(self) -> None:
        """Close the local attachment to the daemon-owned shared-memory region."""
        if self._shm is None:
            return
        try:
            self._shm.close()
        except Exception:
            logger.warning("Failed to close shared-memory handle", exc_info=True)
        self._shm = None

    def _remove_control_socket_path(self) -> None:
        """Remove the filesystem entry backing the control IPC endpoint."""
        try:
            self._control_socket_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning(
                "Failed to remove shared-slot control socket file %s",
                self._control_socket_path,
                exc_info=True,
            )


class SharedSlotVideoWorker:
    """Background worker that writes packets into daemon-owned shared-memory slots."""

    _instance: SharedSlotVideoWorker | None = None
    _refcount = 0
    _instance_lock = threading.Lock()

    def __init__(self, registry: SharedSlotRegistry) -> None:
        self._registry = registry
        self._queue: queue.Queue[QueuedSharedSlotPacket | None] = queue.Queue(
            maxsize=max(1, registry.slot_count)
        )
        self._error: Exception | None = None
        self._error_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="shared-slot-video-worker",
            daemon=True,
        )
        self._thread.start()

    @classmethod
    def acquire(cls, registry: SharedSlotRegistry) -> "SharedSlotVideoWorker":
        """Acquire a singleton worker instance for isolated unit tests."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(registry)
            cls._refcount += 1
            return cls._instance

    @classmethod
    def release_shared_instance(cls) -> None:
        """Release one singleton test worker reference."""
        with cls._instance_lock:
            if cls._instance is None:
                return
            cls._refcount = max(0, cls._refcount - 1)
            if cls._refcount > 0:
                return
            instance = cls._instance
            cls._instance = None
        instance.close()

    @classmethod
    def reset_shared_instance_for_tests(cls) -> None:
        """Tear down the singleton test worker, if any."""
        with cls._instance_lock:
            instance = cls._instance
            cls._instance = None
            cls._refcount = 0
        if instance is not None:
            instance.close()

    def enqueue_packet(
        self,
        *,
        packet: QueuedSharedSlotPacket,
    ) -> None:
        """Queue one complete packet for shared-memory copy and descriptor send."""
        self._ensure_running()
        if packet.packet_length > self._registry.slot_size:
            raise PacketTooLarge(
                "Packet length "
                f"{packet.packet_length} exceeds slot size {self._registry.slot_size}"
            )
        while True:
            self._ensure_running()
            try:
                self._queue.put(packet, timeout=0.1)
                return
            except queue.Full:
                continue

    def close(self) -> None:
        """Stop the worker thread."""
        try:
            self._queue.put(None, timeout=0.1)
        except queue.Full:
            pass
        self._thread.join(timeout=1.0)

    def get_debug_stats(self) -> dict[str, object]:
        """Return a snapshot of worker queue/backlog state."""
        with self._error_lock:
            worker_error = None if self._error is None else str(self._error)
        return {
            "worker_queue_qsize": self._queue.qsize(),
            "worker_queue_maxsize": self._queue.maxsize,
            "worker_thread_alive": self._thread.is_alive(),
            "worker_error": worker_error,
        }

    def is_idle(self) -> bool:
        """Return True when the worker has no queued packets left."""
        return self._queue.qsize() == 0

    def get_error(self) -> Exception | None:
        """Return the worker error, if the background thread failed."""
        with self._error_lock:
            return self._error

    def _ensure_running(self) -> None:
        self._registry.ensure_healthy()
        with self._error_lock:
            if self._error is not None:
                raise RuntimeError("Shared-slot video worker failed") from self._error
        if not self._thread.is_alive():
            raise RuntimeError("Shared-slot video worker is not running")

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    break
                try:
                    self._process_item(item)
                except Exception as exc:
                    with self._error_lock:
                        self._error = exc
                    logger.exception("Shared-slot video worker failed")
                    break
            finally:
                self._queue.task_done()

    def _process_item(self, item: QueuedSharedSlotPacket) -> None:
        slot_id, offset = self._registry.allocate_slot()
        try:
            shm_view = self._registry.shared_memory_view(offset, item.packet_length)
            try:
                header = struct.pack(
                    SHARED_RING_RECORD_HEADER_FORMAT,
                    SHARED_RING_RECORD_MAGIC,
                    len(item.metadata_bytes),
                    len(item.chunk),
                )
                header_end = SHARED_RING_RECORD_HEADER_SIZE
                metadata_end = header_end + len(item.metadata_bytes)
                shm_view[:header_end] = header
                shm_view[header_end:metadata_end] = item.metadata_bytes
                shm_view[metadata_end:item.packet_length] = item.chunk
            finally:
                shm_view.release()

            sequence_id = self._registry.mark_in_flight(slot_id)
            if self._registry.shm_name is None:
                raise RuntimeError("Shared-slot transport is not ready")
            descriptor = SharedSlotDescriptor(
                shm_name=self._registry.shm_name,
                slot_id=slot_id,
                offset=offset,
                length=item.packet_length,
                sequence_id=sequence_id,
                slot_size=self._registry.slot_size,
            )
            envelope = MessageEnvelope(
                producer_id=item.producer_id,
                command=CommandType.SHARED_SLOT_DESCRIPTOR,
                payload={
                    CommandType.SHARED_SLOT_DESCRIPTOR.value: descriptor.to_dict(),
                },
                sequence_number=sequence_id,
            )
            try:
                item.sender.enqueue_envelope(
                    envelope,
                    on_sent=lambda: self._registry.mark_sent(sequence_id),
                    on_failed_send=self._registry.notify_sender_failure,
                )
            except Exception:
                self._registry.rollback_enqueued_slot(sequence_id)
                raise
        finally:
            del item


class SharedSlotVideoTransport:
    """Producer-facing adapter over one daemon-owned shared-slot session."""

    def __init__(
        self,
        *,
        slot_size: int = DEFAULT_VIDEO_SLOT_SIZE,
        slot_count: int = DEFAULT_VIDEO_SLOT_COUNT,
        ack_timeout_s: float = DEFAULT_VIDEO_ACK_TIMEOUT_SECONDS,
        allocate_timeout_s: float = DEFAULT_VIDEO_SLOT_ALLOCATE_TIMEOUT_SECONDS,
    ) -> None:
        self._registry = SharedSlotRegistry(
            slot_size=slot_size,
            slot_count=slot_count,
            ack_timeout_s=_env_float(
                "NCD_VIDEO_ACK_TIMEOUT_SECONDS",
                ack_timeout_s,
            ),
            allocate_timeout_s=_env_float(
                "NCD_VIDEO_SLOT_ALLOCATE_TIMEOUT_SECONDS",
                allocate_timeout_s,
            ),
        )
        self._worker = SharedSlotVideoWorker(self._registry)
        self._announced = False

    @property
    def slot_size(self) -> int:
        """Return the configured fixed slot size."""
        return self._registry.slot_size

    def open_payload(self) -> OpenFixedSharedSlotsModel:
        """Return the setup request payload and mark the transport announced."""
        self._announced = True
        return self._registry.request_payload()

    def is_announced(self) -> bool:
        """Return True when setup has been announced to the daemon."""
        return self._announced

    def is_ready(self) -> bool:
        """Return True when the daemon has opened the shared-memory session."""
        return self._registry.is_ready()

    def wait_until_ready(self) -> bool:
        """Block until the daemon has opened the shared-memory session."""
        return self._registry.wait_until_ready()

    def enqueue_packet(
        self,
        *,
        producer_id: str,
        sender: ProducerChannelMessageSender,
        metadata: dict[str, str | int | None],
        chunk: bytes | bytearray | memoryview,
    ) -> None:
        """Serialize one transport packet and hand it to the background worker."""
        metadata_bytes, packet_length = build_shared_frame_packet_metadata(
            metadata, chunk
        )
        self._worker.enqueue_packet(
            packet=QueuedSharedSlotPacket(
                producer_id=producer_id,
                sender=sender,
                metadata_bytes=metadata_bytes,
                chunk=chunk,
                packet_length=packet_length,
            )
        )

    def next_sequence_number(self) -> int:
        """Reserve a channel-scoped sequence number for control messages."""
        with self._registry._condition:
            sequence_id = self._registry._sequence_id
            self._registry._sequence_id += 1
            return sequence_id

    def is_healthy(self) -> bool:
        """Return True while the transport can accept new video writes."""
        return self._registry.is_healthy()

    def notify_sender_failure(self) -> None:
        """Mark the shared-slot transport unhealthy after sender failure."""
        self._registry.notify_sender_failure()

    def wait_until_drained(self, timeout_s: float = 30.0) -> None:
        """Wait until all queued packets and in-flight credits are settled."""
        deadline = time.monotonic() + timeout_s
        last_stats: ProducerSharedRingBufferDebugStats | None = None

        while time.monotonic() < deadline:
            last_stats = self.get_stats()

            if self._worker.get_error() is not None:
                raise RuntimeError(
                    "Shared-slot transport worker failed before drain completed. "
                    f"last_stats={last_stats}"
                )

            if self._is_drained():
                return

            time.sleep(0.05)

        raise RuntimeError(
            "Timed out waiting for shared-slot transport to drain before close. "
            f"last_stats={last_stats}"
        )

    def close(self) -> None:
        """Release this channel's shared-slot runtime."""
        self._worker.close()
        self._registry.close()

    def get_stats(self) -> ProducerSharedRingBufferDebugStats:
        """Return a best-effort debug snapshot during the transport migration."""
        registry_stats = self._registry.get_debug_stats()
        worker_stats = self._worker.get_debug_stats()
        return ProducerSharedRingBufferDebugStats(
            shared_ring_buffer_name=self._registry.shm_name,
            shared_ring_buffer_size=self._registry.total_shared_memory_bytes,
            shared_ring_open=ProducerTransportTimingStats(),
            shared_ring_write=ProducerTransportTimingStats(),
            shared_ring_write_bytes=0,
            slot_count=int(registry_stats["slot_count"]),
            free_slot_count=int(registry_stats["free_slot_count"]),
            in_flight_slot_count=int(registry_stats["in_flight_slot_count"]),
            max_in_flight_slot_count=int(registry_stats["max_in_flight_slot_count"]),
            acked_sequence_count=int(registry_stats["acked_sequence_count"]),
            ack_timeout_count=int(registry_stats["ack_timeout_count"]),
            worker_queue_qsize=int(worker_stats["worker_queue_qsize"]),
            worker_queue_maxsize=int(worker_stats["worker_queue_maxsize"]),
            worker_thread_alive=bool(worker_stats["worker_thread_alive"]),
            worker_error=(
                None
                if worker_stats["worker_error"] is None
                else str(worker_stats["worker_error"])
            ),
            last_acked_sequence_id=(
                None
                if registry_stats["last_acked_sequence_id"] is None
                else int(registry_stats["last_acked_sequence_id"])
            ),
            last_ack_latency_s=(
                None
                if registry_stats["last_ack_latency_s"] is None
                else float(registry_stats["last_ack_latency_s"])
            ),
            max_ack_latency_s=float(registry_stats["max_ack_latency_s"]),
            unhealthy_reason=(
                None
                if registry_stats["unhealthy_reason"] is None
                else str(registry_stats["unhealthy_reason"])
            ),
        )

    def _is_drained(self) -> bool:
        """Return True when shutdown can proceed without queued local work."""
        return self._worker.is_idle() and self._registry.get_in_flight_count() == 0
