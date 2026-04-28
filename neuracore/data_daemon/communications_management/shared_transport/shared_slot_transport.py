"""Fixed shared-slot transport for producer-side video packets."""

from __future__ import annotations

import json
import logging
import os
import queue
import struct
import threading
import time

from neuracore.data_daemon.const import (
    DEFAULT_VIDEO_ACK_TIMEOUT_SECONDS,
    DEFAULT_VIDEO_SLOT_ALLOCATE_TIMEOUT_SECONDS,
    DEFAULT_VIDEO_SLOT_COUNT,
    DEFAULT_VIDEO_SLOT_SIZE,
    SHARED_MEMORY_RECORD_HEADER_FORMAT,
    SHARED_MEMORY_RECORD_HEADER_SIZE,
    SHARED_MEMORY_RECORD_MAGIC,
)
from neuracore.data_daemon.models import (
    CommandType,
    MessageEnvelope,
    OpenFixedSharedSlotsModel,
    SharedSlotDescriptor,
)

from ..producer.models import (
    ProducerSharedMemoryDebugStats,
    ProducerTransportTimingStats,
)
from ..producer.producer_channel_message_sender import ProducerChannelMessageSender
from .models import QueuedSharedSlotPacket
from .registry import SharedSlotRegistry

logger = logging.getLogger(__name__)


def _int_snapshot_value(snapshot: dict[str, object], key: str) -> int:
    """Return an integer debug snapshot value with a zero default."""
    value = snapshot.get(key)
    if value is None:
        return 0
    if isinstance(value, (bool, int)):
        return int(value)
    if isinstance(value, (str, bytes, bytearray)):
        return int(value)
    raise TypeError(f"Unsupported int snapshot value for {key}: {type(value)!r}")


def _float_snapshot_value(snapshot: dict[str, object], key: str) -> float:
    """Return a float debug snapshot value with a zero default."""
    value = snapshot.get(key)
    if value is None:
        return 0.0
    if isinstance(value, (bool, int, float)):
        return float(value)
    if isinstance(value, (str, bytes, bytearray)):
        return float(value)
    raise TypeError(f"Unsupported float snapshot value for {key}: {type(value)!r}")


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


def build_shared_frame_packet(
    metadata: dict[str, str | int | None],
    chunk: bytes | bytearray | memoryview,
) -> bytes:
    """Build the self-describing packet stored in one shared slot."""
    metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    payload = bytes(chunk)
    return (
        struct.pack(
            SHARED_MEMORY_RECORD_HEADER_FORMAT,
            SHARED_MEMORY_RECORD_MAGIC,
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
    packet_length = SHARED_MEMORY_RECORD_HEADER_SIZE + len(metadata_bytes) + chunk_len
    return metadata_bytes, packet_length


def parse_shared_frame_packet(packet: bytes) -> tuple[dict[str, object], bytes]:
    """Parse one self-describing packet copied out of a shared slot."""
    metadata, chunk_start, chunk_end = parse_shared_frame_packet_view(
        memoryview(packet)
    )
    return metadata, packet[chunk_start:chunk_end]


def parse_shared_frame_packet_view(
    packet: memoryview,
) -> tuple[dict[str, object], int, int]:
    """Parse one shared-slot packet view without copying the payload chunk."""
    if len(packet) < SHARED_MEMORY_RECORD_HEADER_SIZE:
        raise ValueError("Shared-slot packet shorter than record header")
    magic, metadata_len, chunk_len = struct.unpack(
        SHARED_MEMORY_RECORD_HEADER_FORMAT,
        packet[:SHARED_MEMORY_RECORD_HEADER_SIZE],
    )
    if magic != SHARED_MEMORY_RECORD_MAGIC:
        raise ValueError("Shared-slot packet missing shared record magic")
    expected = SHARED_MEMORY_RECORD_HEADER_SIZE + metadata_len + chunk_len
    if len(packet) < expected:
        raise ValueError("Shared-slot packet shorter than declared lengths")
    if len(packet) > expected:
        raise ValueError("Shared-slot packet contains trailing bytes")
    metadata_start = SHARED_MEMORY_RECORD_HEADER_SIZE
    chunk_start = metadata_start + metadata_len
    metadata = json.loads(packet[metadata_start:chunk_start].tobytes().decode("utf-8"))
    return metadata, chunk_start, expected


class SharedSlotVideoWorker:
    """Background worker that writes packets into daemon-owned shared-memory slots."""

    _instance: SharedSlotVideoWorker | None = None
    _refcount = 0
    _instance_lock = threading.Lock()

    def __init__(self, registry: SharedSlotRegistry) -> None:
        """Start the background worker for one shared-slot registry."""
        self._registry = registry
        self._queue: queue.Queue[QueuedSharedSlotPacket | None] = queue.Queue(
            maxsize=max(1, registry.slot_count)
        )
        self._active_items = 0
        self._active_items_lock = threading.Lock()
        self._error: Exception | None = None
        self._error_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="shared-slot-video-worker",
            daemon=True,
        )
        self._thread.start()

    @classmethod
    def acquire(cls, registry: SharedSlotRegistry) -> SharedSlotVideoWorker:
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

    def is_idle(self) -> bool:
        """Return True when the worker has no queued packets left."""
        with self._active_items_lock:
            return self._queue.qsize() == 0 and self._active_items == 0

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
                with self._active_items_lock:
                    self._active_items += 1
                try:
                    self._process_item(item)
                except Exception as exc:
                    with self._error_lock:
                        self._error = exc
                    logger.exception("Shared-slot video worker failed")
                    break
                finally:
                    with self._active_items_lock:
                        self._active_items = max(0, self._active_items - 1)
            finally:
                self._queue.task_done()

    def _process_item(self, item: QueuedSharedSlotPacket) -> None:
        slot_id, offset = self._registry.allocate_slot()
        try:
            shm_view = self._registry.shared_memory_view(offset, item.packet_length)
            try:
                header = struct.pack(
                    SHARED_MEMORY_RECORD_HEADER_FORMAT,
                    SHARED_MEMORY_RECORD_MAGIC,
                    len(item.metadata_bytes),
                    len(item.chunk),
                )
                header_end = SHARED_MEMORY_RECORD_HEADER_SIZE
                metadata_end = header_end + len(item.metadata_bytes)
                shm_view[:header_end] = header
                shm_view[header_end:metadata_end] = item.metadata_bytes
                shm_view[metadata_end : item.packet_length] = item.chunk
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
        """Initialize a producer-facing shared-slot transport.

        Args:
            slot_size: Bytes available in each daemon-owned shared-memory slot.
            slot_count: Number of shared-memory slots to provision.
            ack_timeout_s: Timeout for receiving slot credits after send.
            allocate_timeout_s: Timeout while waiting for a writable slot.
        """
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
        return self._registry.next_sequence_number()

    def is_healthy(self) -> bool:
        """Return True while the transport can accept new video writes."""
        return self._registry.is_healthy()

    def notify_sender_failure(self) -> None:
        """Mark the shared-slot transport unhealthy after sender failure."""
        self._registry.notify_sender_failure()

    def finish_recording_session(self) -> None:
        """Reset transport state so the next recording opens a fresh session."""
        self._registry.reset_session()
        self._announced = False

    def wait_until_drained(self, timeout_s: float = 30.0) -> None:
        """Wait until all queued packets and in-flight credits are settled."""
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            if self._worker.get_error() is not None:
                raise RuntimeError(
                    "Shared-slot transport worker failed before drain completed"
                )

            if self._is_drained():
                return

            time.sleep(0.05)

        raise RuntimeError(
            "Timed out waiting for shared-slot transport to drain before close"
        )

    def wait_until_payload_handed_off(self, timeout_s: float = 30.0) -> None:
        """Wait until queued payloads have been copied and descriptor-enqueued."""
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            if self._worker.get_error() is not None:
                raise RuntimeError(
                    "Shared-slot transport worker failed before payload "
                    "handoff completed"
                )

            if self._worker.is_idle():
                return

            time.sleep(0.01)

        raise RuntimeError(
            "Timed out waiting for shared-slot payload handoff before stop"
        )

    def close(self) -> None:
        """Release this channel's shared-slot runtime."""
        self._worker.close()
        self._registry.close()

    def get_stats(self) -> ProducerSharedMemoryDebugStats:
        """Return a best-effort debug snapshot for shared-memory transport."""
        snapshot = self._registry.debug_snapshot()
        return ProducerSharedMemoryDebugStats(
            shared_memory_name=self._registry.shm_name,
            shared_memory_size=self._registry.total_shared_memory_bytes,
            shared_memory_open=ProducerTransportTimingStats(),
            shared_memory_write=ProducerTransportTimingStats(),
            shared_memory_write_bytes=0,
            slot_count=_int_snapshot_value(snapshot, "slot_count"),
            free_slot_count=_int_snapshot_value(snapshot, "free_slot_count"),
            in_flight_slot_count=_int_snapshot_value(snapshot, "in_flight_slot_count"),
            max_in_flight_slot_count=_int_snapshot_value(
                snapshot, "max_in_flight_slot_count"
            ),
            acked_sequence_count=_int_snapshot_value(snapshot, "acked_sequence_count"),
            ack_timeout_count=_int_snapshot_value(snapshot, "ack_timeout_count"),
            worker_queue_qsize=self._worker._queue.qsize(),
            worker_queue_maxsize=self._worker._queue.maxsize,
            worker_thread_alive=self._worker._thread.is_alive(),
            worker_error=(
                None
                if self._worker.get_error() is None
                else str(self._worker.get_error())
            ),
            last_acked_sequence_id=(
                None
                if snapshot.get("last_acked_sequence_id") is None
                else _int_snapshot_value(snapshot, "last_acked_sequence_id")
            ),
            last_ack_latency_s=(
                None
                if snapshot.get("last_ack_latency_s") is None
                else _float_snapshot_value(snapshot, "last_ack_latency_s")
            ),
            max_ack_latency_s=_float_snapshot_value(snapshot, "max_ack_latency_s"),
            unhealthy_reason=(
                None
                if snapshot.get("unhealthy_reason") is None
                else str(snapshot.get("unhealthy_reason"))
            ),
        )

    def _is_drained(self) -> bool:
        """Return True when shutdown can proceed without queued local work."""
        return self._worker.is_idle() and self._registry.get_in_flight_count() == 0
