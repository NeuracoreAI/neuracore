"""Producer-side transport message and debug data models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from neuracore.data_daemon.models import MessageEnvelope


@dataclass
class QueuedEnvelope:
    """A socket message plus optional failure callback."""

    envelope: MessageEnvelope
    on_sent: Callable[[], None] | None = None
    on_failed_send: Callable[[], None] | None = None


@dataclass(frozen=True)
class ProducerTransportTimingStats:
    """Aggregate timing metrics for one transport operation."""

    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0

    @property
    def avg_s(self) -> float:
        """Return the average observed duration in seconds."""
        return self.total_s / self.count if self.count else 0.0


@dataclass(frozen=True)
class ProducerChannelMessageSenderDebugStats:
    """Debug snapshot for the producer socket sender thread."""

    send_queue_qsize: int
    send_queue_maxsize: int
    last_enqueued_sequence_number: int
    last_socket_sent_sequence_number: int
    pending_sequence_count: int
    sender_thread_alive: bool
    queue_put: ProducerTransportTimingStats
    socket_send: ProducerTransportTimingStats
    shared_memory_dispatch: ProducerTransportTimingStats
    send_error_count: int
    last_send_error: str | None

    def to_dict(self) -> dict[str, object]:
        """Serialize sender debug statistics to a plain dictionary."""
        return {
            "send_queue_qsize": self.send_queue_qsize,
            "send_queue_maxsize": self.send_queue_maxsize,
            "last_enqueued_sequence_number": self.last_enqueued_sequence_number,
            "last_socket_sent_sequence_number": self.last_socket_sent_sequence_number,
            "pending_sequence_count": self.pending_sequence_count,
            "sender_thread_alive": self.sender_thread_alive,
            "queue_put_count": self.queue_put.count,
            "queue_put_total_s": self.queue_put.total_s,
            "queue_put_avg_s": self.queue_put.avg_s,
            "queue_put_max_s": self.queue_put.max_s,
            "socket_send_count": self.socket_send.count,
            "socket_send_total_s": self.socket_send.total_s,
            "socket_send_avg_s": self.socket_send.avg_s,
            "socket_send_max_s": self.socket_send.max_s,
            "shared_memory_dispatch_count": self.shared_memory_dispatch.count,
            "shared_memory_dispatch_total_s": self.shared_memory_dispatch.total_s,
            "shared_memory_dispatch_avg_s": self.shared_memory_dispatch.avg_s,
            "shared_memory_dispatch_max_s": self.shared_memory_dispatch.max_s,
            "send_error_count": self.send_error_count,
            "last_send_error": self.last_send_error,
        }


@dataclass(frozen=True)
class ProducerSharedMemoryDebugStats:
    """Debug snapshot for producer shared-memory transport state."""

    shared_memory_name: str | None
    shared_memory_size: int
    shared_memory_open: ProducerTransportTimingStats
    shared_memory_write: ProducerTransportTimingStats
    shared_memory_write_bytes: int
    slot_count: int = 0
    free_slot_count: int = 0
    in_flight_slot_count: int = 0
    max_in_flight_slot_count: int = 0
    acked_sequence_count: int = 0
    ack_timeout_count: int = 0
    worker_queue_qsize: int = 0
    worker_queue_maxsize: int = 0
    worker_thread_alive: bool = False
    worker_error: str | None = None
    last_acked_sequence_id: int | None = None
    last_ack_latency_s: float | None = None
    max_ack_latency_s: float = 0.0
    unhealthy_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize shared-memory debug statistics to a plain dictionary."""
        return {
            "shared_memory_name": self.shared_memory_name,
            "shared_memory_size": self.shared_memory_size,
            "shared_memory_open_count": self.shared_memory_open.count,
            "shared_memory_open_total_s": self.shared_memory_open.total_s,
            "shared_memory_open_avg_s": self.shared_memory_open.avg_s,
            "shared_memory_open_max_s": self.shared_memory_open.max_s,
            "shared_memory_write_count": self.shared_memory_write.count,
            "shared_memory_write_bytes": self.shared_memory_write_bytes,
            "shared_memory_write_total_s": self.shared_memory_write.total_s,
            "shared_memory_write_avg_s": self.shared_memory_write.avg_s,
            "shared_memory_write_max_s": self.shared_memory_write.max_s,
            "slot_count": self.slot_count,
            "free_slot_count": self.free_slot_count,
            "in_flight_slot_count": self.in_flight_slot_count,
            "max_in_flight_slot_count": self.max_in_flight_slot_count,
            "acked_sequence_count": self.acked_sequence_count,
            "ack_timeout_count": self.ack_timeout_count,
            "worker_queue_qsize": self.worker_queue_qsize,
            "worker_queue_maxsize": self.worker_queue_maxsize,
            "worker_thread_alive": self.worker_thread_alive,
            "worker_error": self.worker_error,
            "last_acked_sequence_id": self.last_acked_sequence_id,
            "last_ack_latency_s": self.last_ack_latency_s,
            "max_ack_latency_s": self.max_ack_latency_s,
            "unhealthy_reason": self.unhealthy_reason,
        }


@dataclass(frozen=True)
class ProducerTransportDebugStats:
    """Combined producer transport debug statistics."""

    channel_id: str
    recording_id: str | None
    trace_id: str | None
    chunk_size: int
    heartbeat_thread_alive: bool
    shared_memory: ProducerSharedMemoryDebugStats
    message_sender: ProducerChannelMessageSenderDebugStats

    def to_dict(self) -> dict[str, object]:
        """Serialize all transport debug statistics to a plain dictionary."""
        payload: dict[str, object] = {
            "channel_id": self.channel_id,
            "recording_id": self.recording_id,
            "trace_id": self.trace_id,
            "chunk_size": self.chunk_size,
            "heartbeat_thread_alive": self.heartbeat_thread_alive,
        }
        payload.update(self.shared_memory.to_dict())
        payload.update(self.message_sender.to_dict())
        return payload

    def __getitem__(self, key: str) -> object:
        """Return one serialized debug field by key."""
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Return one serialized debug field with a fallback default."""
        return self.to_dict().get(key, default)
