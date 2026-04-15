"""Typed debug models for producer transport stats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProducerTransportTimingStats:
    """Timing/counter summary for one transport operation category."""

    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0

    @property
    def avg_s(self) -> float:
        """Return the average observed duration."""
        return self.total_s / self.count if self.count else 0.0


@dataclass(frozen=True)
class ProducerChannelMessageSenderDebugStats:
    """Debug stats for the ordered sender service."""

    send_queue_qsize: int
    send_queue_maxsize: int
    last_enqueued_sequence_number: int
    last_socket_sent_sequence_number: int
    pending_sequence_count: int
    sender_thread_alive: bool
    queue_put: ProducerTransportTimingStats
    socket_send: ProducerTransportTimingStats
    shared_ring_dispatch: ProducerTransportTimingStats
    send_error_count: int
    last_send_error: str | None

    def to_dict(self) -> dict[str, object]:
        """Serialize to the legacy flat dict shape."""
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
            "shared_ring_dispatch_count": self.shared_ring_dispatch.count,
            "shared_ring_dispatch_total_s": self.shared_ring_dispatch.total_s,
            "shared_ring_dispatch_avg_s": self.shared_ring_dispatch.avg_s,
            "shared_ring_dispatch_max_s": self.shared_ring_dispatch.max_s,
            "send_error_count": self.send_error_count,
            "last_send_error": self.last_send_error,
        }


@dataclass(frozen=True)
class ProducerSharedRingBufferDebugStats:
    """Debug stats for the shared ring buffer transport."""

    shared_ring_buffer_name: str | None
    shared_ring_buffer_size: int
    shared_ring_open: ProducerTransportTimingStats
    shared_ring_write: ProducerTransportTimingStats
    shared_ring_write_bytes: int

    def to_dict(self) -> dict[str, object]:
        """Serialize to the legacy flat dict shape."""
        return {
            "shared_ring_buffer_name": self.shared_ring_buffer_name,
            "shared_ring_buffer_size": self.shared_ring_buffer_size,
            "shared_ring_open_count": self.shared_ring_open.count,
            "shared_ring_open_total_s": self.shared_ring_open.total_s,
            "shared_ring_open_avg_s": self.shared_ring_open.avg_s,
            "shared_ring_open_max_s": self.shared_ring_open.max_s,
            "shared_ring_write_count": self.shared_ring_write.count,
            "shared_ring_write_bytes": self.shared_ring_write_bytes,
            "shared_ring_write_total_s": self.shared_ring_write.total_s,
            "shared_ring_write_avg_s": self.shared_ring_write.avg_s,
            "shared_ring_write_max_s": self.shared_ring_write.max_s,
        }


@dataclass(frozen=True)
class ProducerTransportDebugStats:
    """Top-level producer transport debug model."""

    channel_id: str
    recording_id: str | None
    trace_id: str | None
    chunk_size: int
    heartbeat_thread_alive: bool
    shared_ring: ProducerSharedRingBufferDebugStats
    message_sender: ProducerChannelMessageSenderDebugStats

    def to_dict(self) -> dict[str, object]:
        """Serialize to the legacy flat dict shape."""
        payload: dict[str, object] = {
            "channel_id": self.channel_id,
            "recording_id": self.recording_id,
            "trace_id": self.trace_id,
            "chunk_size": self.chunk_size,
            "heartbeat_thread_alive": self.heartbeat_thread_alive,
        }
        payload.update(self.shared_ring.to_dict())
        payload.update(self.message_sender.to_dict())
        return payload

    def __getitem__(self, key: str) -> object:
        """Provide dict-like access for callers during migration."""
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Provide dict-like access with a default for callers."""
        return self.to_dict().get(key, default)
