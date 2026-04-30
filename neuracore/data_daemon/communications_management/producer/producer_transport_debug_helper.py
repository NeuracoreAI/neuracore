"""Debug-only transport stats helpers for producer channel services."""

from __future__ import annotations

import threading
import time

from neuracore.data_daemon.helpers import is_debug_mode

from .producer_transport_debug_models import (
    ProducerChannelMessageSenderDebugStats,
    ProducerSharedMemoryDebugStats,
    ProducerTransportTimingStats,
)


class ProducerTransportDebugHelper:
    """Collect transport timings and counters only when debug mode is enabled."""

    def __init__(self, *, enabled: bool | None = None) -> None:
        """Initialize debug counters and optional debug-mode override."""
        self._enabled = is_debug_mode() if enabled is None else enabled
        self._lock = threading.Lock()
        self._queue_put_count = 0
        self._queue_put_total_s = 0.0
        self._queue_put_max_s = 0.0
        self._socket_send_count = 0
        self._socket_send_total_s = 0.0
        self._socket_send_max_s = 0.0
        self._shared_memory_dispatch_count = 0
        self._shared_memory_dispatch_total_s = 0.0
        self._shared_memory_dispatch_max_s = 0.0
        self._shared_memory_open_count = 0
        self._shared_memory_open_total_s = 0.0
        self._shared_memory_open_max_s = 0.0
        self._shared_memory_write_count = 0
        self._shared_memory_write_bytes = 0
        self._shared_memory_write_total_s = 0.0
        self._shared_memory_write_max_s = 0.0
        self._send_error_count = 0
        self._last_send_error: str | None = None

    @property
    def enabled(self) -> bool:
        """Return True when debug stats collection is active."""
        return self._enabled

    def start_timer(self) -> float | None:
        """Start a timing sample when debug stats are enabled."""
        if not self._enabled:
            return None
        return time.perf_counter()

    def record_queue_put(self, started_at: float | None) -> None:
        """Record queue put timing."""
        elapsed_s = self._elapsed_from(started_at)
        if elapsed_s is None:
            return
        with self._lock:
            self._queue_put_count += 1
            self._queue_put_total_s += elapsed_s
            self._queue_put_max_s = max(self._queue_put_max_s, elapsed_s)

    def record_socket_send(self, started_at: float | None) -> None:
        """Record socket send timing."""
        elapsed_s = self._elapsed_from(started_at)
        if elapsed_s is None:
            return
        with self._lock:
            self._socket_send_count += 1
            self._socket_send_total_s += elapsed_s
            self._socket_send_max_s = max(self._socket_send_max_s, elapsed_s)

    def record_shared_memory_dispatch(self, started_at: float | None) -> None:
        """Record shared-memory dispatch timing from the sender thread."""
        elapsed_s = self._elapsed_from(started_at)
        if elapsed_s is None:
            return
        with self._lock:
            self._shared_memory_dispatch_count += 1
            self._shared_memory_dispatch_total_s += elapsed_s
            self._shared_memory_dispatch_max_s = max(
                self._shared_memory_dispatch_max_s,
                elapsed_s,
            )

    def record_shared_memory_open(self, started_at: float | None) -> None:
        """Record shared-memory open timing."""
        elapsed_s = self._elapsed_from(started_at)
        if elapsed_s is None:
            return
        with self._lock:
            self._shared_memory_open_count += 1
            self._shared_memory_open_total_s += elapsed_s
            self._shared_memory_open_max_s = max(
                self._shared_memory_open_max_s,
                elapsed_s,
            )

    def record_shared_memory_write(
        self,
        *,
        started_at: float | None,
        bytes_written: int,
    ) -> None:
        """Record shared-memory write timing and payload size."""
        elapsed_s = self._elapsed_from(started_at)
        if elapsed_s is None:
            return
        with self._lock:
            self._shared_memory_write_count += 1
            self._shared_memory_write_bytes += bytes_written
            self._shared_memory_write_total_s += elapsed_s
            self._shared_memory_write_max_s = max(
                self._shared_memory_write_max_s,
                elapsed_s,
            )

    def record_send_error(self, exc: Exception) -> None:
        """Record a transport send error in debug mode."""
        if not self._enabled:
            return
        with self._lock:
            self._send_error_count += 1
            self._last_send_error = str(exc)

    def sender_stats(
        self,
        *,
        send_queue_qsize: int,
        send_queue_maxsize: int,
        last_enqueued_sequence_number: int,
        last_socket_sent_sequence_number: int,
        sender_thread_alive: bool,
    ) -> ProducerChannelMessageSenderDebugStats:
        """Return sender stats, with debug timings when enabled."""
        pending_sequence_count = max(
            0,
            last_enqueued_sequence_number - last_socket_sent_sequence_number,
        )
        with self._lock:
            return ProducerChannelMessageSenderDebugStats(
                send_queue_qsize=send_queue_qsize,
                send_queue_maxsize=send_queue_maxsize,
                last_enqueued_sequence_number=last_enqueued_sequence_number,
                last_socket_sent_sequence_number=last_socket_sent_sequence_number,
                pending_sequence_count=pending_sequence_count,
                sender_thread_alive=sender_thread_alive,
                queue_put=ProducerTransportTimingStats(
                    count=self._queue_put_count,
                    total_s=self._queue_put_total_s,
                    max_s=self._queue_put_max_s,
                ),
                socket_send=ProducerTransportTimingStats(
                    count=self._socket_send_count,
                    total_s=self._socket_send_total_s,
                    max_s=self._socket_send_max_s,
                ),
                shared_memory_dispatch=ProducerTransportTimingStats(
                    count=self._shared_memory_dispatch_count,
                    total_s=self._shared_memory_dispatch_total_s,
                    max_s=self._shared_memory_dispatch_max_s,
                ),
                send_error_count=self._send_error_count,
                last_send_error=self._last_send_error,
            )

    def shared_memory_stats(
        self,
        *,
        shared_memory_name: str | None,
        shared_memory_size: int,
    ) -> ProducerSharedMemoryDebugStats:
        """Return shared-memory stats, with debug timings when enabled."""
        with self._lock:
            return ProducerSharedMemoryDebugStats(
                shared_memory_name=shared_memory_name,
                shared_memory_size=shared_memory_size,
                shared_memory_open=ProducerTransportTimingStats(
                    count=self._shared_memory_open_count,
                    total_s=self._shared_memory_open_total_s,
                    max_s=self._shared_memory_open_max_s,
                ),
                shared_memory_write=ProducerTransportTimingStats(
                    count=self._shared_memory_write_count,
                    total_s=self._shared_memory_write_total_s,
                    max_s=self._shared_memory_write_max_s,
                ),
                shared_memory_write_bytes=self._shared_memory_write_bytes,
            )

    def _elapsed_from(self, started_at: float | None) -> float | None:
        if started_at is None:
            return None
        return time.perf_counter() - started_at
