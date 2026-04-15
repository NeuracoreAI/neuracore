"""Ordered sender service for producer channels."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass

from neuracore.data_daemon.communications_management.communications_manager import (
    CommunicationsManager,
    MessageEnvelope,
)
from neuracore.data_daemon.models import CommandType

from .producer_transport_debug_helper import ProducerTransportDebugHelper
from .producer_transport_debug_models import ProducerChannelMessageSenderDebugStats

logger = logging.getLogger(__name__)


@dataclass
class QueuedSharedRingWrite:
    """A shared-ring data write to be processed by the sender thread."""

    metadata: dict[str, str | int | None]
    chunk: bytes | bytearray | memoryview


class ProducerChannelMessageSender:
    """Ordered producer-side dispatcher for socket messages and ring writes."""

    def __init__(
        self,
        *,
        producer_id: str,
        comm: CommunicationsManager,
        send_queue_maxsize: int,
        write_shared_ring_record: Callable[
            [dict[str, str | int | None], bytes | bytearray | memoryview],
            None,
        ],
    ) -> None:
        """Initialize ordered dispatch state for socket messages and ring writes."""
        self._producer_id = producer_id
        self._comm = comm
        self._write_shared_ring_record = write_shared_ring_record
        self._send_queue: queue.Queue[
            MessageEnvelope | QueuedSharedRingWrite | None
        ] = queue.Queue(maxsize=send_queue_maxsize)
        self._sender_thread: threading.Thread | None = threading.Thread(
            target=self._sender_loop,
            name="producer-channel-sender",
            daemon=True,
        )
        self._next_sequence_number = 1
        self._last_enqueued_sequence_number = 0
        self._last_socket_sent_sequence_number = 0
        self._sequence_cv = threading.Condition()
        self._enqueue_lock = threading.Lock()
        self._debug_helper = ProducerTransportDebugHelper()
        self._sender_thread.start()

    @property
    def queue(
        self,
    ) -> queue.Queue[MessageEnvelope | QueuedSharedRingWrite | None]:
        """Expose the underlying send queue for compatibility/testing."""
        return self._send_queue

    def close(self, *, join_timeout_s: float = 2.0) -> None:
        """Stop the sender thread and release queue waiters."""
        self._send_queue.put(None)
        if self._sender_thread is not None:
            self._sender_thread.join(timeout=join_timeout_s)
            self._sender_thread = None
        with self._sequence_cv:
            self._sequence_cv.notify_all()

    def send(self, command: CommandType, payload: dict | None = None) -> int:
        """Enqueue a command message for ordered socket delivery."""
        with self._enqueue_lock:
            envelope = self._build_envelope(command, payload)
            started_at = self._debug_helper.start_timer()
            self._send_queue.put(envelope)
            self._debug_helper.record_queue_put(started_at)
            return int(envelope.sequence_number or 0)

    def enqueue_shared_ring_write(
        self,
        *,
        metadata: dict[str, str | int | None],
        chunk: bytes | bytearray | memoryview,
    ) -> None:
        """Enqueue a shared-ring write behind previously queued work."""
        started_at = self._debug_helper.start_timer()
        self._send_queue.put(QueuedSharedRingWrite(metadata=metadata, chunk=chunk))
        self._debug_helper.record_queue_put(started_at)

    def get_last_sent_sequence_number(self) -> int:
        """Return the most recent sequence number successfully sent on the socket."""
        with self._sequence_cv:
            return self._last_socket_sent_sequence_number

    def get_last_enqueued_sequence_number(self) -> int:
        """Return the most recent sequence number enqueued for the sender thread."""
        with self._sequence_cv:
            return self._last_enqueued_sequence_number

    def wait_until_sequence_sent(self, sequence_number: int) -> bool:
        """Block until the sender thread has sent up to `sequence_number`."""
        if sequence_number <= 0:
            return True
        with self._sequence_cv:
            while self._last_socket_sent_sequence_number < sequence_number:
                sender_thread = self._sender_thread
                if sender_thread is None or not sender_thread.is_alive():
                    return False
                self._sequence_cv.wait()
            return True

    def get_stats(self) -> ProducerChannelMessageSenderDebugStats:
        """Return a lightweight snapshot of ordered sender state."""
        with self._sequence_cv:
            last_enqueued_sequence_number = self._last_enqueued_sequence_number
            last_socket_sent_sequence_number = self._last_socket_sent_sequence_number

        sender_thread = self._sender_thread
        return self._debug_helper.sender_stats(
            send_queue_qsize=self._send_queue.qsize(),
            send_queue_maxsize=self._send_queue.maxsize,
            last_enqueued_sequence_number=last_enqueued_sequence_number,
            last_socket_sent_sequence_number=last_socket_sent_sequence_number,
            sender_thread_alive=(
                sender_thread.is_alive() if sender_thread is not None else False
            ),
        )

    def _build_envelope(
        self,
        command: CommandType,
        payload: dict | None = None,
    ) -> MessageEnvelope:
        """Reserve a sequence number and build a transport envelope."""
        with self._sequence_cv:
            sequence_number = self._next_sequence_number
            self._next_sequence_number += 1
            self._last_enqueued_sequence_number = sequence_number
        return MessageEnvelope(
            producer_id=self._producer_id,
            command=command,
            payload=payload or {},
            sequence_number=sequence_number,
        )

    def _sender_loop(self) -> None:
        """Serialize shared-ring writes and socket messages on one thread."""
        while True:
            item = self._send_queue.get()
            try:
                if item is None:
                    break

                try:
                    if isinstance(item, QueuedSharedRingWrite):
                        started_at = self._debug_helper.start_timer()
                        self._write_shared_ring_record(item.metadata, item.chunk)
                        self._debug_helper.record_shared_ring_dispatch(started_at)
                    else:
                        started_at = self._debug_helper.start_timer()
                        self._comm.send_message(item)
                        self._debug_helper.record_socket_send(started_at)
                        if item.sequence_number is not None:
                            with self._sequence_cv:
                                if (
                                    item.sequence_number
                                    > self._last_socket_sent_sequence_number
                                ):
                                    self._last_socket_sent_sequence_number = (
                                        item.sequence_number
                                    )
                                self._sequence_cv.notify_all()
                except Exception as exc:
                    self._debug_helper.record_send_error(exc)
                    logger.warning("Send failed: %s", exc)
            finally:
                self._send_queue.task_done()

        with self._sequence_cv:
            self._sequence_cv.notify_all()
