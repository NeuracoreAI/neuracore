"""Ordered sender service for producer channels."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable

from neuracore.data_daemon.models import CommandType

from ..shared_transport.communications_manager import (
    CommunicationsManager,
    MessageEnvelope,
)
from .models import QueuedEnvelope
from .producer_transport_debug_helper import ProducerTransportDebugHelper
from .producer_transport_debug_models import ProducerChannelMessageSenderDebugStats

logger = logging.getLogger(__name__)


class ProducerChannelMessageSender:
    """Ordered producer-side dispatcher for socket messages and ring writes."""

    def __init__(
        self,
        *,
        producer_id: str,
        comm: CommunicationsManager,
        send_queue_maxsize: int,
    ) -> None:
        """Initialize ordered dispatch state for socket messages."""
        self._producer_id = producer_id
        self._comm = comm
        self._send_queue: queue.Queue[QueuedEnvelope | None] = queue.Queue(
            maxsize=send_queue_maxsize
        )
        self._sender_thread: threading.Thread | None = threading.Thread(
            target=self._sender_loop,
            name="producer-channel-sender",
            daemon=True,
        )
        self._next_sequence_number = 1
        self._last_enqueued_sequence_number = 0
        self._last_socket_sent_sequence_number = 0
        self._sender_error: Exception | None = None
        self._sequence_cv = threading.Condition()
        self._enqueue_lock = threading.Lock()
        self._debug_helper = ProducerTransportDebugHelper()
        self._sender_thread.start()

    @property
    def producer_id(self) -> str:
        """Return the sender's producer/channel identifier."""
        return self._producer_id

    @property
    def queue(self) -> queue.Queue[QueuedEnvelope | None]:
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

    def send(
        self,
        command: CommandType,
        payload: dict | None = None,
        *,
        sequence_number: int | None = None,
        on_sent: Callable[[], None] | None = None,
        on_failed_send: Callable[[], None] | None = None,
    ) -> int:
        """Enqueue a command message for ordered socket delivery."""
        with self._enqueue_lock:
            envelope = self._build_envelope(
                command,
                payload,
                sequence_number=sequence_number,
            )
            self._enqueue_envelope_locked(
                envelope,
                on_sent=on_sent,
                on_failed_send=on_failed_send,
            )
            return int(envelope.sequence_number or 0)

    def enqueue_envelope(
        self,
        envelope: MessageEnvelope,
        *,
        on_sent: Callable[[], None] | None = None,
        on_failed_send: Callable[[], None] | None = None,
    ) -> None:
        """Enqueue a prebuilt envelope for ordered socket delivery."""
        with self._enqueue_lock:
            self._enqueue_envelope_locked(
                envelope,
                on_sent=on_sent,
                on_failed_send=on_failed_send,
            )

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
                if self._sender_error is not None:
                    return False
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
        *,
        sequence_number: int | None = None,
    ) -> MessageEnvelope:
        """Reserve a sequence number and build a transport envelope."""
        with self._sequence_cv:
            if sequence_number is None:
                sequence_number = self._next_sequence_number
                self._next_sequence_number += 1
            else:
                sequence_number = int(sequence_number)
                if sequence_number >= self._next_sequence_number:
                    self._next_sequence_number = sequence_number + 1
            if sequence_number > self._last_enqueued_sequence_number:
                self._last_enqueued_sequence_number = sequence_number
        return MessageEnvelope(
            producer_id=self._producer_id,
            command=command,
            payload=payload or {},
            sequence_number=sequence_number,
        )

    def _enqueue_envelope_locked(
        self,
        envelope: MessageEnvelope,
        *,
        on_sent: Callable[[], None] | None = None,
        on_failed_send: Callable[[], None] | None = None,
    ) -> None:
        with self._sequence_cv:
            if self._sender_error is not None:
                raise RuntimeError("Sender thread is no longer healthy")
        started_at = self._debug_helper.start_timer()
        self._send_queue.put(
            QueuedEnvelope(
                envelope=envelope,
                on_sent=on_sent,
                on_failed_send=on_failed_send,
            )
        )
        self._debug_helper.record_queue_put(started_at)

    def _sender_loop(self) -> None:
        """Serialize socket messages on one thread."""
        while True:
            item = self._send_queue.get()
            try:
                if item is None:
                    break

                try:
                    started_at = self._debug_helper.start_timer()
                    self._comm.send_message(item.envelope)
                    self._debug_helper.record_socket_send(started_at)
                    if item.on_sent is not None:
                        try:
                            item.on_sent()
                        except Exception:
                            logger.exception("Sender success callback crashed")
                    if item.envelope.sequence_number is not None:
                        with self._sequence_cv:
                            if (
                                item.envelope.sequence_number
                                > self._last_socket_sent_sequence_number
                            ):
                                self._last_socket_sent_sequence_number = (
                                    item.envelope.sequence_number
                                )
                            self._sequence_cv.notify_all()
                except Exception as exc:
                    self._debug_helper.record_send_error(exc)
                    if item.on_failed_send is not None:
                        try:
                            item.on_failed_send()
                        except Exception:
                            logger.exception("Sender failure callback crashed")
                    with self._sequence_cv:
                        self._sender_error = exc
                        self._sequence_cv.notify_all()
                    logger.exception("Send failed")
                    break
            finally:
                self._send_queue.task_done()

        with self._sequence_cv:
            self._sequence_cv.notify_all()
