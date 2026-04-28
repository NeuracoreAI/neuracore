"""Ordered sender service for producer channels."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

from neuracore.data_daemon.communications_management.communications_manager import (
    CommunicationsManager,
    MessageEnvelope,
)
from neuracore.data_daemon.models import CommandType

from .global_producer_sender_executor import get_global_producer_sender_executor
from .producer_transport_debug_helper import ProducerTransportDebugHelper
from .producer_transport_debug_models import ProducerChannelMessageSenderDebugStats

logger = logging.getLogger(__name__)


@dataclass
class QueuedSharedRingWrite:
    """A shared-ring data write to be processed by the sender thread."""

    metadata: dict[str, str | int | None]
    chunk: bytes | bytearray | memoryview


class ProducerChannelMessageSender:
    """Ordered dispatcher using a shared thread pool."""

    def __init__(
        self,
        *,
        producer_id: str,
        comm: CommunicationsManager,
        write_shared_ring_record: Callable[
            [dict[str, str | int | None], bytes | bytearray | memoryview],
            None,
        ],
    ) -> None:
        """Initialize the sender.

        Args:
            producer_id: The producer ID.
            comm: the communications manager.
            write_shared_ring_record: the shared ring record writer.
        """
        self._producer_id = producer_id
        self._comm = comm
        self._write_shared_ring_record = write_shared_ring_record
        self._executor = get_global_producer_sender_executor()

        self._next_sequence_number = 1
        self._last_enqueued_sequence_number = 0
        self._last_socket_sent_sequence_number = 0

        self._sequence_cv = threading.Condition()
        self._debug_helper = ProducerTransportDebugHelper()

    def close(self) -> None:
        """Cleanup sender state."""
        with self._sequence_cv:
            self._sequence_cv.notify_all()

    def send(self, command: CommandType, payload: dict | None = None) -> int:
        """Enqueue a command message for ordered socket delivery."""
        envelope = self._build_envelope(command, payload)
        started_at = self._debug_helper.start_timer()

        self._executor.submit(
            self._producer_id, self._dispatch_item, envelope, started_at
        )
        return int(envelope.sequence_number or 0)

    def enqueue_shared_ring_write(
        self,
        *,
        metadata: dict[str, str | int | None],
        chunk: bytes | bytearray | memoryview,
    ) -> None:
        """Enqueue a shared-ring write behind previously queued work."""
        item = QueuedSharedRingWrite(metadata=metadata, chunk=chunk)
        started_at = self._debug_helper.start_timer()
        self._executor.submit(self._producer_id, self._dispatch_item, item, started_at)

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
            return self._sequence_cv.wait_for(
                lambda: self._last_socket_sent_sequence_number >= sequence_number,
            )

    def get_stats(self) -> ProducerChannelMessageSenderDebugStats:
        """Return a lightweight snapshot of ordered sender state."""
        with self._sequence_cv:
            return self._debug_helper.sender_stats(
                send_queue_qsize=0,  # Queue is now managed by Executor
                send_queue_maxsize=0,
                last_enqueued_sequence_number=self._last_enqueued_sequence_number,
                last_socket_sent_sequence_number=self._last_socket_sent_sequence_number,
                sender_thread_alive=True,  # Executor handles threads
            )

    def _build_envelope(
        self,
        command: CommandType,
        payload: dict | None = None,
    ) -> MessageEnvelope:
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

    def _dispatch_item(
        self, item: MessageEnvelope | QueuedSharedRingWrite, started_at: float | None
    ) -> None:
        """The task executed by the thread pool."""
        try:
            if isinstance(item, QueuedSharedRingWrite):
                self._write_shared_ring_record(item.metadata, item.chunk)
                self._debug_helper.record_shared_ring_dispatch(started_at)
            else:
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
            logger.warning("Send failed in pool: %s", exc)
