"""Ordered sender service for producer channels."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable

from neuracore.data_daemon.debug_profiling import observe_value, record_duration
from neuracore.data_daemon.models import CommandType

from ..shared_transport.communications_manager import (
    CommunicationsManager,
    MessageEnvelope,
)
from .models import (
    ProducerChannelMessageSenderDebugStats,
    ProducerTransportTimingStats,
    QueuedEnvelope,
)

logger = logging.getLogger(__name__)
_MAX_BATCHED_DATA_CHUNKS = 64


def _profile_label(envelope: MessageEnvelope) -> str:
    command = envelope.command.value
    payload = envelope.payload or {}

    if envelope.command == CommandType.BATCHED_JOINT_DATA:
        batch = payload.get(CommandType.BATCHED_JOINT_DATA.value, {})
        data_type = batch.get("data_type") or "unknown"
        item_count = len(batch.get("items") or [])
        return f"{command}:{data_type}:{item_count}items"

    if envelope.command == CommandType.DATA_CHUNK:
        chunk = payload.get("data_chunk", payload)
        data_type_name = (
            chunk.get("data_type_name") or chunk.get("data_type") or "unknown"
        )
        return f"{command}:{data_type_name}"

    if envelope.command == CommandType.ENVELOPE_BATCH:
        inner = list(payload.get("envelopes") or [])
        return f"{command}:{len(inner)}items"

    return command


class ProducerChannelMessageSender:
    """Ordered producer-side dispatcher for socket messages and ring writes."""

    def __init__(
        self,
        *,
        comm: CommunicationsManager,
        send_queue_maxsize: int,
        producer_id: str | None = None,
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
        self._queue_put_count = 0
        self._queue_put_total_s = 0.0
        self._queue_put_max_s = 0.0
        self._socket_send_count = 0
        self._socket_send_total_s = 0.0
        self._socket_send_max_s = 0.0
        self._send_error_count = 0
        self._sequence_cv = threading.Condition()
        self._enqueue_lock = threading.Lock()
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
        producer_id: str | None = None,
        sequence_number: int | None = None,
        on_sent: Callable[[], None] | None = None,
        on_failed_send: Callable[[], None] | None = None,
    ) -> int:
        """Enqueue a command message for ordered socket delivery."""
        with self._enqueue_lock:
            envelope = self._build_envelope(
                producer_id=producer_id or self._producer_id,
                command=command,
                payload=payload,
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
        """Return a lightweight debug snapshot for this sender."""
        with self._sequence_cv:
            sender_thread = self._sender_thread
            return ProducerChannelMessageSenderDebugStats(
                send_queue_qsize=self._send_queue.qsize(),
                send_queue_maxsize=self._send_queue.maxsize,
                last_enqueued_sequence_number=self._last_enqueued_sequence_number,
                last_socket_sent_sequence_number=self._last_socket_sent_sequence_number,
                pending_sequence_count=(
                    self._last_enqueued_sequence_number
                    - self._last_socket_sent_sequence_number
                ),
                sender_thread_alive=(
                    sender_thread.is_alive() if sender_thread is not None else False
                ),
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
                shared_memory_dispatch=ProducerTransportTimingStats(),
                send_error_count=self._send_error_count,
                last_send_error=(
                    None if self._sender_error is None else str(self._sender_error)
                ),
            )

    def _build_envelope(
        self,
        producer_id: str | None,
        command: CommandType,
        payload: dict | None = None,
        *,
        sequence_number: int | None = None,
    ) -> MessageEnvelope:
        """Reserve a sequence number and build a transport envelope."""
        if not producer_id:
            raise ValueError("producer_id is required to build a transport envelope")
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
            producer_id=producer_id,
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
        label = _profile_label(envelope)
        qsize_before_put = self._send_queue.qsize()
        started = time.monotonic()
        self._send_queue.put(
            QueuedEnvelope(
                envelope=envelope,
                on_sent=on_sent,
                on_failed_send=on_failed_send,
            )
        )
        record_duration(
            "producer.queue_put",
            label,
            time.monotonic() - started,
        )
        queue_put_elapsed = time.monotonic() - started
        with self._sequence_cv:
            self._queue_put_count += 1
            self._queue_put_total_s += queue_put_elapsed
            self._queue_put_max_s = max(self._queue_put_max_s, queue_put_elapsed)
        observe_value(
            "producer.queue_qsize_before_put",
            label,
            float(qsize_before_put),
        )
        observe_value(
            "producer.queue_capacity",
            label,
            float(self._send_queue.maxsize),
        )

    def _sender_loop(self) -> None:
        """Serialize socket messages on one thread."""
        pending_item: QueuedEnvelope | None = None
        stop_after_batch = False
        while True:
            item = pending_item if pending_item is not None else self._send_queue.get()
            pending_item = None
            batch: list[QueuedEnvelope] = [item] if item is not None else []
            try:
                if item is None:
                    break

                try:
                    batch, pending_item, stop_after_batch = self._collect_batch(item)
                    envelope = self._build_transport_envelope(batch)
                    label = _profile_label(envelope)
                    qsize_after_get = self._send_queue.qsize()
                    started = time.monotonic()
                    self._comm.send_message(envelope)
                    socket_send_elapsed = time.monotonic() - started
                    with self._sequence_cv:
                        self._socket_send_count += 1
                        self._socket_send_total_s += socket_send_elapsed
                        self._socket_send_max_s = max(
                            self._socket_send_max_s,
                            socket_send_elapsed,
                        )
                    record_duration(
                        "producer.socket_send",
                        label,
                        socket_send_elapsed,
                    )
                    observe_value(
                        "producer.queue_qsize_after_get",
                        label,
                        float(qsize_after_get),
                    )
                    highest_sequence = 0
                    for queued_item in batch:
                        if queued_item.on_sent is not None:
                            try:
                                queued_item.on_sent()
                            except Exception:
                                logger.exception("Sender success callback crashed")
                        if queued_item.envelope.sequence_number is not None:
                            highest_sequence = max(
                                highest_sequence,
                                queued_item.envelope.sequence_number,
                            )
                    if highest_sequence > 0:
                        with self._sequence_cv:
                            if highest_sequence > self._last_socket_sent_sequence_number:
                                self._last_socket_sent_sequence_number = highest_sequence
                            self._sequence_cv.notify_all()
                except Exception as exc:
                    for failed_item in batch:
                        if failed_item.on_failed_send is not None:
                            try:
                                failed_item.on_failed_send()
                            except Exception:
                                logger.exception("Sender failure callback crashed")
                    with self._sequence_cv:
                        self._sender_error = exc
                        self._send_error_count += 1
                        self._sequence_cv.notify_all()
                    logger.exception("Send failed")
                    break
            finally:
                self._send_queue.task_done()
                for _ in range(max(0, len(batch) - 1)):
                    self._send_queue.task_done()
                if stop_after_batch:
                    self._send_queue.task_done()
            if stop_after_batch:
                break

        with self._sequence_cv:
            self._sequence_cv.notify_all()

    def _collect_batch(
        self,
        first_item: QueuedEnvelope,
    ) -> tuple[list[QueuedEnvelope], QueuedEnvelope | None, bool]:
        """Gather additional queue items that can share one socket send."""
        batch = [first_item]
        envelope = first_item.envelope
        if envelope.command != CommandType.DATA_CHUNK:
            return batch, None, False

        while len(batch) < _MAX_BATCHED_DATA_CHUNKS:
            try:
                next_item = self._send_queue.get_nowait()
            except queue.Empty:
                return batch, None, False
            if next_item is None:
                return batch, None, True
            if next_item.envelope.command != CommandType.DATA_CHUNK:
                return batch, next_item, False
            batch.append(next_item)
        return batch, None, False

    def _build_transport_envelope(
        self,
        batch: list[QueuedEnvelope],
    ) -> MessageEnvelope:
        """Return the on-wire envelope for one or more queued items."""
        if len(batch) == 1:
            return batch[0].envelope
        return MessageEnvelope(
            producer_id=None,
            command=CommandType.ENVELOPE_BATCH,
            payload={
                "envelopes": [queued_item.envelope.to_dict() for queued_item in batch],
            },
            sequence_number=None,
        )
