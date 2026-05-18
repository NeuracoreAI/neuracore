"""Daemon-side shared-slot transport helpers."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from multiprocessing.shared_memory import SharedMemory
from typing import Protocol

import zmq

from neuracore.data_daemon.const import SHARED_SLOT_SHM_PREFIX
from neuracore.data_daemon.helpers import env_float
from neuracore.data_daemon.models import (
    CommandType,
    MessageEnvelope,
    OpenFixedSharedSlotsModel,
    SharedMemoryChunkMetadata,
    SharedSlotCreditReturn,
    SharedSlotDescriptor,
    SharedSlotOpenFailedModel,
    SharedSlotReadyModel,
)

from ..consumer.bridge_chunk_spool import BridgeChunkSpool, ChunkSpoolRef
from ..consumer.models import ChannelState
from .communications_manager import CommunicationsManager
from .models import SharedSlotTransportResult
from .shared_memory_budget import SharedMemoryBudget
from .shared_slot_transport import parse_shared_frame_packet_view

logger = logging.getLogger(__name__)

SHARED_SLOT_REOPEN_DRAIN_TIMEOUT_S = 1


class SharedSlotDescriptorAbandoned(RuntimeError):
    """Raised when a queued descriptor belongs to an abandoned slot session."""


class _AckSenderSocket(Protocol):
    def close(self, linger: int = 0) -> None: ...

    def connect(self, addr: str) -> None: ...

    def send(self, data: bytes) -> None: ...

    def setsockopt(self, option: int, value: int) -> None: ...


class SharedSlotDaemonHandler:
    """Own daemon-side shared-slot transport mechanics."""

    def __init__(
        self,
        comm: CommunicationsManager,
        reopen_drain_timeout_s: float = SHARED_SLOT_REOPEN_DRAIN_TIMEOUT_S,
    ) -> None:
        """Initialize daemon-side caches for shared memory and ACK sockets."""
        self._comm = comm
        self._shared_memory_cache: dict[str, SharedMemory] = {}
        self._ack_sender_sockets: dict[str, _AckSenderSocket] = {}
        self._shared_memory_budget = SharedMemoryBudget()
        self._reopen_drain_timeout_s = env_float(
            "NCD_SHARED_SLOT_REOPEN_DRAIN_TIMEOUT_S",
            float(reopen_drain_timeout_s),
        )
        self._descriptor_delay_once_s = env_float(
            "NCD_TEST_SHARED_SLOT_DESCRIPTOR_DELAY_ONCE_S",
            0.0,
        )
        self._descriptor_delay_lock = threading.Lock()
        self._pending_by_shm: dict[str, set[tuple[str, int]]] = {}
        self._abandoned_descriptors: set[tuple[str, str, int]] = set()
        self._pending_condition = threading.Condition()

    def _cleanup_previous_shared_slots(
        self, channel: ChannelState, control_endpoint: str | None = None
    ) -> None:
        """Clean up a producer's previous shared-slot resources, if any."""
        previous_shm_name = channel.shared_slot.shm_name
        previous_endpoint = channel.shared_slot.control_endpoint

        if previous_shm_name:
            old = self._shared_memory_cache.pop(previous_shm_name, None)
            if old is not None:
                try:
                    old.close()
                finally:
                    try:
                        old.unlink()
                    except FileNotFoundError:
                        pass

            self._shared_memory_budget.release(previous_shm_name)

        if previous_endpoint and previous_endpoint != control_endpoint:
            old_socket = self._ack_sender_sockets.pop(previous_endpoint, None)
            if old_socket is not None:
                old_socket.close(0)

        channel.shared_slot.reset()

    def handle_open(
        self,
        channel: ChannelState,
        payload: dict,
        on_abandoned_sequences: Callable[[str, list[int]], None] | None = None,
    ) -> None:
        """Open daemon-owned fixed shared slots for one channel."""
        request = OpenFixedSharedSlotsModel(**payload)

        if channel.shared_slot.shm_name is not None:
            abandoned_sequences = self._wait_or_abandon_previous_session(channel)
            if abandoned_sequences and on_abandoned_sequences is not None:
                on_abandoned_sequences(channel.producer_id, abandoned_sequences)
            self._cleanup_previous_shared_slots(channel, request.control_endpoint)

        shm_name = f"{SHARED_SLOT_SHM_PREFIX}{uuid.uuid4().hex[:16]}"

        reservation = None
        shm: SharedMemory | None = None

        try:
            reservation = self._shared_memory_budget.reserve(
                shm_name=shm_name,
                slot_size=request.slot_size,
                requested_slot_count=request.slot_count,
            )

            shm = SharedMemory(
                name=shm_name,
                create=True,
                size=reservation.allocated_bytes,
            )

            self._shared_memory_cache[shm_name] = shm

            channel.mark_shared_slot_transport_open(
                control_endpoint=request.control_endpoint,
                shm_name=shm_name,
            )

            self._send_ready_message(
                endpoint=request.control_endpoint,
                ready=SharedSlotReadyModel(
                    shm_name=shm_name,
                    slot_size=request.slot_size,
                    slot_count=reservation.slot_count,
                ),
            )

        except Exception as exc:
            error_message = str(exc) or exc.__class__.__name__
            self._shared_memory_cache.pop(shm_name, None)

            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    logger.warning(
                        "Failed to clean up shared memory after open failure %s",
                        shm_name,
                        exc_info=True,
                    )

            self._shared_memory_budget.rollback(shm_name)
            channel.shared_slot.reset()
            if request.control_endpoint:
                try:
                    self._send_open_failed_message(
                        endpoint=request.control_endpoint,
                        failure=SharedSlotOpenFailedModel(error_message=error_message),
                    )
                except Exception:
                    logger.warning(
                        "Failed to send shared-slot open failure "
                        "producer_id=%s endpoint=%s",
                        channel.producer_id,
                        request.control_endpoint,
                        exc_info=True,
                    )
            raise

    def handle_descriptor(
        self,
        channel: ChannelState,
        payload: dict,
        chunk_spool: BridgeChunkSpool,
    ) -> SharedSlotTransportResult:
        """Spool, credit, and parse one shared-slot descriptor."""
        self._delay_pending_descriptor_processing(channel, payload)
        descriptor = SharedSlotDescriptor.from_dict(payload)
        self._raise_if_descriptor_abandoned(channel, descriptor)
        spool_failed = False
        try:
            metadata_dict, chunk_spool_ref = self._spool_shared_slot_packet(
                descriptor, chunk_spool
            )
        except Exception:
            spool_failed = True
            logger.exception(
                "Shared-slot copy failed " "producer_id=%s sequence_id=%s slot_id=%s",
                channel.producer_id,
                descriptor.sequence_id,
                descriptor.slot_id,
            )
            raise
        finally:
            try:
                self._send_slot_credit_return(channel, descriptor)
            except Exception:
                if spool_failed:
                    logger.exception(
                        "Failed to return shared-slot credit after copy failure "
                        "producer_id=%s sequence_id=%s slot_id=%s",
                        channel.producer_id,
                        descriptor.sequence_id,
                        descriptor.slot_id,
                    )
                else:
                    raise

        channel.mark_shared_slot_descriptor_seen(
            shm_name=descriptor.shm_name,
        )

        chunk_metadata = SharedMemoryChunkMetadata.from_dict(metadata_dict)
        return SharedSlotTransportResult(
            descriptor=descriptor,
            chunk_metadata=chunk_metadata,
            chunk_spool_ref=chunk_spool_ref,
            trace_id=chunk_metadata.trace_id,
            trace_metadata=chunk_metadata.trace_metadata,
        )

    def _delay_pending_descriptor_processing(
        self,
        channel: ChannelState,
        payload: dict,
    ) -> None:
        """Optional one-shot descriptor delay used by integration tests."""
        with self._descriptor_delay_lock:
            delay_s = self._descriptor_delay_once_s
            self._descriptor_delay_once_s = 0.0
        if delay_s <= 0.0:
            return

        logger.warning(
            "Delaying shared-slot descriptor processing for test "
            "producer_id=%s shm_name=%s sequence_id=%s delay=%.3fs",
            channel.producer_id,
            payload.get("shm_name"),
            payload.get("sequence_id"),
            delay_s,
        )
        time.sleep(delay_s)

    def mark_descriptor_pending(
        self,
        channel: ChannelState,
        payload: dict,
    ) -> SharedSlotDescriptor:
        """Track a shared-slot descriptor until the spool worker has handled it."""
        descriptor = SharedSlotDescriptor.from_dict(payload)
        with self._pending_condition:
            descriptor_key = self._descriptor_key(channel, descriptor)
            if descriptor_key in self._abandoned_descriptors:
                raise SharedSlotDescriptorAbandoned(
                    "Shared-slot descriptor belongs to an abandoned session "
                    f"producer_id={channel.producer_id} "
                    f"shm_name={descriptor.shm_name} "
                    f"sequence_id={descriptor.sequence_id}"
                )
            self._pending_by_shm.setdefault(descriptor.shm_name, set()).add(
                (channel.producer_id, descriptor.sequence_id)
            )
            self._pending_condition.notify_all()
        return descriptor

    def mark_descriptor_completed(
        self,
        producer_id: str,
        descriptor: SharedSlotDescriptor,
    ) -> None:
        """Clear daemon-side pending tracking for one shared-slot descriptor."""
        with self._pending_condition:
            pending = self._pending_by_shm.get(descriptor.shm_name)
            if pending is not None:
                pending.discard((producer_id, descriptor.sequence_id))
                if not pending:
                    self._pending_by_shm.pop(descriptor.shm_name, None)
            self._abandoned_descriptors.discard(
                (producer_id, descriptor.shm_name, descriptor.sequence_id)
            )
            self._pending_condition.notify_all()

    def _descriptor_key(
        self,
        channel: ChannelState,
        descriptor: SharedSlotDescriptor,
    ) -> tuple[str, str, int]:
        return (channel.producer_id, descriptor.shm_name, descriptor.sequence_id)

    def _raise_if_descriptor_abandoned(
        self,
        channel: ChannelState,
        descriptor: SharedSlotDescriptor,
    ) -> None:
        with self._pending_condition:
            descriptor_key = self._descriptor_key(channel, descriptor)
            if descriptor_key not in self._abandoned_descriptors:
                return
        raise SharedSlotDescriptorAbandoned(
            "Skipping abandoned shared-slot descriptor "
            f"producer_id={channel.producer_id} "
            f"shm_name={descriptor.shm_name} "
            f"sequence_id={descriptor.sequence_id}"
        )

    def _wait_or_abandon_previous_session(self, channel: ChannelState) -> list[int]:
        """Wait for old descriptors before reusing a producer's shared-slot state."""
        shm_name = channel.shared_slot.shm_name
        if shm_name is None:
            return []

        deadline = time.monotonic() + self._reopen_drain_timeout_s
        with self._pending_condition:
            while True:
                pending = self._pending_by_shm.get(shm_name, set())
                producer_pending = {
                    sequence_id
                    for producer_id, sequence_id in pending
                    if producer_id == channel.producer_id
                }
                if not producer_pending:
                    return []

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    abandoned = sorted(producer_pending)
                    for sequence_id in abandoned:
                        pending.discard((channel.producer_id, sequence_id))
                        self._abandoned_descriptors.add(
                            (channel.producer_id, shm_name, sequence_id)
                        )
                    if not pending:
                        self._pending_by_shm.pop(shm_name, None)
                    self._pending_condition.notify_all()
                    break
                self._pending_condition.wait(timeout=min(0.1, remaining))

        logger.warning(
            "Abandoning stalled shared-slot session before reopen "
            "producer_id=%s shm_name=%s pending_sequences=%s "
            "timeout=%.3fs",
            channel.producer_id,
            shm_name,
            abandoned,
            self._reopen_drain_timeout_s,
        )
        return abandoned

    def cleanup_channel_resources(self, channel: ChannelState) -> None:
        """Close daemon-side shared-slot resources associated with one channel."""
        shm_name = channel.shared_slot.shm_name
        if shm_name:
            shm = self._shared_memory_cache.pop(shm_name, None)

            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    logger.warning(
                        "Failed to close cached shared memory %s",
                        shm_name,
                        exc_info=True,
                    )
                finally:
                    self._shared_memory_budget.release(shm_name)
            else:
                self._shared_memory_budget.release(shm_name)

        endpoint = channel.shared_slot.control_endpoint
        if endpoint:
            socket_obj = self._ack_sender_sockets.pop(endpoint, None)
            if socket_obj is not None:
                try:
                    socket_obj.close(0)
                except Exception:
                    logger.warning(
                        "Failed to close shared-slot ACK sender %s",
                        endpoint,
                        exc_info=True,
                    )

        channel.shared_slot.reset()

    def close(self) -> None:
        """Close all daemon-side shared-slot handles during shutdown."""
        for socket_obj in self._ack_sender_sockets.values():
            try:
                socket_obj.close(0)
            except Exception:
                logger.warning("Failed to close shared-slot ACK sender", exc_info=True)
        self._ack_sender_sockets.clear()

        for shm_name, shm in list(self._shared_memory_cache.items()):
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                logger.warning("Failed to close cached shared memory", exc_info=True)
            finally:
                self._shared_memory_budget.release(shm_name)

        self._shared_memory_cache.clear()

    def _spool_shared_slot_packet(
        self,
        descriptor: SharedSlotDescriptor,
        chunk_spool: BridgeChunkSpool,
    ) -> tuple[dict[str, object], ChunkSpoolRef]:
        """Copy one payload chunk from shared memory into the disk-backed spool."""
        packet_view = self._shared_slot_packet_view(descriptor)
        try:
            metadata, chunk_start, chunk_end = parse_shared_frame_packet_view(
                packet_view
            )
            chunk_view = packet_view[chunk_start:chunk_end]
            try:
                chunk_spool_ref = chunk_spool.append(chunk_view)
            finally:
                chunk_view.release()
            return metadata, chunk_spool_ref
        finally:
            packet_view.release()

    def _shared_slot_packet_view(self, descriptor: SharedSlotDescriptor) -> memoryview:
        """Return one packet view out of cached shared memory."""
        shm = self._shared_memory_cache.get(descriptor.shm_name)
        if shm is None:
            raise RuntimeError(
                "Shared-slot shared memory handle missing from daemon cache. "
                "Expected handle to be cached during handle_open() "
                f"for shm_name={descriptor.shm_name}"
            )

        return shm.buf[descriptor.offset : descriptor.offset + descriptor.length]

    def _send_ready_message(
        self,
        endpoint: str,
        ready: SharedSlotReadyModel,
    ) -> None:
        """Send one daemon-owned shared-slot ready message."""
        socket_obj = self._get_or_create_ack_sender_socket(endpoint)
        socket_obj.send(
            MessageEnvelope(
                producer_id=None,
                command=CommandType.SHARED_SLOT_READY,
                payload={CommandType.SHARED_SLOT_READY.value: ready.model_dump()},
            ).to_bytes()
        )

    def _send_open_failed_message(
        self,
        endpoint: str,
        failure: SharedSlotOpenFailedModel,
    ) -> None:
        """Send one daemon-owned shared-slot open failure message."""
        socket_obj = self._get_or_create_ack_sender_socket(endpoint)
        socket_obj.send(
            MessageEnvelope(
                producer_id=None,
                command=CommandType.SHARED_SLOT_OPEN_FAILED,
                payload={
                    CommandType.SHARED_SLOT_OPEN_FAILED.value: failure.model_dump()
                },
            ).to_bytes()
        )

    def _get_or_create_ack_sender_socket(self, endpoint: str) -> _AckSenderSocket:
        """Return a cached PUSH socket for one producer control endpoint."""
        socket_obj = self._ack_sender_sockets.get(endpoint)
        if socket_obj is None:
            socket_obj = self._comm._context.socket(zmq.PUSH)
            socket_obj.setsockopt(zmq.LINGER, 0)
            socket_obj.connect(endpoint)
            self._ack_sender_sockets[endpoint] = socket_obj
        return socket_obj

    def _send_slot_credit_return(
        self,
        channel: ChannelState,
        descriptor: SharedSlotDescriptor,
    ) -> None:
        """Return one writable slot credit immediately after shared-memory copy-out."""
        endpoint = channel.shared_slot.control_endpoint
        if not endpoint:
            raise RuntimeError("Shared-slot control endpoint is not available")
        socket_obj = self._get_or_create_ack_sender_socket(endpoint)
        credit = SharedSlotCreditReturn(
            shm_name=descriptor.shm_name,
            slot_id=descriptor.slot_id,
            sequence_id=descriptor.sequence_id,
        )
        try:
            socket_obj.send(
                MessageEnvelope(
                    producer_id=None,
                    command=CommandType.SHARED_SLOT_CREDIT_RETURN,
                    payload={
                        CommandType.SHARED_SLOT_CREDIT_RETURN.value: credit.to_dict()
                    },
                ).to_bytes()
            )
        except Exception:
            logger.exception(
                "Failed to return shared-slot credit producer_id=%s sequence_id=%s",
                channel.producer_id,
                descriptor.sequence_id,
            )
