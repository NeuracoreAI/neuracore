"""Daemon-side shared-slot transport helpers."""

from __future__ import annotations

import logging
import time
import uuid
from multiprocessing.shared_memory import SharedMemory
from typing import Protocol

import zmq

from neuracore.data_daemon.models import (
    CommandType,
    MessageEnvelope,
    OpenFixedSharedSlotsModel,
    SharedMemoryChunkMetadata,
    SharedSlotCreditReturn,
    SharedSlotDescriptor,
    SharedSlotReadyModel,
)

from ..consumer.bridge_chunk_spool import BridgeChunkSpool, ChunkSpoolRef
from ..consumer.models import ChannelState
from .communications_manager import CommunicationsManager
from .models import SharedSlotTransportResult
from .shared_memory_budget import SharedMemoryBudget
from .shared_slot_transport import parse_shared_frame_packet_view

logger = logging.getLogger(__name__)


class _AckSenderSocket(Protocol):
    def close(self, linger: int = 0) -> None: ...

    def connect(self, addr: str) -> None: ...

    def send(self, data: bytes) -> None: ...

    def setsockopt(self, option: int, value: int) -> None: ...


class SharedSlotDaemonHandler:
    """Own daemon-side shared-slot transport mechanics."""

    def __init__(self, comm: CommunicationsManager) -> None:
        """Initialize daemon-side caches for shared memory and ACK sockets."""
        self._comm = comm
        self._shared_memory_cache: dict[str, SharedMemory] = {}
        self._ack_sender_sockets: dict[str, _AckSenderSocket] = {}
        self._shared_memory_budget = SharedMemoryBudget()

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
    ) -> None:
        """Open daemon-owned fixed shared slots for one channel."""
        request = OpenFixedSharedSlotsModel(**payload)

        if channel.shared_slot.shm_name is not None:
            self._cleanup_previous_shared_slots(channel, request.control_endpoint)

        shm_name = f"neuracore-slots-{uuid.uuid4().hex}-{int(time.time() * 1000)}"

        reservation = self._shared_memory_budget.reserve(
            shm_name=shm_name,
            slot_size=request.slot_size,
            requested_slot_count=request.slot_count,
        )

        shm: SharedMemory | None = None

        try:
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

        except Exception:
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
            raise

        logger.info(
            "Opened daemon-owned fixed shared slots for producer_id=%s "
            "shm_name=%s slot_size=%d requested_slot_count=%d "
            "actual_slot_count=%d total_shared_memory_bytes=%d "
            "max_in_flight_packets=%d",
            channel.producer_id,
            shm_name,
            request.slot_size,
            request.slot_count,
            reservation.slot_count,
            reservation.allocated_bytes,
            reservation.slot_count,
        )

    def handle_descriptor(
        self,
        channel: ChannelState,
        payload: dict,
        chunk_spool: BridgeChunkSpool,
    ) -> SharedSlotTransportResult:
        """Spool, credit, and parse one shared-slot descriptor."""
        descriptor = SharedSlotDescriptor.from_dict(payload)
        try:
            metadata_dict, chunk_spool_ref = self._spool_shared_slot_packet(
                descriptor, chunk_spool
            )
        except Exception:
            logger.exception(
                "Shared-slot copy failed " "producer_id=%s sequence_id=%s slot_id=%s",
                channel.producer_id,
                descriptor.sequence_id,
                descriptor.slot_id,
            )
            raise
        self._send_slot_credit_return(channel, descriptor)

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
            shm = SharedMemory(name=descriptor.shm_name, create=False)

            self._shared_memory_cache[descriptor.shm_name] = shm

        return shm.buf[descriptor.offset : descriptor.offset + descriptor.length]

    def _send_ready_message(
        self,
        *,
        endpoint: str,
        ready: SharedSlotReadyModel,
    ) -> None:
        """Send one daemon-owned shared-slot ready message."""
        socket_obj = self._ack_sender_sockets.get(endpoint)
        if socket_obj is None:
            socket_obj = self._comm._context.socket(zmq.PUSH)
            socket_obj.setsockopt(zmq.LINGER, 0)
            socket_obj.connect(endpoint)
            self._ack_sender_sockets[endpoint] = socket_obj
        socket_obj.send(
            MessageEnvelope(
                producer_id=None,
                command=CommandType.SHARED_SLOT_READY,
                payload={CommandType.SHARED_SLOT_READY.value: ready.model_dump()},
            ).to_bytes()
        )

    def _send_slot_credit_return(
        self,
        channel: ChannelState,
        descriptor: SharedSlotDescriptor,
    ) -> None:
        """Return one writable slot credit immediately after shared-memory copy-out."""
        endpoint = channel.shared_slot.control_endpoint
        if not endpoint:
            raise RuntimeError("Shared-slot control endpoint is not available")
        socket_obj = self._ack_sender_sockets.get(endpoint)
        if socket_obj is None:
            socket_obj = self._comm._context.socket(zmq.PUSH)
            socket_obj.setsockopt(zmq.LINGER, 0)
            socket_obj.connect(endpoint)
            self._ack_sender_sockets[endpoint] = socket_obj
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
