"""Daemon-side shared-slot transport helpers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING

import zmq

from neuracore.data_daemon.models import (
    OpenFixedSharedSlotsModel,
    SharedSlotCreditReturn,
    SharedRingChunkMetadata,
    SharedSlotDescriptor,
    SharedSlotReadyModel,
    TraceTransportMetadata,
    MessageEnvelope,
    CommandType,
)

from .communications_manager import CommunicationsManager
from .shared_slot_transport import parse_shared_frame_packet

if TYPE_CHECKING:
    from neuracore.data_daemon.communications_management.data_bridge import (
        ChannelState,
    )

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SharedSlotTransportResult:
    """Parsed result of one shared-slot descriptor."""

    descriptor: SharedSlotDescriptor
    chunk_metadata: SharedRingChunkMetadata
    chunk_data: bytes
    trace_id: str
    trace_metadata: TraceTransportMetadata | None

class SharedSlotDaemonHandler:
    """Own daemon-side shared-slot transport mechanics."""

    def __init__(self, comm: CommunicationsManager) -> None:
        self._comm = comm
        self._shared_memory_cache: dict[str, SharedMemory] = {}
        self._ack_sender_sockets: dict[str, object] = {}

    def _cleanup_previous_shared_slots(
            self, 
            channel: ChannelState, 
            control_endpoint: str | None = None
            ) -> None:
        """"Cleanup previous shared slot if needed."""
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
        if previous_endpoint and previous_endpoint != control_endpoint:
            old_socket = self._ack_sender_sockets.pop(previous_endpoint, None)
            if old_socket is not None:
                old_socket.close(0)

    def handle_open(
        self,
        channel: ChannelState,
        payload: dict,
    ) -> None:
        """Open daemon-owned fixed shared slots for one channel."""
        request = OpenFixedSharedSlotsModel(**payload)
        shm_name = f"neuracore-slots-{channel.producer_id}-{int(time.time() * 1000)}"
        shm = SharedMemory(
            name=shm_name,
            create=True,
            size=request.slot_size * request.slot_count,
        )
        try:
            # So multiprocessing doesn't think it owns the memory cleanup
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            logger.debug(
                "Failed to unregister daemon-owned shared memory %s",
                shm_name,
                exc_info=True,
            )

        if channel.shared_slot.shm_name is not None:
            self._cleanup_previous_shared_slots(channel, request.control_endpoint)

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
                slot_count=request.slot_count,
            ),
        )
        logger.debug(
            "Opened daemon-owned fixed shared slots for producer_id=%s slot_size=%d slot_count=%d "
            "total_shared_memory_bytes=%d max_in_flight_packets=%d",
            channel.producer_id,
            request.slot_size,
            request.slot_count,
            request.slot_size * request.slot_count,
            request.slot_count,
        )

    def handle_descriptor(
        self,
        channel: ChannelState,
        payload: dict,
    ) -> SharedSlotTransportResult:
        """Copy, credit, parse, and account for one shared-slot descriptor."""
        descriptor = SharedSlotDescriptor.from_dict(payload)

        logger.debug(
            "Shared-slot descriptor received "
            "producer_id=%s sequence_id=%s slot_id=%s "
            "offset=%s length=%s shm_name=%s control_endpoint=%s",
            channel.producer_id,
            descriptor.sequence_id,
            descriptor.slot_id,
            descriptor.offset,
            descriptor.length,
            descriptor.shm_name,
            channel.shared_slot.control_endpoint,
        )

        copy_start = time.monotonic()
        try:
            packet = self._copy_shared_slot_packet(descriptor)
        except Exception:
            logger.exception(
                "Shared-slot copy failed "
                "producer_id=%s sequence_id=%s slot_id=%s",
                channel.producer_id,
                descriptor.sequence_id,
                descriptor.slot_id,
            )
            raise

        copy_elapsed = time.monotonic() - copy_start

        ack_start = time.monotonic()
        self._send_slot_credit_return(channel, descriptor)
        ack_elapsed = time.monotonic() - ack_start

        channel.mark_shared_slot_descriptor_seen(
            shm_name=descriptor.shm_name,
            copied_bytes=len(packet),
        )

        logger.debug(
            "Shared-slot daemon progress "
            "producer_id=%s last_sequence_id=%s "
            "descriptors_received=%d completed_messages=%d "
            "copied_mib=%.2f pending_traces=%d",
            channel.producer_id,
            descriptor.sequence_id,
            channel.shared_slot.descriptors_received,
            channel.shared_slot.completed_messages,
            channel.shared_slot.copied_bytes / (1024 * 1024),
            len(channel.socket_pending_messages),
        )

        metadata_dict, chunk_data = parse_shared_frame_packet(packet)
        chunk_metadata = SharedRingChunkMetadata.from_dict(metadata_dict)
        return SharedSlotTransportResult(
            descriptor=descriptor,
            chunk_metadata=chunk_metadata,
            chunk_data=chunk_data,
            trace_id=chunk_metadata.trace_id,
            trace_metadata=chunk_metadata.trace_metadata,
            copy_elapsed=copy_elapsed,
            ack_elapsed=ack_elapsed,
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

    def close(self) -> None:
        """Close all daemon-side shared-slot handles during shutdown."""
        for socket_obj in self._ack_sender_sockets.values():
            try:
                socket_obj.close(0)
            except Exception:
                logger.warning("Failed to close shared-slot ACK sender", exc_info=True)
        self._ack_sender_sockets.clear()

        for shm in self._shared_memory_cache.values():
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                logger.warning("Failed to close cached shared memory", exc_info=True)
        self._shared_memory_cache.clear()


    def _copy_shared_slot_packet(self, descriptor: SharedSlotDescriptor) -> bytes:
        """Copy one packet out of cached shared memory."""
        shm = self._shared_memory_cache.get(descriptor.shm_name)
        if shm is None:
            shm = SharedMemory(name=descriptor.shm_name, create=False)

            try:
                resource_tracker.unregister(shm._name, "shared_memory")
            except Exception:
                logger.debug(
                    "Failed to unregister daemon shared-memory handle %s",
                    descriptor.shm_name,
                    exc_info=True,
                )

            self._shared_memory_cache[descriptor.shm_name] = shm

        return bytes(shm.buf[descriptor.offset : descriptor.offset + descriptor.length])

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
