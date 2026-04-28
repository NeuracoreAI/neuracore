"""Daemon-side shared-slot transport helpers."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING

import zmq

from neuracore.data_daemon.models import (
    OpenFixedSharedSlotsModel,
    SharedRingChunkMetadata,
    SharedSlotDescriptor,
    SlotReleaseAck,
    TraceTransportMetadata,
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
    copy_elapsed: float
    ack_elapsed: float
    should_trace: bool


class SharedSlotDaemonHandler:
    """Own daemon-side shared-slot transport mechanics."""

    def __init__(self, comm: CommunicationsManager) -> None:
        self._comm = comm
        self._shared_memory_cache: dict[str, SharedMemory] = {}
        self._ack_sender_sockets: dict[str, object] = {}

    def handle_open(
        self,
        channel: ChannelState,
        payload: dict,
    ) -> None:
        """Open fixed shared slots for one channel."""
        setup = OpenFixedSharedSlotsModel(**payload)
        channel.mark_shared_slot_transport_open(setup)
        logger.info(
            "Opened fixed shared slots for producer_id=%s slot_size=%d slot_count=%d "
            "total_shared_memory_bytes=%d max_in_flight_packets=%d",
            channel.producer_id,
            setup.slot_size,
            setup.slot_count,
            setup.slot_size * setup.slot_count,
            setup.slot_count,
        )

    def handle_descriptor(
        self,
        channel: ChannelState,
        payload: dict,
    ) -> SharedSlotTransportResult:
        """Copy, ACK, parse, and account for one shared-slot descriptor."""
        descriptor = SharedSlotDescriptor.from_dict(payload)
        should_trace = self._should_trace(descriptor.sequence_id)

        if should_trace:
            logger.info(
                "Shared-slot descriptor received "
                "producer_id=%s sequence_id=%s slot_id=%s "
                "offset=%s length=%s shm_name=%s ack_endpoint=%s",
                channel.producer_id,
                descriptor.sequence_id,
                descriptor.slot_id,
                descriptor.offset,
                descriptor.length,
                descriptor.shm_name,
                descriptor.ack_endpoint,
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
        if should_trace or copy_elapsed > 0.5:
            logger.info(
                "Shared-slot copied "
                "producer_id=%s sequence_id=%s slot_id=%s "
                "bytes=%d copy_elapsed=%.3fs",
                channel.producer_id,
                descriptor.sequence_id,
                descriptor.slot_id,
                len(packet),
                copy_elapsed,
            )

        ack_start = time.monotonic()
        self._send_slot_release_ack(channel, descriptor)
        ack_elapsed = time.monotonic() - ack_start

        if should_trace or ack_elapsed > 0.1:
            logger.info(
                "Shared-slot ACK sent "
                "producer_id=%s sequence_id=%s slot_id=%s "
                "ack_elapsed=%.3fs endpoint=%s",
                channel.producer_id,
                descriptor.sequence_id,
                descriptor.slot_id,
                ack_elapsed,
                descriptor.ack_endpoint,
            )

        channel.mark_shared_slot_descriptor_seen(
            ack_endpoint=descriptor.ack_endpoint,
            shm_name=descriptor.shm_name,
            copied_bytes=len(packet),
        )

        if channel.shared_slot.descriptors_received % 100 == 0:
            logger.info(
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
            should_trace=should_trace,
        )

    def cleanup_channel_resources(self, channel: ChannelState) -> None:
        """Close daemon-side shared-slot resources associated with one channel."""
        shm_name = channel.shared_slot.shm_name
        if shm_name:
            shm = self._shared_memory_cache.pop(shm_name, None)
            if shm is not None:
                try:
                    shm.close()
                except Exception:
                    logger.warning(
                        "Failed to close cached shared memory %s",
                        shm_name,
                        exc_info=True,
                    )

        endpoint = channel.shared_slot.ack_endpoint
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
            except Exception:
                logger.warning("Failed to close cached shared memory", exc_info=True)
        self._shared_memory_cache.clear()

    @staticmethod
    def _should_trace(sequence_id: int) -> bool:
        return sequence_id < 10 or sequence_id % 100 == 0 or 1090 <= sequence_id <= 1130

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

    def _send_slot_release_ack(
        self,
        channel: ChannelState,
        descriptor: SharedSlotDescriptor,
    ) -> None:
        """Send the slot-release ACK immediately after shared-memory copy-out."""
        endpoint = descriptor.ack_endpoint
        socket_obj = self._ack_sender_sockets.get(endpoint)
        if socket_obj is None:
            socket_obj = self._comm._context.socket(zmq.PUSH)
            socket_obj.setsockopt(zmq.LINGER, 0)
            socket_obj.connect(endpoint)
            self._ack_sender_sockets[endpoint] = socket_obj
        ack = SlotReleaseAck(
            shm_name=descriptor.shm_name,
            slot_id=descriptor.slot_id,
            sequence_id=descriptor.sequence_id,
        )
        try:
            socket_obj.send(json.dumps(ack.to_dict()).encode("utf-8"))
        except Exception:
            logger.exception(
                "Failed to send shared-slot ACK producer_id=%s sequence_id=%s",
                channel.producer_id,
                descriptor.sequence_id,
            )
