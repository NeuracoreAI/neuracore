"""Shared transport data structures used by producer and daemon components."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import TYPE_CHECKING

import zmq

from neuracore.data_daemon.models import (
    SharedMemoryChunkMetadata,
    SharedSlotDescriptor,
    TraceTransportMetadata,
)

from ..consumer.bridge_chunk_spool import ChunkSpoolRef

if TYPE_CHECKING:
    from ..producer.producer_channel_message_sender import ProducerChannelMessageSender


@dataclass(frozen=True)
class SharedSlotReservation:
    """Shared-memory reservation details returned by the budget manager."""

    slot_count: int
    allocated_bytes: int


@dataclass(frozen=True)
class QueuedSharedSlotPacket:
    """One shared-slot payload waiting to be written and announced."""

    producer_id: str
    sender: ProducerChannelMessageSender
    metadata_bytes: bytes
    chunk: bytes | bytearray | memoryview
    packet_length: int


@dataclass(frozen=True)
class SharedSlotTransportResult:
    """Daemon-side result of reading one shared-slot descriptor."""

    descriptor: SharedSlotDescriptor
    chunk_metadata: SharedMemoryChunkMetadata
    chunk_spool_ref: ChunkSpoolRef
    trace_id: str
    trace_metadata: TraceTransportMetadata | None


@dataclass(frozen=True)
class SharedSlotRegistryConfig:
    """Configuration values for one producer shared-slot registry."""

    slot_size: int
    slot_count: int
    ack_timeout_s: float
    allocate_timeout_s: float


@dataclass(frozen=True)
class InFlightSlot:
    """Metadata for one shared-memory slot awaiting credit return."""

    shm_name: str
    slot_id: int
    sequence_id: int
    reserved_at: float
    socket_sent_at: float | None = None


@dataclass
class SharedSlotRegistryState:
    """Mutable producer-side shared-slot transport state."""

    shm_name: str | None = None
    shm: SharedMemory | None = None
    free_slots: deque[int] = field(default_factory=deque)
    sequence_id: int = 1
    healthy: bool = True
    ready: bool = False
    in_flight: dict[int, InFlightSlot] = field(default_factory=dict)
    max_in_flight_count: int = 0
    acked_sequence_count: int = 0
    ack_timeout_count: int = 0
    last_acked_sequence_id: int | None = None
    last_ack_latency_s: float | None = None
    max_ack_latency_s: float = 0.0
    last_credit_return_at: float | None = None
    closed: bool = False
    unhealthy_reason: str | None = None
    failure_message: str | None = None


@dataclass
class SharedSlotControlRuntime:
    """Threading and socket resources for shared-slot control messages."""

    control_socket_path: Path
    control_endpoint: str
    context: zmq.Context
    control_socket: zmq.Socket
    stop_event: threading.Event
    control_thread: threading.Thread
    watchdog_thread: threading.Thread

    @classmethod
    def build(
        cls,
        *,
        socket_path: Path,
        control_listener_target: Callable[[], None],
        watchdog_target: Callable[[], None],
    ) -> SharedSlotControlRuntime:
        """Build a control runtime with bound IPC socket and worker threads."""
        control_endpoint = f"ipc://{socket_path}"
        context = zmq.Context()
        control_socket = context.socket(zmq.PULL)
        control_socket.setsockopt(zmq.LINGER, 0)
        control_socket.bind(control_endpoint)

        stop_event = threading.Event()
        control_thread = threading.Thread(
            target=control_listener_target,
            name="shared-slot-control-listener",
            daemon=True,
        )
        watchdog_thread = threading.Thread(
            target=watchdog_target,
            name="shared-slot-watchdog",
            daemon=True,
        )
        return cls(
            control_socket_path=socket_path,
            control_endpoint=control_endpoint,
            context=context,
            control_socket=control_socket,
            stop_event=stop_event,
            control_thread=control_thread,
            watchdog_thread=watchdog_thread,
        )

    def start(self) -> None:
        """Start the control listener and watchdog threads."""
        self.control_thread.start()
        self.watchdog_thread.start()
