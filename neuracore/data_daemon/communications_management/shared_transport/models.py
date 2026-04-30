from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import zmq

from neuracore.data_daemon.models import (
    SharedMemoryChunkMetadata,
    SharedSlotDescriptor,
    TraceTransportMetadata,
)

from ..consumer.bridge_chunk_spool import ChunkSpoolRef

@dataclass(frozen=True)
class SharedSlotReservation:
    slot_count: int
    allocated_bytes: int

@dataclass(frozen=True)
class QueuedSharedSlotPacket:
    producer_id: str
    sender: "ProducerChannelMessageSender"
    metadata_bytes: bytes
    chunk: bytes | bytearray | memoryview
    packet_length: int


@dataclass(frozen=True)
class SharedSlotTransportResult:
    descriptor: SharedSlotDescriptor
    chunk_metadata: SharedMemoryChunkMetadata
    chunk_spool_ref: ChunkSpoolRef
    trace_id: str
    trace_metadata: TraceTransportMetadata | None


@dataclass(frozen=True)
class SharedSlotRegistryConfig:
    slot_size: int
    slot_count: int
    ack_timeout_s: float
    allocate_timeout_s: float


@dataclass(frozen=True)
class InFlightSlot:
    shm_name: str
    slot_id: int
    sequence_id: int
    reserved_at: float
    socket_sent_at: float | None = None


@dataclass
class SharedSlotRegistryState:
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


@dataclass
class SharedSlotControlRuntime:
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
    ) -> "SharedSlotControlRuntime":
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
        self.control_thread.start()
        self.watchdog_thread.start()
