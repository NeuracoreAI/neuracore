"""Internal runtime models and helpers for shared-slot transport."""

from __future__ import annotations

import os
import threading
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import zmq

from neuracore.data_daemon.const import ACK_BASE_DIR


@dataclass(frozen=True)
class SharedSlotRegistryConfig:
    """Normalized shared-slot transport configuration."""

    slot_size: int
    slot_count: int
    ack_timeout_s: float
    allocate_timeout_s: float


@dataclass(frozen=True)
class InFlightSlot:
    """Tracked producer-side state for one descriptor awaiting slot credit."""

    shm_name: str
    slot_id: int
    sequence_id: int
    reserved_at: float
    socket_sent_at: float | None = None


@dataclass
class SharedSlotRegistryState:
    """Mutable shared-slot session state for one producer transport."""

    shm_name: str | None = None
    shm: SharedMemory | None = None
    free_slots: deque[int] = field(default_factory=deque)
    sequence_id: int = 1
    healthy: bool = True
    ready: bool = False
    in_flight: dict[int, InFlightSlot] = field(default_factory=dict)
    closed: bool = False
    unhealthy_reason: str | None = None


def create_control_socket_path(base_dir: Path = ACK_BASE_DIR) -> Path:
    """Create a unique filesystem path for the shared-slot control socket."""
    base_dir.mkdir(parents=True, exist_ok=True)
    socket_path = base_dir / f"slot_control_{os.getpid()}_{uuid.uuid4().hex}.ipc"
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass
    return socket_path


@dataclass
class SharedSlotControlRuntime:
    """Owned producer-side resources for control messages and watchdog threads."""

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
        control_listener_target: Callable[[], None],
        watchdog_target: Callable[[], None],
    ) -> SharedSlotControlRuntime:
        """Build the bound ZMQ runtime and background threads."""
        socket_path = create_control_socket_path()
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
        """Start the background threads for this runtime."""
        self.control_thread.start()
        self.watchdog_thread.start()
