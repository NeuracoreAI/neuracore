"""Producer-side registry for daemon-owned shared-slot transport."""

from __future__ import annotations

import logging
import os
import threading
import time
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory

import zmq

from neuracore.data_daemon.const import (
    DEFAULT_VIDEO_ACK_TIMEOUT_SECONDS,
    DEFAULT_VIDEO_SLOT_ALLOCATE_TIMEOUT_SECONDS,
    DEFAULT_VIDEO_SLOT_COUNT,
    DEFAULT_VIDEO_SLOT_SIZE,
)
from neuracore.data_daemon.models import (
    CommandType,
    MessageEnvelope,
    OpenFixedSharedSlotsModel,
    SharedSlotCreditReturn,
    SharedSlotReadyModel,
)

from .runtime import (
    InFlightSlot,
    SharedSlotControlRuntime,
    SharedSlotRegistryConfig,
    SharedSlotRegistryState,
)

logger = logging.getLogger(__name__)


class SharedSlotTimeout(TimeoutError):
    """Raised when shared-slot setup, allocation, or credit return times out."""


class SharedSlotRegistry:
    """Producer-side session state for one daemon-owned shared-slot transport."""

    _instance: SharedSlotRegistry | None = None
    _refcount = 0
    _instance_lock = threading.Lock()

    def __init__(
        self,
        *,
        slot_size: int,
        slot_count: int,
        ack_timeout_s: float,
        allocate_timeout_s: float,
    ) -> None:
        self._config = SharedSlotRegistryConfig(
            slot_size=int(slot_size),
            slot_count=int(slot_count),
            ack_timeout_s=float(ack_timeout_s),
            allocate_timeout_s=float(allocate_timeout_s),
        )
        self._state = SharedSlotRegistryState()
        self._condition = threading.Condition()
        self._runtime = SharedSlotControlRuntime.build(
            control_listener_target=self._control_listener_loop,
            watchdog_target=self._watchdog_loop,
        )
        self._runtime.start()

    @classmethod
    def acquire(
        cls,
        *,
        slot_size: int = DEFAULT_VIDEO_SLOT_SIZE,
        slot_count: int = DEFAULT_VIDEO_SLOT_COUNT,
        ack_timeout_s: float | None = None,
        allocate_timeout_s: float | None = None,
    ) -> "SharedSlotRegistry":
        """Acquire a singleton registry instance for isolated unit tests."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(
                    slot_size=slot_size,
                    slot_count=slot_count,
                    ack_timeout_s=_env_float(
                        "NCD_VIDEO_ACK_TIMEOUT_SECONDS",
                        DEFAULT_VIDEO_ACK_TIMEOUT_SECONDS
                        if ack_timeout_s is None
                        else ack_timeout_s,
                    ),
                    allocate_timeout_s=_env_float(
                        "NCD_VIDEO_SLOT_ALLOCATE_TIMEOUT_SECONDS",
                        DEFAULT_VIDEO_SLOT_ALLOCATE_TIMEOUT_SECONDS
                        if allocate_timeout_s is None
                        else allocate_timeout_s,
                    ),
                )
            cls._refcount += 1
            return cls._instance

    @classmethod
    def release_shared_instance(cls) -> None:
        """Release one singleton test registry reference."""
        with cls._instance_lock:
            if cls._instance is None:
                return
            cls._refcount = max(0, cls._refcount - 1)
            if cls._refcount > 0:
                return
            instance = cls._instance
            cls._instance = None
        instance.close()

    @classmethod
    def reset_shared_instance_for_tests(cls) -> None:
        """Tear down the singleton test registry, if any."""
        with cls._instance_lock:
            instance = cls._instance
            cls._instance = None
            cls._refcount = 0
        if instance is not None:
            instance.close()

    @property
    def slot_size(self) -> int:
        """Return the configured slot size."""
        return self._config.slot_size

    @property
    def slot_count(self) -> int:
        """Return the configured slot count."""
        return self._config.slot_count

    @property
    def ack_timeout_s(self) -> float:
        """Return the configured ACK timeout."""
        return self._config.ack_timeout_s

    @property
    def allocate_timeout_s(self) -> float:
        """Return the configured allocation timeout."""
        return self._config.allocate_timeout_s

    @property
    def shm_name(self) -> str | None:
        """Return the current attached shared-memory name, if any."""
        return self._state.shm_name

    @property
    def control_endpoint(self) -> str:
        """Return the producer-side control endpoint."""
        return self._runtime.control_endpoint

    @property
    def total_shared_memory_bytes(self) -> int:
        """Return the total bytes reserved in shared memory."""
        return self.slot_size * self.slot_count

    def request_payload(self) -> OpenFixedSharedSlotsModel:
        """Return the setup request payload for daemon-owned fixed shared slots."""
        return OpenFixedSharedSlotsModel(
            control_endpoint=self.control_endpoint,
            slot_size=self.slot_size,
            slot_count=self.slot_count,
        )

    def is_ready(self) -> bool:
        """Return True when the daemon has opened the shared memory session."""
        with self._condition:
            return self._state.ready

    def wait_until_ready(self) -> bool:
        """Block until the daemon has opened the shared-slot session."""
        deadline = time.monotonic() + self.allocate_timeout_s
        with self._condition:
            while not self._state.ready:
                if not self._is_healthy_locked():
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._mark_unhealthy_locked("open_timeout")
                    return False
                self._condition.wait(timeout=min(0.1, remaining))
            return True

    def is_healthy(self) -> bool:
        """Return True while the shared-slot transport is still healthy."""
        with self._condition:
            return self._is_healthy_locked()

    def ensure_healthy(self) -> None:
        """Raise when the shared-slot transport is unhealthy."""
        if self.is_healthy():
            return
        raise RuntimeError("Shared-slot transport is unhealthy")

    def allocate_slot(self) -> tuple[int, int]:
        """Reserve one free slot or fail when backpressure persists."""
        deadline = time.monotonic() + self.allocate_timeout_s
        with self._condition:
            while True:
                self._check_for_timeouts_locked()
                if not self._is_healthy_locked():
                    raise RuntimeError("Shared-slot transport is unhealthy")
                if self._state.ready and self._state.free_slots:
                    slot_id = self._state.free_slots.popleft()
                    return int(slot_id), int(slot_id) * self.slot_size
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if not self._state.ready:
                        self._mark_unhealthy_locked("open_timeout")
                        raise SharedSlotTimeout(
                            "Timed out waiting for daemon-owned shared slots to open"
                        )
                    raise SharedSlotTimeout("Timed out waiting for a free shared slot")
                self._condition.wait(timeout=min(0.1, remaining))

    def mark_in_flight(self, slot_id: int) -> int:
        """Record that the slot now backs one sent descriptor."""
        with self._condition:
            self._check_for_timeouts_locked()
            if not self._is_healthy_locked():
                raise RuntimeError("Shared-slot transport is unhealthy")
            return self._reserve_slot_for_descriptor_locked(slot_id)

    def mark_sent(self, sequence_id: int) -> None:
        """Start the credit-return timeout clock once the descriptor is on the socket."""
        with self._condition:
            self._mark_descriptor_sent_locked(sequence_id)

    def next_sequence_number(self) -> int:
        """Reserve the next shared-slot sequence number."""
        with self._condition:
            sequence_id = self._state.sequence_id
            self._state.sequence_id += 1
            return sequence_id

    def release_slot(self, shm_name: str, slot_id: int, sequence_id: int) -> bool:
        """Release one in-flight slot after a matching credit return arrives."""
        with self._condition:
            self._check_for_timeouts_locked()
            return self._apply_slot_credit_locked(
                shm_name=shm_name,
                slot_id=slot_id,
                sequence_id=sequence_id,
            )

    def rollback_enqueued_slot(self, sequence_id: int) -> None:
        """Return a slot immediately when descriptor enqueue fails."""
        with self._condition:
            self._release_sequence_locked(sequence_id)

    def notify_sender_failure(self) -> None:
        """Fail fast when a descriptor could not be written to ZMQ."""
        with self._condition:
            if not self._is_healthy_locked():
                return
            self._mark_unhealthy_locked("sender_failure")

    def get_in_flight_count(self) -> int:
        """Return the number of descriptors still awaiting slot credit."""
        with self._condition:
            return len(self._state.in_flight)

    def get_unhealthy_reason(self) -> str | None:
        """Return the current unhealthy reason, if any."""
        with self._condition:
            return self._state.unhealthy_reason

    def debug_snapshot(self) -> dict[str, object]:
        """Return a narrow debug snapshot of the current registry state."""
        with self._condition:
            return {
                "ready": self._state.ready,
                "healthy": self._state.healthy,
                "closed": self._state.closed,
                "free_slot_count": len(self._state.free_slots),
                "in_flight_sequence_ids": tuple(self._state.in_flight),
                "unhealthy_reason": self._state.unhealthy_reason,
            }

    def shared_memory_view(self, offset: int, length: int) -> memoryview:
        """Return a writable view into one slot-sized shared-memory span."""
        shm = self._state.shm
        if shm is None:
            raise RuntimeError("Shared-slot transport is not ready")
        return shm.buf[offset : offset + length]

    def close(self) -> None:
        """Stop threads and close local handles."""
        with self._condition:
            if self._state.closed:
                return
            self._mark_closed_locked()

        self._runtime.stop_event.set()
        self._runtime.control_thread.join(timeout=1.0)
        self._runtime.watchdog_thread.join(timeout=1.0)
        self._close_control_resources()
        self._close_shared_memory()
        self._remove_control_socket_path()

    def _control_listener_loop(self) -> None:
        logger.info(
            "Shared-slot control receiver started endpoint=%s",
            self.control_endpoint,
        )
        poller = zmq.Poller()
        poller.register(self._runtime.control_socket, zmq.POLLIN)

        try:
            while not self._runtime.stop_event.is_set():
                try:
                    events = dict(poller.poll(100))
                except zmq.ZMQError:
                    logger.exception(
                        "Shared-slot control receiver poll failed endpoint=%s",
                        self.control_endpoint,
                    )
                    break

                if self._runtime.control_socket not in events:
                    continue

                try:
                    message = MessageEnvelope.from_bytes(
                        self._runtime.control_socket.recv()
                    )
                    self._process_control_message(message)
                except Exception:
                    logger.exception("Failed to process shared-slot control message")
        finally:
            with self._condition:
                closed = self._state.closed
                healthy = self._state.healthy
            log_fn = logger.info if self._runtime.stop_event.is_set() and closed else logger.warning
            log_fn(
                "Shared-slot control receiver exiting endpoint=%s stop_event=%s closed=%s healthy=%s",
                self.control_endpoint,
                self._runtime.stop_event.is_set(),
                closed,
                healthy,
            )

    def _process_control_message(self, message: MessageEnvelope) -> None:
        if message.command == CommandType.SHARED_SLOT_READY:
            ready = SharedSlotReadyModel(**message.payload[message.command.value])
            self._apply_ready_message(ready)
            return
        if message.command == CommandType.SHARED_SLOT_CREDIT_RETURN:
            credit = SharedSlotCreditReturn.from_dict(message.payload[message.command.value])
            self._process_slot_credit_return(credit)
            return
        logger.warning("Ignoring unexpected shared-slot control command %s", message.command)

    def _apply_ready_message(self, ready: SharedSlotReadyModel) -> None:
        try:
            shm = SharedMemory(name=ready.shm_name, create=False)
            try:
                resource_tracker.unregister(shm._name, "shared_memory")
            except Exception:
                logger.debug(
                    "Failed to unregister daemon-owned shared-memory handle %s",
                    ready.shm_name,
                    exc_info=True,
                )
        except Exception:
            logger.exception("Failed to attach daemon-owned shared memory %s", ready.shm_name)
            with self._condition:
                self._mark_unhealthy_locked("attach_failed")
            return

        with self._condition:
            if self._state.closed:
                shm.close()
                return
            self._set_ready_shared_memory_locked(shm=shm, ready=ready)

    def _watchdog_loop(self) -> None:
        while not self._runtime.stop_event.wait(0.1):
            with self._condition:
                self._check_for_timeouts_locked()

    def _set_ready_shared_memory_locked(
        self,
        *,
        shm: SharedMemory,
        ready: SharedSlotReadyModel,
    ) -> None:
        """Swap in a newly attached shared-memory region and mark the registry ready."""
        if self._state.shm is not None:
            self._state.shm.close()
        self._config = SharedSlotRegistryConfig(
            slot_size=int(ready.slot_size),
            slot_count=int(ready.slot_count),
            ack_timeout_s=self._config.ack_timeout_s,
            allocate_timeout_s=self._config.allocate_timeout_s,
        )
        self._state.shm = shm
        self._state.shm_name = ready.shm_name
        self._state.free_slots = type(self._state.free_slots)(range(self.slot_count))
        self._state.ready = True
        self._condition.notify_all()

    def _check_for_timeouts_locked(self) -> None:
        if not self._is_healthy_locked():
            return
        now = time.monotonic()
        timed_out = [
            sequence_id
            for sequence_id, entry in self._state.in_flight.items()
            if entry.socket_sent_at is not None
            and now - entry.socket_sent_at >= self.ack_timeout_s
        ]
        if not timed_out:
            return
        last_reason: str | None = None
        for sequence_id in timed_out:
            entry = self._state.in_flight.get(sequence_id)
            if entry is not None:
                last_reason = (
                    f"credit_timeout(sequence_id={entry.sequence_id},slot_id={entry.slot_id})"
                )
                logger.warning(
                    "Shared-slot credit timed out shm_name=%s slot_id=%s sequence_id=%s",
                    entry.shm_name,
                    entry.slot_id,
                    entry.sequence_id,
                )
        self._mark_unhealthy_locked(last_reason or "credit_timeout", sequence_ids=timed_out)

    def _release_sequence_locked(self, sequence_id: int) -> None:
        entry = self._state.in_flight.pop(sequence_id, None)
        if entry is None:
            return
        self._state.free_slots.append(entry.slot_id)
        self._condition.notify_all()

    def _reserve_slot_for_descriptor_locked(self, slot_id: int) -> int:
        """Create in-flight tracking for a reserved slot."""
        shm_name = self._state.shm_name
        if shm_name is None:
            raise RuntimeError("Shared-slot transport is not ready")
        sequence_id = self._state.sequence_id
        self._state.sequence_id += 1
        self._state.in_flight[sequence_id] = InFlightSlot(
            shm_name=shm_name,
            slot_id=int(slot_id),
            sequence_id=sequence_id,
            reserved_at=time.monotonic(),
        )
        return sequence_id

    def _mark_descriptor_sent_locked(self, sequence_id: int) -> None:
        """Start the credit timeout clock for one in-flight descriptor."""
        entry = self._state.in_flight.get(sequence_id)
        if entry is None or entry.socket_sent_at is not None:
            return
        self._state.in_flight[sequence_id] = InFlightSlot(
            shm_name=entry.shm_name,
            slot_id=entry.slot_id,
            sequence_id=entry.sequence_id,
            reserved_at=entry.reserved_at,
            socket_sent_at=time.monotonic(),
        )

    def _apply_slot_credit_locked(
        self,
        *,
        shm_name: str,
        slot_id: int,
        sequence_id: int,
    ) -> bool:
        """Apply one returned slot credit to the in-flight state."""
        entry = self._state.in_flight.get(sequence_id)
        if entry is None or entry.shm_name != shm_name or entry.slot_id != slot_id:
            logger.warning(
                "Ignoring stale or unknown slot credit shm_name=%s slot_id=%s sequence_id=%s",
                shm_name,
                slot_id,
                sequence_id,
            )
            return False

        self._release_sequence_locked(sequence_id)
        return True

    def _process_slot_credit_return(self, credit: SharedSlotCreditReturn) -> None:
        """Apply one returned slot credit."""
        self.release_slot(
            credit.shm_name,
            credit.slot_id,
            credit.sequence_id,
        )

    def _is_healthy_locked(self) -> bool:
        """Return True when the registry can still accept work."""
        return self._state.healthy and not self._state.closed

    def _mark_closed_locked(self) -> None:
        """Mark the registry closed and wake any waiters."""
        self._state.closed = True
        self._state.healthy = False
        self._condition.notify_all()

    def _mark_unhealthy_locked(
        self,
        reason: str,
        *,
        sequence_ids: list[int] | None = None,
    ) -> None:
        """Transition to unhealthy state and release affected slots."""
        self._state.healthy = False
        self._state.unhealthy_reason = reason
        ids_to_release = (
            list(self._state.in_flight) if sequence_ids is None else sequence_ids
        )
        for sequence_id in ids_to_release:
            self._release_sequence_locked(sequence_id)
        self._condition.notify_all()

    def _close_control_resources(self) -> None:
        """Close the producer-side control socket and its ZMQ context."""
        try:
            self._runtime.control_socket.close(0)
        except Exception:
            logger.warning("Failed to close shared-slot control socket", exc_info=True)
        try:
            self._runtime.context.term()
        except Exception:
            logger.warning("Failed to terminate shared-slot control context", exc_info=True)

    def _close_shared_memory(self) -> None:
        """Close the local attachment to the daemon-owned shared-memory region."""
        shm = self._state.shm
        if shm is None:
            return
        try:
            shm.close()
        except Exception:
            logger.warning("Failed to close shared-memory handle", exc_info=True)
        self._state.shm = None

    def _remove_control_socket_path(self) -> None:
        """Remove the filesystem entry backing the control IPC endpoint."""
        try:
            self._runtime.control_socket_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning(
                "Failed to remove shared-slot control socket file %s",
                self._runtime.control_socket_path,
                exc_info=True,
            )


def _env_float(name: str, default: float) -> float:
    """Read a float override from the environment, falling back safely."""
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return float(default)
