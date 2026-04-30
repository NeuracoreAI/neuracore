import logging
import shutil
import threading
from dataclasses import dataclass, field

from neuracore.data_daemon.communications_management.shared_transport.models import (
    SharedSlotReservation,
)

logger = logging.getLogger(__name__)

BYTES_PER_MIB = 1024**2


@dataclass(frozen=True)
class SHMBytesAllocation:
    shm_name: str
    allocated_bytes: int


@dataclass
class SHMBytesAllocationRegistry:
    _allocations: dict[str, SHMBytesAllocation] = field(default_factory=dict)

    def add(self, allocation: SHMBytesAllocation) -> None:
        self._allocations[allocation.shm_name] = allocation

    def pop(self, shm_name: str) -> SHMBytesAllocation | None:
        return self._allocations.pop(shm_name, None)


class SharedMemoryBudget:
    def __init__(
        self,
        *,
        shm_path: str = "/dev/shm",
        budget_fraction: float = 0.75,
    ) -> None:
        self._shm_path = shm_path
        self._budget_fraction = budget_fraction
        self._lock = threading.Lock()
        self._reserved_bytes = 0
        self._allocations = SHMBytesAllocationRegistry()

    def reserve(
        self,
        *,
        shm_name: str,
        slot_size: int,
        requested_slot_count: int,
    ) -> SharedSlotReservation:
        usage = shutil.disk_usage(self._shm_path)
        total_budget = int(usage.total * self._budget_fraction)

        with self._lock:
            remaining_budget = total_budget - self._reserved_bytes

            if remaining_budget < slot_size:
                raise RuntimeError(
                    "Not enough shared-memory for data throughput requirements. "
                    "Next steps: 1) increase shared memory size, and/or "
                    "2) reduce volume of data logged. "
                    f"slot_size={slot_size / BYTES_PER_MIB:.2f}MiB, "
                    f"remaining={remaining_budget / BYTES_PER_MIB:.2f}MiB, "
                    f"reserved={self._reserved_bytes / BYTES_PER_MIB:.2f}MiB, "
                    f"budget={total_budget / BYTES_PER_MIB:.2f}MiB, "
                    f"shm_total={usage.total / BYTES_PER_MIB:.2f}MiB"
                )

            slot_count = min(
                requested_slot_count,
                remaining_budget // slot_size,
            )

            allocated_bytes = slot_size * slot_count

            allocation = SHMBytesAllocation(
                shm_name=shm_name,
                allocated_bytes=allocated_bytes,
            )

            self._reserved_bytes += allocated_bytes
            self._allocations.add(allocation)

            reserved_bytes = self._reserved_bytes

        logger.debug(
            "Reserved shared-memory budget shm_name=%s slot_size=%.2fMiB "
            "requested_slot_count=%d actual_slot_count=%d allocated=%.2fMiB "
            "reserved_total=%.2fMiB budget=%.2fMiB shm_total=%.2fMiB",
            shm_name,
            slot_size / BYTES_PER_MIB,
            requested_slot_count,
            slot_count,
            allocated_bytes / BYTES_PER_MIB,
            reserved_bytes / BYTES_PER_MIB,
            total_budget / BYTES_PER_MIB,
            usage.total / BYTES_PER_MIB,
        )

        return SharedSlotReservation(
            slot_count=int(slot_count),
            allocated_bytes=int(allocated_bytes),
        )

    def release(self, shm_name: str) -> None:
        with self._lock:
            allocation = self._allocations.pop(shm_name)

            if allocation is None:
                return

            self._reserved_bytes = max(
                0,
                self._reserved_bytes - allocation.allocated_bytes,
            )
            reserved_bytes = self._reserved_bytes

        logger.debug(
            "Released shared-memory budget shm_name=%s released=%.2fMiB "
            "reserved_total=%.2fMiB",
            allocation.shm_name,
            allocation.allocated_bytes / BYTES_PER_MIB,
            reserved_bytes / BYTES_PER_MIB,
        )

    def rollback(self, shm_name: str) -> None:
        self.release(shm_name)
