from collections import namedtuple

from neuracore.data_daemon.communications_management.shared_transport import (
    shared_memory_budget as shared_memory_budget_module,
)

SharedMemoryBudget = shared_memory_budget_module.SharedMemoryBudget


def test_shared_memory_budget_caps_slot_count_to_remaining_budget(
    monkeypatch,
) -> None:
    budget = SharedMemoryBudget()
    usage = namedtuple("usage", ["total", "used", "free"])(
        128 * 1024**2,
        88 * 1024**2,
        40 * 1024**2,
    )
    slot_size = 31 * 1024**2

    monkeypatch.setattr(
        "neuracore.data_daemon.communications_management.shared_transport"
        ".shared_memory_budget.shutil.disk_usage",
        lambda _path: usage,
    )

    reservation = budget.reserve(
        shm_name="test-shm",
        slot_size=slot_size,
        requested_slot_count=4,
    )

    assert reservation.slot_count == 3
    assert reservation.allocated_bytes == slot_size * 3
