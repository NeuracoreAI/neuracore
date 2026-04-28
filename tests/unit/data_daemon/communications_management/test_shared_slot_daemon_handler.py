import pytest

from neuracore.data_daemon.communications_management.shared_transport.shared_memory_budget import (
    BYTES_PER_MIB,
    SharedMemoryBudget,
)


def test_shared_memory_budget_raises_when_budget_cannot_fit_one_slot(monkeypatch) -> None:
    budget = SharedMemoryBudget()
    usage = type(
        "usage",
        (),
        {
            "total": 40 * BYTES_PER_MIB,
            "free": 40 * BYTES_PER_MIB,
        },
    )

    monkeypatch.setattr(
        "neuracore.data_daemon.communications_management.shared_transport.shared_memory_budget.shutil.disk_usage",
        lambda _path: usage,
    )

    with pytest.raises(
        RuntimeError,
        match=r"Not enough shared-memory for data throughput requirements",
    ):
        budget.reserve(
            shm_name="shm-test",
            slot_size=31 * BYTES_PER_MIB,
            requested_slot_count=4,
        )
