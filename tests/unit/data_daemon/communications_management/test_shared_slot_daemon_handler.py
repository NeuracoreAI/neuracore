from collections import namedtuple

import pytest

from neuracore.data_daemon.communications_management.consumer import (
    models as consumer_models,
)
from neuracore.data_daemon.communications_management.shared_transport import (
    shared_memory_budget as shared_memory_budget_module,
)
from neuracore.data_daemon.communications_management.shared_transport import (
    shared_slot_daemon_handler as shared_slot_daemon_handler_module,
)

SharedMemoryBudget = shared_memory_budget_module.SharedMemoryBudget
SharedSlotDaemonHandler = shared_slot_daemon_handler_module.SharedSlotDaemonHandler


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


def test_handle_descriptor_returns_slot_credit_when_spool_fails(monkeypatch) -> None:
    handler = SharedSlotDaemonHandler(comm=object())  # type: ignore[arg-type]
    channel = consumer_models.ChannelState(producer_id="producer-1")
    channel.mark_shared_slot_transport_open(
        control_endpoint="ipc://credits",
        shm_name="existing-shm",
    )
    descriptor_payload = {
        "shm_name": "descriptor-shm",
        "slot_id": 7,
        "offset": 0,
        "length": 128,
        "sequence_id": 42,
        "slot_size": 256,
    }
    returned_credits: list[tuple[str, int, int]] = []

    def fake_spool(*_args, **_kwargs):
        raise RuntimeError("spool failed")

    def fake_return_credit(channel_arg, descriptor_arg) -> None:
        returned_credits.append((
            channel_arg.producer_id,
            descriptor_arg.slot_id,
            descriptor_arg.sequence_id,
        ))

    monkeypatch.setattr(handler, "_spool_shared_slot_packet", fake_spool)
    monkeypatch.setattr(handler, "_send_slot_credit_return", fake_return_credit)

    with pytest.raises(RuntimeError, match="spool failed"):
        handler.handle_descriptor(channel, descriptor_payload, chunk_spool=object())

    assert returned_credits == [("producer-1", 7, 42)]
    assert channel.shared_slot.shm_name == "existing-shm"


def test_handle_descriptor_preserves_spool_error_when_credit_return_fails(
    monkeypatch,
) -> None:
    handler = SharedSlotDaemonHandler(comm=object())  # type: ignore[arg-type]
    channel = consumer_models.ChannelState(producer_id="producer-1")
    channel.mark_shared_slot_transport_open(
        control_endpoint="ipc://credits",
        shm_name="existing-shm",
    )
    descriptor_payload = {
        "shm_name": "descriptor-shm",
        "slot_id": 7,
        "offset": 0,
        "length": 128,
        "sequence_id": 42,
        "slot_size": 256,
    }

    def fake_spool(*_args, **_kwargs):
        raise RuntimeError("spool failed")

    def fake_return_credit(*_args, **_kwargs) -> None:
        raise RuntimeError("credit failed")

    monkeypatch.setattr(handler, "_spool_shared_slot_packet", fake_spool)
    monkeypatch.setattr(handler, "_send_slot_credit_return", fake_return_credit)

    with pytest.raises(RuntimeError, match="spool failed"):
        handler.handle_descriptor(channel, descriptor_payload, chunk_spool=object())
