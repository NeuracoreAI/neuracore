from collections import namedtuple
from unittest.mock import patch

import pytest

from neuracore.data_daemon.communications_management.shared_slot_daemon_handler import (
    SharedSlotDaemonHandler,
)


def test_get_safe_slot_count_raises_when_budget_cannot_fit_one_slot() -> None:
    handler = SharedSlotDaemonHandler(comm=None)
    usage = namedtuple("usage", ["total", "used", "free"])(
        128 * 1024**2,
        88 * 1024**2,
        40 * 1024**2,
    )
    slot_size = 31 * 1024**2

    with patch(
        "neuracore.data_daemon.communications_management.shared_slot_daemon_handler"
        ".shutil.disk_usage",
        return_value=usage,
    ):
        with pytest.raises(
            RuntimeError,
            match=(
                r"Not enough /dev/shm for one shared slot: "
                r"slot_size=31\.00MiB, "
                r"budget=30\.00MiB, "
                r"free=40\.00MiB, "
                r"total=128\.00MiB"
            ),
        ):
            handler._get_safe_slot_count(
                slot_size=slot_size,
                requested_slot_count=4,
            )
