"""The bridge sends lifecycle timestamps to the daemon in microseconds."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from neuracore.data_daemon import bridge

TIMESTAMP_S = 12.5
TIMESTAMP_US = 12_500_000
ROBOT_ID = "robot-1"
ROBOT_INSTANCE = 2


@pytest.fixture
def native(monkeypatch) -> MagicMock:
    native_module = MagicMock()
    monkeypatch.setattr(bridge, "_load_native", lambda: native_module)
    monkeypatch.setattr(bridge, "ensure_daemon_running", lambda: None)
    return native_module


@pytest.mark.parametrize(
    ("timestamp", "expected_us"), [(TIMESTAMP_S, TIMESTAMP_US), (None, None)]
)
def test_start_recording_sends_microseconds(
    native: MagicMock, timestamp: float | None, expected_us: int | None
) -> None:
    bridge.RecordingContext().start_recording(
        ROBOT_ID, ROBOT_INSTANCE, timestamp=timestamp
    )

    native.start_recording.assert_called_once_with(
        ROBOT_ID, ROBOT_INSTANCE, None, None, None, expected_us
    )


@pytest.mark.parametrize("operation", ["stop_recording", "cancel_recording"])
@pytest.mark.parametrize(
    ("timestamp", "expected_us"), [(TIMESTAMP_S, TIMESTAMP_US), (None, None)]
)
def test_stop_and_cancel_send_microseconds(
    native: MagicMock,
    operation: str,
    timestamp: float | None,
    expected_us: int | None,
) -> None:
    context = bridge.RecordingContext()
    context.bind_source(ROBOT_ID, ROBOT_INSTANCE)

    getattr(context, operation)(timestamp)

    getattr(native, operation).assert_called_once_with(
        ROBOT_ID, ROBOT_INSTANCE, expected_us
    )
