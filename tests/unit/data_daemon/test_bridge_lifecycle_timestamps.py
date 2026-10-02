"""The bridge passes lifecycle timestamps to the daemon in microseconds."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from neuracore.data_daemon import bridge

TIMESTAMP_US = 12_500_000
ROBOT_ID = "robot-1"
ROBOT_INSTANCE = 2


@pytest.fixture
def native(monkeypatch) -> MagicMock:
    native_module = MagicMock()
    monkeypatch.setattr(bridge, "_load_native", lambda: native_module)
    monkeypatch.setattr(bridge, "ensure_daemon_running", lambda: None)
    return native_module


def test_start_recording_sends_microseconds(native: MagicMock) -> None:
    bridge.RecordingContext().start_recording(
        ROBOT_ID, ROBOT_INSTANCE, timestamp_us=TIMESTAMP_US
    )

    native.start_recording.assert_called_once_with(
        ROBOT_ID, ROBOT_INSTANCE, None, None, None, TIMESTAMP_US
    )


@pytest.mark.parametrize("operation", ["stop_recording", "cancel_recording"])
def test_stop_and_cancel_send_microseconds(native: MagicMock, operation: str) -> None:
    context = bridge.RecordingContext()
    context.bind_source(ROBOT_ID, ROBOT_INSTANCE)

    getattr(context, operation)(TIMESTAMP_US)

    getattr(native, operation).assert_called_once_with(
        ROBOT_ID, ROBOT_INSTANCE, TIMESTAMP_US
    )
