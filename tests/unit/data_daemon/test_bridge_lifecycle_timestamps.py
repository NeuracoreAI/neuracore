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


def test_start_local_recording_sends_the_local_details(native: MagicMock) -> None:
    started_at = bridge.RecordingContext().start_local_recording(
        ROBOT_ID,
        ROBOT_INSTANCE,
        "local-dataset-1",
        timestamp_us=TIMESTAMP_US,
        local_robot=bridge.LocalRobot(name="arm"),
        local_dataset=bridge.LocalDataset(
            name="warehouse", shared=True, description="d", tags=["t"]
        ),
    )

    native.start_local_recording.assert_called_once_with(
        ROBOT_ID,
        ROBOT_INSTANCE,
        "local-dataset-1",
        TIMESTAMP_US,
        local_robot_name="arm",
        local_robot_shared=False,
        local_robot_overwrite_urdf=False,
        local_dataset_name="warehouse",
        local_dataset_shared=True,
        local_dataset_description="d",
        local_dataset_tags=["t"],
    )
    assert started_at is native.start_local_recording.return_value
    native.start_recording.assert_not_called()


def test_start_local_recording_with_only_a_local_robot(native: MagicMock) -> None:
    context = bridge.RecordingContext()
    context.start_local_recording(
        ROBOT_ID,
        ROBOT_INSTANCE,
        "cloud-dataset-1",
        timestamp_us=TIMESTAMP_US,
        local_robot=bridge.LocalRobot(name="arm", shared=True, overwrite_urdf=True),
        local_dataset=None,
    )

    kwargs = native.start_local_recording.call_args.kwargs
    assert kwargs["local_robot_name"] == "arm"
    assert kwargs["local_robot_shared"] is True
    assert kwargs["local_robot_overwrite_urdf"] is True
    assert kwargs["local_dataset_name"] is None
    assert kwargs["local_dataset_tags"] == []
    # Later stop/cancel calls go to the same source.
    context.stop_recording(TIMESTAMP_US)
    native.stop_recording.assert_called_once_with(
        ROBOT_ID, ROBOT_INSTANCE, TIMESTAMP_US
    )


@pytest.mark.parametrize("operation", ["stop_recording", "cancel_recording"])
def test_stop_and_cancel_send_microseconds(native: MagicMock, operation: str) -> None:
    context = bridge.RecordingContext()
    context.bind_source(ROBOT_ID, ROBOT_INSTANCE)

    getattr(context, operation)(TIMESTAMP_US)

    getattr(native, operation).assert_called_once_with(
        ROBOT_ID, ROBOT_INSTANCE, TIMESTAMP_US
    )
