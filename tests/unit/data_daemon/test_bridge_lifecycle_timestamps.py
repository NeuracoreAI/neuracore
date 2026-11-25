"""The bridge passes lifecycle timestamps to the daemon in microseconds."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from neuracore.data_daemon import bridge

TIMESTAMP_US = 12_500_000
ROBOT_ID = "robot-1"
ROBOT_INSTANCE = 2
ROBOT_SOURCE = bridge.RobotInstanceSource(ROBOT_ID, ROBOT_INSTANCE)
IMPORT_SOURCE = bridge.ImportItemSource(ROBOT_ID, "session-1", "episode-7")


@pytest.fixture
def native(monkeypatch) -> MagicMock:
    native_module = MagicMock()
    monkeypatch.setattr(bridge, "_load_native", lambda: native_module)
    monkeypatch.setattr(bridge, "ensure_daemon_running", lambda: None)
    return native_module


def test_start_recording_sends_microseconds(native: MagicMock) -> None:
    bridge.RecordingContext().start_recording(ROBOT_SOURCE, timestamp_us=TIMESTAMP_US)

    native.RecordingSource.robot_instance.assert_called_once_with(
        ROBOT_ID, ROBOT_INSTANCE
    )
    native.start_recording.assert_called_once_with(
        native.RecordingSource.robot_instance.return_value,
        None,
        None,
        None,
        TIMESTAMP_US,
    )


def test_an_import_start_is_sent_under_the_item_source(native: MagicMock) -> None:
    bridge.RecordingContext().start_recording(
        IMPORT_SOURCE, "robot", "dataset-1", timestamp_us=TIMESTAMP_US
    )

    native.RecordingSource.import_item.assert_called_once_with(
        ROBOT_ID, "session-1", "episode-7"
    )
    native.RecordingSource.robot_instance.assert_not_called()
    native.start_recording.assert_called_once_with(
        native.RecordingSource.import_item.return_value,
        "robot",
        "dataset-1",
        None,
        TIMESTAMP_US,
    )


@pytest.mark.parametrize("source", [ROBOT_SOURCE, IMPORT_SOURCE])
@pytest.mark.parametrize(
    ("operation", "trailing_args"),
    [("stop_recording", ([],)), ("cancel_recording", ())],
)
def test_stop_and_cancel_send_microseconds_for_the_bound_source(
    native: MagicMock,
    operation: str,
    trailing_args: tuple,
    source: bridge.RecordingSource,
) -> None:
    context = bridge.RecordingContext()
    context.bind_source(source)

    getattr(context, operation)(TIMESTAMP_US)

    native_source = (
        native.RecordingSource.import_item.return_value
        if isinstance(source, bridge.ImportItemSource)
        else native.RecordingSource.robot_instance.return_value
    )
    getattr(native, operation).assert_called_once_with(
        native_source, TIMESTAMP_US, *trailing_args
    )


@pytest.mark.parametrize("operation", ["stop_recording", "cancel_recording"])
def test_an_unbound_context_does_not_reach_the_daemon(
    native: MagicMock, operation: str
) -> None:
    getattr(bridge.RecordingContext(), operation)(TIMESTAMP_US)

    getattr(native, operation).assert_not_called()


def test_rebinding_the_same_source_reuses_its_native_handle(
    native: MagicMock,
) -> None:
    context = bridge.RecordingContext()
    context.bind_source(bridge.RobotInstanceSource(ROBOT_ID, ROBOT_INSTANCE))
    context.bind_source(bridge.RobotInstanceSource(ROBOT_ID, ROBOT_INSTANCE))

    native.RecordingSource.robot_instance.assert_called_once()
    assert context.source == ROBOT_SOURCE


@pytest.mark.parametrize(
    "build",
    [
        lambda: bridge.RobotInstanceSource(""),
        lambda: bridge.ImportItemSource("", "session-1", "episode-7"),
        lambda: bridge.ImportItemSource(ROBOT_ID, "", "episode-7"),
        lambda: bridge.ImportItemSource(ROBOT_ID, "session-1", ""),
    ],
)
def test_a_source_needs_every_identifier(build) -> None:
    with pytest.raises(ValueError):
        build()


def test_two_items_of_one_session_are_distinct_sources() -> None:
    first = bridge.ImportItemSource(ROBOT_ID, "session-1", "episode-1")
    second = bridge.ImportItemSource(ROBOT_ID, "session-1", "episode-2")

    assert first != second
    assert first != bridge.RobotInstanceSource(ROBOT_ID)
