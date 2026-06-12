"""Tests for RecordingCloseRegistry close-state transitions."""

from datetime import datetime, timezone

from neuracore.data_daemon.communications_management.consumer.models import (
    RecordingCloseRegistry,
    RecordingClosingState,
)


def _closing_state() -> RecordingClosingState:
    return RecordingClosingState(
        producer_stop_sequence_numbers={"producer-1": 5},
        stop_requested_at=datetime.now(timezone.utc),
    )


def test_mark_closing_then_close_marks_recording_closed() -> None:
    registry = RecordingCloseRegistry()
    registry.mark_closing("recording-1", _closing_state())

    assert not registry.is_closed("recording-1")
    assert registry.get_closing("recording-1") is not None

    registry.close("recording-1")

    assert registry.is_closed("recording-1")
    assert registry.get_closing("recording-1") is None
    assert list(registry.items()) == []


def test_mark_closing_ignored_for_closed_recording() -> None:
    """A closed recording must never re-enter the closing set.

    Re-entering would allow a second close to emit a zeroed
    expected_trace_count after the unique-trace registry was cleared.
    """
    registry = RecordingCloseRegistry()
    registry.mark_closing("recording-1", _closing_state())
    registry.close("recording-1")

    registry.mark_closing("recording-1", _closing_state())

    assert registry.get_closing("recording-1") is None
    assert list(registry.items()) == []
    assert registry.is_closed("recording-1")


def test_mark_closing_unrelated_recording_unaffected_by_closed_one() -> None:
    registry = RecordingCloseRegistry()
    registry.mark_closing("recording-1", _closing_state())
    registry.close("recording-1")

    registry.mark_closing("recording-2", _closing_state())

    assert registry.get_closing("recording-2") is not None
    assert [recording_id for recording_id, _ in registry.items()] == ["recording-2"]
