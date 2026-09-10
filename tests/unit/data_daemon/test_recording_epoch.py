from __future__ import annotations

from unittest.mock import MagicMock

from neuracore.data_daemon import bridge as recording_context


def test_the_epoch_comes_from_the_bound_source(monkeypatch) -> None:
    """The epoch is per source, so it must carry the source the context is
    bound to — a process contributing to another's recording included."""
    native = MagicMock()
    native.recording_epoch.return_value = 1_700
    monkeypatch.setattr(recording_context, "_load_native", lambda: native)

    context = recording_context.RecordingContext()
    context.bind_source("robot-id-1", robot_instance=3)

    assert context.recording_epoch() == 1_700
    native.recording_epoch.assert_called_once_with("robot-id-1", 3)


def test_an_unbound_context_has_no_epoch(monkeypatch) -> None:
    """Nothing has a recording before a source is bound, and asking the native
    layer without one would key the cache on nothing."""
    native = MagicMock()
    monkeypatch.setattr(recording_context, "_load_native", lambda: native)

    assert recording_context.RecordingContext().recording_epoch() is None
    native.recording_epoch.assert_not_called()
