from __future__ import annotations

from unittest.mock import MagicMock

from neuracore.data_daemon import bridge as recording_context


def _bound_context(monkeypatch) -> tuple[recording_context.RecordingContext, MagicMock]:
    native = MagicMock()
    monkeypatch.setattr(recording_context, "_load_native", lambda: native)
    context = recording_context.RecordingContext()
    context.bind_source("robot-id-1", robot_instance=3)
    return context, native


def test_stop_publishes_the_tags_with_the_stop(monkeypatch) -> None:
    context, native = _bound_context(monkeypatch)

    context.stop_recording(1_000, tags=["good-run", "edge"])

    native.stop_recording.assert_called_once_with(
        "robot-id-1", 3, 1_000, ["good-run", "edge"]
    )


def test_stop_without_tags_publishes_an_empty_list(monkeypatch) -> None:
    context, native = _bound_context(monkeypatch)

    context.stop_recording(1_000)

    native.stop_recording.assert_called_once_with("robot-id-1", 3, 1_000, [])
