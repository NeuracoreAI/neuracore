from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from neuracore.data_daemon.recording_encoding_disk_manager.encoding import (
    video_trace as video_trace_module,
)
from neuracore.data_daemon.recording_encoding_disk_manager.encoding.json_trace import (
    JsonTrace,
)


def test_json_trace_finish_writes_empty_array(tmp_path: Path) -> None:
    out_dir = tmp_path / "trace"
    trace = JsonTrace(output_dir=out_dir, chunk_size=32)
    trace.finish()

    content = (out_dir / "trace.json").read_text(encoding="utf-8")
    assert content == "[]"


def test_json_trace_writes_valid_json_array(tmp_path: Path) -> None:
    out_dir = tmp_path / "trace"
    trace = JsonTrace(output_dir=out_dir, chunk_size=32)

    trace.add_frame({"a": 1})
    trace.add_frame({"b": 2})
    trace.finish()

    content = (out_dir / "trace.json").read_text(encoding="utf-8")
    parsed = json.loads(content)
    assert parsed == [{"a": 1}, {"b": 2}]


def test_json_trace_comma_handling_no_trailing_or_leading_commas(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "trace"
    trace = JsonTrace(output_dir=out_dir, chunk_size=32)

    trace.add_frame({"a": 1})
    trace.finish()

    content = (out_dir / "trace.json").read_text(encoding="utf-8")
    assert content.startswith("[")
    assert content.endswith("]")
    assert ",]" not in content
    assert "[," not in content
    assert json.loads(content) == [{"a": 1}]


def test_json_trace_chunking_still_produces_valid_json(tmp_path: Path) -> None:
    out_dir = tmp_path / "trace"
    trace = JsonTrace(output_dir=out_dir, chunk_size=8)

    for i in range(50):
        trace.add_frame({"i": i, "payload": "x" * 20})
    trace.finish()

    parsed = json.loads((out_dir / "trace.json").read_text(encoding="utf-8"))
    assert len(parsed) == 50
    assert parsed[0]["i"] == 0
    assert parsed[-1]["i"] == 49


def test_json_trace_preserves_unicode_when_ensure_ascii_false(tmp_path: Path) -> None:
    out_dir = tmp_path / "trace"
    trace = JsonTrace(output_dir=out_dir, chunk_size=32)

    trace.add_frame({"msg": "café"})
    trace.finish()

    content = (out_dir / "trace.json").read_text(encoding="utf-8")
    assert "café" in content
    assert json.loads(content) == [{"msg": "café"}]


class _FakeDiskVideoEncoder:
    def __init__(self, *, filepath: Path, **kwargs: Any) -> None:
        self.filepath = filepath
        self.calls: list[tuple[float, tuple[int, ...]]] = []
        self._finished = False
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def add_frame(self, *, timestamp: float, np_frame: Any) -> None:
        shape = getattr(np_frame, "shape", ())
        self.calls.append((float(timestamp), tuple(int(x) for x in shape)))

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self.filepath.write_bytes(b"mp4")


@pytest.fixture
def patched_video_trace(monkeypatch: pytest.MonkeyPatch):
    """
    Patch VideoTrace to avoid real PyAV/FFmpeg encoding (libx264 issues).

    We still test:
    - metadata parsing (dict + list)
    - width/height required before frames
    - frame size validation
    - trace.json writing + frame_idx behaviour
    - timestamp fallback logic
    """
    monkeypatch.setattr(
        video_trace_module,
        "DiskVideoEncoder",
        _FakeDiskVideoEncoder,
        raising=True,
    )
    return video_trace_module


def test_video_trace_requires_metadata_before_frames(
    tmp_path: Path,
    patched_video_trace,
) -> None:
    pytest.importorskip("numpy")

    VideoTrace = patched_video_trace.VideoTrace
    out_dir = tmp_path / "video"
    vt = VideoTrace(output_dir=out_dir)

    frame = bytes([0, 0, 0] * (2 * 2))
    with pytest.raises(RuntimeError, match="width/height"):
        vt.add_payload(frame)


def test_video_trace_accepts_metadata_list(tmp_path: Path, patched_video_trace) -> None:
    pytest.importorskip("numpy")

    VideoTrace = patched_video_trace.VideoTrace
    out_dir = tmp_path / "video"
    vt = VideoTrace(output_dir=out_dir)

    w, h = 4, 3
    metas: list[dict[str, Any]] = [
        {"width": w, "height": h, "timestamp": 1.0},
        {"timestamp": 2.0},
    ]
    vt.add_payload(json.dumps(metas).encode("utf-8"))

    frame = bytes([10, 20, 30] * (w * h))
    vt.add_payload(frame)
    vt.finish()

    assert (out_dir / "lossy.mp4").is_file()
    assert (out_dir / "lossless.mp4").is_file()
    assert (out_dir / "trace.json").is_file()

    trace = json.loads((out_dir / "trace.json").read_text(encoding="utf-8"))
    assert isinstance(trace, list)
    assert len(trace) == 2
    assert trace[0]["width"] == w
    assert trace[0]["height"] == h
    assert trace[0]["frame_idx"] == 0
    assert trace[1]["frame_idx"] == 1


def test_video_trace_validates_frame_size(tmp_path: Path, patched_video_trace) -> None:
    pytest.importorskip("numpy")

    VideoTrace = patched_video_trace.VideoTrace
    out_dir = tmp_path / "video"
    vt = VideoTrace(output_dir=out_dir)

    meta = {"width": 4, "height": 3, "timestamp": 1.0}
    vt.add_payload(json.dumps(meta).encode("utf-8"))

    wrong = b"x" * 10
    with pytest.raises(ValueError, match="frame size"):
        vt.add_payload(wrong)


def test_video_trace_writes_outputs_when_metadata_then_frames(
    tmp_path: Path,
    patched_video_trace,
) -> None:
    pytest.importorskip("numpy")

    VideoTrace = patched_video_trace.VideoTrace
    out_dir = tmp_path / "video"
    vt = VideoTrace(output_dir=out_dir)

    w, h = 4, 3
    meta: dict[str, Any] = {"width": w, "height": h, "timestamp": 123.0}
    vt.add_payload(json.dumps(meta).encode("utf-8"))

    frame = bytes([10, 20, 30] * (w * h))
    vt.add_payload(frame)
    vt.add_payload(frame)
    vt.finish()

    assert (out_dir / "lossy.mp4").is_file()
    assert (out_dir / "lossless.mp4").is_file()
    assert (out_dir / "trace.json").is_file()

    trace = json.loads((out_dir / "trace.json").read_text(encoding="utf-8"))
    assert isinstance(trace, list)
    assert trace[0]["width"] == w
    assert trace[0]["height"] == h


def test_video_trace_timestamp_fallback_does_not_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patched_video_trace,
) -> None:
    pytest.importorskip("numpy")

    VideoTrace = patched_video_trace.VideoTrace
    out_dir = tmp_path / "video"
    vt = VideoTrace(output_dir=out_dir)

    monkeypatch.setattr(time, "time", lambda: 999.0)

    w, h = 4, 3
    meta: dict[str, Any] = {"width": w, "height": h}
    vt.add_payload(json.dumps(meta).encode("utf-8"))

    frame = bytes([10, 20, 30] * (w * h))
    vt.add_payload(frame)
    vt.finish()

    assert (out_dir / "lossy.mp4").is_file()
    assert (out_dir / "lossless.mp4").is_file()
    assert (out_dir / "trace.json").is_file()
