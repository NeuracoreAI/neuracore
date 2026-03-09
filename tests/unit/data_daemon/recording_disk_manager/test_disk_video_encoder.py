from __future__ import annotations

from pathlib import Path

import numpy as np

from neuracore.data_daemon.recording_encoding_disk_manager.encoding import (
    disk_video_encoder as disk_video_encoder_module,
)


class _FakeEncoder:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def add_frame(self, *, timestamp: float, np_frame: np.ndarray) -> None:
        return None

    def finish(self) -> None:
        return None


def _make_kwargs(tmp_path: Path) -> dict:
    return {
        "filepath": tmp_path / "video.mp4",
        "width": 4,
        "height": 3,
        "codec": "libx264",
        "pixel_format": "yuv420p",
        "codec_context_options": {"qp": "23"},
    }


def test_factory_prefers_ffmpeg_when_available(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(disk_video_encoder_module, "_is_ffmpeg_available", lambda: True)
    monkeypatch.setattr(
        disk_video_encoder_module, "FfmpegDiskVideoEncoder", _FakeEncoder
    )

    class _ShouldNotBeUsed(_FakeEncoder):
        def __init__(self, **kwargs) -> None:  # noqa: ARG002
            raise AssertionError("PyAV encoder should not be used")

    monkeypatch.setattr(
        disk_video_encoder_module, "PyAVDiskVideoEncoder", _ShouldNotBeUsed
    )

    encoder = disk_video_encoder_module.DiskVideoEncoder(**_make_kwargs(tmp_path))
    assert isinstance(encoder, _FakeEncoder)


def test_factory_uses_pyav_when_ffmpeg_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        disk_video_encoder_module, "_is_ffmpeg_available", lambda: False
    )
    monkeypatch.setattr(disk_video_encoder_module, "PyAVDiskVideoEncoder", _FakeEncoder)

    encoder = disk_video_encoder_module.DiskVideoEncoder(**_make_kwargs(tmp_path))
    assert isinstance(encoder, _FakeEncoder)


def test_factory_falls_back_to_pyav_on_ffmpeg_init_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(disk_video_encoder_module, "_is_ffmpeg_available", lambda: True)

    class _FailingFfmpeg(_FakeEncoder):
        def __init__(self, **kwargs) -> None:  # noqa: ARG002
            raise RuntimeError("ffmpeg init failed")

    monkeypatch.setattr(
        disk_video_encoder_module, "FfmpegDiskVideoEncoder", _FailingFfmpeg
    )
    monkeypatch.setattr(disk_video_encoder_module, "PyAVDiskVideoEncoder", _FakeEncoder)

    encoder = disk_video_encoder_module.DiskVideoEncoder(**_make_kwargs(tmp_path))
    assert isinstance(encoder, _FakeEncoder)
