"""Tests for the RGB-only gating of the lossy video codec in _EncoderManager.

Only RGB honours a lossy codec; depth always keeps its lossless storage even
under ``h264_medium``. Tests inject a ConfigWatcher seeded with a fixed codec.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from neuracore_types import DataType

from neuracore.data_daemon.config_manager.config_watcher import ConfigWatcher
from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.recording_encoding_disk_manager.lifecycle import (
    encoder_manager as em,
)


def _resolve(codec: str | None, data_type: DataType) -> dict[str, str] | None:
    """Resolve lossy-only options for `data_type` given a fixed cached codec.

    The encoder path re-resolves the watcher synchronously before reading the
    codec (so a codec set just before recording is honoured), so the injected
    resolver returns the same fixed codec the cache is seeded with.
    """
    config = DaemonConfig(video_codec=codec)
    watcher = ConfigWatcher(initial_config=config, resolver=lambda: config)
    fake_self = SimpleNamespace(_config_watcher=watcher)
    return em._EncoderManager._resolve_lossy_only_options(fake_self, data_type)


def test_rgb_honours_lossy_codec() -> None:
    assert _resolve("h264_medium", DataType.RGB_IMAGES) == {
        "crf": "23",
        "preset": "medium",
    }


@pytest.mark.parametrize(
    "data_type",
    [DataType.DEPTH_IMAGES, DataType.JOINT_POSITIONS, DataType.POINT_CLOUDS],
)
def test_non_rgb_keeps_lossless_even_when_lossy_configured(
    data_type: DataType,
) -> None:
    # Gated off before the codec matters: depth keeps its lossless storage;
    # non-video types never carry lossy-only options.
    assert _resolve("h264_medium", data_type) is None


def test_rgb_default_codec_uses_default_encoders() -> None:
    assert _resolve(None, DataType.RGB_IMAGES) is None
