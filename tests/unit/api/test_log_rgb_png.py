import io

import numpy as np
import pytest
from PIL import Image

import neuracore as nc
from neuracore.api.logging import _png_dimensions


def _png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


def test_png_dimensions_reads_header():
    assert _png_dimensions(_png(5, 4)) == (5, 4)


def test_log_rgb_png_rejects_non_png():
    with pytest.raises(ValueError, match="not a PNG"):
        nc.log_rgb_png("cam", b"not a png", dry_run=True)


def test_log_rgb_png_dry_run_does_not_need_a_robot():
    nc.log_rgb_png("cam", _png(5, 4), dry_run=True)


def _robot_with_stream(monkeypatch, live: bool):
    from unittest.mock import MagicMock

    from neuracore.api import logging as api_logging

    robot = MagicMock()
    robot.get_data_stream.return_value = None
    monkeypatch.setattr(api_logging, "_get_robot", lambda *_: robot)
    manager = MagicMock()
    manager.is_disabled.return_value = not live
    monkeypatch.setattr(
        api_logging, "get_provide_live_data_enabled_manager", lambda: manager
    )
    publish = MagicMock()
    monkeypatch.setattr(api_logging, "_publish_video_to_p2p", publish)
    return robot, publish


def test_log_rgb_png_skips_decode_when_live_data_is_off(monkeypatch):
    robot, publish = _robot_with_stream(monkeypatch, live=False)

    nc.log_rgb_png("cam", _png(5, 4), timestamp=1.0)

    robot._get_daemon_recording_context().log_rgb_png.assert_called_once()
    publish.assert_not_called()


def test_log_rgb_png_decodes_for_live_data(monkeypatch):
    robot, publish = _robot_with_stream(monkeypatch, live=True)

    nc.log_rgb_png("cam", _png(5, 4), timestamp=1.0)

    robot._get_daemon_recording_context().log_rgb_png.assert_called_once()
    frame = publish.call_args.args[4]
    assert frame.shape == (4, 5, 3)
