"""Tests for the depth video source."""

import json
from unittest.mock import MagicMock

import numpy as np
from neuracore_types import DataType, DepthCameraData, RobotStreamTrack

from neuracore.core.streaming.p2p.consumer.sync_point_parser import parse_sync_point
from neuracore.core.streaming.p2p.provider.video_source import DepthVideoSource
from neuracore.core.utils.depth_utils import depth_to_log_gray


def _depth_source() -> DepthVideoSource:
    source = DepthVideoSource(stream_enabled=MagicMock())
    source.custom_data_source = MagicMock()
    return source


def test_depth_data_channel_round_trips_every_sensor_value():
    """Send uint16 depth that the consumer parses back unchanged with its scale."""
    source = _depth_source()
    frame = (np.arange(48, dtype=np.uint16).reshape(6, 8) * 1300) % 65535
    frame[0, 0] = 0
    camera_data = DepthCameraData(timestamp=2.5, depth_scale_m=1e-4)

    source.add_frame(camera_data, frame)

    message = source.custom_data_source.publish.call_args.args[0]
    track = RobotStreamTrack(
        robot_id="robot_1",
        robot_instance=0,
        stream_id="stream_1",
        data_type=DataType.DEPTH_IMAGES,
        label="depth",
        mid="mid_1",
    )
    parsed = parse_sync_point(json.dumps(message), track)
    depth = parsed.data[DataType.DEPTH_IMAGES]["depth"]
    assert depth.depth_scale_m == 1e-4
    np.testing.assert_array_equal(depth.frame, frame)


def test_depth_video_sends_log_grey_levels_of_absolute_depth():
    """Send each depth as its log-curve grey level whatever else the frame holds."""
    source = _depth_source()
    camera_data = DepthCameraData(timestamp=1.0, depth_scale_m=1e-3)
    frame = np.full((4, 4), 500, dtype=np.uint16)
    frame[0, 0] = 6000
    frame[0, 1] = 0
    source.add_frame(camera_data, frame)

    rgb = source.get_last_frame().to_ndarray(format="rgb24")

    expected = depth_to_log_gray(frame.astype(np.float32) * np.float32(1e-3))
    for channel in range(3):
        np.testing.assert_array_equal(rgb[..., channel], expected)
    assert rgb[0, 1].tolist() == [0, 0, 0]
