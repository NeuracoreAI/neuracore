"""Tests for the payload the live video source publishes."""

from unittest.mock import MagicMock

import numpy as np
from neuracore_types import RGBCameraData

from neuracore.core.streaming.p2p.provider.video_source import VideoSource


def test_published_frame_carries_the_timestamp_in_both_units():
    source = VideoSource(stream_enabled=MagicMock())
    source.custom_data_source = MagicMock()

    source.add_frame(
        RGBCameraData(timestamp_us=12_500_000), np.zeros((8, 8, 3), dtype=np.uint8)
    )

    published = source.custom_data_source.publish.call_args.args[0]
    assert (published["timestamp"], published["timestamp_us"]) == (12.5, 12_500_000)
