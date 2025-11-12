"""Video streaming module for WebRTC media tracks.

This module provides video source management and streaming capabilities using
aiortc and av libraries. It supports both regular RGB video streams and depth
video streams with automatic normalization.

Constants:
    STREAMING_FPS: Target frames per second for video streaming.
    VIDEO_CLOCK_RATE: Clock rate for video timing (90kHz standard).
    VIDEO_TIME_BASE: Time base for video frames.
    TIMESTAMP_DELTA: Timestamp increment between frames.
"""

import asyncio
import fractions
import math
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

import av
import numpy as np
from aiortc import MediaStreamTrack
from neuracore_types import CameraData

from neuracore.core.streaming.p2p.provider.json_source import JSONSource
from neuracore.core.utils.image_string_encoder import ImageStringEncoder

from ..enabled_manager import EnabledManager

av.logging.set_level(None)

STREAMING_FPS = 30
VIDEO_CLOCK_RATE = 90000
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
TIMESTAMP_DELTA = int(VIDEO_CLOCK_RATE / STREAMING_FPS)


@dataclass
class VideoSource:
    """A video source for streaming RGB video frames.

    This class manages video frame data and provides methods to add new frames
    and retrieve the most recent frame for streaming. Each video source has a
    unique media identifier and maintains the last received frame.

    Attributes:
        stream_enabled: Manager for controlling stream state.
        mid: Unique media identifier for this video source.
        _last_frame: The most recently added video frame data.
    """

    stream_enabled: EnabledManager
    mid: str = field(default_factory=lambda: uuid4().hex)
    _last_frame: np.ndarray = field(
        default_factory=lambda: np.zeros((480, 640, 3), dtype=np.uint8)
    )
    _last_camera_data: Optional[CameraData] = None
    custom_data_source: Optional[JSONSource] = None

    def add_frame(self, frame_data: np.ndarray, camera_data: CameraData) -> None:
        """Add a new video frame to the source.

        Args:
            frame_data: RGB video frame data as a numpy array.
                Should be in HWC format (Height, Width, Channels).
            camera_data: Extra metadata about the frame.
        """
        self._last_frame = frame_data
        self._last_camera_data = camera_data
        if self.custom_data_source:
            self.custom_data_source.publish({
                **camera_data.model_dump(mode="json"),
                "frame": ImageStringEncoder.encode_image(frame_data, cap_size=True),
            })

    def get_last_frame(self) -> av.VideoFrame:
        """Get the most recent video frame.

        Returns:
            av.VideoFrame: The last frame converted to av.VideoFrame format
                with RGB24 pixel format.
        """
        return av.VideoFrame.from_ndarray(self._last_frame, format="rgb24")

    def get_video_track(self) -> "VideoTrack":
        """Create a video track for streaming this source.

        Returns:
            VideoTrack: A new video track instance configured for this source.

        Raises:
            RuntimeError: If streaming is not currently enabled.
        """
        if self.stream_enabled.is_disabled():
            raise RuntimeError("Streaming is not enabled")
        consumer = VideoTrack(self)
        self.stream_enabled.add_listener(EnabledManager.DISABLED, consumer.stop)
        return consumer

    def get_neuracore_custom_track(
        self,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> JSONSource:
        """Gets a data source for the video frames encoded as dataUri's.

        Args:
            loop: the event loop to run on. Defaults to the running loop if not
                    provided.


        Returns:
            JSONSource: the source for the video frames.
        """
        if not self.custom_data_source:
            self.custom_data_source = JSONSource(
                mid=self.mid, stream_enabled=self.stream_enabled, loop=loop
            )
            if self._last_frame is not None and self._last_camera_data is not None:
                self.custom_data_source.publish({
                    **self._last_camera_data.model_dump(mode="json"),
                    "frame": ImageStringEncoder.encode_image(
                        self._last_frame, cap_size=True
                    ),
                })

        return self.custom_data_source


@dataclass
class DepthVideoSource(VideoSource):
    """A specialized video source for streaming depth video data.

    This class extends VideoSource to handle depth data by automatically
    normalizing depth values to the 0-1 range and converting them to RGB
    grayscale images for streaming.

    Attributes:
        _maximum_depth: Maximum depth value seen so far.
        _minimum_depth: Minimum depth value seen so far.
    """

    _maximum_depth: float = field(default=-math.inf, init=False)
    _minimum_depth: float = field(default=math.inf, init=False)

    def get_last_frame(self) -> av.VideoFrame:
        """Get the most recent depth frame normalized to RGB format.

        The depth data is normalized using the running min/max values and
        converted to a 3-channel RGB image where all channels contain the
        same grayscale depth representation.

        Returns:
            av.VideoFrame: Normalized depth frame as RGB24 format where each
                channel contains the same grayscale depth visualization.
        """
        # Ensure _last_frame is in [0, 1] range
        self._maximum_depth = max(self._maximum_depth, self._last_frame.max())
        self._minimum_depth = min(self._minimum_depth, self._last_frame.min())

        normalized_frame = np.clip(
            (self._last_frame - self._minimum_depth)
            / (self._maximum_depth - self._minimum_depth),
            0,
            1,
        )

        # Convert to uint8 safely
        uint8_frame = (normalized_frame * 255).astype(np.uint8)
        # Stack three identical grayscale frames into an RGB image
        rgb_frame = np.stack([uint8_frame] * 3, axis=-1)
        return av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")


class VideoTrack(MediaStreamTrack):
    """A WebRTC media track for streaming video from a VideoSource.

    This class implements the MediaStreamTrack interface to provide video
    streaming capabilities with proper timing and frame delivery. It maintains
    consistent frame rates and handles timestamp generation for WebRTC.

    Attributes:
        kind: Media track type, always "video".
        source: The video source providing frame data.
        _mid: Media identifier copied from the source.
        _ended: Flag indicating if the track has ended.
        _start: Start time for timestamp calculations.
        _timestamp: Current timestamp for frame timing.
    """

    kind = "video"

    def __init__(self, source: VideoSource) -> None:
        """Initialize the video track with a video source.

        Args:
            source (VideoSource): The video source to stream from.
        """
        super().__init__()
        self.source = source
        self._mid = source.mid
        self._ended: bool = False
        self._start: Optional[float] = None
        self._timestamp: int = 0

    @property
    def mid(self) -> str:
        """Get the media identifier for this track.

        Returns:
            str: The unique media identifier.
        """
        return self._mid

    async def next_timestamp(self) -> int:
        """Calculate and wait for the next frame timestamp.

        This method implements frame rate control by calculating the appropriate
        timestamp for the next frame and sleeping if necessary to maintain the
        target frame rate.

        Returns:
            int: The timestamp for the next frame in video clock units.
        """
        if self._start is None:
            self._start = time.time()
            return self._timestamp

        self._timestamp += TIMESTAMP_DELTA
        wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        return self._timestamp

    async def recv(self) -> av.VideoFrame:
        """Receive the next video frame for streaming.

        This method is called by the WebRTC framework to get the next frame
        for transmission. It handles timing, frame retrieval, and proper
        timestamp assignment.

        Returns:
            av.VideoFrame: The next video frame with proper PTS and time base
                settings for WebRTC streaming.

        Raises:
            Exception: Re-raises any exception that occurs during frame
                processing after logging the error.
        """
        try:
            pts = await self.next_timestamp()
            frame_data = self.source.get_last_frame()
            frame_data.time_base = VIDEO_TIME_BASE
            frame_data.pts = pts
            return frame_data
        except Exception as e:
            print(f"Error in receiving frame: {self.mid=} {e}")
            raise
