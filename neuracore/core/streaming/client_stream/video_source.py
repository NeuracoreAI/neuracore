import asyncio
import fractions
import math
import time
import weakref
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

import av
import numpy as np
from aiortc import MediaStreamTrack

av.logging.set_level(None)

STREAMING_FPS = 30
VIDEO_CLOCK_RATE = 90000
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
TIMESTAMP_DELTA = int(VIDEO_CLOCK_RATE / STREAMING_FPS)


@dataclass
class VideoSource:
    mid: str = field(default_factory=lambda: uuid4().hex)
    _last_frame: np.ndarray = field(
        default_factory=lambda: np.zeros((480, 640, 3), dtype=np.uint8)
    )
    _consumers: weakref.WeakSet["VideoTrack"] = field(default_factory=weakref.WeakSet)

    def add_frame(self, frame_data: np.ndarray):
        self._last_frame = frame_data

    def get_last_frame(self) -> av.VideoFrame:
        return av.VideoFrame.from_ndarray(self._last_frame, format="rgb24")

    def get_video_track(self):
        consumer = VideoTrack(self)
        self._consumers.add(consumer)
        return consumer

    def stop(self):
        """Stop the source"""
        for consumer in self._consumers:
            consumer.stop()


@dataclass
class DepthVideoSource(VideoSource):
    _maximum_depth = -math.inf
    _minimum_depth = math.inf

    def get_last_frame(self) -> av.VideoFrame:
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
    kind = "video"

    def __init__(self, source: VideoSource):
        super().__init__()
        self.source = source
        self._mid = source.mid
        self._ended: bool = False
        self._start: Optional[float] = None
        self._timestamp: int = 0

    @property
    def mid(self) -> str:
        return self._mid

    async def next_timestamp(self) -> int:
        if self._start is None:
            self._start = time.time()
            return self._timestamp

        self._timestamp += TIMESTAMP_DELTA
        wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        return self._timestamp

    async def recv(self) -> av.VideoFrame:
        """Receive the next frame"""
        try:
            if self._ended:
                raise Exception("Track has ended")
            pts = await self.next_timestamp()
            frame_data = self.source.get_last_frame()
            frame_data.time_base = VIDEO_TIME_BASE
            frame_data.pts = pts

            return frame_data
        except Exception as e:
            print(f"Error in receiving frame: {self.mid=} {e}")
            raise

    def stop(self):
        """Stop the track"""
        self._ended = True
