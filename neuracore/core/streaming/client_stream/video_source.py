import asyncio
import fractions
import time
import weakref
from dataclasses import dataclass, field
from typing import Optional

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
    pixel_format: str
    _last_frame: np.ndarray[np.uint8] = field(
        default_factory=lambda: np.zeros((480, 640, 3), dtype=np.uint8)
    )
    _consumers: weakref.WeakSet["VideoTrack"] = field(default_factory=weakref.WeakSet)

    def add_frame(self, frame_data: np.ndarray):
        self._last_frame = frame_data

    def get_last_frame(self) -> av.VideoFrame:
        return av.VideoFrame.from_ndarray(self._last_frame, format=self.pixel_format)

    def get_video_track(self):
        consumer = VideoTrack(self)
        self._consumers.add(consumer)
        return consumer

    def stop(self):
        """Stop the source"""
        for consumer in self._consumers:
            consumer.stop()


class VideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, source: VideoSource):
        super().__init__()
        self.source = source
        self._ended: bool = False
        self._start: Optional[float] = None
        self._timestamp: int = 0

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
        if self._ended:
            raise Exception("Track has ended")

        pts = await self.next_timestamp()
        frame_data = self.source.get_last_frame()
        frame_data.time_base = VIDEO_TIME_BASE
        frame_data.pts = pts

        return frame_data

    def stop(self):
        """Stop the track"""
        self._ended = True
