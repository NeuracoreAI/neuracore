
import asyncio
import time
from typing import Optional

from aiortc import VideoStreamTrack
from av import VideoFrame
import numpy as np


MAX_STREAMING_FPS = 30

class VideoTrack(VideoStreamTrack):
    """A media stream track for video"""

    def __init__(self):
        super().__init__()
        self._ended = False
        self._last_frame_time: float = time.time()
        self._queue: asyncio.Queue[VideoFrame] = asyncio.Queue(
            maxsize=MAX_STREAMING_FPS
        )

    def add_frame(self, frame_data: np.ndarray):
        """Add a frame to the queue with rate limiting and dropping old frames"""
        if self._ended:
            return

        current_time = time.time()
        time_diff = current_time - self._last_frame_time

        if time_diff < 1 / MAX_STREAMING_FPS:
            return  # drop frames that are to fast

        self._last_frame_time = current_time

        if self._queue.full():
            self._queue.get_nowait()

        self._queue.put_nowait(VideoFrame.from_ndarray(frame_data))

    async def get_frame(self) -> Optional[VideoFrame]:
        """Get the next frame from the queue"""
        if self._ended:
            return None

        try:
            return await asyncio.wait_for(self._queue.get(), timeout=1)
        except asyncio.TimeoutError:
            return None

    async def recv(self):
        """Receive the next frame"""
        if self._ended:
            raise Exception("Track has ended")

        frame_data = await self.get_frame()

        if frame_data is None:
            self._ended = True
            raise Exception("Track has ended")

        pts, time_base = await self.next_timestamp()
        frame_data.pts = pts
        frame_data.time_base = time_base
        return frame_data

    def stop(self):
        """Stop the track"""
        self._ended = True
