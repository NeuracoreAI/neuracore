import io
from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Generator

import av
import numpy as np

from neuracore_new_data_format.ncdata import NCData


class MarshallingOutput(ABC):
    pass


class DataMarshaller(ABC):
    @abstractmethod
    def write(self, data: NCData) -> None:
        raise NotImplementedError("write not implemented")

    @abstractmethod
    def read(
        self,
    ) -> Generator["NCData", None, None]:
        raise NotImplementedError("read not implemented")

    def close(self) -> None:
        pass


PTS_FRACT = 1000000  # Timebase for pts in microseconds


class CameraDataEncoder:
    TABLE_NAME = "camera_data"

    def __init__(
        self,
        codec: str = "libx264",
        pixel_format: str = "yuv444p10le",
    ):

        self.codec = codec
        self.pixel_format = pixel_format

        self.buffer = io.BytesIO()
        self.container = av.open(
            self.buffer,
            mode="w",
            format="mp4",
            options={"movflags": "frag_keyframe+empty_moov"},
        )
        self.stream = None

        self.start_ts = None  # first timestamp
        self.time_base = None  # time base of the stream
        self.last_pts = None  # last pts

    def add_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """
        Add a numpy RGB frame to the encoder with a timestamp in seconds.
        PTS is calculated relative to the first frame.
        """
        if self.stream is None:
            h, w, _ = frame.shape
            self.stream = self.container.add_stream(self.codec)
            self.stream.width = w
            self.stream.height = h
            self.stream.pix_fmt = self.pixel_format
            self.stream.options = {
                "preset": "ultrafast",
            }
            self.stream.codec_context.options = {
                "qp": "0",  # lossless quantization
                "preset": "ultrafast",  # low compression fast speed
            }
            # let PyAV pick time_base automatically (usually 1/1000 or 1/90000)
            self.time_base = Fraction(1, PTS_FRACT)
            self.stream.time_base = self.time_base

        if self.start_ts is None:
            self.start_ts = timestamp

        rel_ts = timestamp - self.start_ts
        pts = int(rel_ts * PTS_FRACT)  # Convert to microseconds

        # Ensure pts is monotonically increasing (required by most codecs)
        if self.last_pts is not None and pts <= self.last_pts:
            pts = self.last_pts + 1

        self.last_pts = pts

        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        av_frame = av_frame.reformat(format=self.pixel_format)
        av_frame.pts = pts
        av_frame.time_base = self.time_base

        for packet in self.stream.encode(av_frame):
            self.container.mux(packet)

    def read_frames(self, blob: bytes) -> list[np.ndarray]:
        """
        Decode MP4 bytes back into frames.
        Returns list of frame ndarray
        """
        buffer = io.BytesIO(blob)
        container = av.open(buffer, mode="r", format="mp4")

        frames = []
        for packet in container.demux(video=0):
            for frame in packet.decode():
                frames.append(frame.to_ndarray(format="rgb24"))
        return frames

    def get_blob(self) -> bytes:
        """Finalize encoding and write MP4 bytes to SQLite."""
        if self.stream is not None:
            for packet in self.stream.encode(None):
                self.container.mux(packet)
        self.container.close()

        return self.buffer.getvalue()
