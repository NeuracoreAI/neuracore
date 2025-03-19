import io
import json
import logging
import time
from typing import Optional

import av
import numpy as np
import requests

from neuracore.core.auth import get_auth
from neuracore.core.const import API_URL
from neuracore.core.streaming.resumable_upload import ResumableUpload

logger = logging.getLogger(__name__)

PTS_FRACT = 1000000  # Timebase for pts in microseconds


class StreamingVideoEncoder:
    """A video encoder that handles variable framerate and streams."""

    def __init__(
        self,
        resumable_upload: ResumableUpload,
        width: int,
        height: int,
        codec: str = "libx264rgb",
        pixel_format: str = "rgb24",
        chunk_size: int = 256 * 1024,  # 256 KB default chunk size
        framerate_cap: Optional[float] = None,
    ):
        """
        Initialize a streaming video encoder.

        Args:
            resumable_upload: Resumable upload handler
            width: Frame width
            height: Frame height
            codec: Video codec
            pixel_format: Pixel format
            chunk_size: Size of chunks to upload
            framerate_cap: Optional framerate cap (frames per second) excessive frames will be dropped
        """
        self.uploader = resumable_upload
        self.width = width
        self.height = height
        self.min_frame_interval = 0 if framerate_cap is None else 1.0 / self.framerate_cap

        # Ensure chunk_size is a multiple of 256 KiB
        if chunk_size % (256 * 1024) != 0:
            self.chunk_size = ((chunk_size // (256 * 1024)) + 1) * 256 * 1024
            logger.info(
                f"Adjusted chunk size to {self.chunk_size/1024:.0f} "
                "KiB to ensure it's a multiple of 256 KiB"
            )
        else:
            self.chunk_size = chunk_size

        # Create in-memory buffer
        self.buffer = io.BytesIO()

        # Open output container to write to memory buffer
        self.container = av.open(
            self.buffer,
            mode="w",
            format="mp4",
            options={"movflags": "frag_keyframe+empty_moov"},
        )

        # Create video stream
        self.stream = self.container.add_stream(codec)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = pixel_format
        self.stream.codec_context.options = {"crf": "0", "preset": "ultrafast"}

        # Set a very precise timebase (microseconds)
        from fractions import Fraction

        self.stream.time_base = Fraction(1, PTS_FRACT)

        # Keep track of timestamps
        self.first_timestamp = None
        self.last_pts = None
        self.last_frame_time: float = 0.0

        # Track bytes and buffer positions
        self.total_bytes_written = 0
        self.last_upload_position = 0

        # Create a dedicated buffer for upload chunks
        self.upload_buffer = bytearray()
        self.last_write_position = 0
        self.timestamps = []

    def add_frame(self, frame_data: np.ndarray, timestamp: float) -> None:
        """
        Add frame to the video with timestamp and stream if buffer large enough.

        Args:
            frame_data: RGB frame data as numpy array with shape (height, width, 3)
            timestamp: Frame timestamp in seconds (can be irregular)
        """

        # Handle first frame timestamp
        if self.first_timestamp is None:
            self.first_timestamp = timestamp

        # Framerate cap check
        if timestamp - self.last_frame_time < self.min_frame_interval:
            return  # Discard frame if below cap

        self.last_frame_time = timestamp

        # Calculate pts in timebase units (microseconds)
        relative_time = timestamp - self.first_timestamp
        pts = int(relative_time * PTS_FRACT)  # Convert to microseconds

        if self.last_pts is not None and pts <= self.last_pts:
            pts = self.last_pts + 1

        self.last_pts = pts

        frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        frame.pts = pts

        for packet in self.stream.encode(frame):
            self.container.mux(packet)

        current_pos = self.buffer.tell()
        current_chunk_size = current_pos - self.last_write_position
        if current_chunk_size >= self.chunk_size:
            self.buffer.seek(self.last_write_position)
            chunk_data = self.buffer.read(current_chunk_size)
            self.upload_buffer.extend(chunk_data)
            self.last_write_position = current_pos
            self.buffer.seek(current_pos)
            self._upload_chunks()

        self.total_bytes_written = current_pos
        self.timestamps.append(timestamp)

    def _upload_chunks(self) -> None:
        while len(self.upload_buffer) >= self.chunk_size:
            chunk = bytes(self.upload_buffer[: self.chunk_size])
            self.upload_buffer = self.upload_buffer[self.chunk_size :]
            success = self.uploader.upload_chunk(chunk, is_final=False)

            if not success:
                logger.warning("Failed to upload chunk, retrying once more...")
                time.sleep(1)
                success = self.uploader.upload_chunk(chunk, is_final=False)
                if not success:
                    logger.error("Failed to upload chunk again, will try later")
                    self.upload_buffer = bytearray(chunk) + self.upload_buffer

    def finish(self) -> None:
        for packet in self.stream.encode(None):
            self.container.mux(packet)

        self.container.close()

        current_pos = self.buffer.tell()
        current_chunk_size = current_pos - self.last_write_position
        self.buffer.seek(self.last_write_position)
        chunk_data = self.buffer.read(current_chunk_size)
        self.upload_buffer.extend(chunk_data)
        self.last_write_position = current_pos

        final_chunk = bytes(self.upload_buffer)
        success = self.uploader.upload_chunk(final_chunk, is_final=True)

        if not success:
            logger.warning("Failed to upload final chunk, retrying...")
            time.sleep(1)
            success = self.uploader.upload_chunk(final_chunk, is_final=True)
            if not success:
                logger.error("Failed to upload final chunk.")

        logger.info(
            "Video encoding and upload complete: "
            f"{self.uploader.total_bytes_uploaded} bytes"
        )
        self._upload_timestamps()

    def _upload_timestamps(self) -> None:
        """
        Upload timestamps to the server.
        """
        camera_id = f"{self.uploader.sensor_type.value}_{self.uploader.sensor_name}"
        stream_name = f"cameras/{camera_id}/timestamps.json"
        upload_url_response = requests.get(
            f"{API_URL}/recording/{self.uploader.recording_id}/json_upload_url?filename={stream_name}",
            headers=get_auth().get_headers(),
        )
        upload_url_response.raise_for_status()
        upload_url = upload_url_response.json()["url"]
        data = json.dumps(self.timestamps)
        logger.info(f"Uploading {len(data)} bytes to {upload_url}")
        response = requests.put(
            upload_url, headers={"Content-Length": str(len(data))}, data=data
        )
        response.raise_for_status()
        self.timestamps = []
