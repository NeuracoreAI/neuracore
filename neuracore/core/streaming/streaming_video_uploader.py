import io
import json
import logging
import queue
import threading
from fractions import Fraction
from typing import Callable

import av
import numpy as np
import requests

from ..auth import get_auth
from ..const import API_URL
from ..nc_types import CameraMetaData
from ..streaming.resumable_upload import ResumableUpload

logger = logging.getLogger(__name__)

PTS_FRACT = 1000000  # Timebase for pts in microseconds
CHUNK_MULTIPLE = 256 * 1024  # Chunk size multiple of 256 KiB
MB_CHUNK = 4 * CHUNK_MULTIPLE
CHUNK_SIZE = 64 * MB_CHUNK


class StreamingVideoUploader:
    """A video encoder that handles variable framerate and streams."""

    def __init__(
        self,
        recording_id: str,
        path: str,
        width: int,
        height: int,
        transform_frame: Callable[[np.ndarray], np.ndarray] | None = None,
        codec: str = "libx264",
        pixel_format: str = "yuv444p10le",
        chunk_size: int = CHUNK_SIZE,
    ):
        """
        Initialize a streaming video encoder.

        Args:
            recording_id: Recording ID
            camera_id: Camera ID
            width: Frame width
            height: Frame height
            transform_frame: Frame transformation function
            codec: Video codec
            pixel_format: Pixel format
            chunk_size: Size of chunks to upload
            framerate_cap: Optional framerate cap (frames per second) excessive frames will be dropped
        """
        self.recording_id = recording_id
        self.path = path
        self.width = width
        self.height = height
        self.transform_frame = transform_frame
        self.codec = codec
        self.pixel_format = pixel_format
        self.chunk_size = chunk_size
        self._streaming_done = False
        self._upload_queue = queue.Queue()
        # Thread will continue, even if main thread exits
        self._upload_thread = threading.Thread(target=self._upload_loop, daemon=False)
        self._upload_thread.start()

    def _thread_setup(self) -> None:
        """Setup thread for upload loop."""

        # Ensure chunk_size is a multiple of 256 KiB
        if self.chunk_size % CHUNK_MULTIPLE != 0:
            self.chunk_size = ((self.chunk_size // CHUNK_MULTIPLE) + 1) * CHUNK_MULTIPLE
            logger.info(
                f"Adjusted chunk size to {self.chunk_size/1024:.0f} "
                "KiB to ensure it's a multiple of {CHUNK_MULTIPLE} MiB"
            )

        self.uploader = ResumableUpload(
            self.recording_id, f"{self.path}/video.mp4", "video/mp4"
        )

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
        self.stream = self.container.add_stream(self.codec)
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = self.pixel_format
        self.stream.codec_context.options = {"qp": "0", "preset": "ultrafast"}

        self.stream.time_base = Fraction(1, PTS_FRACT)

        # Keep track of timestamps
        self.first_timestamp = None
        self.last_pts = None

        # Track bytes and buffer positions
        self.total_bytes_written = 0
        self.last_upload_position = 0

        # Create a dedicated buffer for upload chunks
        self.upload_buffer = bytearray()
        self.last_write_position = 0
        self.frame_metadatas: list[CameraMetaData] = []

    def _upload_loop(self) -> None:
        """
        Upload chunks in a separate thread.
        """
        self._thread_setup()
        # If final has not been called, or we still have items in the queue
        while not self._streaming_done or self._upload_queue.qsize() > 0:
            try:
                frame_data, json_data = self._upload_queue.get(timeout=0.1)
                if frame_data is None:
                    break
                self._add_frame(frame_data, json_data)
            except queue.Empty:
                continue

        # Flush encoder
        for packet in self.stream.encode(None):
            self.container.mux(packet)

        # Close the container to finalize the MP4
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
            raise RuntimeError("Failed to upload final chunk")

        logger.info(
            "Video encoding and upload complete: "
            f"{self.uploader.total_bytes_uploaded} bytes"
        )
        self._upload_json_data()

    def add_frame(self, frame_data: np.ndarray, metadata: CameraMetaData) -> None:
        """
        Add frame to the video with timestamp and stream if buffer large enough.

        Args:
            frame_data: RGB frame data as numpy array with shape (height, width, 3)
            json_data: JSON data to log with the frame
        """
        self._upload_queue.put((frame_data, metadata))

    def _add_frame(
        self, frame_data: np.ndarray, frame_metadata: CameraMetaData
    ) -> None:
        """
        Add frame to the video with timestamp and stream if buffer large enough.

        Args:
            frame_data: RGB frame data as numpy array with shape (height, width, 3)
            json_data: JSON data to log with the frame
        """

        if self.transform_frame is not None:
            frame_data = self.transform_frame(frame_data)

        # Handle first frame timestamp
        if self.first_timestamp is None:
            self.first_timestamp = frame_metadata.timestamp

        # Calculate pts in timebase units (microseconds)
        relative_time = frame_metadata.timestamp - self.first_timestamp
        pts = int(relative_time * PTS_FRACT)  # Convert to microseconds

        if self.last_pts is not None and pts <= self.last_pts:
            pts = self.last_pts + 1

        self.last_pts = pts

        frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        frame = frame.reformat(format=self.pixel_format)
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
        self.frame_metadatas.append(frame_metadata)

    def _upload_chunks(self) -> None:
        while len(self.upload_buffer) >= self.chunk_size:
            chunk = bytes(self.upload_buffer[: self.chunk_size])
            self.upload_buffer = self.upload_buffer[self.chunk_size :]
            success = self.uploader.upload_chunk(chunk, is_final=False)
            logger.info(f"Uploaded {len(chunk)} bytes")

            if not success:
                raise RuntimeError("Failed to upload chunk")

    def finish(self) -> threading.Thread:
        """
        Finish encoding and upload any remaining data.
        """
        # Note we dont join on the (non-daemon) thread as we dont want to block
        self._upload_queue.put((None, None))
        self._streaming_done = True
        return self._upload_thread

    def _upload_json_data(self) -> None:
        """
        Upload timestamps to the server.
        """
        params = {
            "filepath": f"{self.path}/metadata.json",
            "content_type": "application/json",
        }
        upload_url_response = requests.get(
            f"{API_URL}/recording/{self.uploader.recording_id}/resumable_upload_url",
            params=params,
            headers=get_auth().get_headers(),
        )
        upload_url_response.raise_for_status()
        upload_url = upload_url_response.json()["url"]
        for i in range(0, len(self.frame_metadatas)):
            self.frame_metadatas[i].frame_idx = i
        data = json.dumps([fm.model_dump() for fm in self.frame_metadatas])
        logger.info(f"Uploading {len(data)} bytes to {upload_url}")
        response = requests.put(
            upload_url, headers={"Content-Length": str(len(data))}, data=data
        )
        response.raise_for_status()
        self.frame_metadatas = []
