"""Streaming video encoder with chunked upload support for robot recordings.

This module provides a streaming video uploader that encodes video frames in
real-time to MP4 format and uploads them to cloud storage using resumable
uploads. It supports variable frame rates, frame transformations, and includes
metadata handling for robot camera data.
"""

import io
import json
import logging
import queue
import threading
from fractions import Fraction
from typing import Callable, Optional

import av
import numpy as np
import requests
from neuracore_types import CameraData

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.exceptions import EncodingError
from neuracore.core.streaming.recording_state_manager import get_recording_state_manager

from .bucket_uploader import BucketUploader
from .resumable_upload import ResumableUpload

logger = logging.getLogger(__name__)

PTS_FRACT = 1000000  # Timebase for pts in microseconds
CHUNK_MULTIPLE = 256 * 1024  # Chunk size multiple of 256 KiB
MB_CHUNK = 4 * CHUNK_MULTIPLE
CHUNK_SIZE = 64 * MB_CHUNK


class StreamingVideoUploader(BucketUploader):
    """A video encoder that handles variable framerate streaming and chunked uploads.

    This class provides real-time video encoding with configurable codecs, pixel
    formats, and frame transformations. It encodes video frames to MP4 format
    and uploads them in chunks while maintaining frame metadata for synchronization.
    The encoding and upload process runs asynchronously to avoid blocking frame
    capture.
    """

    def __init__(
        self,
        recording_id: str,
        path: str,
        width: int,
        height: int,
        transform_frame: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        codec: str = "libx264",
        pixel_format: str = "yuv444p10le",
        chunk_size: int = CHUNK_SIZE,
        video_name: str = "video.mp4",
        codec_context_options: Optional[dict[str, str]] = None,
    ):
        """Initialize a streaming video encoder.

        Sets up the video encoding pipeline with specified parameters and starts
        the asynchronous upload thread. The encoder supports frame transformations
        and various codec configurations for different quality and performance
        requirements.

        Args:
            recording_id: Unique identifier for the recording session.
            path: Target directory path in the cloud storage bucket.
            width: Frame width in pixels.
            height: Frame height in pixels.
            transform_frame: Optional function to transform frames before encoding.
                Should accept and return numpy arrays of shape (height, width, 3).
            codec: Video codec to use for encoding (e.g., "libx264", "libx265").
            pixel_format: Pixel format for encoding (e.g., "yuv444p10le", "yuv420p").
            chunk_size: Size in bytes of each upload chunk. Will be adjusted
                to be a multiple of 256 KiB if necessary.
            video_name: Filename for the output video file.
            codec_context_options: Additional codec-specific options for fine-tuning
                encoding parameters.
        """
        super().__init__(recording_id)
        self.path = path
        self.width = width
        self.height = height
        self.transform_frame = transform_frame
        self.codec = codec
        self.pixel_format = pixel_format
        self.chunk_size = chunk_size
        self.video_name = video_name
        self.codec_context_options = codec_context_options
        self._streaming_done = False
        self.container_format = "mp4"
        self._check_codec_support()
        self._upload_queue: queue.Queue = queue.Queue()
        # Thread will continue, even if main thread exits
        self._upload_thread = threading.Thread(target=self._upload_loop, daemon=False)
        self._recording_manager = get_recording_state_manager()
        self._upload_thread.start()
        self._update_num_active_streams(1)

    def _thread_setup(self) -> None:
        """Setup thread-local resources for the video encoding and upload loop.

        Initializes the video encoder, adjusts chunk size requirements, creates
        the MP4 container with fragmented headers for streaming, and sets up
        tracking variables for timestamp management and buffer handling.
        """
        # Ensure chunk_size is a multiple of 256 KiB
        if self.chunk_size % CHUNK_MULTIPLE != 0:
            self.chunk_size = ((self.chunk_size // CHUNK_MULTIPLE) + 1) * CHUNK_MULTIPLE
            logger.debug(
                f"Adjusted chunk size to {self.chunk_size/1024:.0f} "
                "KiB to ensure it's a multiple of {CHUNK_MULTIPLE} MiB"
            )

        self.uploader = ResumableUpload(
            self.recording_id, f"{self.path}/{self.video_name}", "video/mp4"
        )

        # Create in-memory buffer
        self.buffer = io.BytesIO()

        # Open output container to write to memory buffer
        self.container = av.open(
            self.buffer,
            mode="w",
            format=self.container_format,
            options={"movflags": "frag_keyframe+empty_moov"},
        )

        # Create video stream
        self.stream = self.container.add_stream(self.codec, rate=PTS_FRACT)
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = self.pixel_format
        if self.codec_context_options is not None:
            self.stream.codec_context.options = self.codec_context_options

        self.stream.time_base = Fraction(1, PTS_FRACT)

        # Keep track of timestamps
        self.first_timestamp: Optional[float] = None
        self.last_pts: Optional[int] = None

        # Track bytes and buffer positions
        self.total_bytes_written = 0
        self.last_upload_position = 0

        # Create a dedicated buffer for upload chunks
        self.upload_buffer = bytearray()
        self.last_write_position = 0
        self.frame_metadatas: list[CameraData] = []

    def _check_codec_support(self) -> None:
        """Check if the current FFmpeg/PyAV setup supports our encoding target."""
        if self.container_format not in av.formats_available:
            raise EncodingError(
                f"The container` format '{self.container_format}' is not supported "
                "by PyAV. Make sure your PyAV build supports this container format."
            )

        if self.codec not in av.codecs_available:
            raise EncodingError(
                f"The codec '{self.codec}' is not available in your PyAV/FFmpeg build. "
                "Please check your FFmpeg installation and PyAV build configuration."
            )

        codec = av.Codec(self.codec, "w")
        supported_pix_fmts = {format.name for format in codec.video_formats}
        if self.pixel_format not in supported_pix_fmts:
            supported_formats_str = ", ".join(sorted(supported_pix_fmts)) or "unknown"
            raise EncodingError(
                f"The codec '{self.codec}' does not support pixel format "
                "'{self.pixel_format}'.\n"
                f"Supported formats: {supported_formats_str}\n\n"
                "You may need to:\n"
                " 1. Install full FFmpeg (e.g. `sudo apt install -y ffmpeg`)\n"
                " 2. Rebuild PyAV against it (`pip install --no-binary av av`)"
            )

    def _upload_loop(self) -> None:
        """Main video encoding and upload loop running in a separate thread.

        Processes queued frame data, encodes frames to MP4, manages chunked
        uploads, and handles encoder flushing and container finalization.
        Also uploads frame metadata as a separate JSON file after video
        encoding is complete.
        """
        # Skipping uploads if recording is expired
        if self._recording_manager.is_recording_expired(self.recording_id):
            self.finish()
            return
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

        logger.debug(
            "Video encoding and upload complete: "
            f"{self.uploader.total_bytes_uploaded} bytes"
        )
        self._upload_json_data()
        self._update_num_active_streams(-1)

    def add_frame(self, frame_data: np.ndarray, metadata: CameraData) -> None:
        """Add a video frame to the encoding queue.

        Queues a frame and its associated metadata for asynchronous encoding
        and upload. The frame will be processed in the order received and
        assigned appropriate timestamps based on the metadata.

        Args:
            frame_data: RGB frame data as numpy array with shape (height, width, 3).
                Should match the width and height specified during initialization.
            metadata: Camera data containing timestamp and other frame information
                that will be preserved for synchronization purposes.
        """
        self._upload_queue.put((frame_data, metadata))

    def _add_frame(self, frame_data: np.ndarray, frame_metadata: CameraData) -> None:
        """Encode a video frame and upload chunks when buffer threshold is reached.

        Applies frame transformations if configured, handles timestamp management
        for variable frame rates, encodes the frame to the configured format,
        and triggers chunk uploads when the buffer reaches the specified size.

        Args:
            frame_data: RGB frame data as numpy array with shape (height, width, 3).
            frame_metadata: Camera data containing timestamp and frame information.
        """
        if self.transform_frame is not None:
            frame_data = self.transform_frame(frame_data)

        # Handle first frame timestamp
        if self.first_timestamp is None:
            self.first_timestamp = frame_metadata.timestamp

        # Calculate pts in timebase units (microseconds)
        relative_time = frame_metadata.timestamp - self.first_timestamp
        pts = int(relative_time * PTS_FRACT)  # Convert to microseconds

        # Ensure pts is monotonically increasing (required by most codecs)
        if self.last_pts is not None and pts <= self.last_pts:
            pts = self.last_pts + 1

        self.last_pts = pts

        # Create video frame from numpy array
        frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        frame = frame.reformat(format=self.pixel_format)
        frame.pts = pts

        # Encode and mux
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

        # Get current buffer position after encoding
        current_pos = self.buffer.tell()
        current_chunk_size = current_pos - self.last_write_position
        if current_chunk_size >= self.chunk_size:
            self.buffer.seek(self.last_write_position)
            chunk_data = self.buffer.read(current_chunk_size)
            self.upload_buffer.extend(chunk_data)
            self.last_write_position = current_pos
            self.buffer.seek(current_pos)
            self._upload_chunks()

        # Total bytes written
        self.total_bytes_written = current_pos
        self.frame_metadatas.append(frame_metadata)

    def _upload_chunks(self) -> None:
        """Upload complete chunks of the configured size.

        Processes the upload buffer and uploads chunks of exactly chunk_size
        bytes when enough data is available. Continues until insufficient
        data remains for a complete chunk.

        Raises:
            RuntimeError: If any chunk upload fails.
        """
        if self._recording_manager.is_recording_expired(self.recording_id):
            self.upload_buffer = bytearray()
            return

        # Upload complete chunks while we have enough data
        while len(self.upload_buffer) >= self.chunk_size:
            # Extract a chunk of exactly chunk_size bytes
            chunk = bytes(self.upload_buffer[: self.chunk_size])

            # Remove this chunk from our upload buffer
            self.upload_buffer = self.upload_buffer[self.chunk_size :]

            # Upload the chunk
            success = self.uploader.upload_chunk(chunk, is_final=False)

            if not success:
                raise RuntimeError("Failed to upload chunk")

    def finish(self) -> threading.Thread:
        """Complete the video encoding process and initiate final upload.

        Signals the encoding thread that no more frames will be added, allowing
        it to flush the encoder, finalize the MP4 container, and upload any
        remaining data including frame metadata.

        Returns:
            The upload thread that can be joined to wait for completion.
        """
        # Note we dont join on the (non-daemon) thread as we dont want to block
        self._upload_queue.put((None, None))
        self._streaming_done = True
        return self._upload_thread

    def _upload_json_data(self) -> None:
        """Upload frame metadata as a JSON file to cloud storage.

        Creates a separate metadata file containing all frame information
        including timestamps and frame indices for synchronization with
        other data streams in the recording.

        Raises:
            requests.HTTPError: If the metadata upload fails.
        """
        if self._recording_manager.is_recording_expired(self.recording_id):
            return
        params = {
            "filepath": f"{self.path}/metadata.json",
            "content_type": "application/json",
        }
        org_id = get_current_org()
        try:
            upload_url_response = requests.get(
                f"{API_URL}/org/{org_id}/recording/{self.uploader.recording_id}/resumable_upload_url",
                params=params,
                headers=get_auth().get_headers(),
            )
        except requests.exceptions.RequestException as e:
            logger.debug(e)
            pass
        upload_url_response.raise_for_status()
        upload_url = upload_url_response.json()["url"]
        for i in range(0, len(self.frame_metadatas)):
            self.frame_metadatas[i].frame_idx = i
        data = json.dumps([fm.model_dump(mode="json") for fm in self.frame_metadatas])
        response = requests.put(
            upload_url, headers={"Content-Length": str(len(data))}, data=data
        )
        response.raise_for_status()
        self.frame_metadatas = []
