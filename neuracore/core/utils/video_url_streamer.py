"""Streaming video decoder for processing remote video files without full download.

This module provides classes for streaming video files from URLs and decoding
frames on-the-fly without downloading the entire file. It uses a custom
buffered IO reader to provide seamless streaming capabilities with PyAV.
"""

import io
import logging
from typing import Any, Iterator, Optional

import av
import numpy as np
import requests

logger = logging.getLogger(__name__)

CHUNK_SIZE = 256 * 1024  # Multiples of 256KB


class StreamingReader(io.BufferedIOBase):
    """A custom IO reader that combines buffer and streaming response data.

    This class provides a file-like interface that allows PyAV to read video
    data as if it were a complete local file, while actually streaming from
    a remote source. It maintains a buffer for the beginning of the file and
    seamlessly transitions to reading from the streaming response.
    """

    def __init__(self, buffer: io.BytesIO, response: requests.Response):
        """Initialize the streaming reader with buffer and response.

        Args:
            buffer: BytesIO buffer containing the initial portion of the file
                needed for format detection and metadata parsing.
            response: Streaming HTTP response from requests for the video file.
        """
        self.buffer = buffer
        self.response = response
        self.response_iter = response.iter_content(chunk_size=8192)
        self.position = 0
        self.buffer_size = buffer.getbuffer().nbytes
        self.eof = False
        self.leftover = None

    def read(self, size: Optional[int] = -1) -> bytes:
        """Read bytes from the combined buffer and streaming response.

        Reads data first from the buffer, then seamlessly transitions to
        reading from the streaming response. Handles partial reads and
        maintains proper position tracking.

        Args:
            size: Number of bytes to read. -1 means read all available data.

        Returns:
            The requested bytes, which may be fewer than requested if
            end of stream is reached.
        """
        if size is None:
            size = -1

        if size == 0:
            return b""

        # If we have leftover data from a previous read, use it first
        result = b""
        if self.leftover:
            if len(self.leftover) <= size or size == -1:
                result = self.leftover
                self.leftover = None
                if size != -1:
                    size -= len(result)
            else:
                result = self.leftover[:size]
                self.leftover = self.leftover[size:]
                size = 0

        # If we need more data and haven't reached EOF
        if (size > 0 or size == -1) and not self.eof:
            # Try to get more data from buffer or response
            if self.position < self.buffer_size:
                # Read from buffer
                self.buffer.seek(self.position)
                buffer_data = self.buffer.read(size if size != -1 else None)
                self.position += len(buffer_data)
                result += buffer_data

                # If we still need more and reached the end of buffer, get from response
                if (
                    len(buffer_data) < size or size == -1
                ) and self.position >= self.buffer_size:
                    try:
                        response_data = self._read_from_response(
                            size - len(buffer_data) if size != -1 else -1
                        )
                        result += response_data
                    except StopIteration:
                        self.eof = True
            else:
                # Read directly from response
                try:
                    response_data = self._read_from_response(size)
                    result += response_data
                except StopIteration:
                    self.eof = True

        # Update position
        self.position += len(result) - (len(self.leftover) if self.leftover else 0)
        self.leftover = None

        return result

    def _read_from_response(self, size: int) -> bytes:
        """Read data from the streaming HTTP response iterator.

        Accumulates data from the response stream until the requested amount
        is available or the stream ends. Handles partial chunks and stores
        overflow data for subsequent reads.

        Args:
            size: Number of bytes to read. -1 means read all available data.

        Returns:
            The requested bytes from the response stream.

        Raises:
            StopIteration: If no more data is available from the stream.
        """
        result = b""
        try:
            while size == -1 or len(result) < size:
                chunk = next(self.response_iter)
                if not chunk:
                    break

                if size == -1 or len(result) + len(chunk) <= size:
                    result += chunk
                else:
                    # Take what we need, store the rest
                    needed = size - len(result)
                    result += chunk[:needed]
                    self.leftover = chunk[needed:]
                    break
        except StopIteration:
            if not result:
                raise

        return result

    def seekable(self) -> bool:
        """Check if the stream supports seeking operations.

        Returns:
            Always False, as streaming responses cannot be sought.
        """
        return False

    def readable(self) -> bool:
        """Check if the stream supports read operations.

        Returns:
            Always True, as the stream supports reading.
        """
        return True


class VideoStreamer:
    """Streams and decodes video frames from a remote URL without full download.

    This class provides an iterator interface for processing video frames
    from remote sources. It uses streaming HTTP requests and progressive
    decoding to minimize memory usage and enable processing of large video
    files without downloading them entirely.
    """

    def __init__(self, video_url: str, buffer_size: int = CHUNK_SIZE):
        """Initialize the video streamer for a remote video URL.

        Args:
            video_url: HTTP/HTTPS URL of the video file to stream.
            buffer_size: Size of the initial buffer to download for format
                detection and metadata parsing.
        """
        self.video_url = video_url
        self.buffer_size = buffer_size
        self.response: Optional[requests.Response] = None
        self.container = None
        self.video_stream = None
        self.frame_count = 0

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over video frames as numpy arrays.

        Starts the streaming request, initializes the video decoder, and
        yields frames as RGB numpy arrays. The iterator handles the entire
        streaming and decoding pipeline automatically.

        Returns:
            Iterator that yields numpy arrays of shape (height, width, 3)
            in RGB format representing video frames.

        Raises:
            Exception: If the video URL is inaccessible, contains no video
                streams, or if decoding fails.
        """
        # Start the streaming request
        self.response = requests.get(self.video_url, stream=True)

        # Check if request was successful
        if self.response.status_code != 200:
            status = self.response.status_code or "unknown"
            msg = f"Failed to access video. Status code: {status}"
            raise Exception(msg)

        # Create a growing buffer for the initial part of the file
        buffer = io.BytesIO()

        # Initialize the container with the buffer
        buffer_reader = StreamingReader(buffer, self.response)
        self.container = av.open(buffer_reader)

        # Find the video stream
        if self.container and self.container.streams.video:
            self.video_stream = self.container.streams.video[0]
            # Only decode video frames
            self.video_stream.thread_type = "AUTO"  # Enable multithreading
            logger.debug(
                "Video info: Resolution: "
                f"{self.video_stream.width}x{self.video_stream.height}, "
                f"FPS: {float(self.video_stream.average_rate):.2f}, "
                f"Codec: {self.video_stream.codec_context.name}"
            )
            if self.container.duration:
                duration = self.container.duration * float(self.video_stream.time_base)
                logger.debug(f"Duration: {duration:.2f} seconds")
        else:
            raise Exception("No video stream found in the container")

        # Yield frames
        try:
            for frame in self.container.decode(video=0):
                self.frame_count += 1
                frame_array = frame.to_rgb().to_ndarray()
                yield frame_array
        except Exception:
            logger.error("Error during streaming.", exc_info=True)
        finally:
            self.close()

    def close(self) -> None:
        """Close the video stream and release all resources.

        Properly closes the video container and HTTP response to prevent
        resource leaks. Should be called when done processing the video.
        """
        if self.container:
            self.container.close()
        if self.response:
            self.response.close()
        logger.debug(f"Stream closed. Processed {self.frame_count} frames.")

    def __enter__(self: "VideoStreamer") -> "VideoStreamer":
        """Context manager entry point.

        Returns:
            Self for use in with statements.
        """
        return self

    def __exit__(
        self: "VideoStreamer", exc_type: Any, exc_val: Any, exc_tb: Any
    ) -> None:
        """Context manager exit point that ensures proper cleanup.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        self.close()
