import io
import logging
from typing import Iterator

import av
import numpy as np
import requests

logger = logging.getLogger(__name__)

CHUNK_SIZE = 256 * 1024  # Multiples of 256KB


class StreamingReader(io.BufferedIOBase):
    """
    A custom IO reader that reads from a buffer and then from a streaming response.
    This allows PyAV to read data as if it were a complete file.
    """

    def __init__(self, buffer: io.BytesIO, response: requests.Response):
        """
        Initialize the streaming reader.

        Args:
            buffer: A BytesIO buffer containing the beginning of the file
            response: The streaming response from requests
        """
        self.buffer = buffer
        self.response = response
        self.response_iter = response.iter_content(chunk_size=8192)
        self.position = 0
        self.buffer_size = buffer.getbuffer().nbytes
        self.eof = False
        self.leftover = None

    def read(self, size: int = -1) -> bytes:
        """
        Read size bytes from the stream.

        Args:
            size: Number of bytes to read, -1 means read all

        Returns:
            bytes: The read data
        """
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
        """
        Read data from the response iterator.

        Args:
            size: Number of bytes to read, -1 means read all available

        Returns:
            bytes: The read data

        Raises:
            StopIteration: If no more data is available
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
        """
        Check if the stream is seekable.

        Returns:
            bool: False, as we can't seek in a streaming response
        """
        return False

    def readable(self) -> bool:
        """
        Check if the stream is readable.

        Returns:
            bool: True, as we can read from the stream
        """
        return True


class VideoStreamer:
    """
    A class that streams a video from a URL, decoding frames on the fly.
    Yields frames as numpy arrays without downloading the entire video first.
    """

    def __init__(self, video_url: str, buffer_size: int = CHUNK_SIZE):
        """
        Initialize the video streamer.

        Args:
            video_url: URL of the video
            buffer_size: Size of the initial buffer to download (default: 10MB)
        """
        self.video_url = video_url
        self.buffer_size = buffer_size
        self.response = None
        self.container = None
        self.video_stream = None
        self.frame_count = 0

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Iterator that yields frames as numpy arrays.

        Returns:
            Iterator yielding numpy arrays of shape (height, width, 3) in RGB format
        """
        # Start the streaming request
        # self.response = requests.get(self.video_url, stream=True)
        self.response = requests.get(self.video_url, stream=True)

        # Check if request was successful
        if self.response.status_code != 200:
            raise Exception(
                f"Failed to access video. Status code: {self.response.status_code}"
            )

        # Create a growing buffer for the initial part of the file
        buffer = io.BytesIO()

        # Initialize the container with the buffer
        buffer_reader = StreamingReader(buffer, self.response)
        self.container = av.open(buffer_reader)

        # Find the video stream
        if self.container.streams.video:
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
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
        finally:
            self.close()

    def close(self) -> None:
        """Close the video stream and release resources."""
        if self.container:
            self.container.close()
        if self.response:
            self.response.close()
        logger.debug(f"Stream closed. Processed {self.frame_count} frames.")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit, ensures streamer is closed."""
        self.close()
