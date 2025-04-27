import io
import json
import logging
import queue
import threading
from typing import Any, Dict

from .bucket_uploader import BucketUploader
from .resumable_upload import ResumableUpload

logger = logging.getLogger(__name__)

CHUNK_MULTIPLE = 256 * 1024  # Chunk size multiple of 256 KiB
MB_CHUNK = 4 * CHUNK_MULTIPLE
CHUNK_SIZE = 64 * MB_CHUNK


class StreamingJsonUploader(BucketUploader):
    """A JSON data streamer that handles chunked uploads."""

    def __init__(
        self,
        recording_id: str,
        filepath: str,
        chunk_size: int = CHUNK_SIZE,
    ):
        """
        Initialize a streaming JSON uploader.

        Args:
            recording_id: Recording ID
            stream_id: Stream ID
            chunk_size: Size of chunks to upload
        """
        super().__init__(recording_id)
        self.filepath = filepath
        self.chunk_size = chunk_size
        self._streaming_done = False
        self._upload_queue = queue.Queue()
        # Thread will continue, even if main thread exits
        self._upload_thread = threading.Thread(target=self._upload_loop, daemon=False)
        self._upload_thread.start()
        self._update_num_active_streams(1)

    def _thread_setup(self) -> None:
        """Setup thread for upload loop."""

        # Ensure chunk_size is a multiple of 256 KiB
        if self.chunk_size % CHUNK_MULTIPLE != 0:
            self.chunk_size = ((self.chunk_size // CHUNK_MULTIPLE) + 1) * CHUNK_MULTIPLE
            logger.debug(
                f"Adjusted chunk size to {self.chunk_size/1024:.0f} "
                f"KiB to ensure it's a multiple of {CHUNK_MULTIPLE/1024:.0f} KiB"
            )

        self.uploader = ResumableUpload(
            self.recording_id, self.filepath, "application/json"
        )

        # Create in-memory buffer
        self.buffer = io.BytesIO()

        # Track bytes and buffer positions
        self.total_bytes_written = 0

        # Create a dedicated buffer for upload chunks
        self.upload_buffer = bytearray()
        self.last_write_position = 0

        # Store all data entries
        self.data_entries = []

        # Track if we've already started the JSON array
        self.json_array_started = False

    def _upload_loop(self) -> None:
        """
        Upload chunks in a separate thread.
        """
        self._thread_setup()

        # Write the opening bracket of the JSON array
        self.buffer.write(b"[")
        self.json_array_started = True

        # Variable to track if this is the first entry
        first_entry = True

        # Process queue until streaming is done and queue is empty
        while not self._streaming_done or self._upload_queue.qsize() > 0:
            try:
                data_entry = self._upload_queue.get(timeout=0.1)
                if data_entry is None:
                    break

                # Add comma for all entries except the first one
                if not first_entry:
                    self.buffer.write(b",")
                else:
                    first_entry = False

                # Add the JSON entry
                self._add_entry(data_entry)
            except queue.Empty:
                continue

        # Write closing bracket for JSON array
        self.buffer.write(b"]")

        # Get current position
        current_pos = self.buffer.tell()

        # Read any remaining data since last write position
        if current_pos > self.last_write_position:
            self.buffer.seek(self.last_write_position)
            remaining_data = self.buffer.read(current_pos - self.last_write_position)
            self.upload_buffer.extend(remaining_data)
            self.last_write_position = current_pos

        # Upload any remaining data in the upload buffer
        if len(self.upload_buffer) > 0:
            final_chunk = bytes(self.upload_buffer)
            success = self.uploader.upload_chunk(final_chunk, is_final=True)

            if not success:
                raise RuntimeError("Failed to upload final chunk")

        logger.debug(
            "JSON streaming and upload complete: "
            f"{self.uploader.total_bytes_uploaded} bytes"
        )
        self._update_num_active_streams(-1)

    def add_frame(self, data_entry: Dict[str, Any]) -> None:
        """
        Add a JSON data entry to the stream.

        Args:
            data_entry: Dictionary containing timestamp and data
        """
        self._upload_queue.put(data_entry)

    def _add_entry(self, data_entry: Dict[str, Any]) -> None:
        """
        Add a JSON data entry to the stream and upload if buffer is large enough.

        Args:
            data_entry: Dictionary containing timestamp and data
        """
        # Serialize the entry to JSON and encode to bytes
        entry_json = json.dumps(data_entry)
        entry_bytes = entry_json.encode("utf-8")

        # Write to buffer
        self.buffer.write(entry_bytes)

        # Store the entry for potential further processing
        self.data_entries.append(data_entry)

        # Get current buffer position after writing
        current_pos = self.buffer.tell()
        current_chunk_size = current_pos - self.last_write_position

        if current_chunk_size >= self.chunk_size:
            # Read the chunk to upload
            self.buffer.seek(self.last_write_position)
            chunk_data = self.buffer.read(current_chunk_size)

            # Add to upload buffer
            self.upload_buffer.extend(chunk_data)

            # Update last write position
            self.last_write_position = current_pos

            # Return to end of buffer for further writing
            self.buffer.seek(current_pos)

            # Upload complete chunks
            self._upload_chunks()

        # Total bytes written
        self.total_bytes_written = current_pos

    def _upload_chunks(self) -> None:
        """
        Upload chunks of exactly chunk_size bytes if enough data is available.
        """
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
        """
        Finish encoding and upload any remaining data.
        Returns the upload thread which can be joined if needed.
        """
        # Signal the upload thread that we're done
        self._upload_queue.put(None)
        self._streaming_done = True
        return self._upload_thread
