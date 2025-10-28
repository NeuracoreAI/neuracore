"""Streaming JSON data uploader with chunked upload support.

This module provides a streaming JSON uploader that collects data entries,
formats them as a JSON array, and uploads them to cloud storage using
resumable uploads with configurable chunk sizes.
"""

import io
import json
import logging
import queue
import threading
from typing import Any, Dict, List

from neuracore.core.streaming.recording_state_manager import get_recording_state_manager

from .bucket_uploader import BucketUploader
from .resumable_upload import ResumableUpload

logger = logging.getLogger(__name__)

CHUNK_MULTIPLE = 256 * 1024  # Chunk size multiple of 256 KiB
MB_CHUNK = 4 * CHUNK_MULTIPLE
CHUNK_SIZE = 64 * MB_CHUNK


class StreamingJsonUploader(BucketUploader):
    """A JSON data streamer that handles chunked uploads to cloud storage.

    This class provides asynchronous streaming of JSON data entries by collecting
    them in a queue, formatting them as a valid JSON array, and uploading them
    in configurable chunks using resumable uploads. The upload process runs in
    a separate thread to avoid blocking data collection.
    """

    def __init__(
        self,
        recording_id: str,
        filepath: str,
        chunk_size: int = CHUNK_SIZE,
    ):
        """Initialize a streaming JSON uploader.

        Sets up the upload queue, starts the upload thread, and increments the
        active stream count for the recording. The uploader will collect JSON
        data entries and stream them as a properly formatted JSON array.

        Args:
            recording_id: Unique identifier for the recording session.
            filepath: Target file path in the cloud storage bucket.
            chunk_size: Size in bytes of each upload chunk. Will be adjusted
                to be a multiple of 256 KiB if necessary.
        """
        super().__init__(recording_id)
        self.filepath = filepath
        self.chunk_size = chunk_size
        self._streaming_done = False
        self._upload_queue: queue.Queue = queue.Queue()
        # Thread will continue, even if main thread exits
        self._upload_thread = threading.Thread(target=self._upload_loop, daemon=False)
        self._recording_manager = get_recording_state_manager()
        self._upload_thread.start()
        self._update_num_active_streams(1)

    def _thread_setup(self) -> None:
        """Setup thread-local resources for the upload loop.

        Initializes the resumable uploader, adjusts chunk size to meet requirements,
        creates in-memory buffers, and sets up tracking variables for the JSON
        streaming process.
        """
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
        self.data_entries: List[Dict[str, Any]] = []

        # Track if we've already started the JSON array
        self.json_array_started = False

    def _upload_loop(self) -> None:
        """Upload chunks in a separate thread.

        Main upload loop that processes queued data entries, formats them as
        a JSON array, and uploads them in chunks. Runs until streaming is
        complete and all queued data has been processed.
        """
        # Skipping uploads of expired recording
        if self._recording_manager.is_recording_expired(self.recording_id):
            self.finish()
            return
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
        """Add a JSON data entry to the streaming queue.

        Queues a data entry for asynchronous processing and upload. The entry
        will be serialized to JSON and included in the output array in the
        order received.

        Args:
            data_entry: Dictionary containing the data to be included in the
                JSON stream. Typically contains timestamp and sensor data.
        """
        self._upload_queue.put(data_entry)

    def _add_entry(self, data_entry: Dict[str, Any]) -> None:
        """Add a JSON data entry to the buffer and upload if threshold is reached.

        Serializes the data entry to JSON, writes it to the buffer, and triggers
        chunk uploads when the buffer reaches the configured chunk size.

        Args:
            data_entry: Dictionary containing timestamp and data to be serialized.

        Raises:
            RuntimeError: If chunk upload fails during the process.
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
        """Complete the streaming process and initiate final upload.

        Signals the upload thread that no more data will be added, allowing it
        to finalize the JSON array and upload any remaining data. The thread
        will complete the upload process including the closing JSON bracket.

        Returns:
            The upload thread that can be joined to wait for completion.
        """
        # Signal the upload thread that we're done
        self._upload_queue.put(None)
        self._streaming_done = True
        return self._upload_thread
