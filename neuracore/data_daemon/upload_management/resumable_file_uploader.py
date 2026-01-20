"""Resumable file uploader for Neuracore Data Daemon.

This module provides a file uploader that handles chunked uploads to cloud
storage with crash recovery and retry logic.
"""

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

import aiohttp

from neuracore.data_daemon.auth_management.auth_manager import get_auth
from neuracore.data_daemon.const import API_URL

logger = logging.getLogger(__name__)


class ResumableFileUploader:
    """Upload a single file with resumable chunked uploads.

    This is a pure utility class that uploads files to cloud storage.
    It uses a progress callback to report upload progress and does not
    manage any persistent state.
    """

    CHUNK_SIZE = 64 * 1024 * 1024
    MAX_RETRIES = 5

    def __init__(
        self,
        recording_id: str,
        filepath: str,
        cloud_filepath: str,
        content_type: str,
        client_session: aiohttp.ClientSession,
        bytes_uploaded: int = 0,
        progress_callback: Callable[[int], None] | None = None,
    ) -> None:
        """Initialize the file uploader.

        Args:
            recording_id: Recording identifier
            filepath: Local filesystem path to file
            cloud_filepath: Cloud storage path
            content_type: MIME type
            client_session: aiohttp ClientSession for HTTP requests
            bytes_uploaded: Starting offset for resume
            progress_callback: Called after each chunk to report progress
        """
        self._recording_id = recording_id
        self._filepath = filepath
        self._cloud_filepath = cloud_filepath
        self._content_type = content_type
        self._session = client_session
        self._bytes_uploaded = bytes_uploaded
        self._progress_callback = progress_callback

        self._session_uri: str | None = None
        self._total_bytes = 0

    async def _get_upload_session_uri(self) -> str:
        """Get a resumable upload session URI from the backend.

        Makes an API call to obtain a resumable upload session URL from
        Google Cloud Storage that will be used for all chunk uploads.

        Returns:
            The resumable upload session URI from Google Cloud Storage.

        Raises:
            aiohttp.ClientError: If the API request fails.
        """
        params = {
            "filepath": self._cloud_filepath,
            "content_type": self._content_type,
        }

        auth = get_auth()
        org_id = auth.get_org_id()

        timeout = aiohttp.ClientTimeout(total=30)
        async with self._session.get(
            f"{API_URL}/org/{org_id}/recording/{self._recording_id}/resumable_upload_url",
            params=params,
            headers=auth.get_headers(),
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return data["url"]

    async def upload(self) -> tuple[bool, int, str | None]:
        """Upload the file with resumable chunks.

        Reads the file from disk starting at the bytes_uploaded offset and
        uploads it in chunks to cloud storage. Calls progress_callback after
        each successful chunk if provided.

        Returns:
            Tuple of (success, total_bytes_uploaded, error_message)

        Raises:
            FileNotFoundError: If the local file does not exist.
        """
        logger.info(
            f"Starting upload for {self._recording_id}/{self._filepath}: "
            f"{self._bytes_uploaded} bytes already uploaded"
        )

        # Validate file exists and get size
        file_path = Path(self._filepath)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {self._filepath}")

        self._total_bytes = file_path.stat().st_size

        # Get upload session URI
        try:
            self._session_uri = await self._get_upload_session_uri()
        except aiohttp.ClientError as e:
            error_msg = f"Failed to get upload session URI: {e}"
            logger.error(error_msg)
            return (False, self._bytes_uploaded, error_msg)

        # Upload file in chunks
        success, error_message = await self._upload_file_in_chunks()

        if success:
            logger.info(
                f"Upload complete for {self._recording_id}/{self._filepath}: "
                f"{self._total_bytes} bytes"
            )
            return (True, self._bytes_uploaded, None)
        else:
            logger.warning(
                f"Upload failed for {self._recording_id}/{self._filepath} "
                f"at offset {self._bytes_uploaded}/{self._total_bytes}: {error_message}"
            )
            return (False, self._bytes_uploaded, error_message)

    async def _upload_file_in_chunks(self) -> tuple[bool, str | None]:
        """Read file from disk and upload in chunks.

        Opens the file, seeks to the resume point, and uploads remaining
        data in chunks. Calls progress_callback after each successful chunk.

        Returns:
            Tuple of (success, error_message)

        Raises:
            IOError: If there's an error reading the file.
        """
        try:
            with open(self._filepath, "rb") as f:
                # Seek to resume point
                f.seek(self._bytes_uploaded)

                while True:
                    # Read next chunk
                    chunk = f.read(self.CHUNK_SIZE)
                    if not chunk:
                        break  # End of file

                    # Calculate byte range
                    chunk_start = self._bytes_uploaded
                    chunk_end = chunk_start + len(chunk) - 1
                    is_final = (chunk_end + 1) >= self._total_bytes

                    # Upload chunk with retry logic
                    success, error_msg = await self._upload_chunk(
                        chunk, chunk_start, chunk_end, is_final
                    )

                    if not success:
                        return (False, error_msg)

                    # Update progress
                    chunk_size = len(chunk)
                    self._bytes_uploaded += chunk_size

                    # Notify callback
                    if self._progress_callback:
                        self._progress_callback(chunk_size)

                    logger.debug(
                        f"Uploaded chunk: {self._bytes_uploaded}/"
                        f"{self._total_bytes} bytes"
                    )

                return (True, None)

        except OSError as e:
            error_msg = f"File I/O error: {e}"
            logger.error(error_msg)
            return (False, error_msg)

    async def _upload_chunk(
        self,
        data: bytes,
        chunk_start: int,
        chunk_end: int,
        is_final: bool,
    ) -> tuple[bool, str | None]:
        """Upload a single chunk with exponential backoff retry.

        Uploads a chunk of data to the resumable upload session with proper
        Content-Range headers. Handles session expiration (410 Gone) by
        obtaining a new session URI. Returns False immediately on network
        errors.

        Args:
            data: Binary data chunk to upload
            chunk_start: Starting byte offset
            chunk_end: Ending byte offset
            is_final: Whether this is the final chunk

        Returns:
            Tuple of (success, error_message)
        """
        # Prepare headers
        headers = {"Content-Length": str(len(data))}

        if is_final:
            # Final chunk - include total size
            total_size = chunk_end + 1
            headers["Content-Range"] = f"bytes {chunk_start}-{chunk_end}/{total_size}"
        else:
            headers["Content-Range"] = f"bytes {chunk_start}-{chunk_end}/*"

        # Retry with exponential backoff
        for attempt in range(self.MAX_RETRIES):
            try:
                if self._session_uri is None:
                    return (False, "No upload session URI available")

                timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes
                async with self._session.put(
                    self._session_uri,
                    headers=headers,
                    data=data,
                    timeout=timeout,
                ) as response:
                    status_code = response.status

                    if status_code in (200, 201, 308):
                        return (True, None)
                    elif status_code == 410:
                        # Session expired - get new session
                        logger.info("Upload session expired, obtaining new session")
                        self._session_uri = await self._get_upload_session_uri()
                        continue
                    else:
                        logger.warning(
                            f"Upload chunk failed "
                            f"(attempt {attempt + 1}/{self.MAX_RETRIES}): "
                            f"HTTP {status_code}"
                        )

            except aiohttp.ClientConnectorError as e:
                logger.warning(f"Network connection error (attempt {attempt + 1})")
                # Network error - don't retry
                return (False, f"Network connection error: {e}")

            except asyncio.TimeoutError:
                logger.warning(f"Upload chunk timeout (attempt {attempt + 1})")

            except Exception as e:
                logger.error(f"Unexpected error uploading chunk: {e}")
                return (False, f"Unexpected error: {e}")

            if attempt < self.MAX_RETRIES - 1:
                await asyncio.sleep(2**attempt)

        error_msg = f"Upload chunk failed after {self.MAX_RETRIES} attempts"
        logger.error(error_msg)
        return (False, error_msg)
