"""Resumable file uploader for Neuracore Data Daemon.

This module provides a file uploader that handles chunked uploads to cloud
storage with crash recovery and retry logic.
"""

import asyncio
import base64
import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

from neuracore.data_daemon.auth_management.auth_manager import get_auth
from neuracore.data_daemon.const import API_URL

logger = logging.getLogger(__name__)


@dataclass
class FinalResponseData:
    """Container for extracted HTTP response data.

    Holds headers and optional JSON body extracted from an aiohttp.ClientResponse
    before the response context is closed. This allows checksum verification to
    access the data after the async with block exits.
    """

    headers: dict[str, str] = field(default_factory=dict)
    json_body: dict[str, Any] | None = None


class ResumableFileUploader:
    """Upload a single file with resumable chunked uploads.

    This is a pure utility class that uploads files to cloud storage.
    It uses a progress callback to report upload progress and does not
    manage any persistent state.
    """

    CHUNK_SIZE = 64 * 1024 * 1024
    MAX_RETRIES = 5
    MAX_BACKOFF_SECONDS = 300
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
    FINAL_SUCCESS_CODES = {200, 201}
    RESUME_INCOMPLETE_CODE = 308
    SESSION_EXPIRED_CODE = 410
    FAST_FAIL_STATUS_CODES = {403, 404}

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

        # Get upload session URI if needed
        if self._session_uri is None:
            try:
                self._session_uri = await self._get_upload_session_uri()
            except aiohttp.ClientError as e:
                error_msg = f"Failed to get upload session URI: {e}"
                logger.error(error_msg)
                return (False, self._bytes_uploaded, error_msg)

        # Sync resume point with server state to avoid duplicates
        success, error_message = await self._sync_with_server_upload_position()
        if not success:
            return (False, self._bytes_uploaded, error_message)

        # Upload file in chunks
        success, error_message, final_response = await self._upload_file_in_chunks()

        if success:
            checksum_ok, checksum_error = await self._verify_checksum(final_response)
            if not checksum_ok:
                logger.warning(
                    f"Upload failed checksum verification for {self._filepath}: "
                    f"{checksum_error}"
                )
                return (False, self._bytes_uploaded, checksum_error)
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

    async def _upload_file_in_chunks(
        self,
    ) -> tuple[bool, str | None, FinalResponseData | None]:
        """Read file from disk and upload in chunks.

        Opens the file, seeks to the resume point, and uploads remaining
        data in chunks. Calls progress_callback after each successful chunk.

        Returns:
            Tuple of (success, error_message, final_response)

        Raises:
            IOError: If there's an error reading the file.
        """
        try:
            with open(self._filepath, "rb") as f:
                # Seek to resume point
                f.seek(self._bytes_uploaded)

                final_response: FinalResponseData | None = None

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
                    success, error_msg, final_response = await self._upload_chunk(
                        chunk, chunk_start, chunk_end, is_final
                    )

                    if not success:
                        return (False, error_msg, None)

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

                if final_response is None:
                    error_msg = "Upload did not finalize successfully"
                    logger.error(error_msg)
                    return (False, error_msg, None)

                return (True, None, final_response)

        except OSError as e:
            error_msg = f"File I/O error: {e}"
            logger.error(error_msg)
            return (False, error_msg, None)

    async def _upload_chunk(
        self,
        data: bytes,
        chunk_start: int,
        chunk_end: int,
        is_final: bool,
    ) -> tuple[bool, str | None, FinalResponseData | None]:
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
            Tuple of (success, error_message, final_response)
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
                    return (False, "No upload session URI available", None)

                timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes
                async with self._session.put(
                    self._session_uri,
                    headers=headers,
                    data=data,
                    timeout=timeout,
                ) as response:
                    status_code = response.status

                    if status_code in self.FINAL_SUCCESS_CODES:
                        if is_final:
                            # Extract response data before context exits
                            final_data = FinalResponseData(
                                headers=dict(response.headers)
                            )
                            try:
                                final_data.json_body = await response.json()
                            except Exception:
                                pass  # JSON body is optional
                            return (True, None, final_data)
                        return (True, None, None)
                    if status_code == self.RESUME_INCOMPLETE_CODE:
                        if is_final:
                            if await self._is_server_upload_complete():
                                finalized, error_msg, final_response = (
                                    await self._finalize_upload()
                                )
                                if finalized:
                                    return (True, None, final_response)
                                return (False, error_msg, None)
                            return (False, "Finalization incomplete", None)
                        return (True, None, None)
                    if status_code == self.SESSION_EXPIRED_CODE:
                        # Session expired - get new session
                        logger.info("Upload session expired, obtaining new session")
                        self._session_uri = await self._get_upload_session_uri()
                        continue
                    if status_code == 403:
                        if await self._is_signed_url_expired(response):
                            logger.info("Signed URL expired, re-acquiring session URL")
                            self._session_uri = await self._get_upload_session_uri()
                            continue
                        return (False, "Permission denied (403)", None)
                    if status_code == 404:
                        return (False, "Bucket not found (404)", None)
                    if status_code in self.RETRYABLE_STATUS_CODES:
                        if is_final and await self._is_server_upload_complete():
                            finalized, error_msg, final_response = (
                                await self._finalize_upload()
                            )
                            if finalized:
                                return (True, None, final_response)
                            return (False, error_msg, None)
                        logger.warning(
                            f"Upload chunk failed "
                            f"(attempt {attempt + 1}/{self.MAX_RETRIES}): "
                            f"HTTP {status_code}"
                        )
                    else:
                        return (False, f"Upload failed with HTTP {status_code}", None)

            except aiohttp.ClientConnectorError as e:
                logger.warning(f"Network connection error (attempt {attempt + 1})")
                if attempt < self.MAX_RETRIES - 1:
                    await self._sleep_backoff(attempt)
                    continue
                return (False, f"Network connection error: {e}", None)

            except aiohttp.ClientSSLError as e:
                logger.warning(f"SSL error (attempt {attempt + 1})")
                if attempt < self.MAX_RETRIES - 1:
                    await self._sleep_backoff(attempt)
                    continue
                return (False, f"SSL error: {e}", None)

            except asyncio.TimeoutError:
                logger.warning(f"Upload chunk timeout (attempt {attempt + 1})")

            except Exception as e:
                logger.error(f"Unexpected error uploading chunk: {e}")
                return (False, f"Unexpected error: {e}", None)

            if attempt < self.MAX_RETRIES - 1:
                await self._sleep_backoff(attempt)

        error_msg = f"Upload chunk failed after {self.MAX_RETRIES} attempts"
        logger.error(error_msg)
        return (False, error_msg, None)

    async def _sleep_backoff(self, attempt: int) -> None:
        """Sleep with exponential backoff, capped at MAX_BACKOFF_SECONDS."""
        delay = min(2**attempt, self.MAX_BACKOFF_SECONDS)
        await asyncio.sleep(delay)

    async def _is_signed_url_expired(self, response: aiohttp.ClientResponse) -> bool:
        """Detect signed URL expiration from response headers/body."""
        header_value = response.headers.get("X-Signed-Url-Expired", "").lower()
        if header_value == "true":
            return True
        try:
            body_text = await response.text()
        except Exception:
            body_text = ""
        return "expired" in body_text.lower()

    async def _check_session_status(self) -> int | None:
        """Check current uploaded bytes on the server.

        Returns:
            Bytes uploaded on server, or None if session is invalid/expired.
        """
        if self._session_uri is None:
            return None
        headers = {"Content-Length": "0", "Content-Range": "bytes */*"}
        timeout = aiohttp.ClientTimeout(total=30)
        async with self._session.put(
            self._session_uri, headers=headers, data=b"", timeout=timeout
        ) as response:
            if response.status in self.FINAL_SUCCESS_CODES:
                return self._total_bytes
            if response.status == self.RESUME_INCOMPLETE_CODE:
                range_header = response.headers.get("Range")
                if range_header:
                    last_byte = int(range_header.split("-")[1])
                    return last_byte + 1
                return 0
            if response.status in {404, self.SESSION_EXPIRED_CODE}:
                return None
            if response.status == 403 and await self._is_signed_url_expired(response):
                return None
            raise aiohttp.ClientResponseError(
                response.request_info,
                response.history,
                status=response.status,
                message=f"Unexpected status checking: {response.status}",
            )

    async def _sync_with_server_upload_position(self) -> tuple[bool, str | None]:
        """Ensure local resume offset matches server state to avoid duplicates."""
        try:
            server_bytes = await self._check_session_status()
        except aiohttp.ClientSSLError as e:
            return (False, f"SSL error: {e}")
        except aiohttp.ClientError as e:
            return (False, f"Failed to check upload status: {e}")

        if server_bytes is None:
            try:
                self._session_uri = await self._get_upload_session_uri()
            except aiohttp.ClientError as e:
                return (False, f"Failed to refresh upload session URI: {e}")
            return (True, None)

        if server_bytes != self._bytes_uploaded:
            logger.info(
                "Adjusting resume offset from %s to %s based on server status",
                self._bytes_uploaded,
                server_bytes,
            )
            self._bytes_uploaded = server_bytes
        return (True, None)

    async def _is_server_upload_complete(self) -> bool:
        """Check if server has all bytes uploaded for this file."""
        try:
            server_bytes = await self._check_session_status()
        except aiohttp.ClientError:
            return False
        return server_bytes == self._total_bytes

    async def _finalize_upload(
        self,
    ) -> tuple[bool, str | None, FinalResponseData | None]:
        """Finalize an upload without re-sending data."""
        if self._session_uri is None:
            return (False, "No upload session URI available", None)
        headers = {
            "Content-Length": "0",
            "Content-Range": f"bytes */{self._total_bytes}",
        }
        for attempt in range(self.MAX_RETRIES):
            try:
                timeout = aiohttp.ClientTimeout(total=30)
                async with self._session.put(
                    self._session_uri, headers=headers, data=b"", timeout=timeout
                ) as response:
                    if response.status in self.FINAL_SUCCESS_CODES:
                        # Extract response data before context exits
                        final_data = FinalResponseData(headers=dict(response.headers))
                        try:
                            final_data.json_body = await response.json()
                        except Exception:
                            pass  # JSON body is optional
                        return (True, None, final_data)
                    if response.status == self.RESUME_INCOMPLETE_CODE:
                        return (False, "Finalization incomplete", None)
                    if response.status in self.RETRYABLE_STATUS_CODES:
                        if attempt < self.MAX_RETRIES - 1:
                            await self._sleep_backoff(attempt)
                            continue
                    if response.status == self.SESSION_EXPIRED_CODE:
                        return (
                            False,
                            "Upload session expired during finalization",
                            None,
                        )
                    return (
                        False,
                        f"Finalization failed with HTTP {response.status}",
                        None,
                    )
            except aiohttp.ClientError as e:
                if attempt < self.MAX_RETRIES - 1:
                    await self._sleep_backoff(attempt)
                    continue
                return (False, f"Finalization request failed: {e}", None)
        return (False, "Finalization failed after retries", None)

    async def _verify_checksum(
        self, response_data: FinalResponseData | None
    ) -> tuple[bool, str | None]:
        """Verify file integrity after finalization using server checksum."""
        if response_data is None:
            return (False, "No finalization response available for checksum")

        server_md5_hex: str | None = None
        headers = response_data.headers
        if "X-Checksum-MD5" in headers:
            server_md5_hex = headers["X-Checksum-MD5"].strip().lower()
        elif "x-goog-hash" in headers:
            hashes = headers["x-goog-hash"]
            for part in hashes.split(","):
                part = part.strip()
                if part.startswith("md5="):
                    b64_hash = part.split("=", 1)[1]
                    server_md5_hex = base64.b64decode(b64_hash).hex()
                    break
        else:
            body = response_data.json_body
            if body and "md5Hash" in body:
                server_md5_hex = base64.b64decode(body["md5Hash"]).hex()

        if not server_md5_hex:
            return (False, "Missing checksum from server response")

        md5_hash = hashlib.md5()
        try:
            with open(self._filepath, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    md5_hash.update(chunk)
        except OSError as e:
            return (False, f"Failed to compute checksum: {e}")

        local_md5_hex = md5_hash.hexdigest().lower()
        if local_md5_hex != server_md5_hex:
            return (False, "Checksum mismatch after finalization")
        return (True, None)
