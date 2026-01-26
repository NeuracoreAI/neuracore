"""Resumable upload functionality for Google Cloud Storage integration.

This module provides a ResumableUpload class that handles chunked uploads to
Google Cloud Storage with retry logic and progress tracking. It supports
resuming interrupted uploads and provides proper error handling with
exponential backoff.
"""

import logging
import time

import requests

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL

logger = logging.getLogger(__name__)


class ResumableUpload:
    """Handles resumable uploads to Google Cloud Storage.

    This class manages the lifecycle of a resumable upload session, including
    obtaining upload URLs, tracking progress, handling retries, and managing
    chunk uploads with proper HTTP range headers. It supports resuming
    interrupted uploads by checking server status.
    """

    def __init__(self, recording_id: str, filepath: str, content_type: str):
        """Initialize a resumable upload to GCS.

        Args:
            recording_id: Unique identifier for the recording.
            filepath: Target file path in the storage bucket.
            content_type: MIME type of the file being uploaded.
        """
        self.recording_id = recording_id
        self.filepath = filepath
        self.content_type = content_type
        logger.info(
            "ResumableUpload init: recording_id=%s filepath=%s content_type=%s",
            recording_id,
            filepath,
            content_type,
        )
        self.session_uri = self._get_upload_session_uri()
        self.total_bytes_uploaded = 0
        self.max_retries = 5
        logger.info(
            "ResumableUpload ready: recording_id=%s session_uri=%s",
            recording_id,
            (
                self.session_uri[:80] + "..."
                if len(self.session_uri) > 80
                else self.session_uri
            ),
        )

    def _get_upload_session_uri(self) -> str:
        """Get a resumable upload session URI from the backend.

        Makes an API call to obtain a resumable upload session URL that will
        be used for all subsequent chunk uploads.

        Returns:
            The resumable upload session URI from Google Cloud Storage.

        Raises:
            requests.HTTPError: If the API request fails.
        """
        auth = get_auth()
        params = {
            "filepath": self.filepath,
            "content_type": self.content_type,
        }
        org_id = get_current_org()
        url = (
            f"{API_URL}/org/{org_id}/recording/{self.recording_id}/resumable_upload_url"
        )
        logger.info(
            "GET resumable_upload_url: recording_id=%s filepath=%s",
            self.recording_id,
            self.filepath,
        )
        response = requests.get(url, params=params, headers=auth.get_headers())
        logger.info(
            "GET resumable_upload_url response: status=%d recording_id=%s",
            response.status_code,
            self.recording_id,
        )
        response.raise_for_status()
        return response.json()["url"]

    def upload_chunk(self, data: bytes, is_final: bool = False) -> bool:
        """Upload a chunk of data to the resumable upload session.

        Uploads a chunk of data with proper range headers and retry logic.
        Automatically handles upload position verification and exponential
        backoff on failures.

        Args:
            data: Binary data chunk to upload.
            is_final: Whether this is the final chunk of the upload.

        Returns:
            True if the upload was successful, False otherwise.

        Raises:
            Exception: If there's a mismatch between local and server upload positions.
        """
        if len(data) == 0 and not is_final:
            return True  # Nothing to upload

        # First, check if the session is still valid and get current uploaded bytes
        actual_uploaded_bytes = self.check_status()
        if actual_uploaded_bytes != self.total_bytes_uploaded:
            raise Exception(
                "Upload position mismatch: "
                f"Local={self.total_bytes_uploaded}, Server={actual_uploaded_bytes}"
            )

        # Prepare headers
        headers = {
            "Content-Length": str(len(data)),
        }

        # Set content range header
        chunk_first_byte = self.total_bytes_uploaded
        chunk_last_byte = self.total_bytes_uploaded + len(data) - 1

        if is_final:
            # Final chunk, include total size
            total_size = self.total_bytes_uploaded + len(data)
            headers["Content-Range"] = (
                f"bytes {chunk_first_byte}-{chunk_last_byte}/{total_size}"
            )
        else:
            # Not final chunk, use '*' for total size
            headers["Content-Range"] = f"bytes {chunk_first_byte}-{chunk_last_byte}/*"

        # Attempt the upload with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    "PUT chunk: rec=%s bytes=%d range=%s final=%s attempt=%d",
                    self.recording_id,
                    len(data),
                    headers.get("Content-Range"),
                    is_final,
                    attempt + 1,
                )
                response = requests.put(self.session_uri, headers=headers, data=data)
                status_code = response.status_code
                logger.info(
                    "PUT upload_chunk response: status=%d recording_id=%s",
                    status_code,
                    self.recording_id,
                )

                if status_code == 200 or status_code == 201:
                    self.total_bytes_uploaded += len(data)
                    logger.info(
                        "Upload complete: recording_id=%s total_bytes=%d",
                        self.recording_id,
                        self.total_bytes_uploaded,
                    )
                    return True
                elif status_code == 308:
                    # Resume Incomplete, more data expected
                    self.total_bytes_uploaded += len(data)
                    return True
                else:
                    # Error occurred
                    logger.warning(
                        "Upload chunk failed: status=%d recording_id=%s response=%s",
                        status_code,
                        self.recording_id,
                        response.text[:200] if response.text else "",
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(2**attempt)  # Exponential backoff

            except Exception:
                logger.error(
                    f"Exception during upload (attempt {attempt+1})", exc_info=True
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff

        logger.error(
            "Upload chunk failed after %d retries: recording_id=%s",
            self.max_retries,
            self.recording_id,
        )
        return False

    def check_status(self) -> int:
        """Check the status of the resumable upload session.

        Queries the upload session to determine how many bytes have been
        successfully uploaded so far. This is used for resuming interrupted
        uploads and verifying upload progress.

        Returns:
            Number of bytes that have been successfully uploaded to the server.

        Raises:
            Exception: If the server returns an unexpected status code.
        """
        headers = {"Content-Length": "0", "Content-Range": "bytes */*"}

        logger.debug("PUT check_status: recording_id=%s", self.recording_id)
        response = requests.put(self.session_uri, headers=headers)
        logger.info(
            "PUT check_status response: status=%d recording_id=%s",
            response.status_code,
            self.recording_id,
        )
        if response.status_code == 200 or response.status_code == 201:
            logger.info(
                "check_status: upload complete recording_id=%s total_bytes=%d",
                self.recording_id,
                self.total_bytes_uploaded,
            )
            return self.total_bytes_uploaded
        elif response.status_code == 308 and "Range" in response.headers:
            range_header = response.headers["Range"]
            uploaded_bytes = int(range_header.split("-")[1]) + 1
            logger.info(
                "check_status: in progress recording_id=%s uploaded_bytes=%d",
                self.recording_id,
                uploaded_bytes,
            )
            return uploaded_bytes
        elif response.status_code == 308:
            logger.info(
                "check_status: no bytes uploaded yet recording_id=%s",
                self.recording_id,
            )
            return 0

        logger.error(
            "check_status: unexpected status=%d recording_id=%s",
            response.status_code,
            self.recording_id,
        )
        raise Exception(f"Unexpected status code: {response.status_code}")
