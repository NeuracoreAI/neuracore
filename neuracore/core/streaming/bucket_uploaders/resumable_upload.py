import logging
import time

import requests

from neuracore.core.auth import get_auth
from neuracore.core.const import API_URL

logger = logging.getLogger(__name__)


class ResumableUpload:
    """
    Handles resumable uploads to Google Cloud Storage.
    """

    def __init__(self, recording_id: str, filepath: str, content_type: str):
        """
        Initialize a resumable upload to GCS.

        Args:
            recording_id: Recording ID
            camera_id: Name of the sensor
        """
        self.recording_id = recording_id
        self.filepath = filepath
        self.content_type = content_type
        self.session_uri = self._get_upload_session_uri()
        self.total_bytes_uploaded = 0
        self.max_retries = 5

    def _get_upload_session_uri(self) -> str:
        """
        Get a resumable upload session URI from the backend.

        Returns:
            str: Resumable upload session URI
        """
        auth = get_auth()
        params = {
            "filepath": self.filepath,
            "content_type": self.content_type,
        }
        response = requests.get(
            f"{API_URL}/recording/{self.recording_id}/resumable_upload_url",
            params=params,
            headers=auth.get_headers(),
        )
        response.raise_for_status()
        return response.json()["url"]

    def upload_chunk(self, data: bytes, is_final: bool = False) -> bool:
        """
        Upload a chunk of data to the resumable upload session.

        Args:
            data: Chunk of data to upload
            is_final: Whether this is the final chunk

        Returns:
            bool: Whether the upload was successful
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
                response = requests.put(self.session_uri, headers=headers, data=data)
                status_code = response.status_code

                if status_code == 200 or status_code == 201:
                    self.total_bytes_uploaded += len(data)
                    return True
                elif status_code == 308:
                    # Resume Incomplete, more data expected
                    self.total_bytes_uploaded += len(data)
                    return True
                else:
                    # Error occurred
                    if attempt < self.max_retries - 1:
                        time.sleep(2**attempt)  # Exponential backoff

            except Exception as e:
                logger.error(f"Exception during upload (attempt {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff

        return False

    def check_status(self) -> int:
        """
        Check the status of the resumable upload.

        Returns:
            int: Bytes uploaded so far
        """
        headers = {"Content-Length": "0", "Content-Range": "bytes */*"}

        response = requests.put(self.session_uri, headers=headers)
        if response.status_code == 200 or response.status_code == 201:
            logger.debug("Upload complete")
            return self.total_bytes_uploaded
        elif response.status_code == 308 and "Range" in response.headers:
            range_header = response.headers["Range"]
            return int(range_header.split("-")[1]) + 1
        elif response.status_code == 308:
            return 0

        raise Exception(f"Unexpected status code: {response.status_code}")
