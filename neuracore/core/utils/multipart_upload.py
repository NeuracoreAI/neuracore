"""Multipart uploads through the backend's v2 upload routes."""

from __future__ import annotations

import base64
import logging
import math
import os
import time
from collections.abc import Mapping
from http import HTTPStatus
from typing import Any, BinaryIO

import google_crc32c
import requests

from neuracore.core.auth import get_auth
from neuracore.core.utils.http_errors import extract_error_detail, extract_error_type
from neuracore.core.utils.http_session import (
    RETRYABLE_STATUS_CODES,
    thread_local_session,
)

logger = logging.getLogger(__name__)

_PART_URL_WINDOW = 8
"""Parts signed per request for more part URLs, as many as the start route signs."""

_MAX_PART_ATTEMPTS = 5
"""Attempts to send one part before the upload fails."""

_BACKOFF_BASE_SECONDS = 1.0
"""Base of the exponential backoff between part attempts (1, 2, 4, 8 s)."""

_EXPIRED_URL_MARKERS = ("ExpiredToken", "Request has expired")
"""Body text of a 400 that a store sends for an expired presigned URL.

GCS's XML API answers with ``ExpiredToken``, and S3 with ``Request has expired``.
"""


class UploadError(Exception):
    """A file could not be uploaded."""


class _RestartUploadError(UploadError):
    """The backend lost the upload or rejected its checksum."""


def _raise_for_status(response: requests.Response) -> None:
    """Raise UploadError for an unsuccessful backend response.

    Args:
        response: Response from a backend upload route.

    Raises:
        UploadError: If the response status is not successful.
    """
    if not response.ok:
        raise UploadError(
            f"{response.request.method} {response.url} failed with HTTP "
            f"{response.status_code}: {extract_error_detail(response)}"
        )


def _post(url: str, body: Mapping[str, Any]) -> requests.Response:
    """Send an authenticated JSON request to a backend upload route.

    Args:
        url: The route.
        body: JSON request body.

    Returns:
        The response.

    Raises:
        UploadError: If the request failed in transport.
    """
    session = thread_local_session(retry_transient=True)
    try:
        return session.post(url, headers=get_auth().get_headers(), json=body)
    except requests.RequestException as error:
        raise UploadError(f"POST {url} failed: {error}") from error


def _by_part_number(part_urls: Mapping[str, str]) -> dict[int, str]:
    """Key part URLs from a JSON response by their integer part number.

    Args:
        part_urls: Part URLs keyed by part number as a string.

    Returns:
        The same URLs keyed by part number.
    """
    return {int(part_number): url for part_number, url in part_urls.items()}


def _put_part_response(url: str, part: bytes) -> requests.Response | None:
    """Send a part to its presigned URL, without the backend's auth header.

    Args:
        url: Presigned PUT URL of the part.
        part: Bytes of the part.

    Returns:
        The response, or None if the request failed in transport.
    """
    try:
        return thread_local_session().put(url, data=part)
    except requests.RequestException as error:
        logger.warning("Part upload failed in transport: %s", type(error).__name__)
        return None


def _part_url_expired(response: requests.Response) -> bool:
    """Whether a store rejected a part because its presigned URL expired.

    Args:
        response: Response to a part PUT.

    Returns:
        True for a 403, or a 400 whose body names an expired URL.
    """
    if response.status_code == HTTPStatus.FORBIDDEN:
        return True
    return response.status_code == HTTPStatus.BAD_REQUEST and any(
        marker in response.text for marker in _EXPIRED_URL_MARKERS
    )


class MultipartUploader:
    """Uploads files as S3 multipart uploads through the backend's v2 routes.

    A resource's start route returns the part size and the first part URLs.
    Parts go one at a time to presigned URLs, and the backend checks the
    file's CRC32C when the upload completes. A file whose upload the backend
    lost, or whose checksum it rejected, starts again once.
    """

    def __init__(self, base_url: str) -> None:
        """Initialize the uploader.

        Args:
            base_url: Root of the v2 routes that serve ``/uploads/parts`` and
                ``/uploads/complete``, such as ``{API_URL}/v2/org/{org_id}``.
        """
        self._base_url = base_url

    def upload(
        self,
        start_url: str,
        stream: BinaryIO,
        content_type: str,
        start_fields: Mapping[str, str] | None = None,
    ) -> None:
        """Upload a file through a resource's start route.

        Args:
            start_url: The resource's v2 start route.
            stream: Seekable binary stream of the file, read one part at a time.
            content_type: MIME type of the file.
            start_fields: Fields the start route takes besides the content type
                and size, such as ``filepath``.

        Raises:
            UploadError: If the file could not be uploaded.
        """
        size_bytes = stream.seek(0, os.SEEK_END)
        start_body = {
            **(start_fields or {}),
            "content_type": content_type,
            "size_bytes": size_bytes,
        }
        try:
            self._upload_once(start_url, start_body, stream, size_bytes)
        except _RestartUploadError as error:
            logger.warning("Restarting upload to %s: %s", start_url, error)
            self._upload_once(start_url, start_body, stream, size_bytes)

    def _upload_once(
        self,
        start_url: str,
        start_body: Mapping[str, Any],
        stream: BinaryIO,
        size_bytes: int,
    ) -> None:
        """Start an upload, send every part and complete it.

        Args:
            start_url: The resource's v2 start route.
            start_body: Request body of the start route.
            stream: Seekable binary stream of the file.
            size_bytes: Size of the file.

        Raises:
            UploadError: If the file could not be uploaded.
        """
        response = _post(start_url, start_body)
        _raise_for_status(response)
        started = response.json()
        upload_id = started["upload_id"]
        part_size_bytes = started["part_size_bytes"]
        part_urls = _by_part_number(started["part_urls"])
        part_count = max(1, math.ceil(size_bytes / part_size_bytes))
        checksum = 0
        stream.seek(0)
        for part_number in range(1, part_count + 1):
            part = stream.read(part_size_bytes)
            checksum = google_crc32c.extend(checksum, part)
            if part_number not in part_urls:
                part_urls.update(self._sign_parts(upload_id, part_number, part_count))
            self._put_part(upload_id, part_number, part_count, part, part_urls)
        self._complete(upload_id, checksum)

    def _put_part(
        self,
        upload_id: str,
        part_number: int,
        part_count: int,
        part: bytes,
        part_urls: dict[int, str],
    ) -> None:
        """Send one part, retrying transient failures and refreshing expired URLs.

        Args:
            upload_id: ID of the upload.
            part_number: Number of the part.
            part_count: Number of parts in the file.
            part: Bytes of the part.
            part_urls: Part URLs by part number, refreshed in place when the
                part's URL has expired.

        Raises:
            UploadError: If the part could not be sent.
        """
        status_code = None
        for attempt in range(_MAX_PART_ATTEMPTS):
            response = _put_part_response(part_urls[part_number], part)
            status_code = None if response is None else response.status_code
            if status_code == HTTPStatus.OK:
                return
            if response is not None and _part_url_expired(response):
                part_urls.update(self._sign_parts(upload_id, part_number, part_count))
                continue
            if status_code is not None and status_code not in RETRYABLE_STATUS_CODES:
                break
            if attempt + 1 < _MAX_PART_ATTEMPTS:
                time.sleep(_BACKOFF_BASE_SECONDS * 2**attempt)
        raise UploadError(
            f"Part {part_number} of upload {upload_id} failed, last status "
            f"{status_code}"
        )

    def _sign_parts(
        self, upload_id: str, first_part: int, part_count: int
    ) -> dict[int, str]:
        """Get URLs for the next window of parts.

        Args:
            upload_id: ID of the upload.
            first_part: Number of the first part to sign.
            part_count: Number of parts in the file.

        Returns:
            Part URLs by part number.

        Raises:
            UploadError: If the request fails.
            _RestartUploadError: If the backend no longer holds the upload.
        """
        last_part = min(first_part + _PART_URL_WINDOW - 1, part_count)
        response = _post(
            f"{self._base_url}/uploads/parts",
            {
                "uploads": [{
                    "upload_id": upload_id,
                    "part_numbers": list(range(first_part, last_part + 1)),
                }]
            },
        )
        _raise_for_status(response)
        signed = response.json()
        if upload_id in signed["missing_upload_ids"]:
            raise _RestartUploadError(f"Upload {upload_id} is no longer open")
        return _by_part_number(signed["uploads"][0]["part_urls"])

    def _complete(self, upload_id: str, checksum: int) -> None:
        """Complete an upload with the CRC32C of its file.

        Args:
            upload_id: ID of the upload.
            checksum: CRC32C of the whole file.

        Raises:
            UploadError: If the request fails.
            _RestartUploadError: If the backend no longer holds the upload or
                the object does not match the checksum.
        """
        response = _post(
            f"{self._base_url}/uploads/complete",
            {
                "upload_id": upload_id,
                "crc32c": base64.b64encode(checksum.to_bytes(4, "big")).decode(),
            },
        )
        if response.status_code == HTTPStatus.NOT_FOUND:
            raise _RestartUploadError(f"Upload {upload_id} is no longer open")
        if extract_error_type(response) == "CHECKSUM_MISMATCH":
            raise _RestartUploadError(f"Upload {upload_id} failed its checksum")
        _raise_for_status(response)
