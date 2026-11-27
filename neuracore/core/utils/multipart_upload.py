"""Multipart uploads through the backend's v2 upload routes."""

from __future__ import annotations

import base64
import logging
import math
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, BinaryIO, TypeVar
from xml.sax.saxutils import escape

import google_crc32c
import requests

from neuracore.core.auth import get_auth
from neuracore.core.utils.http_errors import extract_error_detail
from neuracore.core.utils.http_session import (
    RETRYABLE_STATUS_CODES,
    thread_local_session,
)

logger = logging.getLogger(__name__)

_PART_URL_WINDOW = 8
"""Number of part URLs to request per call. The start route signs as many."""

_MAX_PART_ATTEMPTS = 5
"""Attempts to send one part, or the Complete request, before the upload fails."""

_BACKOFF_BASE_SECONDS = 1.0
"""Base of the exponential backoff between attempts (1, 2, 4, 8 s)."""

_EXPIRED_URL_MARKERS = ("ExpiredToken", "Request has expired")
"""Body text of a 400 that a store sends for an expired presigned URL.

GCS's XML API answers with ``ExpiredToken``, and S3 with ``Request has expired``.
"""

_MISSING_UPLOAD_ERRORS = ("NoSuchUpload", "NoSuchPart", "InvalidPart")
"""Error codes of a Complete request whose upload is gone or whose listed parts
the store does not hold."""

_XML_QUOTE = {'"': "&quot;"}

_CRC32C_HEADER = "x-amz-checksum-crc32c"
"""Complete response header with the object's CRC32C.

A store that does not send it puts ``<ChecksumCRC32C>`` in the response body.
"""


_Result = TypeVar("_Result")


class UploadError(Exception):
    """A file could not be uploaded."""


class _RestartUploadError(UploadError):
    """The store lost the upload, or the object does not match the file."""


@dataclass(frozen=True)
class _Retry:
    """An attempt that failed in a way the next attempt may not.

    Attributes:
        status_code: HTTP status of the response, or None if the request failed
            in transport.
        backoff: Whether to wait before the next attempt.
    """

    status_code: int | None
    backoff: bool = True


class MultipartUploader:
    """Uploads files as S3 multipart uploads through the backend's v2 routes.

    A resource's start route returns the part size, the first part URLs and the
    Complete URL. Parts go one at a time to presigned URLs. The uploader keeps
    the ETag of each part, completes the upload with them and compares the
    CRC32C the store returns with the file's CRC32C. A file whose upload the
    store lost, or whose CRC32C does not match, starts again once.
    """

    def __init__(self, base_url: str) -> None:
        """Initialize the uploader.

        Args:
            base_url: Root of the v2 routes that serve ``/uploads/parts``, such
                as ``{API_URL}/v2/org/{org_id}``.
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
        response = self._post(start_url, start_body)
        self._raise_for_status(response)
        started = response.json()
        upload_id = started["upload_id"]
        part_size_bytes = started["part_size_bytes"]
        part_urls = self._by_part_number(started["part_urls"])
        complete_url = started["complete_url"]
        part_count = max(1, math.ceil(size_bytes / part_size_bytes))
        checksum = 0
        etags: dict[int, str] = {}
        stream.seek(0)
        for part_number in range(1, part_count + 1):
            part = stream.read(part_size_bytes)
            checksum = google_crc32c.extend(checksum, part)
            if part_number not in part_urls:
                part_urls.update(self._sign_parts(upload_id, part_number, part_count))
            etags[part_number] = self._put_part(
                upload_id, part_number, part_count, part, part_urls
            )
        store_checksum = self._complete(upload_id, complete_url, etags)
        if store_checksum != checksum:
            raise _RestartUploadError(
                f"CRC32C mismatch for upload {upload_id}: local={checksum:#010x} "
                f"store={store_checksum:#010x}"
            )

    def _put_part(
        self,
        upload_id: str,
        part_number: int,
        part_count: int,
        part: bytes,
        part_urls: dict[int, str],
    ) -> str:
        """Send one part, retrying transient failures and refreshing expired URLs.

        Args:
            upload_id: ID of the upload.
            part_number: Number of the part.
            part_count: Number of parts in the file.
            part: Bytes of the part.
            part_urls: Part URLs by part number, refreshed in place when the
                part's URL has expired.

        Returns:
            The ETag the store returned for the part.

        Raises:
            UploadError: If the part could not be sent.
        """
        return self._retry(
            lambda: self._put_part_once(
                upload_id, part_number, part_count, part, part_urls
            ),
            f"Part {part_number} of upload {upload_id}",
        )

    def _put_part_once(
        self,
        upload_id: str,
        part_number: int,
        part_count: int,
        part: bytes,
        part_urls: dict[int, str],
    ) -> str | _Retry:
        """Make one attempt to send a part.

        Args:
            upload_id: ID of the upload.
            part_number: Number of the part.
            part_count: Number of parts in the file.
            part: Bytes of the part.
            part_urls: Part URLs by part number, refreshed in place when the
                part's URL has expired.

        Returns:
            The part's ETag, or a retry for a failure the next attempt may not
            repeat.

        Raises:
            UploadError: If the store rejects the part for good.
        """
        response = self._put_part_response(part_urls[part_number], part)
        if response is None:
            return _Retry(status_code=None)
        if response.status_code == HTTPStatus.OK:
            etag = response.headers.get("ETag")
            if etag is None:
                raise UploadError(
                    f"Part {part_number} of upload {upload_id} returned no ETag"
                )
            return etag
        if self._part_url_expired(response):
            part_urls.update(self._sign_parts(upload_id, part_number, part_count))
            return _Retry(status_code=response.status_code, backoff=False)
        if response.status_code in RETRYABLE_STATUS_CODES:
            return _Retry(status_code=response.status_code)
        raise self._failed(
            f"Part {part_number} of upload {upload_id}", response.status_code
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
        response = self._post(
            f"{self._base_url}/uploads/parts",
            {
                "uploads": [{
                    "upload_id": upload_id,
                    "part_numbers": list(range(first_part, last_part + 1)),
                }]
            },
        )
        self._raise_for_status(response)
        signed = response.json()
        if upload_id in signed["missing_upload_ids"]:
            raise _RestartUploadError(f"Upload {upload_id} is no longer open")
        return self._by_part_number(signed["uploads"][0]["part_urls"])

    def _complete(
        self, upload_id: str, complete_url: str, etags: Mapping[int, str]
    ) -> int:
        """Complete an upload with its parts and return the object's CRC32C.

        Args:
            upload_id: ID of the upload.
            complete_url: Presigned URL that completes the upload.
            etags: ETag of each part by part number.

        Returns:
            CRC32C of the object the store assembled.

        Raises:
            UploadError: If the request fails.
            _RestartUploadError: If the store no longer holds the upload or its
                parts, the URL expired, or the response has no CRC32C.
        """
        body = self._complete_body(etags)
        return self._retry(
            lambda: self._complete_once(upload_id, complete_url, body),
            f"Complete of upload {upload_id}",
        )

    def _complete_once(
        self, upload_id: str, complete_url: str, body: str
    ) -> int | _Retry:
        """Make one attempt to complete an upload.

        Args:
            upload_id: ID of the upload.
            complete_url: Presigned URL that completes the upload.
            body: ``CompleteMultipartUpload`` XML body.

        Returns:
            CRC32C of the object the store assembled, or a retry for a failure
            the next attempt may not repeat.

        Raises:
            UploadError: If the store rejects the request for good.
            _RestartUploadError: If the store no longer holds the upload or its
                parts, the URL expired, or the response has no CRC32C.
        """
        response = self._post_complete_response(complete_url, body)
        if response is None:
            return _Retry(status_code=None)
        error_code = self._error_code(response.text)
        if response.status_code == HTTPStatus.OK and error_code is None:
            checksum = self._response_crc32c(response)
            if checksum is None:
                raise _RestartUploadError(
                    f"Complete of upload {upload_id} returned no CRC32C"
                )
            return checksum
        if self._complete_upload_missing(response, error_code):
            raise _RestartUploadError(
                f"Upload {upload_id} is no longer open: HTTP "
                f"{response.status_code} {error_code}"
            )
        if (
            response.status_code != HTTPStatus.OK
            and response.status_code not in RETRYABLE_STATUS_CODES
        ):
            raise self._failed(f"Complete of upload {upload_id}", response.status_code)
        return _Retry(status_code=response.status_code)

    @staticmethod
    def _retry(
        attempt_once: Callable[[], _Result | _Retry], description: str
    ) -> _Result:
        """Repeat an attempt with exponential backoff until it succeeds.

        Args:
            attempt_once: Makes one attempt. It returns the result, or a retry
                for a failure that the next attempt may not repeat, and raises
                for a failure that no attempt can fix.
            description: What the attempts do, for the error message.

        Returns:
            The result of the first successful attempt.

        Raises:
            UploadError: If every attempt asks for a retry.
        """
        last_status_code = None
        for attempt in range(_MAX_PART_ATTEMPTS):
            result = attempt_once()
            if not isinstance(result, _Retry):
                return result
            last_status_code = result.status_code
            if result.backoff and attempt + 1 < _MAX_PART_ATTEMPTS:
                time.sleep(_BACKOFF_BASE_SECONDS * 2**attempt)
        raise MultipartUploader._failed(description, last_status_code)

    @staticmethod
    def _failed(description: str, status_code: int | None) -> UploadError:
        """Build the error for a request that did not succeed.

        Args:
            description: What the request does.
            status_code: HTTP status of the last response, or None if the
                request failed in transport.

        Returns:
            The error to raise.
        """
        return UploadError(f"{description} failed, last status {status_code}")

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _by_part_number(part_urls: Mapping[str, str]) -> dict[int, str]:
        """Key part URLs from a JSON response by their integer part number.

        Args:
            part_urls: Part URLs keyed by part number as a string.

        Returns:
            The same URLs keyed by part number.
        """
        return {int(part_number): url for part_number, url in part_urls.items()}

    @staticmethod
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

    @staticmethod
    def _post_complete_response(url: str, body: str) -> requests.Response | None:
        """POST a Complete request to its presigned URL, without the auth header.

        Args:
            url: Presigned POST URL that completes the upload.
            body: ``CompleteMultipartUpload`` XML body.

        Returns:
            The response, or None if the request failed in transport.
        """
        try:
            return thread_local_session().post(url, data=body.encode())
        except requests.RequestException as error:
            logger.warning("Complete failed in transport: %s", type(error).__name__)
            return None

    @staticmethod
    def _complete_body(etags: Mapping[int, str]) -> str:
        """Build the ``CompleteMultipartUpload`` body that lists every part.

        Args:
            etags: ETag of each part by part number.

        Returns:
            The XML body, with the parts in ascending order.
        """
        parts = "".join(
            f"<Part><PartNumber>{part_number}</PartNumber>"
            f"<ETag>{escape(etags[part_number], _XML_QUOTE)}"
            "</ETag></Part>"
            for part_number in sorted(etags)
        )
        return f"<CompleteMultipartUpload>{parts}</CompleteMultipartUpload>"

    @staticmethod
    def _xml_element(xml: str, tag: str) -> str | None:
        """Return the text of the first ``tag`` element in an XML document.

        Args:
            xml: The XML document.
            tag: Name of the element.

        Returns:
            The element's text, or None if the document has no such element.
        """
        start = xml.find(f"<{tag}>")
        if start < 0:
            return None
        start += len(tag) + 2
        end = xml.find(f"</{tag}>", start)
        if end < 0:
            return None
        return xml[start:end].strip()

    @staticmethod
    def _error_code(body: str) -> str | None:
        """Return the error code of a store response body.

        A store can send an error body with HTTP 200.

        Args:
            body: Response body.

        Returns:
            The ``<Code>`` of an ``<Error>`` body, an empty string for an error
            body without a code, or None if the body is not an error.
        """
        if "<Error>" not in body:
            return None
        return MultipartUploader._xml_element(body, "Code") or ""

    @staticmethod
    def _response_crc32c(response: requests.Response) -> int | None:
        """Read the object's CRC32C from a Complete response.

        Args:
            response: Successful response to a Complete request.

        Returns:
            The CRC32C from the response header, else from the body, or None if
            neither holds a valid one.
        """
        encoded = response.headers.get(
            _CRC32C_HEADER
        ) or MultipartUploader._xml_element(response.text, "ChecksumCRC32C")
        if encoded is None:
            return None
        try:
            checksum = base64.b64decode(encoded, validate=True)
        except ValueError:
            return None
        if len(checksum) != 4:
            return None
        return int.from_bytes(checksum, "big")

    @staticmethod
    def _complete_upload_missing(
        response: requests.Response, error_code: str | None
    ) -> bool:
        """Whether a Complete failed because the upload, a part or the URL is gone.

        Args:
            response: Response to a Complete request.
            error_code: The ``<Code>`` of its error body, if any.

        Returns:
            True for a 403 or 404, a missing upload or part, or an expired URL.
        """
        return (
            response.status_code in (HTTPStatus.FORBIDDEN, HTTPStatus.NOT_FOUND)
            or error_code in _MISSING_UPLOAD_ERRORS
            or any(marker in response.text for marker in _EXPIRED_URL_MARKERS)
        )

    @staticmethod
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
