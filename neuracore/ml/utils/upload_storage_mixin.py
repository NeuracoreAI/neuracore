"""Shared upload helpers for storage handlers."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Protocol

from neuracore.core.utils.http_session import thread_local_session

logger = logging.getLogger(__name__)

CHUNK_SIZE = 16 * 1024 * 1024
"""Size of each resumable-upload chunk, in bytes.

Must be a multiple of 256 KiB (the GCS resumable-upload requirement for every
non-final chunk); 16 MiB = 64 x 256 KiB. Keeping chunks bounded means a
transient connection stall costs at most one chunk retry instead of
restarting the whole upload.
"""

_MAX_CHUNK_RETRIES = 5
"""Maximum attempts for a single chunk before the upload is given up on."""

_BACKOFF_BASE_SECONDS = 1.0
"""Base for the exponential backoff between chunk retry attempts (1,2,4,8s)."""

_RETRYABLE_UPLOAD_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
"""Transient statuses for the upload PUT itself: the session is still good,
only this chunk needs to be resent."""

_SESSION_EXPIRED_STATUS_CODES = frozenset({404, 410})
"""Statuses GCS returns when a resumable upload session URI has expired."""


class _UploadFailedError(Exception):
    """A chunk could not be delivered after exhausting all retries."""


class UploadResponseLike(Protocol):
    """Minimal response interface returned by upload operations."""

    @property
    def status_code(self) -> int:
        """HTTP status code."""

    @property
    def text(self) -> str:
        """Response body text."""

    @property
    def headers(self) -> Mapping[str, str]:
        """Response headers."""


def _payload_size(data: bytes | IO[bytes]) -> int:
    """Return the total size of a payload without disturbing its position."""
    if isinstance(data, bytes):
        return len(data)
    position = data.tell()
    data.seek(0, 2)
    size = data.tell()
    data.seek(position)
    return size


def _read_chunk(data: bytes | IO[bytes], offset: int, length: int) -> bytes:
    """Read exactly ``length`` bytes starting at ``offset``.

    Always seeks to ``offset`` first so resuming after a partial 308 commit
    (which can land anywhere, not just at a chunk boundary) reads the right
    bytes.
    """
    if isinstance(data, bytes):
        return data[offset : offset + length]
    data.seek(offset)
    return data.read(length)


def _parse_committed_offset(headers: Mapping[str, str], attempted_end: int) -> int:
    """Parse the byte offset GCS reports as committed from a 308 response.

    Args:
        headers: Response headers from the 308 response.
        attempted_end: Inclusive end byte of the chunk we attempted to send,
            used as a fallback if the ``Range`` header is malformed.

    Returns:
        The offset to resume uploading from. GCS omits the ``Range`` header
        entirely when zero bytes are committed.
    """
    range_header = headers.get("Range")
    if not range_header:
        return 0
    try:
        return int(range_header.split("-")[1]) + 1
    except (IndexError, ValueError):
        return attempted_end + 1


class UploadStorageMixin:
    """Mixin that provides upload_file and upload_bytes helpers.

    Uploads go through the GCS resumable-upload protocol in bounded chunks
    (``CHUNK_SIZE``) rather than a single PUT of the whole payload: a
    transient connection stall only costs a retry of the current chunk, not
    the whole file, and each chunk PUT's write phase stays comfortably within
    the shared HTTP session's per-``send()`` timeout (see
    ``neuracore.core.utils.http_session``).
    """

    log_to_cloud: bool

    def _get_upload_url(self, filepath: str, content_type: str) -> str:
        raise NotImplementedError

    def _execute_upload(
        self,
        upload_url: str,
        data: bytes,
        content_type: str,
        headers: dict[str, str] | None = None,
    ) -> UploadResponseLike:
        """Send one PUT to the (resumable) upload URL.

        Args:
            upload_url: The signed/resumable URL to PUT to.
            data: The chunk (or whole small payload) to send.
            content_type: MIME type of the file being uploaded.
            headers: Extra headers (``Content-Range``, ``Content-Length``)
                merged over the default ``Content-Type`` header.
        """
        session = thread_local_session(retry_transient=True)
        return session.put(
            upload_url,
            data=data,
            headers={"Content-Type": content_type, **(headers or {})},
        )

    def _put_chunk_with_retry(
        self,
        upload_url: str,
        chunk: bytes,
        chunk_start: int,
        total: int,
        content_type: str,
    ) -> tuple[int, bool]:
        """PUT one chunk, retrying transient failures with backoff.

        Args:
            upload_url: The resumable session URI to PUT to.
            chunk: The chunk bytes to send.
            chunk_start: Offset of ``chunk`` within the full payload.
            total: Total size of the full payload.
            content_type: MIME type of the file being uploaded.

        Returns:
            A ``(next_offset, session_expired)`` tuple. ``next_offset`` is
            the byte offset to resume from — usually ``chunk_start +
            len(chunk)``, but can be less if GCS only committed a prefix of
            the chunk. ``session_expired`` signals the resumable session URI
            needs to be refreshed and the upload restarted from offset 0.

        Raises:
            _UploadFailedError: The chunk could not be delivered after
                ``_MAX_CHUNK_RETRIES`` attempts.
        """
        chunk_end = chunk_start + len(chunk) - 1
        is_final = chunk_start + len(chunk) >= total
        if total == 0:
            content_range = "bytes */0"
        elif is_final:
            content_range = f"bytes {chunk_start}-{chunk_end}/{total}"
        else:
            content_range = f"bytes {chunk_start}-{chunk_end}/*"
        headers = {"Content-Length": str(len(chunk)), "Content-Range": content_range}

        last_error = "unknown error"
        for attempt in range(_MAX_CHUNK_RETRIES):
            try:
                response = self._execute_upload(
                    upload_url, data=chunk, content_type=content_type, headers=headers
                )
            except Exception as e:  # connection-level failure, e.g. write timeout
                last_error = str(e)
            else:
                if response.status_code in (200, 201):
                    return chunk_start + len(chunk), False
                if response.status_code == 308:
                    return _parse_committed_offset(response.headers, chunk_end), False
                if response.status_code in _SESSION_EXPIRED_STATUS_CODES:
                    return 0, True
                if response.status_code not in _RETRYABLE_UPLOAD_STATUS_CODES:
                    raise _UploadFailedError(
                        f"HTTP {response.status_code}: {response.text}"
                    )
                last_error = f"HTTP {response.status_code}: {response.text}"

            if attempt < _MAX_CHUNK_RETRIES - 1:
                time.sleep(_BACKOFF_BASE_SECONDS * 2**attempt)

        raise _UploadFailedError(
            f"chunk at offset {chunk_start} failed after "
            f"{_MAX_CHUNK_RETRIES} attempts: {last_error}"
        )

    def _upload_chunked(
        self,
        data: bytes | IO[bytes],
        remote_filepath: str,
        content_type: str,
    ) -> None:
        """Upload a payload to cloud storage in ``CHUNK_SIZE`` chunks.

        Args:
            data: The payload to upload — an open file handle or in-memory
                bytes. A payload smaller than one chunk still resolves in
                exactly one PUT, matching the previous single-shot behavior.
            remote_filepath: Destination path within cloud storage.
            content_type: MIME type of the file being uploaded.

        Raises:
            _UploadFailedError: A chunk could not be delivered after
                exhausting retries.
        """
        total = _payload_size(data)
        upload_url = self._get_upload_url(
            filepath=remote_filepath, content_type=content_type
        )

        if total == 0:
            self._put_chunk_with_retry(upload_url, b"", 0, 0, content_type)
            return

        offset = 0
        while offset < total:
            chunk = _read_chunk(data, offset, min(CHUNK_SIZE, total - offset))
            next_offset, session_expired = self._put_chunk_with_retry(
                upload_url, chunk, offset, total, content_type
            )
            if session_expired:
                logger.warning(
                    "Upload session expired for %s; fetching a new session "
                    "and restarting from the beginning",
                    remote_filepath,
                )
                upload_url = self._get_upload_url(
                    filepath=remote_filepath, content_type=content_type
                )
                offset = 0
                continue
            offset = next_offset

    def _upload_payload(
        self,
        data: bytes | IO[bytes],
        remote_filepath: str,
        content_type: str,
        payload_type: str,
    ) -> bool:
        if not self.log_to_cloud:
            return False

        try:
            self._upload_chunked(data, remote_filepath, content_type)
        except _UploadFailedError as e:
            logger.error(
                "Failed to upload %s to cloud path %s: %s",
                payload_type,
                remote_filepath,
                e,
            )
            return False
        return True

    def upload_file(
        self,
        local_path: Path,
        remote_filepath: str,
        content_type: str = "application/octet-stream",
    ) -> bool:
        """Upload a local file to cloud storage."""
        if not self.log_to_cloud:
            return False
        if not local_path.exists() or not local_path.is_file():
            return False

        with open(local_path, "rb") as f:
            return self._upload_payload(
                data=f,
                remote_filepath=remote_filepath,
                content_type=content_type,
                payload_type="file",
            )

    def upload_bytes(
        self,
        data: bytes,
        remote_filepath: str,
        content_type: str = "application/octet-stream",
    ) -> bool:
        """Upload bytes content to cloud storage."""
        return self._upload_payload(
            data=data,
            remote_filepath=remote_filepath,
            content_type=content_type,
            payload_type="bytes",
        )
