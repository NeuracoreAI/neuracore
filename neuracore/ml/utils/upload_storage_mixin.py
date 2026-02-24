"""Shared upload helpers for storage handlers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import IO, Protocol

logger = logging.getLogger(__name__)


class UploadResponseLike(Protocol):
    """Minimal response interface returned by upload operations."""

    @property
    def status_code(self) -> int:
        """HTTP status code."""

    @property
    def text(self) -> str:
        """Response body text."""


class UploadStorageMixin:
    """Mixin that provides upload_file and upload_bytes helpers."""

    log_to_cloud: bool

    def _get_upload_url(self, filepath: str, content_type: str) -> str:
        raise NotImplementedError

    def _execute_upload(
        self,
        upload_url: str,
        data: bytes | IO[bytes],
        content_type: str,
    ) -> UploadResponseLike:
        raise NotImplementedError

    def _upload_payload(
        self,
        data: bytes | IO[bytes],
        remote_filepath: str,
        content_type: str,
        payload_type: str,
    ) -> bool:
        if not self.log_to_cloud:
            return False

        upload_url = self._get_upload_url(
            filepath=remote_filepath,
            content_type=content_type,
        )
        response = self._execute_upload(
            upload_url=upload_url,
            data=data,
            content_type=content_type,
        )
        if response.status_code != 200:
            logger.error(
                "Failed to upload %s to cloud path %s: %s",
                payload_type,
                remote_filepath,
                response.text,
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
