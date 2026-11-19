"""Shared upload helpers for storage handlers."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import BinaryIO

from neuracore.core.const import API_URL
from neuracore.core.utils.multipart_upload import MultipartUploader, UploadError

logger = logging.getLogger(__name__)


class UploadStorageMixin:
    """Mixin that provides upload_file and upload_bytes helpers.

    Each payload goes up as a multipart upload that starts at the handler's v2
    start route (see ``MultipartUploader``).
    """

    log_to_cloud: bool
    org_id: str

    def _upload_start_url(self) -> str:
        """Return the handler's v2 upload start route."""
        raise NotImplementedError

    def _upload_base_url(self) -> str:
        """Return the root of the v2 routes that serve upload parts."""
        return f"{API_URL}/v2/org/{self.org_id}"

    def _upload_payload(
        self,
        data: BinaryIO,
        remote_filepath: str,
        content_type: str,
        payload_type: str,
    ) -> bool:
        if not self.log_to_cloud:
            return False

        try:
            MultipartUploader(self._upload_base_url()).upload(
                self._upload_start_url(),
                data,
                content_type,
                {"filepath": remote_filepath},
            )
        except UploadError as e:
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
            data=io.BytesIO(data),
            remote_filepath=remote_filepath,
            content_type=content_type,
            payload_type="bytes",
        )
