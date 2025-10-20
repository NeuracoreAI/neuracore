from __future__ import annotations
from typing import Iterable
import requests

from .bucket_chunk_uploader import BucketChunkUploader

_HTTP_TIMEOUT_SECONDS = 60


class S3BucketChunkUploader(BucketChunkUploader):
    """
    S3-style single-shot PUT against a presigned URL. `finalize` is ignored.
    """

    def __init__(self, upload_session_url: str, content_type: str) -> None:
        super().__init__(upload_session_url, content_type)
        self.supports_midstream_uploads = False


    def upload_chunks(self, chunks: Iterable[bytes], *, finalize: bool = False) -> None:
        payload = b"".join(chunks)

        headers = {"Content-Length": str(len(payload))}
        if self.content_type:
            headers["Content-Type"] = self.content_type

        resp = requests.put(
            self.upload_session_url,
            headers=headers,
            data=payload,
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        if resp.status_code not in (200, 201, 204):
            raise RuntimeError(f"S3 PUT failed: {resp.status_code} {resp.text}")
