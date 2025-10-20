from __future__ import annotations
from typing import Iterable

from .bucket_chunk_uploader import BucketChunkUploader
from .resumable_upload import ResumableUpload  


class GCPBucketChunkUploader(BucketChunkUploader):
    """
    GCS-style resumable uploader. Maintains a single session and only finalizes once.
    """

    def __init__(self, upload_session_url: str, content_type: str) -> None:
        super().__init__(upload_session_url, content_type)
        self._session = ResumableUpload.from_session_url(upload_session_url, content_type)
        self._is_finalized = False

    def upload_chunks(self, chunks: Iterable[bytes], *, finalize: bool = False) -> None:
        if self._is_finalized:
            raise RuntimeError("Resumable session already finalized")

        segments = list(chunks)
        for idx, segment in enumerate(segments):
            is_last_in_call = (idx == len(segments) - 1)
            should_finalize_now = finalize and is_last_in_call
            ok = self._session.upload_chunk(segment, is_final=should_finalize_now)
            if not ok:
                raise RuntimeError("Failed to upload chunk to GCP")

        if finalize:
            self._is_finalized = True
