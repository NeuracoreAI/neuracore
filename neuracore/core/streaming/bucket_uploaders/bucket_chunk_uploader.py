from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable


class BucketChunkUploader(ABC):
    """
    Strategy interface for uploading object data in segments.

    Implementations must:
      - Accept one or more byte segments per call to `upload_chunks`.
      - Honour `finalize=True` by marking the LAST segment of the LAST call
        as the terminal segment for the upload session.
    """

    def __init__(self, upload_session_url: str, content_type: str) -> None:
        self.upload_session_url = upload_session_url
        self.content_type = content_type
        # Whether the backend supports incremental mid-stream appends (GCP supports, S3 does not support)
        self.supports_midstream_uploads: bool = True


    @abstractmethod
    def upload_chunks(
        self,
        chunks: Iterable[bytes],
        *,
        finalize: bool = False,
    ) -> None:
        """Append segments in order; set `finalize=True` only on the final call."""
        ...
