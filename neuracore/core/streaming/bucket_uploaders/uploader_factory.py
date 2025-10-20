from __future__ import annotations
from urllib.parse import urlparse, parse_qs

from .bucket_chunk_uploader import BucketChunkUploader
from .gcp_bucket_chunk_uploader import GCPBucketChunkUploader
from .s3_bucket_chunk_uploader import S3BucketChunkUploader


def make_chunk_uploader(upload_session_url: str, content_type: str) -> BucketChunkUploader:
    """
    Choose implementation by inspecting the presigned/session URL.
    """
    parsed = urlparse(upload_session_url)
    query = parse_qs(parsed.query)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()

    is_s3 = (
        "amazonaws.com" in host
        or any(k.lower().startswith("x-amz-") for k in query.keys())
    )
    is_gcp = (
        "storage.googleapis.com" in host
        or "googleapis.com" in host
        or query.get("uploadType", [""])[0] == "resumable"
        or "resumable" in path
    )

    if is_s3 and not is_gcp:
        return S3BucketChunkUploader(upload_session_url, content_type)

    # Default to GCP bucket
    return GCPBucketChunkUploader(upload_session_url, content_type)
