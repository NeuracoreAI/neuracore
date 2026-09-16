"""Utility functions for downloading files over the shared pooled session."""

import os
import tempfile
from pathlib import Path

from tqdm import tqdm

from neuracore.core.utils.http_session import thread_local_session

DOWNLOAD_CHUNK_SIZE = 1024 * 1024
"""Bytes read from the socket per iteration while streaming a file to disk."""

DOWNLOAD_TIMEOUT_S: tuple[float, float] = (15.0, 120.0)
"""``(connect, read)`` budget for a file download.

The read budget is wide because it bounds the wait for each chunk of a
potentially very large transfer, not a single small response.
"""


def stream_to_file(url: str, destination: Path, progress: tqdm | None = None) -> int:
    """Download a URL to a file over this thread's pooled session.

    Args:
        url: URL to download.
        destination: Path to write the body to.
        progress: Bar to advance as the body arrives. Its total is set from the
            response's ``Content-Length``.

    Returns:
        Total bytes written.
    """
    session = thread_local_session(retry_transient=True, retry_read_timeout=True)
    written = 0
    with session.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_S) as response:
        response.raise_for_status()
        if progress is not None:
            progress.total = int(response.headers.get("Content-Length", 0)) or None
            progress.refresh()
        with open(destination, "wb") as handle:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                if progress is not None:
                    progress.update(len(chunk))
    return written


def download_bytes(url: str) -> bytes:
    """Download a URL and return its body.

    Args:
        url: URL to download.

    Returns:
        The response body.
    """
    session = thread_local_session(retry_transient=True, retry_read_timeout=True)
    with session.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_S) as response:
        response.raise_for_status()
        return response.content


def remote_content_length(url: str) -> int | None:
    """Return the byte length the server reports for a URL.

    Returns None when the server omits Content-Length.

    Args:
        url: URL to query.

    Returns:
        The advertised body length, or None when the server omits it.

    Raises:
        requests.RequestException: The request failed.
        ValueError: The server sent a Content-Length that is not a number.
    """
    session = thread_local_session(retry_transient=True, retry_read_timeout=True)
    with session.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_S) as response:
        response.raise_for_status()
        length = response.headers.get("Content-Length")
        return int(length) if length is not None else None


def download_with_progress(
    url: str, description: str, destination: Path | None = None
) -> Path:
    """Download a file from a URL with a progress bar.

    Write the body to a sibling staging file and move it onto destination once
    the transfer completes.
    Args:
        url: URL of the file to download.
        description: Description for the progress bar.
        destination: Optional path to save the downloaded file.
            If not provided, a temporary file will be created.

    Returns:
        Path to the downloaded file.
    """
    if destination is None:
        destination = Path(tempfile.mkdtemp()) / "model.nc.zip"
    else:
        destination = Path(destination)

    progress_bar = tqdm(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=description,
        bar_format=(
            "{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]"
        ),
    )
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, suffix=".part", delete=False
    ) as handle:
        staging = Path(handle.name)
    try:
        stream_to_file(url, staging, progress=progress_bar)
        os.replace(staging, destination)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    finally:
        progress_bar.close()
    return destination


def download_to_cache(url: str, destination: Path, description: str) -> Path:
    """Download a URL to a cache path unless a complete copy is already there.

    Keep a cached file only when its size matches the length the server
    reports, and download it again otherwise.

    Args:
        url: URL of the file to download.
        destination: Cache path the file is kept at.
        description: Description for the progress bar.

    Returns:
        Path to the cached file.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        expected = remote_content_length(url)
        if destination.stat().st_size == expected:
            return destination
    return download_with_progress(url, description, destination=destination)
