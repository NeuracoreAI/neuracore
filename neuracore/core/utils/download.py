"""Utility functions for downloading files over the shared pooled session."""

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


def download_with_progress(
    url: str, description: str, destination: Path | None = None
) -> Path:
    """Download a file from a URL with a progress bar.

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
    try:
        stream_to_file(url, destination, progress=progress_bar)
    finally:
        progress_bar.close()
    return destination
