"""Utility functions for downloading files with progress indication."""

import tempfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm


def download_with_progress(
    url: str, description: str, destination: Optional[Path] = None
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
    response = requests.get(
        url,
        timeout=120,
        stream=True,
    )
    response.raise_for_status()

    # Get total file size
    total_size = int(response.headers.get("Content-Length", 0))

    # Create a temporary directory and file path
    if destination is None:
        destination = Path(tempfile.mkdtemp()) / "model.nc.zip"
    else:
        destination = Path(destination)

    # Create progress bar based on file size
    progress_bar = tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=description,
        bar_format=(
            "{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]"
        ),
    )

    # Write the file with progress updates
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))

    # Close the progress bar
    progress_bar.close()
    return destination
