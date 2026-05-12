"""HTTP session with keep-alive disabled.

All Neuracore API calls should use ``Session()`` as a context manager rather than
the bare ``requests.*`` module-level functions to ensure keep-alive is disabled and
avoid stale connection issues in multi-threaded contexts.

Example:
    with Session() as session:
        response = session.get(url)
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

_RETRY = Retry(
    total=3,  # cap total retry attempts across all categories
    connect=3,  # retry conn establishment failures (stale keep-alive reuse lands here)
    read=0,  # never retry after bytes left the wire
    status=0,  # no status-code retries; let 5xx raise immediately
    backoff_factor=0.1,  # 0.1s, 0.2s, 0.4s between retries (~0.7s worst case)
    allowed_methods=False,  # type: ignore[arg-type]  # False = retry all methods
)


class Session(requests.Session):
    """A requests Session with keep-alive disabled."""

    def __init__(self, *, keep_alive: bool = False) -> None:
        """Initialize the session and configure retry-enabled HTTP adapters."""
        super().__init__()
        adapter = HTTPAdapter(max_retries=_RETRY)
        self.mount("https://", adapter)
        self.mount("http://", adapter)
        self.keep_alive = keep_alive
