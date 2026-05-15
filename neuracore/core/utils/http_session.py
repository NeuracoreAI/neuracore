"""Shared HTTP session with retry on connection/SSL failures.

All Neuracore API calls should use ``get_session()`` rather than the bare
``requests.*`` module-level functions so that stale keep-alive connections
are retried transparently instead of raising SSLError.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_RETRY = Retry(
    total=3,  # cap total retry attempts across all categories
    connect=3,  # retry conn establishment failures (stale keep-alive reuse lands here)
    read=0,  # never retry after bytes left the wire
    status=0,  # no status-code retries; let 5xx raise immediately
    backoff_factor=0.1,  # 0.1s, 0.2s, 0.4s between retries (~0.7s worst case)
    allowed_methods=False,  # type: ignore[arg-type]  # False = retry all methods
)


def _build_session() -> requests.Session:
    """Build a new requests Session with retry on connection/SSL failures."""
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=_RETRY)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_session: requests.Session | None = None


def get_session() -> requests.Session:
    """Return the shared, retry-configured requests Session."""
    global _session
    if _session is None or _session.adapters == {}:
        _session = _build_session()
    return _session
