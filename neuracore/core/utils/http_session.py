"""Thread- and process-local HTTP session with shared retry policy.

All Neuracore API calls should obtain a session via ``thread_local_session()``
rather than constructing fresh ``requests.Session`` instances or using the
module-level ``requests.*`` helpers. The returned session is cached per OS
thread and per process: forks get a fresh session (since urllib3 connection
pools are not safe to share across forks), and threads do not share a session
with one another.
"""

import os
import threading

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

_thread_local = threading.local()


def thread_local_session() -> requests.Session:
    """Return a retry-enabled Session cached per thread and process."""
    pid = os.getpid()

    session = getattr(_thread_local, "session", None)
    session_pid = getattr(_thread_local, "pid", None)

    if session is None or session_pid != pid:
        session = requests.Session()

        adapter = HTTPAdapter(max_retries=_RETRY)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        _thread_local.session = session
        _thread_local.pid = pid

    return session
