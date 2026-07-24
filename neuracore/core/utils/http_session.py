"""Thread- and process-local HTTP session with shared retry policy.

All Neuracore API calls should obtain a session via ``thread_local_session()``
rather than constructing fresh ``requests.Session`` instances or using the
module-level ``requests.*`` helpers. The returned session is cached per OS
thread and per process: forks get a fresh session (since urllib3 connection
pools are not safe to share across forks), and threads do not share a session
with one another.

For the aiohttp stack, ``retry_stale_connection`` provides the equivalent
policy as a client middleware: pass it via
``ClientSession(middlewares=(retry_stale_connection,))``.
"""

import asyncio
import logging
import os
import threading

import requests
from aiohttp import (
    ClientHandlerType,
    ClientOSError,
    ClientRequest,
    ClientResponse,
    ServerDisconnectedError,
)
from requests.adapters import HTTPAdapter
from urllib3 import Retry
from urllib3.exceptions import ProtocolError

logger = logging.getLogger(__name__)

_DROPPED_CONNECTION_ERRORS = (
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
)


def _is_dropped_connection(error: Exception) -> bool:
    """Whether the error means the connection died without delivering a response.

    Covers stale keep-alive reuse, TCP resets from proxies/NAT (including during
    the TLS handshake) and ``RemoteDisconnected`` (a ``ConnectionResetError``
    subclass). Errors where response bytes did arrive, such as a garbled status
    line, are excluded.
    """
    return isinstance(error, ProtocolError) and any(
        isinstance(arg, _DROPPED_CONNECTION_ERRORS) for arg in error.args
    )


class _DroppedConnectionRetry(Retry):
    """Retry policy that treats dropped connections as connection errors.

    urllib3 classifies a ``ProtocolError`` on an established connection as a
    read error, which this policy disables. A dropped connection never
    delivered a response, so it is safe to retry under the ``connect`` budget,
    mirroring ``retry_connection_failures`` on the aiohttp stack.
    """

    def _is_connection_error(self, err: Exception) -> bool:
        return super()._is_connection_error(err) or _is_dropped_connection(err)

    def _is_read_error(self, err: Exception) -> bool:
        return super()._is_read_error(err) and not _is_dropped_connection(err)


_RETRY = _DroppedConnectionRetry(
    total=3,  # cap total retry attempts across all categories
    connect=3,  # conn establishment failures and dropped connections land here
    read=0,  # never retry once response bytes have arrived
    status=0,  # no status-code retries; let 5xx raise immediately
    backoff_factor=0.1,  # 0.1s, 0.2s, 0.4s between retries (~0.7s worst case)
    allowed_methods=False,  # type: ignore[arg-type]  # False = retry all methods
)

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

_TRANSIENT_RETRY = _RETRY.new(
    status=2,  # retry transient backend statuses twice, so three attempts
    status_forcelist=_RETRYABLE_STATUS_CODES,
    raise_on_status=False,  # return the final response rather than raising
)

_thread_local = threading.local()


def thread_local_session(retry_transient: bool = False) -> requests.Session:
    """Return a retry-enabled Session cached per thread and process.

    Args:
        retry_transient: Retry transient backend statuses with exponential
            backoff, returning the final response when attempts are exhausted.

    Returns:
        The cached Session for this thread, process and retry policy.
    """
    pid = os.getpid()

    if getattr(_thread_local, "pid", None) != pid:
        _thread_local.sessions = {}
        _thread_local.pid = pid

    session = _thread_local.sessions.get(retry_transient)

    if session is None:
        session = requests.Session()

        adapter = HTTPAdapter(
            max_retries=_TRANSIENT_RETRY if retry_transient else _RETRY
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        _thread_local.sessions[retry_transient] = session

    return session


_STALE_CONNECTION_ATTEMPTS = 3


async def retry_connection_failures(
    request: ClientRequest, handler: ClientHandlerType
) -> ClientResponse:
    """Retry aiohttp requests that fail on a stale pooled keep-alive connection.

    Args:
        request: The outgoing client request.
        handler: The next handler in the middleware chain.

    Returns:
        ClientResponse: The response from the first successful attempt.

    Raises:
        ServerDisconnectedError: If all attempts hit a closed connection.
        ClientOSError: If all attempts fail at the socket level.
    """
    for attempt in range(_STALE_CONNECTION_ATTEMPTS):
        try:
            return await handler(request)
        except (ServerDisconnectedError, ClientOSError) as e:
            if attempt == _STALE_CONNECTION_ATTEMPTS - 1:
                raise
            logger.warning(
                "Stale connection on %s %s (attempt %d/%d): %s",
                request.method,
                request.url,
                attempt + 1,
                _STALE_CONNECTION_ATTEMPTS,
                e,
            )
            await asyncio.sleep(0.1 * 2**attempt)
    raise AssertionError("unreachable")
