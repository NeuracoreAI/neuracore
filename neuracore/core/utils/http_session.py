"""Thread- and process-local HTTP session with shared retry policy.

All Neuracore API calls should obtain a session via ``thread_local_session()``
rather than constructing fresh ``requests.Session`` instances or using the
module-level ``requests.*`` helpers. The returned session is cached per OS
thread and per process: forks get a fresh session (since urllib3 connection
pools are not safe to share across forks), and threads do not share a session
with one another.

Sessions returned here bound every request in time. ``requests`` applies no
timeout of its own, and a pooled connection whose flow has been dropped
silently by an intermediary (cloud load balancers commonly drop idle outbound
flows without sending a RST) accepts the write and then never answers, so the
only limit is the kernel's retransmission budget — around 15 minutes under
Linux's default ``tcp_retries2``. ``_DefaultTimeoutAdapter`` supplies
``DEFAULT_TIMEOUT`` whenever a caller passes none, and ``_socket_options()``
enables TCP keep-alive below the idle limits that cause the drop, so the
connection stays alive rather than dying unobserved. A request that does fail
this way is retried on a fresh connection by ``_DroppedConnectionRetry``.

For the aiohttp stack, ``retry_stale_connection`` provides the equivalent
policy as a client middleware: pass it via
``ClientSession(middlewares=(retry_stale_connection,))``.
"""

import asyncio
import logging
import os
import socket
import threading
import time
from collections.abc import Mapping
from typing import Any

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
from urllib3.connection import HTTPConnection
from urllib3.exceptions import ProtocolError

# cspell:ignore IPPROTO KEEPCNT KEEPIDLE KEEPINTVL poolmanager

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT: tuple[float, float] = (15.0, 30.0)
"""``(connect, read)`` seconds applied to requests that set no timeout.

The connect budget also bounds the request body write, not just connection
setup: urllib3 leaves the connect timeout on the socket for the whole of
``conn.request()`` and only swaps in the read timeout at ``getresponse()``. It
therefore has to cover each ``send()`` of a streamed upload once the kernel
send buffer fills and every send waits on the drain rate — five seconds was
enough for connecting but not for that.

Callers whose response is genuinely slow should pass their own timeout rather
than widening this one.
"""

_KEEPALIVE_IDLE_S = 60
"""Seconds of idleness before the first keep-alive probe.

Below the idle window after which intermediaries drop a flow (commonly four
minutes) so the connection keeps being observed as live.
"""

_KEEPALIVE_INTERVAL_S = 20
"""Seconds between keep-alive probes once probing starts."""

_KEEPALIVE_PROBE_COUNT = 3
"""Failed keep-alive probes before the connection is declared dead."""

_USER_TIMEOUT_MS = 30_000
"""Cap on how long Linux retransmits unacknowledged data before giving up."""

_thread_local = threading.local()
"""Per-thread, per-process session cache and HTTP-time accounting."""

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


def _socket_options() -> list[tuple[int, int, int | bytes]]:
    """Build the TCP options applied to every pooled connection.

    Each option is skipped where the platform does not define it: ``TCP_KEEPIDLE``
    is Linux-only and spelled ``TCP_KEEPALIVE`` on macOS, and ``TCP_USER_TIMEOUT``
    is Linux-only.

    Returns:
        Socket options in the ``(level, option, value)`` form urllib3 expects.
    """
    options: list[tuple[int, int, int | bytes]] = [
        *HTTPConnection.default_socket_options,
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    ]

    keepidle = getattr(socket, "TCP_KEEPIDLE", None) or getattr(
        socket, "TCP_KEEPALIVE", None
    )
    for option, value in (
        (keepidle, _KEEPALIVE_IDLE_S),
        (getattr(socket, "TCP_KEEPINTVL", None), _KEEPALIVE_INTERVAL_S),
        (getattr(socket, "TCP_KEEPCNT", None), _KEEPALIVE_PROBE_COUNT),
        (getattr(socket, "TCP_USER_TIMEOUT", None), _USER_TIMEOUT_MS),
    ):
        if option is not None:
            options.append((socket.IPPROTO_TCP, option, value))

    return options


class _DefaultTimeoutAdapter(HTTPAdapter):
    """Adapter that bounds untimed requests and accounts for time spent in them.

    ``requests`` passes ``timeout=None`` when a caller omits it, which means
    "block until the socket resolves". Substituting ``DEFAULT_TIMEOUT`` there
    bounds the wait without overriding callers that chose their own value.
    """

    def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
        """Attach the keep-alive socket options to the underlying pool manager."""
        kwargs["socket_options"] = _socket_options()
        super().init_poolmanager(*args, **kwargs)

    def send(
        self,
        request: requests.PreparedRequest,
        stream: bool = False,
        timeout: None | float | tuple[float, float] | tuple[float, None] = None,
        verify: bool | str = True,
        cert: None | bytes | str | tuple[bytes | str, bytes | str] = None,
        proxies: Mapping[str, str] | None = None,
    ) -> requests.Response:
        """Send a request, defaulting its timeout and recording its duration.

        Args:
            request: The prepared request to send.
            stream: Whether to defer downloading the response body.
            timeout: Caller-supplied timeout, or ``None`` to use the default.
            verify: TLS verification setting, forwarded unchanged.
            cert: Client certificate, forwarded unchanged.
            proxies: Proxy mapping, forwarded unchanged.

        Returns:
            The response from the underlying adapter.
        """
        if timeout is None:
            timeout = DEFAULT_TIMEOUT
        started = time.perf_counter()
        try:
            return super().send(
                request,
                stream=stream,
                timeout=timeout,
                verify=verify,
                cert=cert,
                proxies=proxies,
            )
        finally:
            _record_request_duration(time.perf_counter() - started)


def _record_request_duration(seconds: float) -> None:
    """Add one completed request to this thread's HTTP accounting."""
    _thread_local.http_calls = getattr(_thread_local, "http_calls", 0) + 1
    _thread_local.http_seconds = getattr(_thread_local, "http_seconds", 0.0) + seconds


def http_time_snapshot() -> tuple[int, float]:
    """Return cumulative HTTP request count and seconds for the calling thread.

    Counts every request issued through a session from ``thread_local_session()``,
    including retried attempts, and covers failed requests as well as successful
    ones. Sample it before and after a block of work to attribute that block's
    elapsed time between waiting on the API and everything else.

    Scoped to the calling thread, matching the per-thread sessions: requests made
    by other threads or by subprocesses are not included.

    Returns:
        Tuple of (request count, total seconds spent in those requests).
    """
    return (
        getattr(_thread_local, "http_calls", 0),
        getattr(_thread_local, "http_seconds", 0.0),
    )


_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

_TRANSIENT_RETRY = _RETRY.new(
    status=2,  # retry transient backend statuses twice, so three attempts
    status_forcelist=_RETRYABLE_STATUS_CODES,
    raise_on_status=False,  # return the final response rather than raising
)

_READ_TIMEOUT_RETRY = _RETRY.new(
    read=2,
    allowed_methods=frozenset({"GET", "HEAD"}),
)

_TRANSIENT_READ_TIMEOUT_RETRY = _TRANSIENT_RETRY.new(
    read=2,
    allowed_methods=frozenset({"GET", "HEAD"}),
)


def thread_local_session(
    retry_transient: bool = False,
    retry_read_timeout: bool = False,
) -> requests.Session:
    """Return a retry-enabled Session cached per thread and process.

    Args:
        retry_transient: Retry transient backend status codes with exponential
            backoff.
        retry_read_timeout: Retry idempotent GET and HEAD requests that time out
            while waiting for a response.

    Returns:
        The cached Session for this thread, process and retry policy.
    """
    pid = os.getpid()

    if getattr(_thread_local, "pid", None) != pid:
        # A forked child inherits the parent's accounting along with its unusable
        # pools, so both are discarded together and the child counts only its own.
        _thread_local.sessions = {}
        _thread_local.pid = pid
        _thread_local.http_calls = 0
        _thread_local.http_seconds = 0.0

    session_key = (retry_transient, retry_read_timeout)
    session = _thread_local.sessions.get(session_key)

    if session is None:
        session = requests.Session()

        if retry_transient and retry_read_timeout:
            retry_policy = _TRANSIENT_READ_TIMEOUT_RETRY
        elif retry_read_timeout:
            retry_policy = _READ_TIMEOUT_RETRY
        elif retry_transient:
            retry_policy = _TRANSIENT_RETRY
        else:
            retry_policy = _RETRY

        adapter = _DefaultTimeoutAdapter(max_retries=retry_policy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        _thread_local.sessions[session_key] = session

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
