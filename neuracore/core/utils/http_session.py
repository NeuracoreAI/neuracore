"""Thread- and process-local HTTP session with shared retry policy.

All Neuracore API calls should obtain a session via ``thread_local_session()``
rather than constructing fresh ``requests.Session`` instances or using the
module-level ``requests.*`` helpers. The returned session is cached per OS
thread and per process: forks get a fresh session (since urllib3 connection
pools are not safe to share across forks), and threads do not share a session
with one another.

Sessions returned here carry two protections against a pooled keep-alive
connection whose peer has gone away: TCP keepalive on the socket, so the path's
flow state stays warm and a dead peer surfaces as a normal dropped connection;
and a default request timeout, so a request that never receives a response fails
in seconds rather than blocking on the kernel's retransmission budget. Call
sites may still pass their own ``timeout``, which takes precedence.

For the aiohttp stack, ``retry_stale_connection`` provides the equivalent
policy as a client middleware: pass it via
``ClientSession(middlewares=(retry_stale_connection,))``.
"""

import asyncio
import logging
import os
import socket
import threading
from typing import Any

import requests
from aiohttp import (
    ClientHandlerType,
    ClientOSError,
    ClientRequest,
    ClientResponse,
    ServerDisconnectedError,
)
from requests.adapters import DEFAULT_POOLBLOCK, HTTPAdapter
from urllib3 import Retry
from urllib3.exceptions import ProtocolError

# cspell:ignore IPPROTO KEEPCNT KEEPIDLE KEEPINTVL NODELAY POOLBLOCK poolmanager

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

_READ_TIMEOUT_RETRY = _RETRY.new(
    read=2,
    allowed_methods=frozenset({"GET", "HEAD"}),
)

_TRANSIENT_READ_TIMEOUT_RETRY = _TRANSIENT_RETRY.new(
    read=2,
    allowed_methods=frozenset({"GET", "HEAD"}),
)

KEEPALIVE_IDLE_S = 60
"""Idle seconds before the first keepalive probe.

Sessions are cached for the life of the process, so a pooled connection can sit
idle between two API calls for as long as the work in between takes — tens of
minutes in the integration suites. Probing every minute keeps the flow state
alive in any NAT or load balancer on the path, which is what stops the mapping
being reaped underneath an idle connection.
"""

KEEPALIVE_INTERVAL_S = 15
"""Seconds between keepalive probes once the peer stops answering."""

KEEPALIVE_PROBES = 4
"""Failed probes before the connection is considered dead (~2 min total)."""


def _keepalive_socket_options() -> list[tuple[int, int, int]]:
    """Socket options enabling TCP keepalive, preserving urllib3's TCP_NODELAY.

    A connection whose flow state is dropped *silently* — no FIN, no RST, as
    happens when a NAT mapping is evicted — is indistinguishable from a healthy
    one to urllib3's ``is_connection_dropped`` check, which only detects a
    socket where a FIN arrived. Reusing such a connection blocks until the
    kernel exhausts its retransmission budget (~16 minutes on Linux). Keepalive
    both refreshes the path state and surfaces a dead peer as a normal dropped
    connection, which urllib3 already discards and reconnects.

    Returns:
        Socket options accepted by urllib3's connection pools. Options absent on
        the running platform are skipped — the names differ across Linux and
        macOS, and not every one is exposed by every Python build.
    """
    options: list[tuple[int, int, int]] = [
        (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    ]
    # TCP_KEEPIDLE on Linux; TCP_KEEPALIVE is the macOS spelling of the same.
    idle_option = getattr(socket, "TCP_KEEPIDLE", None) or getattr(
        socket, "TCP_KEEPALIVE", None
    )
    if idle_option is not None:
        options.append((socket.IPPROTO_TCP, idle_option, KEEPALIVE_IDLE_S))
    if hasattr(socket, "TCP_KEEPINTVL"):
        options.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, KEEPALIVE_INTERVAL_S))
    if hasattr(socket, "TCP_KEEPCNT"):
        options.append((socket.IPPROTO_TCP, socket.TCP_KEEPCNT, KEEPALIVE_PROBES))
    return options


class _KeepAliveAdapter(HTTPAdapter):
    """Adapter whose pooled connections use TCP keepalive."""

    def init_poolmanager(
        self,
        connections: int,
        maxsize: int,
        block: bool = DEFAULT_POOLBLOCK,
        **pool_kwargs: Any,
    ) -> None:
        """Build the pool manager with keepalive socket options applied."""
        pool_kwargs.setdefault("socket_options", _keepalive_socket_options())
        super().init_poolmanager(connections, maxsize, block, **pool_kwargs)


DEFAULT_CONNECT_TIMEOUT_S = 10.0
"""Seconds allowed to establish a connection."""

DEFAULT_READ_TIMEOUT_S = 30.0
"""Seconds allowed between response bytes.

Sized from measured backend latency rather than from the largest timeout anyone
has written down: across a sample of 500 staging requests, p50 was 0.05s and p95
0.49s, and the slowest endpoint on this stack was ``GET /robots/{id}/package`` at
10.3s. Thirty seconds leaves roughly three times that headroom.

The long-running endpoints — the SSE notification streams, which the server caps
at 600s — are on the aiohttp stack and never see this value. The two calls that
genuinely need minutes (:mod:`neuracore.core.utils.download` and the pretrained
cache) pass their own timeout, which takes precedence.

This is an inter-byte timeout, not a total one, so a slow streaming upload or
download is unaffected as long as data keeps moving.
"""

DEFAULT_TIMEOUT = (DEFAULT_CONNECT_TIMEOUT_S, DEFAULT_READ_TIMEOUT_S)


class _DefaultTimeoutSession(requests.Session):
    """Session that applies :data:`DEFAULT_TIMEOUT` when a caller omits one.

    ``requests`` defaults to no timeout, so a request that never gets a response
    blocks forever. Defaulting here bounds every call in one place rather than
    at each of the call sites.
    """

    def request(self, *args: Any, **kwargs: Any) -> requests.Response:
        """Send a request, defaulting the timeout when none was supplied."""
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = DEFAULT_TIMEOUT
        return super().request(*args, **kwargs)


_thread_local = threading.local()


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
        _thread_local.sessions = {}
        _thread_local.pid = pid

    session_key = (retry_transient, retry_read_timeout)
    session = _thread_local.sessions.get(session_key)

    if session is None:
        session = _DefaultTimeoutSession()

        if retry_transient and retry_read_timeout:
            retry_policy = _TRANSIENT_READ_TIMEOUT_RETRY
        elif retry_read_timeout:
            retry_policy = _READ_TIMEOUT_RETRY
        elif retry_transient:
            retry_policy = _TRANSIENT_RETRY
        else:
            retry_policy = _RETRY

        adapter = _KeepAliveAdapter(max_retries=retry_policy)
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
