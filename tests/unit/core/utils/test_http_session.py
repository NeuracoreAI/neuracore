import threading
from unittest.mock import MagicMock, patch

import pytest
import requests
import requests_mock
from aiohttp import ClientSession, ServerDisconnectedError, web
from aiohttp.test_utils import TestServer
from requests.adapters import HTTPAdapter

from neuracore.core.utils import http_session
from neuracore.core.utils.http_session import (
    retry_connection_failures,
    thread_local_session,
)

# cspell:ignore poolmanager

_URL = "https://api.neuracore.com"


def _reset_thread_local() -> None:
    """Drop any session cached on the current thread."""
    if hasattr(http_session._thread_local, "session"):
        del http_session._thread_local.session
    if hasattr(http_session._thread_local, "pid"):
        del http_session._thread_local.pid


def test_returns_requests_session():
    _reset_thread_local()
    session = thread_local_session()
    assert isinstance(session, requests.Session)


def test_same_thread_returns_cached_session():
    _reset_thread_local()
    first = thread_local_session()
    second = thread_local_session()
    assert first is second


def test_different_threads_get_independent_sessions():
    _reset_thread_local()
    main_session = thread_local_session()
    captured: dict = {}

    def worker() -> None:
        captured["session"] = thread_local_session()

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert captured["session"] is not main_session


def test_pid_change_invalidates_cached_session():
    _reset_thread_local()
    with patch.object(http_session.os, "getpid", return_value=1):
        original = thread_local_session()
    with patch.object(http_session.os, "getpid", return_value=2):
        post_fork = thread_local_session()
    assert post_fork is not original


def test_session_mounts_http_and_https():
    _reset_thread_local()
    session = thread_local_session()
    assert isinstance(session.get_adapter("https://"), HTTPAdapter)
    assert isinstance(session.get_adapter("http://"), HTTPAdapter)


def test_retry_config():
    _reset_thread_local()
    session = thread_local_session()
    retry = session.get_adapter(_URL).max_retries
    assert retry.connect == 3
    assert retry.read == 0
    assert retry.status == 0
    assert retry.total == 3


def test_retry_allows_all_methods():
    _reset_thread_local()
    session = thread_local_session()
    retry = session.get_adapter(_URL).max_retries
    assert retry.allowed_methods is False


def test_no_retry_on_5xx():
    _reset_thread_local()
    with requests_mock.Mocker() as m:
        m.get(f"{_URL}/health", status_code=500)
        session = thread_local_session()
        response = session.get(f"{_URL}/health")
    assert response.status_code == 500
    assert m.call_count == 1


def test_adapters_share_retry_config():
    _reset_thread_local()
    session = thread_local_session()
    for scheme in ("http://", "https://"):
        adapter = session.get_adapter(scheme)
        assert adapter.max_retries.connect == 3


def _fake_request() -> MagicMock:
    request = MagicMock()
    request.method = "POST"
    request.url = "https://api.neuracore.com/track"
    return request


@pytest.mark.asyncio
async def test_middleware_retries_stale_connection():
    attempts = []

    async def handler(request):
        attempts.append(request)
        if len(attempts) < 3:
            raise ServerDisconnectedError()
        return "response"

    result = await retry_connection_failures(_fake_request(), handler)
    assert result == "response"
    assert len(attempts) == 3


@pytest.mark.asyncio
async def test_middleware_raises_after_exhausting_attempts():
    attempts = []

    async def handler(request):
        attempts.append(request)
        raise ServerDisconnectedError()

    with pytest.raises(ServerDisconnectedError):
        await retry_connection_failures(_fake_request(), handler)
    assert len(attempts) == 3


@pytest.mark.asyncio
async def test_middleware_does_not_retry_other_errors():
    attempts = []

    async def handler(request):
        attempts.append(request)
        raise ValueError("unrelated")

    with pytest.raises(ValueError):
        await retry_connection_failures(_fake_request(), handler)
    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_session_with_middleware_survives_server_disconnects():
    hits = {"count": 0}

    async def handler(request: web.Request) -> web.Response:
        hits["count"] += 1
        if hits["count"] <= 2:
            assert request.transport is not None
            request.transport.close()
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_get("/health", handler)
    server = TestServer(app)
    await server.start_server()
    try:
        async with ClientSession(middlewares=(retry_connection_failures,)) as session:
            async with session.get(server.make_url("/health")) as response:
                assert response.status == 200
                assert await response.json() == {"ok": True}
        assert hits["count"] == 3
    finally:
        await server.close()
