import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
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


class _ScriptedServer:
    """Local HTTP server replaying a scripted status sequence."""

    def __init__(self, statuses: list[int]) -> None:
        self.statuses = statuses
        self.requests: list[tuple[str, bytes]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _respond(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                index = len(server.requests)
                server.requests.append((self.command, body))
                status = server.statuses[min(index, len(server.statuses) - 1)]
                self.send_response(status)
                self.end_headers()
                self.wfile.write(b"body")

            do_GET = _respond
            do_PUT = _respond
            do_POST = _respond

            def log_message(self, *args: object) -> None:
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}/resource"

    def __enter__(self) -> "_ScriptedServer":
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    @property
    def call_count(self) -> int:
        return len(self.requests)


@pytest.fixture
def transient_session():
    """Return a fresh session that retries transient backend statuses."""
    _reset_thread_local()
    yield thread_local_session(retry_transient=True)
    _reset_thread_local()


def _reset_thread_local() -> None:
    """Drop any session cached on the current thread."""
    for attribute in ("sessions", "pid"):
        if hasattr(http_session._thread_local, attribute):
            delattr(http_session._thread_local, attribute)


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


class TestTransientRetrySession:
    def test_transient_session_is_cached_separately(self):
        _reset_thread_local()
        plain = thread_local_session()
        transient = thread_local_session(retry_transient=True)
        assert plain is not transient
        assert thread_local_session(retry_transient=True) is transient

    def test_pid_change_invalidates_both_sessions(self):
        _reset_thread_local()
        with patch.object(http_session.os, "getpid", return_value=1):
            plain = thread_local_session()
            transient = thread_local_session(retry_transient=True)
        with patch.object(http_session.os, "getpid", return_value=2):
            assert thread_local_session() is not plain
            assert thread_local_session(retry_transient=True) is not transient

    def test_transient_retry_config(self):
        _reset_thread_local()
        retry = thread_local_session(retry_transient=True).get_adapter(_URL).max_retries
        assert retry.read == 0
        assert retry.connect == 3


class TestTransientRetryBehaviour:
    def test_retries_transient_status_then_returns_success(self, transient_session):
        with _ScriptedServer([503, 500, 200]) as server:
            response = transient_session.get(server.url)
        assert response.status_code == 200
        assert server.call_count == 3

    def test_returns_final_response_when_attempts_exhausted(self, transient_session):
        with _ScriptedServer([503, 503, 503]) as server:
            response = transient_session.get(server.url)
        assert response.status_code == 503
        assert server.call_count == 3

    def test_does_not_retry_non_transient_status(self, transient_session):
        with _ScriptedServer([403]) as server:
            response = transient_session.get(server.url)
        assert response.status_code == 403
        assert server.call_count == 1

    def test_does_not_retry_unauthorized(self, transient_session):
        with _ScriptedServer([401]) as server:
            response = transient_session.get(server.url)
        assert response.status_code == 401
        assert server.call_count == 1

    def test_retries_rate_limited_status(self, transient_session):
        with _ScriptedServer([429, 200]) as server:
            response = transient_session.get(server.url)
        assert response.status_code == 200
        assert server.call_count == 2

    def test_resends_full_file_body_on_retry(self, transient_session, tmp_path: Path):
        payload = b"IMPORTANT-CHECKPOINT-BYTES"
        upload = tmp_path / "checkpoint.pt"
        upload.write_bytes(payload)

        with _ScriptedServer([503, 200]) as server:
            with open(upload, "rb") as handle:
                response = transient_session.put(server.url, data=handle)

        assert response.status_code == 200
        assert server.call_count == 2
        assert [body for _, body in server.requests] == [payload, payload]

    def test_retries_post_on_transient_status(self, transient_session):
        with _ScriptedServer([503, 200]) as server:
            response = transient_session.post(server.url, json={"a": 1})
        assert response.status_code == 200
        assert server.call_count == 2
        assert [method for method, _ in server.requests] == ["POST", "POST"]

    def test_plain_session_does_not_retry_transient_status(self):
        _reset_thread_local()
        with _ScriptedServer([503, 200]) as server:
            response = thread_local_session().get(server.url)
        assert response.status_code == 503
        assert server.call_count == 1
        _reset_thread_local()
