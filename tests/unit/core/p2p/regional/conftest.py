"""Fake backend and fake region servers for the regional signalling tests."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import json
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from types import ModuleType
from typing import Any

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

ORG_ID = "org-1"
ROBOT_ID = "robot-1"
TOKEN_LIFETIME = timedelta(minutes=15)
RENEW_AFTER = timedelta(minutes=10)


def regional_module(name: str) -> ModuleType:
    """Import a module of the regional client by its name under ``p2p``."""
    return importlib.import_module(f"neuracore.core.streaming.p2p.{name}")


@dataclasses.dataclass
class Recorded:
    """One request a fake server received."""

    path: str
    authorization: str | None
    body: Any


class FakeBackend:
    """The backend token route. Each token is ``token-<n>``."""

    def __init__(self, issued_at: datetime) -> None:
        """Initialize; tokens are issued at ``issued_at``."""
        self.issued_at = issued_at
        self.region_endpoints: list[str] = []
        self.status = 200
        self.calls: list[Recorded] = []
        self.app = web.Application()
        self.app.router.add_post("/api/org/{org_id}/signalling/token", self._token)

    async def _token(self, request: web.Request) -> web.Response:
        body = await request.json() if request.can_read_body else {}
        self.calls.append(
            Recorded(request.path, request.headers.get("Authorization"), body)
        )
        if self.status != 200:
            return web.json_response({"detail": "refused"}, status=self.status)
        return web.json_response({
            "token": f"token-{len(self.calls)}",
            "expires_at": (self.issued_at + TOKEN_LIFETIME).isoformat(),
            "renew_after": (self.issued_at + RENEW_AFTER).isoformat(),
            "region_endpoints": self.region_endpoints,
        })


class FakeRegion:
    """The regional routes a client calls, with controllable answers."""

    def __init__(self) -> None:
        """Initialize with 200 answers and no events."""
        self.registration_status = 200
        self.registration_body: dict[str, Any] = {
            "region": "region-a",
            "canonical_url": "",
            "epoch": 1,
        }
        self.resurrected = False
        self.events: list[tuple[str, dict[str, Any]]] = []
        # Keep each event stream open after its events, as a live region does.
        self.hold_open = False
        self.registrations: list[Recorded] = []
        self.heartbeats: list[Recorded] = []
        self.event_streams = 0
        self.app = web.Application()
        prefix = "/org/{org_id}/signalling"
        self.app.router.add_post(f"{prefix}/registrations", self._register)
        self.app.router.add_post(f"{prefix}/alive/{{stream_id}}", self._alive)
        self.app.router.add_get(f"{prefix}/notifications/{{stream_id}}", self._events)

    async def _register(self, request: web.Request) -> web.Response:
        self.registrations.append(
            Recorded(
                request.path, request.headers.get("Authorization"), await request.json()
            )
        )
        return web.json_response(
            self.registration_body, status=self.registration_status
        )

    async def _alive(self, request: web.Request) -> web.Response:
        self.heartbeats.append(
            Recorded(request.path, request.headers.get("Authorization"), None)
        )
        return web.json_response({"resurrected": self.resurrected})

    async def _events(self, request: web.Request) -> web.StreamResponse:
        self.event_streams += 1
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for name, data in self.events:
            await response.write(f"event:{name}\ndata:{json.dumps(data)}\n\n".encode())
        if self.hold_open:
            await asyncio.sleep(3600)
        await response.write_eof()
        return response


@pytest.fixture
def issued_at() -> datetime:
    """Issue time of the fake backend's tokens."""
    return datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)


@pytest_asyncio.fixture
async def backend(issued_at: datetime) -> AsyncIterator[tuple[FakeBackend, str]]:
    """A fake backend and its URL."""
    fake = FakeBackend(issued_at)
    server = TestServer(fake.app)
    await server.start_server()
    try:
        yield fake, str(server.make_url("")).rstrip("/")
    finally:
        await server.close()


@pytest_asyncio.fixture
async def region() -> AsyncIterator[tuple[FakeRegion, str]]:
    """A fake region and its canonical URL."""
    fake = FakeRegion()
    server = TestServer(fake.app)
    await server.start_server()
    url = str(server.make_url("")).rstrip("/")
    fake.registration_body["canonical_url"] = url
    try:
        yield fake, url
    finally:
        await server.close()


class FakeAuth:
    """The client's API key auth."""

    def get_headers(self) -> dict[str, str]:
        """Return the API key headers."""
        return {"Authorization": "Bearer api-key"}


@pytest.fixture
def auth() -> FakeAuth:
    """The client's API key auth."""
    return FakeAuth()


@pytest.fixture
def regional() -> Any:
    """Import a module of the regional client by name."""
    return regional_module
