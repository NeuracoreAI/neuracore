"""Signalling session tokens and robot registration for regional signalling.

A session holds the token of one robot instance or one user. It gets the token
from the backend with the API key, and renews it after ``renew_after``. A robot
registers in a region with a register token, which carries a new epoch.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from aiohttp import ClientResponse, ClientSession
from neuracore_types import SignallingTokenRequest, SignallingTokenResponse

from neuracore.core.auth import Auth
from neuracore.core.exceptions import AuthenticationError

LOWER_EPOCH = "lower_epoch"
# The backend or the region refuses the party: retrying does not help.
REFUSED_STATUSES = (401, 403)


class LowerEpochError(Exception):
    """The region holds the stream at a higher epoch."""


@dataclass(frozen=True)
class RegistrationResult:
    """A robot instance's registration in a region.

    Attributes:
        region: Name of the region.
        canonical_url: URL that reaches the region directly.
        epoch: Registration epoch that the region holds.
    """

    region: str
    canonical_url: str
    epoch: int


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SignallingSession:
    """Token holder for one robot instance or one user."""

    def __init__(
        self,
        api_url: str,
        org_id: str,
        auth: Auth,
        client_session: ClientSession,
        robot_id: str | None = None,
        robot_instance: int | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Initialize a session with no token.

        Args:
            api_url: Base URL of the backend API.
            org_id: Organization of the session.
            auth: API key auth for the backend.
            client_session: HTTP session for the backend and the regions.
            robot_id: Robot of a robot or user session, or None for the
                organization.
            robot_instance: Robot instance of a robot session.
            clock: Current UTC time.
        """
        self.api_url = api_url
        self.org_id = org_id
        self.auth = auth
        self.client_session = client_session
        self.robot_id = robot_id
        self.robot_instance = robot_instance
        self.clock = clock
        self.region_endpoints: list[str] = []
        self.base_url: str | None = None
        self._token: SignallingTokenResponse | None = None
        # Renewals and registrations both set the token: one at a time.
        self._lock = asyncio.Lock()

    async def headers(self) -> dict[str, str]:
        """Return the region headers, with a token renewed after ``renew_after``.

        Raises:
            AuthenticationError: If the backend refuses the API key.
        """
        async with self._lock:
            if self._token is None or self.clock() >= self._token.renew_after:
                self._token = await self._request_token(register_epoch=False)
            return {"Authorization": f"Bearer {self._token.token}"}

    async def register(self, base_url: str, stream_id: str) -> RegistrationResult:
        """Register the robot instance's stream in a region with a new epoch.

        Raises:
            LowerEpochError: If the region holds the stream at a higher epoch.
            AuthenticationError: If the backend or the region refuses the party.
            aiohttp.ClientResponseError: If the region fails otherwise.
        """
        async with self._lock:
            self._token = await self._request_token(register_epoch=True)
            return await self._post_registration(base_url, stream_id)

    async def _post_registration(
        self, base_url: str, stream_id: str
    ) -> RegistrationResult:
        assert self._token is not None
        async with self.client_session.post(
            f"{base_url}/org/{self.org_id}/signalling/registrations",
            json={
                "robot_id": self.robot_id,
                "robot_instance": self.robot_instance,
                "stream_id": stream_id,
            },
            headers={"Authorization": f"Bearer {self._token.token}"},
        ) as response:
            if response.status == 409 and await _detail(response) == LOWER_EPOCH:
                raise LowerEpochError()
            _raise_for_status(response)
            body = await response.json()
        result = RegistrationResult(
            region=body["region"],
            canonical_url=body["canonical_url"],
            epoch=body["epoch"],
        )
        self.base_url = result.canonical_url
        return result

    async def _request_token(self, register_epoch: bool) -> SignallingTokenResponse:
        request = SignallingTokenRequest(
            robot_id=self.robot_id,
            robot_instance=self.robot_instance,
            register_epoch=register_epoch,
        )
        async with self.client_session.post(
            f"{self.api_url}/org/{self.org_id}/signalling/token",
            json=request.model_dump(mode="json", exclude_defaults=True),
            headers=self.auth.get_headers(),
        ) as response:
            _raise_for_status(response)
            token = SignallingTokenResponse.model_validate(await response.json())
        self.region_endpoints = token.region_endpoints
        return token


async def _detail(response: ClientResponse) -> object:
    try:
        body = await response.json(content_type=None)
    except ValueError:
        return None
    return body.get("detail") if isinstance(body, dict) else None


def _raise_for_status(response: ClientResponse) -> None:
    if response.status in REFUSED_STATUSES:
        raise AuthenticationError(f"signalling refused: {response.status}")
    response.raise_for_status()
