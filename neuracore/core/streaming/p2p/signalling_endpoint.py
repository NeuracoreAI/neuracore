"""Where a party's signalling calls go: the legacy service or its region."""

from collections.abc import Awaitable, Callable

from neuracore.core.auth import Auth
from neuracore.core.const import STREAM_API_URL

# Returns the base URL of the signalling routes and the headers to send.
SignallingEndpoint = Callable[[], Awaitable[tuple[str, dict[str, str]]]]


def legacy_endpoint(org_id: str, auth: Auth) -> SignallingEndpoint:
    """Return the endpoint of the legacy signalling service."""

    async def endpoint() -> tuple[str, dict[str, str]]:
        return f"{STREAM_API_URL}/org/{org_id}/signalling", auth.get_headers()

    return endpoint
