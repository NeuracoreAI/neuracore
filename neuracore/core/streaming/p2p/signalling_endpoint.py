"""Where a party's signalling calls go: its region."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

# Returns the base URL of the signalling routes and the headers to send.
SignallingEndpoint = Callable[[], Awaitable[tuple[str, dict[str, str]]]]


@dataclass(frozen=True)
class SignallingTarget:
    """The stream that signals a connection, and where its calls go."""

    stream_id: str
    endpoint: SignallingEndpoint
