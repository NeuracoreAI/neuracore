"""Convert a caller's timestamp to ticks at the public API."""

from neuracore_types.timestamps import now_ticks, timestamp_to_ticks


def resolve_ticks(timestamp: float | int | None) -> int:
    """Return the ticks for a timestamp given to the public API.

    Args:
        timestamp: Float seconds, integer ticks, or ``None`` for the monotonic
            clock now.

    Raises:
        TypeError: If the timestamp is a bool or not a number.
        ValueError: If the timestamp is not finite or is at or above 2**53 ticks.
    """
    return now_ticks() if timestamp is None else timestamp_to_ticks(timestamp)
