"""Convert a caller's timestamp to microseconds at the public API."""

from neuracore_types.timestamps import TIMESTAMP_US_LIMIT, now_us, seconds_to_us


def resolve_timestamp_us(timestamp: float | None) -> int:
    """Return the microseconds for a timestamp given to the public API.

    Args:
        timestamp: Seconds, or ``None`` for the wall clock now.

    Returns:
        int: The timestamp in microseconds.

    Raises:
        ValueError: If the timestamp is not finite, or its microseconds are
            negative or at or above ``TIMESTAMP_US_LIMIT``.
    """
    if timestamp is None:
        return now_us()
    timestamp_us = seconds_to_us(timestamp)
    if not 0 <= timestamp_us < TIMESTAMP_US_LIMIT:
        raise ValueError(
            f"Timestamp {timestamp} s is outside the supported range of "
            f"0 to {TIMESTAMP_US_LIMIT} microseconds"
        )
    return timestamp_us
