"""Shared token bucket bandwidth limiter for upload management."""

import asyncio
import time


class BandwidthLimiter:
    """Implements a token bucket algorithm to limit upload bandwidth."""

    def __init__(self, bytes_per_second: int) -> None:
        """Initialise the limiter.

        Args:
            bytes_per_second: Maximum aggregate upload rate in bytes/second.
        """
        if bytes_per_second <= 0:
            raise ValueError(
                f"bytes_per_second must be a positive integer, got {bytes_per_second}"
            )
        self._rate = bytes_per_second
        self._tokens = float(bytes_per_second)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, n_bytes: int) -> None:
        """Acquire a number of bytes from the token bucket.

        Args:
            n_bytes: Number of bytes about to be uploaded.
        """
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens += (now - self._last_refill) * self._rate
                self._last_refill = now
                if self._tokens >= n_bytes:
                    self._tokens -= n_bytes
                    return
                wait = (n_bytes - self._tokens) / self._rate
            await asyncio.sleep(wait)
