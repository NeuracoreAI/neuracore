"""Temporary lap timer for benchmarking training run steps."""

import logging
import time

logger = logging.getLogger(__name__)


class StepTimer:
    """Logs time elapsed between consecutive laps, using time.monotonic."""

    def __init__(self, scope: str):
        """Start the timer for the given scope."""
        self.scope = scope
        self._start = self._last = time.monotonic()

    def lap(self, step: str) -> float:
        """Log and return seconds since the previous lap."""
        now = time.monotonic()
        elapsed, self._last = now - self._last, now
        logger.info("[timing] %s/%s: %.2fs", self.scope, step, elapsed)
        return elapsed

    def total(self, step: str = "total") -> float:
        """Log and return seconds since the timer was created."""
        elapsed = time.monotonic() - self._start
        logger.info("[timing] %s/%s: %.2fs", self.scope, step, elapsed)
        return elapsed
