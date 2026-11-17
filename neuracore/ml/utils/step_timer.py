"""Lap timer that logs how long each step of a training run takes."""

import logging
import time

logger = logging.getLogger(__name__)


class StepTimer:
    """Logs ``[timing] <scope>/<step>: X.XXs`` for the time between laps."""

    def __init__(self, scope: str) -> None:
        """Start timing the given scope."""
        self.scope = scope
        self._started_at = self._last_lap_at = time.monotonic()

    def lap(self, step: str) -> None:
        """Log the time since the previous lap as the given step."""
        now = time.monotonic()
        self._log(step, now - self._last_lap_at)
        self._last_lap_at = now

    def total(self) -> None:
        """Log the time since the timer started."""
        self._log("total", time.monotonic() - self._started_at)

    def _log(self, step: str, seconds: float) -> None:
        logger.info("[timing] %s/%s: %.2fs", self.scope, step, seconds)
