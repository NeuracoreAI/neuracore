"""Lightweight opt-in profiling helpers for transport hot paths."""

from __future__ import annotations

import atexit
import logging
import os
import threading
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_ENABLED = os.getenv("NC_PROFILE_TRANSPORT", "").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}


@dataclass
class _Stat:
    count: int = 0
    total: float = 0.0
    max: float = 0.0

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


class _Profiler:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._durations: dict[str, dict[str, _Stat]] = defaultdict(dict)
        self._values: dict[str, dict[str, _Stat]] = defaultdict(dict)

    def record_duration(self, category: str, label: str, seconds: float) -> None:
        if not _ENABLED:
            return
        with self._lock:
            stat = self._durations[category].setdefault(label, _Stat())
            stat.count += 1
            stat.total += seconds
            if seconds > stat.max:
                stat.max = seconds

    def observe_value(self, category: str, label: str, value: float) -> None:
        if not _ENABLED:
            return
        with self._lock:
            stat = self._values[category].setdefault(label, _Stat())
            stat.count += 1
            stat.total += value
            if value > stat.max:
                stat.max = value

    def log_summary(self, *, summary_logger: logging.Logger, prefix: str) -> None:
        if not _ENABLED:
            return
        with self._lock:
            duration_snapshot = {
                category: dict(stats) for category, stats in self._durations.items()
            }
            value_snapshot = {
                category: dict(stats) for category, stats in self._values.items()
            }

        if not duration_snapshot and not value_snapshot:
            return

        summary_logger.info("Transport profile summary (%s):", prefix)
        for category in sorted(duration_snapshot):
            for label, stat in sorted(
                duration_snapshot[category].items(),
                key=lambda item: (-item[1].max, -item[1].total, item[0]),
            ):
                summary_logger.info(
                    "  %s %-40s count=%d avg=%.6fs max=%.6fs total=%.3fs",
                    category,
                    label,
                    stat.count,
                    stat.avg,
                    stat.max,
                    stat.total,
                )
        for category in sorted(value_snapshot):
            for label, stat in sorted(
                value_snapshot[category].items(),
                key=lambda item: (-item[1].max, -item[1].avg, item[0]),
            ):
                summary_logger.info(
                    "  %s %-40s samples=%d avg=%.2f max=%.2f",
                    category,
                    label,
                    stat.count,
                    stat.avg,
                    stat.max,
                )


_PROFILER = _Profiler()


def profiling_enabled() -> bool:
    """Return True when transport profiling is enabled."""
    return _ENABLED


def record_duration(category: str, label: str, seconds: float) -> None:
    """Record one duration sample."""
    _PROFILER.record_duration(category, label, seconds)


def observe_value(category: str, label: str, value: float) -> None:
    """Record one numeric value sample."""
    _PROFILER.observe_value(category, label, value)


def log_summary(*, prefix: str, summary_logger: logging.Logger | None = None) -> None:
    """Emit the current profiling snapshot to logs."""
    _PROFILER.log_summary(
        summary_logger=summary_logger or logger,
        prefix=prefix,
    )


def _dump_at_exit() -> None:
    if not _ENABLED:
        return
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        return
    log_summary(prefix=f"atexit-pid-{os.getpid()}", summary_logger=root_logger)


atexit.register(_dump_at_exit)
