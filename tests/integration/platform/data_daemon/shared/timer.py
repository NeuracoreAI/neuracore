"""Timer primitives for daemon integration tests.

Provides :class:`TimerSpec`, :class:`Timer`, the per-label registry, and the
shared timing constants used across the suite.  Sits at the bottom of the
import graph so any other shared module may depend on it without cycles.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------

MAX_TIME_TO_START_S = 20
"""Maximum seconds allowed for a daemon-startup or API-handshake operation."""

MAX_TIME_TO_LOG_S = 0.5
"""Maximum seconds allowed for a single data-logging call."""

LEAST_TIME_TO_STOP_S = 10
"""Minimum seconds expected for a recording stop."""

HIGH_TIME_TO_DATASET_READY_S = 500
"""Upper bound on waiting for an online dataset to become ready, in seconds."""


# ---------------------------------------------------------------------------
# Per-suite label sets
# ---------------------------------------------------------------------------

PERF_LOG_LABELS: tuple[str, ...] = (
    "nc.log_joint_positions",
    "nc.log_joint_velocities",
    "nc.log_joint_torques",
    "nc.log_custom_1d",
    "nc.log_rgb",
    "nc.start_recording",
    "nc.stop_recording",
)
"""Per-call logging labels that performance suites assert against.

Performance test modules register these labels with ``assert_limit=True`` at
import time so a single noisy log call fails the test rather than only
inflating reported stats.
"""


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimerSpec:
    """Per-label configuration for :class:`Timer`.

    Attributes:
        max_time: Upper time limit in seconds.
        assert_limit: When ``True``, raise :class:`AssertionError` if the
            block exceeds ``max_time``.  Defaults to ``False``: most labels
            are observational and only the performance suites opt in to
            assertions by registering their own specs.
        log: When ``True``, always log the elapsed time at INFO (or WARNING
            when over ``max_time``).  When ``False``, log only when exceeded.
    """

    max_time: float
    assert_limit: bool = False
    log: bool = True


class Timer:
    """Context manager that measures wall-clock elapsed time for a block.

    Resolves per-label configuration from :attr:`REGISTRY` so call sites can
    just write ``Timer("nc.login")`` without repeating limits and flags.
    Stats are accumulated in :attr:`_stats` for end-of-run reporting.

    To customise behaviour for a specific suite (e.g. performance tests that
    must assert per-call timing budgets) register entries in :attr:`REGISTRY`
    at module import time, or pass ``max_time`` / ``assert_limit`` / ``log``
    as keyword overrides.

    Stochastic-timestamp tolerance is centralised: pass ``use_stochastic=True``
    together with a ``deadline`` and the timer asserts the wall-clock start
    landed within :attr:`STOCHASTIC_TIMING_TOLERANCE_S` of the deadline.
    Override the class attribute to tune tolerance suite-wide.
    """

    _stats: dict[str, dict[str, float]] = {}

    STOCHASTIC_TIMING_TOLERANCE_S: float = 0.05
    """Tolerance applied to deadline checks when ``use_stochastic=True``.

    Stochastic timestamp mode jitters log timestamps by a bounded amount, so
    the wall-clock arrival of a logged frame can drift from its scheduled
    deadline.  This value caps the allowed drift and is used by every Timer
    that opts in via ``use_stochastic=True``.  Override class-wide (or per
    test module) to relax or tighten the tolerance.
    """

    DEFAULT_SPEC: TimerSpec = TimerSpec(max_time=MAX_TIME_TO_LOG_S)
    """Spec used when a label is not present in :attr:`REGISTRY`."""

    REGISTRY: dict[str, TimerSpec] = {
        # API / lifecycle (observational; perf tests do not assert these)
        "nc.login": TimerSpec(max_time=MAX_TIME_TO_START_S),
        "nc.create_dataset": TimerSpec(max_time=MAX_TIME_TO_START_S),
        "nc.get_dataset": TimerSpec(max_time=MAX_TIME_TO_START_S),
        "nc.connect_robot": TimerSpec(max_time=MAX_TIME_TO_START_S),
        "nc.start_recording": TimerSpec(max_time=MAX_TIME_TO_START_S),
        "nc.stop_recording": TimerSpec(max_time=MAX_TIME_TO_START_S),
        "nc.cancel_recording": TimerSpec(max_time=MAX_TIME_TO_START_S),
        # Per-call logging (perf suites override with assert_limit=True)
        "nc.log_joint_positions": TimerSpec(max_time=MAX_TIME_TO_LOG_S),
        "nc.log_joint_velocities": TimerSpec(max_time=MAX_TIME_TO_LOG_S),
        "nc.log_joint_torques": TimerSpec(max_time=MAX_TIME_TO_LOG_S),
        "nc.log_custom_1d": TimerSpec(max_time=MAX_TIME_TO_LOG_S),
        "nc.log_rgb": TimerSpec(max_time=MAX_TIME_TO_LOG_S),
        # Daemon control
        "stop_daemon_escalated": TimerSpec(max_time=5.0),
        "stop_daemon[cli]": TimerSpec(max_time=15.0),
        "stop_daemon[sigterm]": TimerSpec(max_time=15.0),
        "stop_daemon[sigint]": TimerSpec(max_time=15.0),
        "stop_daemon[sigkill]": TimerSpec(max_time=15.0),
        # DB readiness — tests pass an explicit max_time= override
        "daemon.offline_db_ready": TimerSpec(max_time=HIGH_TIME_TO_DATASET_READY_S),
    }
    """Registry of per-label timing specs.

    Suites can mutate this at import time (e.g.
    ``Timer.REGISTRY["nc.log_rgb"] = TimerSpec(0.5, assert_limit=True)``)
    to opt their labels into stricter behaviour.
    """

    def __init__(
        self,
        label: str,
        *,
        max_time: float | None = None,
        assert_limit: bool | None = None,
        log: bool | None = None,
        deadline: float | None = None,
        use_stochastic: bool = False,
    ) -> None:
        spec = self.REGISTRY.get(label, self.DEFAULT_SPEC)
        self.label = label
        self.max_time = max_time if max_time is not None else spec.max_time
        self.assert_limit = (
            assert_limit if assert_limit is not None else spec.assert_limit
        )
        self.log = log if log is not None else spec.log
        self.deadline = deadline
        self.timing_tolerance = (
            self.STOCHASTIC_TIMING_TOLERANCE_S if use_stochastic else None
        )

    def __enter__(self) -> Timer:
        self.wall_start = time.time()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> bool | None:
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        had_exception = len(args) > 0 and args[0] is not None

        stats = self._stats.setdefault(
            self.label, {"count": 0.0, "total": 0.0, "max": 0.0}
        )
        stats["count"] += 1
        stats["total"] += self.interval
        stats["max"] = max(stats["max"], self.interval)

        over_limit = self.interval >= self.max_time
        if self.log or over_limit:
            level = logging.WARNING if over_limit else logging.INFO
            logger.log(
                level,
                "Timer %-32s %.3fs (limit=%.3fs)",
                self.label,
                self.interval,
                self.max_time,
            )

        if had_exception:
            return False

        if self.assert_limit:
            if self.deadline is not None and self.timing_tolerance is not None:
                lateness = self.wall_start - self.deadline
                assert abs(lateness) <= self.timing_tolerance, (
                    f"{self.label} logged at wrong moment: "
                    f"lateness={lateness:+.3f}s, "
                    f"tolerance=±{self.timing_tolerance:.3f}s"
                )
            assert self.interval < self.max_time, (
                f"{self.label} took too long: "
                f"{self.interval:.3f}s >= {self.max_time:.3f}s"
            )
        return None

    @classmethod
    def merge_stats(cls, stats: dict[str, dict[str, float]]) -> None:
        """Merge external timer stats (e.g. from a worker process) into the accumulator."""  # noqa: E501
        for label, incoming in stats.items():
            existing = cls._stats.setdefault(
                label, {"count": 0.0, "total": 0.0, "max": 0.0}
            )
            existing["count"] += incoming["count"]
            existing["total"] += incoming["total"]
            existing["max"] = max(existing["max"], incoming["max"])

    @classmethod
    def assert_labels(cls, labels: tuple[str, ...]) -> None:
        """Opt the given labels into ``assert_limit=True`` using their existing spec.

        Used by performance test modules at import time to enforce per-call
        budgets on logging labels that are otherwise observational.
        """
        for label in labels:
            spec = cls.REGISTRY.get(label, cls.DEFAULT_SPEC)
            cls.REGISTRY[label] = TimerSpec(
                max_time=spec.max_time, assert_limit=True, log=spec.log
            )
