"""``Timer`` stat accumulation under concurrent producers."""

from __future__ import annotations

import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager

from tests.integration.platform.data_daemon.shared.process_control import Timer

THREAD_COUNT = 16
ITERATIONS_PER_THREAD = 200
LABEL = "test.concurrent_timer"

# Repeated so a scheduling fluke cannot hide the race.
ROUNDS = 20

# CPython's thread-switch interval; the default is too slow to catch the race.
FAST_SWITCH_INTERVAL_S = 1e-6


@contextmanager
def aggressive_thread_switching() -> Generator[None]:
    """Force CPython to switch threads as often as it is willing to."""
    previous = sys.getswitchinterval()
    sys.setswitchinterval(FAST_SWITCH_INTERVAL_S)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


def _time_blocks_concurrently() -> tuple[dict[str, float], float, int]:
    """Run one round of concurrent timing.

    Returns:
        ``(stats, slowest measured interval, blocks run)``.
    """
    Timer._stats.pop(LABEL, None)
    observed: list[list[float]] = []
    observed_lock = threading.Lock()

    def hammer() -> None:
        intervals: list[float] = []
        for _ in range(ITERATIONS_PER_THREAD):
            with Timer(0.5, label=LABEL, assert_deadline=False) as timer:
                pass
            intervals.append(timer.interval)
        with observed_lock:
            observed.append(intervals)

    threads = [
        threading.Thread(target=hammer, name=f"timer-hammer-{index}")
        for index in range(THREAD_COUNT)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return (
        dict(Timer._stats[LABEL]),
        max(interval for intervals in observed for interval in intervals),
        sum(len(intervals) for intervals in observed),
    )


def test_concurrent_timers_keep_the_worst_interval() -> None:
    """The reported ``max`` is the slowest block, whoever else was timing one."""
    try:
        with aggressive_thread_switching():
            for round_index in range(ROUNDS):
                stats, slowest, total = _time_blocks_concurrently()

                assert stats["count"] == float(total), (
                    f"round {round_index}: lost {total - stats['count']:.0f} of "
                    f"{total} timer counts across {THREAD_COUNT} threads"
                )
                assert stats["max"] == slowest, (
                    f"round {round_index}: reported max {stats['max']:.6f}s is not "
                    f"the slowest block {slowest:.6f}s — a concurrent update "
                    "overwrote it"
                )
    finally:
        Timer._stats.pop(LABEL, None)
