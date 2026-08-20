"""Timer stats published by a pool worker that never returns a result."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from tests.integration.platform.data_daemon.shared import process_control
from tests.integration.platform.data_daemon.shared.process_control import (
    Timer,
    WorkerTimerStats,
    merge_unreturned_worker_stats,
    publish_timer_stats,
    worker_timer_stats_sink,
)

CONTEXT_INDEX = 7


@contextmanager
def publishing_worker(sink: WorkerTimerStats) -> Generator[None]:
    """Make ``publish_timer_stats`` behave as it does inside a pool worker."""
    previous = process_control._worker_stats_sink
    process_control._worker_stats_sink = sink
    try:
        yield
    finally:
        process_control._worker_stats_sink = previous


@contextmanager
def timed_block(label: str) -> Generator[None]:
    """Accumulate one block under *label*, and leave no stats behind."""
    try:
        with Timer(1.0, label=label, assert_deadline=False):
            pass
        yield
    finally:
        Timer._stats.pop(label, None)


def _forget_locally(label: str) -> None:
    """Drop the publisher's own copy, standing in for the process boundary.

    A real worker's ``Timer._stats`` is not the parent's, so what the parent
    ends up with is only ever what it merged.
    """
    Timer._stats.pop(label)


def test_stats_of_a_worker_that_never_returned_reach_the_parent() -> None:
    """The sink carries what the return value would have."""
    label = "test.unreturned"
    sink: WorkerTimerStats = {}
    with timed_block(label):
        with publishing_worker(sink):
            publish_timer_stats(CONTEXT_INDEX)
        _forget_locally(label)

        merge_unreturned_worker_stats(sink, returned_keys=set())

        assert Timer._stats[label]["count"] == 1.0


def test_only_the_latest_publication_of_a_worker_is_merged() -> None:
    """Publications are cumulative, so summing them would double-count."""
    label = "test.latest_publication"
    sink: WorkerTimerStats = {}
    with timed_block(label):
        with publishing_worker(sink):
            publish_timer_stats(CONTEXT_INDEX)
            with Timer(1.0, label=label, assert_deadline=False):
                pass
            publish_timer_stats(CONTEXT_INDEX)
        _forget_locally(label)

        merge_unreturned_worker_stats(sink, returned_keys=set())

        assert Timer._stats[label]["count"] == 2.0


def test_a_worker_that_returned_is_not_merged_twice() -> None:
    """Its result already carried these stats to the parent."""
    label = "test.returned"
    sink: WorkerTimerStats = {}
    with timed_block(label):
        with publishing_worker(sink):
            publish_timer_stats(CONTEXT_INDEX)
        _forget_locally(label)

        merge_unreturned_worker_stats(sink, returned_keys={CONTEXT_INDEX})

        assert label not in Timer._stats


def test_the_managed_sink_carries_a_publication_across_processes() -> None:
    """The real sink is a manager proxy, not the plain dict above."""
    label = "test.managed_sink"
    with timed_block(label):
        with worker_timer_stats_sink() as sink:
            with publishing_worker(sink):
                publish_timer_stats(CONTEXT_INDEX)
            _forget_locally(label)

            merge_unreturned_worker_stats(sink, returned_keys=set())

        assert Timer._stats[label]["count"] == 1.0
