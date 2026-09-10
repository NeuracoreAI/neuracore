"""One spawned test child, from the handshake to a reap that refuses to leak.

Every kind of child a case runs shares the same lifetime: spawn, say when it is
ready, do its work, report once, exit. Only the work differs.
"""

from __future__ import annotations

import multiprocessing
import queue
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tests.integration.platform.data_daemon.shared.process_control import Timer
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    CHILD_PROCESS_JOIN_TIMEOUT_S,
    CHILD_PROCESS_READY_POLL_S,
    CHILD_PROCESS_REPORT_TIMEOUT_S,
    CHILD_PROCESS_TERMINATE_TIMEOUT_S,
    MAX_TIME_TO_START_S,
)


@dataclass(frozen=True, slots=True)
class ChildOutcome:
    """What a child left behind once it was reaped.

    Attributes:
        payload: The child's own report, or ``None`` when it failed or never
            reported — so a caller reads it only after ``failure`` is empty.
        failure: Why the child cannot be trusted, or ``""`` when it exited
            cleanly having reported.
    """

    payload: dict[str, Any] | None
    failure: str


class ChildProcess:
    """A child process the test drives, and the channels it answers on.

    The readiness handshake and the final report are the same whatever the
    child does, so they live here. A child kind supplies its target and any
    channel of its own, built on this child's spawn context.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._context = multiprocessing.get_context("spawn")
        self.ready_event: Any = self._context.Event()
        self.result_queue: Any = self._context.Queue()
        self._process: Any = None

    def queue(self) -> Any:
        """A channel on this child's spawn context."""
        return self._context.Queue()

    def event(self) -> Any:
        """A flag on this child's spawn context."""
        return self._context.Event()

    def start(self, target: Callable[..., None], args: tuple[Any, ...]) -> None:
        """Spawn the child running *target*."""
        self._process = self._context.Process(name=self.name, target=target, args=args)
        self._process.start()

    @property
    def is_alive(self) -> bool:
        """Whether the child is still running."""
        return self._process is not None and self._process.is_alive()

    def await_ready(self) -> bool:
        """Whether the child said it was ready within
        :data:`MAX_TIME_TO_START_S`, giving up early if it dies so its own
        traceback is reported rather than a timeout."""
        deadline = time.time() + MAX_TIME_TO_START_S
        while time.time() < deadline:
            if self.ready_event.wait(timeout=CHILD_PROCESS_READY_POLL_S):
                return True
            if not self.is_alive:
                return False
        return False

    def collect(self) -> ChildOutcome:
        """Take the child's report and reap it, merging its timings in.

        The queue is drained before the join — joining first can deadlock,
        since a process won't exit until its queued item clears the pipe.
        """
        try:
            report = self.result_queue.get(timeout=CHILD_PROCESS_REPORT_TIMEOUT_S)
        except queue.Empty:
            report = None
        if report is not None:
            Timer.merge_stats(report.get("timer_stats", {}))
        self._process.join(timeout=CHILD_PROCESS_JOIN_TIMEOUT_S)
        if self.is_alive:
            self._process.terminate()
            self._process.join(timeout=CHILD_PROCESS_TERMINATE_TIMEOUT_S)
            return ChildOutcome(None, f"child {self.name} did not exit, terminated")
        if report is None:
            return ChildOutcome(None, f"child {self.name} exited without reporting")
        if not report.get("ok"):
            return ChildOutcome(
                None, f"child {self.name} raised:\n{report.get('traceback')}"
            )
        if self._process.exitcode != 0:
            return ChildOutcome(
                None, f"child {self.name} exited with code {self._process.exitcode}"
            )
        return ChildOutcome(report, "")
