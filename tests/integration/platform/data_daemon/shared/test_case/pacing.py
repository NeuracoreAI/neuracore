"""When a stream may offer its next frame: one pacer per stream, per pacing.

The only place a ``producer_pacing`` value is interpreted. Producers ask for a
pacer and obey it, so which engine runs a stream decides nothing about how hard
that stream drives the SDK.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from tests.integration.platform.data_daemon.shared.process_control import (
    MAX_TIME_TO_LOG_S,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    BURST_VIDEO_FRAMES,
    PACING_BURST_VIDEO,
    PACING_SATURATE,
)


def _wait_until(deadline: float, stop_event: threading.Event) -> bool:
    """Wait until *deadline*, unless *stop_event* fires first.

    Returns:
        ``True`` when *stop_event* fired, meaning the caller's loop should stop.
    """
    remaining = deadline - time.time()
    if remaining <= 0:
        return False
    return stop_event.wait(remaining)


class StreamPacer(ABC):
    """Decides when one stream's next frame may be logged."""

    @abstractmethod
    def release_offset_s(self, frame_index: int) -> float:
        """Seconds after the stream started that *frame_index* may be logged.

        Also the single-threaded producer's ordering key, which is why an
        unpaced stream answers it too.
        """

    @abstractmethod
    def wait_until_due(
        self, started_at: float, frame_index: int, stop_event: threading.Event
    ) -> bool:
        """Block until *frame_index* is due.

        Returns:
            ``True`` when *stop_event* fired, meaning the caller's loop should
            stop.
        """

    def observe(self, emit_s: float) -> None:
        """Report how long the last frame's ``nc.log_*`` calls took.

        Ignored by a pacer that holds to a schedule of its own.
        """


@dataclass(slots=True)
class Unpaced(StreamPacer):
    """Offers each frame as soon as the last one returned, yielding only while
    the SDK is stalled, so the daemon's own back-pressure is what holds the
    stream back.

    A stall is a ``nc.log_*`` call that took :data:`MAX_TIME_TO_LOG_S` or more;
    the gap that follows is as long as that stall, and clears on the first call
    that returns promptly. A stream that answers a slowing daemon by queueing
    harder only moves the backlog into the spool.
    """

    fps: int
    _stalled_for_s: float = 0.0

    def release_offset_s(self, frame_index: int) -> float:
        """The nominal schedule position, used for ordering and never waited on."""
        return frame_index / self.fps

    def observe(self, emit_s: float) -> None:
        """Hold the last stall, or clear it once a call returns promptly."""
        self._stalled_for_s = emit_s if emit_s >= MAX_TIME_TO_LOG_S else 0.0

    def wait_until_due(
        self, started_at: float, frame_index: int, stop_event: threading.Event
    ) -> bool:
        """Wait out a gap as long as the last stall; waits at all only after one."""
        return self._stalled_for_s > 0.0 and stop_event.wait(self._stalled_for_s)


@dataclass(frozen=True, slots=True)
class BurstPacer(StreamPacer):
    """Releases *burst_frames* frames at a time, averaging *fps* over the run.

    ``burst_frames=1`` is deadline pacing: one frame per interval, closed-loop
    against the stream's own start.
    """

    fps: int
    burst_frames: int

    def release_offset_s(self, frame_index: int) -> float:
        """When this frame's burst comes due; every frame in it shares one offset."""
        return (frame_index // self.burst_frames) * self.burst_frames / self.fps

    def wait_until_due(
        self, started_at: float, frame_index: int, stop_event: threading.Event
    ) -> bool:
        """Wait out the rest of this frame's burst interval."""
        return _wait_until(started_at + self.release_offset_s(frame_index), stop_event)


def pacer_for(pacing: str, *, fps: int, is_video: bool) -> StreamPacer:
    """The pacer *pacing* gives a stream running at *fps*."""
    if pacing == PACING_SATURATE:
        return Unpaced(fps)
    if pacing == PACING_BURST_VIDEO and is_video:
        return BurstPacer(fps, BURST_VIDEO_FRAMES)
    return BurstPacer(fps, 1)
