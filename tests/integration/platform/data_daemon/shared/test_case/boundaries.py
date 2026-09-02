"""Frame identity and recording-boundary classification.

The vocabulary every producer reports in, and the rule that decides which
frames a recording owns.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import NamedTuple

from tests.integration.platform.data_daemon.shared.test_case.constants import (
    CONDEMNED_PROVENANCE_MARGIN_S,
)


class EmittedFrame(NamedTuple):
    """One frame a producer logged, bracketed on the wall clock.

    A publish stamp is taken inside the ``log_*`` call, so a test only knows it
    lies somewhere in ``[emitted_at, completed_at]`` (see
    :func:`_classify_boundary_frames`).

    Attributes:
        timestamp: The value that reaches disk; the only field compared there.
        frame_index: Session-wide, never resets across recordings — recovers
            the painted frame code (see :func:`rgb_frame_code`).
        handle: Recording handle latched before the call, or ``None``. Reflects
            the local logging gate, not the daemon's actual window.
        deadline_breaches: ``nc.log_*`` calls this frame made that exceeded
            ``MAX_TIME_TO_LOG_S``. Only asserted on when this frame is ``owed``.
    """

    timestamp: float
    frame_index: int
    emitted_at: float
    completed_at: float
    handle: str | None
    deadline_breaches: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TraceClassification:
    """What one recording requires of one trace's frames.

    The shared verdict both classification rules report in (see
    :func:`_classify_boundary_frames`): ``owed`` must reach disk, ``condemned``
    must not, and ``unknowable`` frames count on neither side.

    Attributes:
        condemned: Reasons kept only near this recording's own control calls
            (see :func:`describe_condemnation`), read only if one reaches disk.
    """

    owed: list[EmittedFrame]
    unknowable: list[EmittedFrame] = field(default_factory=list)
    condemned: list[tuple[EmittedFrame, str]] = field(default_factory=list)

    @property
    def owed_timestamps(self) -> list[float]:
        """The capture timestamps the on-disk trace must hold, in order."""
        return [frame.timestamp for frame in self.owed]

    @property
    def unknowable_timestamps(self) -> set[float]:
        """Capture timestamps allowed to be present or absent."""
        return {frame.timestamp for frame in self.unknowable}

    @property
    def condemned_reasons(self) -> dict[float, str]:
        """Why each ruled-out capture timestamp was ruled out."""
        return {frame.timestamp: reason for frame, reason in self.condemned}


@dataclass(frozen=True, slots=True)
class ObservedFrameCodes:
    """Camera frame codes one recording claimed, per camera name.

    Reported separately because a lifetime producer's frame index is
    session-wide, so codes can't be reconstructed from the recording ordinal.
    """

    recording_index: int
    """Producer recording ordinal used to encode this recording's frames."""
    inside: dict[str, list[int]]
    unknowable: dict[str, set[int]]


@dataclass(frozen=True, slots=True)
class RecordingControlBounds:
    """Wall-clock brackets around the two control calls that bound a recording.

    Both bounds are stamped on the publish clock inside the control calls
    themselves, never from a capture timestamp.

    Attributes:
        handles: Every SDK handle this recording held — a set, not one value,
            because a recording holds a client-generated UUID before the
            backend swaps in its own id.
        stop_settled_at: When the window's upper bound is known to have
            passed. A local stop stamps that bound itself, so this is a
            bounded skew after ``stop_called_at``; a remote one is measured,
            a notification later.
    """

    handles: frozenset[str]
    start_called_at: float
    start_returned_at: float
    stop_called_at: float
    stop_settled_at: float


def _classification(
    owed: list[EmittedFrame],
    unknowable: list[EmittedFrame],
    condemned: list[EmittedFrame],
    bounds: RecordingControlBounds,
) -> TraceClassification:
    """Assemble a verdict, attributing and trimming the condemned frames."""
    near_start = bounds.start_called_at - CONDEMNED_PROVENANCE_MARGIN_S
    near_stop = bounds.stop_settled_at + CONDEMNED_PROVENANCE_MARGIN_S
    return TraceClassification(
        owed=owed,
        unknowable=unknowable,
        condemned=[
            (frame, describe_condemnation(frame, bounds))
            for frame in condemned
            if near_start <= frame.emitted_at <= near_stop
        ],
    )


def _classify_boundary_frames(
    frames: list[EmittedFrame],
    bounds: RecordingControlBounds,
) -> TraceClassification:
    """Split one trace's frames into those inside a recording and those unknowable.

    Neither boundary is directly observable, so each is bracketed between two
    clock readings: a frame is **inside** only if its whole ``log_*`` call ran
    between ``start_recording`` returning and ``stop_recording`` being called.
    It is **outside** if the call finished before ``start_recording`` was
    entered, or began after the window's upper bound had certainly passed.
    Everything else straddled a bracket and is **unknowable**: the daemon's
    answer is correct either way, so those frames count on neither side.

    Both rules read clocks only, so they hold for a process that never made
    the control calls itself as much as for the one that did.

    Returns:
        The recording's verdict on these frames, as whole frames rather than
        bare timestamps, since the cloud assertion also needs frame indexes.
    """
    owed: list[EmittedFrame] = []
    unknowable: list[EmittedFrame] = []
    condemned: list[EmittedFrame] = []
    for frame in frames:
        is_inside = (
            frame.emitted_at >= bounds.start_returned_at
            and frame.completed_at <= bounds.stop_called_at
        )
        is_outside = (
            frame.completed_at <= bounds.start_called_at
            or frame.emitted_at >= bounds.stop_settled_at
        )
        if is_inside:
            owed.append(frame)
        elif is_outside:
            condemned.append(frame)
        else:
            unknowable.append(frame)
    return _classification(owed, unknowable, condemned, bounds)


def describe_condemnation(frame: EmittedFrame, bounds: RecordingControlBounds) -> str:
    """Say why the classification ruled *frame* out of this recording."""
    if frame.completed_at <= bounds.start_called_at:
        return "log call finished before start_recording was entered"
    if frame.emitted_at >= bounds.stop_settled_at:
        timing = "log call began after the window's bound had passed"
    elif frame.emitted_at >= bounds.stop_called_at:
        timing = "log call began after stop_recording was entered"
    else:
        timing = "log call ran wholly inside the control calls"
    if frame.handle is None:
        gate = "gate held no recording"
    elif frame.handle in bounds.handles:
        gate = "gate held this recording"
    else:
        gate = "gate held another recording"
    # Interned: one string per category, not per condemned frame.
    return sys.intern(f"{timing}, {gate}")
