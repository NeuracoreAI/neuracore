"""Frame identity and recording-boundary classification.

The vocabulary every producer reports in, and the rule that decides which
frames a recording owns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple


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

    The single verdict every classification rule reports in, so every consumer
    reads one object rather than unpacking a tuple, and a rule that grows a
    third answer grows a field here instead of a return position.

    The lists partition the frames the rule was given: a frame appears in
    exactly one.

    Attributes:
        owed: Frames the recording provably owns. Every one must reach disk.
        unknowable: Frames logged while a boundary was passing. The daemon's
            answer is correct either way, so they are dropped from *both* sides
            of the comparison rather than tolerated on one.
    """

    owed: list[EmittedFrame]
    unknowable: list[EmittedFrame] = field(default_factory=list)

    @property
    def owed_timestamps(self) -> list[float]:
        """The capture timestamps the on-disk trace must hold, in order."""
        return [frame.timestamp for frame in self.owed]

    @property
    def unknowable_timestamps(self) -> set[float]:
        """Capture timestamps allowed to be present or absent."""
        return {frame.timestamp for frame in self.unknowable}


@dataclass(frozen=True, slots=True)
class ObservedFrameCodes:
    """Camera frame codes one recording claimed, per camera name.

    Reported separately because a lifetime producer's frame index is
    session-wide, so codes can't be reconstructed from the recording ordinal.
    """

    inside: dict[str, list[int]]
    unknowable: dict[str, set[int]]


@dataclass(frozen=True, slots=True)
class RecordingControlBounds:
    """Wall-clock brackets around the two control calls that bound a recording.

    The daemon's window is ``[started_at_ns, stopped_at_ns)``, and both bounds
    are stamped on the publish clock *inside* the producer-side
    ``start_recording`` / ``stop_recording`` calls — never from the capture
    timestamp the caller passes. Neither instant is visible from here, but each
    is known to fall within the call that carries it, and the control thread can
    stamp both edges of that call on the same clock the producer threads use.

    Attributes:
        handle: The SDK recording handle for this recording, as the producer
            threads see it while it is current.
        start_called_at: Wall clock immediately before ``nc.start_recording``.
        start_returned_at: Wall clock immediately after it returned. The
            window's lower bound is somewhere in between.
        stop_called_at: Wall clock immediately before ``nc.stop_recording``.
        stop_returned_at: Wall clock immediately after it returned. The window's
            upper bound is somewhere in between.
    """

    handle: str | None
    start_called_at: float
    start_returned_at: float
    stop_called_at: float
    stop_returned_at: float


def _classify_boundary_frames(
    frames: list[EmittedFrame],
    bounds: RecordingControlBounds,
) -> TraceClassification:
    """Split one trace's frames into those inside a recording and those unknowable.

    Neither boundary is directly observable, so each is bracketed between two
    clock readings: a frame is **inside** only if its whole ``log_*`` call ran
    between ``start_recording`` returning and ``stop_recording`` being called.
    It is **outside** if the call finished before ``start_recording`` was
    entered, or if the SDK's gate — a deliberate superset of the window — had
    already refused it after the stop; a gate *admission* proves nothing, since
    the gate opens first, so only a refusal is conclusive. Everything else
    straddled a bracket and is **unknowable**: the daemon's answer is correct
    either way, so those frames count on neither side.

    Returns:
        The recording's verdict on these frames, as whole frames rather than
        bare timestamps, since the cloud assertion also needs frame indexes.
    """
    inside: list[EmittedFrame] = []
    unknowable: list[EmittedFrame] = []
    for frame in frames:
        is_inside = (
            frame.emitted_at >= bounds.start_returned_at
            and frame.completed_at <= bounds.stop_called_at
        )
        is_outside = frame.completed_at <= bounds.start_called_at or (
            frame.emitted_at >= bounds.stop_called_at and frame.handle != bounds.handle
        )
        if is_inside:
            inside.append(frame)
        elif not is_outside:
            unknowable.append(frame)
    return TraceClassification(owed=inside, unknowable=unknowable)
