"""The producer engines, behind one session interface that hides which ran."""

from __future__ import annotations

import random
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from tests.integration.platform.data_daemon.shared.test_case.boundaries import (
    EmittedFrame,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    camera_names,
    depth_camera_names,
    joint_names_for_count,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    PER_THREAD_LOGGING_TAIL_S,
    PRODUCER_OLD_PER_THREAD,
    PRODUCER_PER_THREAD,
    random_phase_jitter_window,
)
from tests.integration.platform.data_daemon.shared.test_case.context_spec import (
    JOINT_STREAM,
    VIDEO_STREAM,
    ContextSpec,
    recording_timestamps,
    stream_phase_seed,
)
from tests.integration.platform.data_daemon.shared.test_case.pacing import (
    StreamPacer,
    pacer_for,
)
from tests.integration.platform.data_daemon.shared.test_case.streams import (
    StreamEmitter,
    StreamPlan,
    _run_stream_threads,
    stream_plans_for_case,
)


@dataclass(frozen=True, slots=True)
class ProducerRequest:
    """Everything a producer needs to run, whatever that producer's lifetime.

    One shape for both engines, so a case can swap producers without its
    caller learning which one it got. Two fields carry the whole difference:
    *duration_sec* bounds each stream at ``fps * duration_sec`` frames, or runs
    it until *stop_event* fires; *recording_index* and *seed_ordinal* name the
    recording, and are ``0`` for a producer that outlives every recording.

    Attributes:
        robot: Connected robot handle, read once per frame for the SDK's local
            logging gate (see :attr:`EmittedFrame.handle`).
        recording_index: Recording ordinal that namespaces painted frame codes.
        seed_ordinal: Recording ordinal that seeds the random-phase offsets.
        plans: Streams to run, in the order the single-threaded producer breaks
            ties between them.
        timestamp_start_s: Capture timestamp the first frame of every stream
            carries.
        pacing: When each stream may offer its next frame, resolved per stream
            by :func:`pacer_for`. Independent of the producer's lifetime.
    """

    robot: object
    robot_name: str
    context_index: int
    recording_index: int
    seed_ordinal: int
    plans: tuple[StreamPlan, ...]
    image_width: int | None
    image_height: int | None
    video_detail: str
    timestamp_start_s: float
    random_phase: bool
    duration_sec: int | None
    pacing: str
    stop_event: threading.Event
    assert_deadline: bool = False

    def frame_budget(self, plan: StreamPlan) -> int | None:
        """Frames *plan* emits, or ``None`` when it runs until stopped."""
        if self.duration_sec is None:
            return None
        return plan.fps * self.duration_sec

    def pacer(self, plan: StreamPlan) -> StreamPacer:
        """The pacer this request's pacing gives *plan*."""
        return pacer_for(self.pacing, fps=plan.fps, is_video=plan.is_video)


def stream_frame_schedule(
    plan: StreamPlan, request: ProducerRequest
) -> Iterator[tuple[int, float]]:
    """Yield ``(frame_index, timestamp)`` for one stream, in emission order.

    Streams of the same kind share a seed, keeping them aligned with each other
    however many threads run them.
    """
    stream = VIDEO_STREAM if plan.is_video else JOINT_STREAM
    rng = random.Random(
        stream_phase_seed(request.context_index, request.seed_ordinal, stream)
    )
    window = random_phase_jitter_window(plan.fps)
    budget = request.frame_budget(plan)
    frame_index = 0
    while budget is None or frame_index < budget:
        offset = rng.uniform(-window, window) if request.random_phase else 0.0
        yield frame_index, request.timestamp_start_s + frame_index / plan.fps + offset
        frame_index += 1


class FrameReport:
    """Every frame the producer logged, keyed by the traces it feeds.

    Filled from one thread per stream; read only after those threads join.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_trace: dict[str, list[EmittedFrame]] = {}

    def record(self, emitter: StreamEmitter, frame: EmittedFrame) -> None:
        """Attribute *frame* to every trace *emitter*'s stream writes."""
        with self._lock:
            for trace_key in emitter.trace_keys:
                self._by_trace.setdefault(trace_key, []).append(frame)

    def merge(self, other: dict[str, list[EmittedFrame]]) -> None:
        """Fold another run's frames in, keeping each trace in emission order."""
        with self._lock:
            for trace_key, frames in other.items():
                self._by_trace.setdefault(trace_key, []).extend(frames)

    def as_dict(self) -> dict[str, list[EmittedFrame]]:
        """Return the frames per trace, in emission order."""
        with self._lock:
            return dict(self._by_trace)


def _build_emitters(request: ProducerRequest) -> list[StreamEmitter]:
    """One emitter per stream, in the request's own plan order."""
    return [
        StreamEmitter(
            plan=plan,
            robot_name=request.robot_name,
            context_index=request.context_index,
            recording_index=request.recording_index,
            assert_deadline=request.assert_deadline,
        )
        for plan in request.plans
    ]


def _emit_and_record(
    emitter: StreamEmitter,
    frame_index: int,
    timestamp: float,
    request: ProducerRequest,
    report: FrameReport,
) -> None:
    """Log one frame, recording everything the test can observe about the call."""
    # Read before the call, to distinguish a refused frame from an admitted one.
    handle = request.robot.get_current_recording_id()
    emitted_at = time.time()
    deadline_breaches = emitter.emit(frame_index, timestamp)
    report.record(
        emitter,
        EmittedFrame(
            timestamp=timestamp,
            frame_index=frame_index,
            emitted_at=emitted_at,
            completed_at=time.time(),
            handle=handle,
            deadline_breaches=deadline_breaches,
        ),
    )


def run_synchronous_logging(request: ProducerRequest) -> dict[str, list[EmittedFrame]]:
    """Log every stream from one thread, whichever stream is due next.

    Scoped to one recording: returns before that recording stops. One thread
    cannot hold two deadlines at once, so a stream held up by another's frame
    logs late rather than being dropped.
    """
    emitters = _build_emitters(request)
    for emitter in emitters:
        emitter.prepare(request.image_width, request.image_height, request.video_detail)

    report = FrameReport()
    pacers = {emitter: request.pacer(emitter.plan) for emitter in emitters}
    schedules = {
        emitter: stream_frame_schedule(emitter.plan, request) for emitter in emitters
    }
    pending = {
        emitter: frame
        for emitter, schedule in schedules.items()
        if (frame := next(schedule, None)) is not None
    }

    started_at = time.time()
    while pending and not request.stop_event.is_set():
        due = min(
            pending,
            key=lambda emitter: pacers[emitter].release_offset_s(pending[emitter][0]),
        )
        frame_index, timestamp = pending[due]
        if pacers[due].wait_until_due(started_at, frame_index, request.stop_event):
            break
        _emit_and_record(due, frame_index, timestamp, request, report)
        next_frame = next(schedules[due], None)
        if next_frame is None:
            del pending[due]
        else:
            pending[due] = next_frame

    return report.as_dict()


def run_threaded_logging(request: ProducerRequest) -> dict[str, list[EmittedFrame]]:
    """Run one thread per stream, each on its own pacer.

    How long the threads live is the request's business, not this function's:
    a bounded *duration_sec* stops each stream at its frame budget, ``None``
    runs it until *stop_event* fires. Every frame is reported, including those
    logged while no recording was current, so the daemon's rejections are
    visible too (see :func:`_classify_boundary_frames`).

    Returns:
        Mapping of trace key -> every frame logged for it, in order.
    """
    emitters = _build_emitters(request)
    report = FrameReport()
    barrier = threading.Barrier(len(emitters))

    def worker(emitter: StreamEmitter) -> None:
        barrier.wait()
        emitter.prepare(request.image_width, request.image_height, request.video_detail)
        pacer = request.pacer(emitter.plan)
        started_at = time.time()
        for frame_index, timestamp in stream_frame_schedule(emitter.plan, request):
            if request.stop_event.is_set():
                break
            if pacer.wait_until_due(started_at, frame_index, request.stop_event):
                break
            _emit_and_record(emitter, frame_index, timestamp, request, report)

    _run_stream_threads(emitters, worker, error_label="Threaded")

    return report.as_dict()


class ProducerSession(ABC):
    """Runs a case's producer, whatever its lifetime, behind one interface.

    Lifetime is the only thing that differs at the call site: a producer scoped
    to a recording runs *inside* the window, while one that outlives every
    recording has to be started before the first and stopped after the last.
    Both are driven the same way here, so a caller states the recording
    protocol once and the case's ``producer_channels`` decides nothing about
    the shape of that code::

        session = make_producer_session(spec, robot=robot, marker_name=...)
        session.start()
        try:
            for ordinal in range(recordings):
                nc.start_recording(...)
                session.run_recording(ordinal)
                nc.stop_recording(...)
        finally:
            session.finish()
        report = session.report()

    The report is uniform too: every producer reports every frame it logged
    with the wall-clock brackets around the call, so one classification decides
    which recording owns which frame (see :func:`_classify_boundary_frames`).
    """

    def __init__(
        self, spec: ContextSpec, robot: object, plans: list[StreamPlan]
    ) -> None:
        self.spec = spec
        self.robot = robot
        self.plans = tuple(plans)
        self.stop_event = threading.Event()

    @property
    def marker_names(self) -> list[str]:
        """The ``CUSTOM_1D`` marker series this producer's streams write."""
        return [plan.marker_name for plan in self.plans if plan.marker_name is not None]

    def _request(
        self, *, recording_ordinal: int, duration_sec: int | None
    ) -> ProducerRequest:
        """Build the request for one run of this session's engine."""
        case = self.spec.case
        return ProducerRequest(
            robot=self.robot,
            robot_name=self.spec.robot_name,
            context_index=self.spec.context_index,
            recording_index=recording_ordinal,
            seed_ordinal=recording_ordinal,
            plans=self.plans,
            image_width=case.image_width,
            image_height=case.image_height,
            video_detail=case.video_detail,
            timestamp_start_s=(
                self.spec.timestamp_start_s + recording_ordinal * case.duration_sec
            ),
            random_phase=case.random_phase,
            duration_sec=duration_sec,
            pacing=case.producer_pacing,
            stop_event=self.stop_event,
            assert_deadline=self.spec.assert_deadline,
        )

    @abstractmethod
    def frame_code_recording_index(self, recording_ordinal: int) -> int:
        """Recording ordinal *recording_ordinal*'s frames were painted under."""

    @abstractmethod
    def start(self) -> None:
        """Begin producing, if this producer starts before any recording."""

    @abstractmethod
    def run_recording(self, recording_ordinal: int) -> None:
        """Produce this recording's frames, and return once it may be stopped."""

    @abstractmethod
    def finish(self) -> None:
        """Stop producing and surface anything a producer thread raised."""

    @abstractmethod
    def report(self) -> dict[str, list[EmittedFrame]]:
        """Return every frame logged so far, keyed by trace."""


class BoundedProducerSession(ProducerSession):
    """A producer that lives and dies inside a single recording: returns only
    once every thread has joined, so no frame is in flight at a boundary."""

    def __init__(
        self,
        spec: ContextSpec,
        robot: object,
        plans: list[StreamPlan],
        engine: Callable[[ProducerRequest], dict[str, list[EmittedFrame]]],
    ) -> None:
        super().__init__(spec, robot, plans)
        self._engine = engine
        self._report = FrameReport()
        self._runs: list[tuple[int, dict[str, list[EmittedFrame]]]] = []

    def frame_code_recording_index(self, recording_ordinal: int) -> int:
        """Its own ordinal: this producer's frame index restarts every recording."""
        return recording_ordinal

    def start(self) -> None:
        """Nothing to start: this producer only runs inside a recording."""

    def run_recording(self, recording_ordinal: int) -> None:
        """Log this recording's whole frame budget, then join every thread."""
        frames = self._engine(
            self._request(
                recording_ordinal=recording_ordinal,
                duration_sec=self.spec.case.duration_sec,
            )
        )
        self._runs.append((recording_ordinal, frames))
        self._report.merge(frames)

    def finish(self) -> None:
        """Check each run against the schedule it was supposed to emit.

        Checked here rather than in :meth:`run_recording`, to keep the
        comparison out of the recording window.
        """
        for recording_ordinal, frames in self._runs:
            joint_timestamps, video_timestamps = recording_timestamps(
                self.spec, recording_ordinal
            )
            for plan in self.plans:
                expected = video_timestamps if plan.is_video else joint_timestamps
                for trace_key in plan.trace_keys:
                    logged = [frame.timestamp for frame in frames.get(trace_key, ())]
                    assert logged == expected, (
                        f"{self.spec.case.producer_channels} producer logged "
                        f"{len(logged)} frames for {trace_key} in recording "
                        f"{recording_ordinal}, expected {len(expected)}"
                    )

    def report(self) -> dict[str, list[EmittedFrame]]:
        """Return every frame logged, across all this context's recordings."""
        return self._report.as_dict()


class LifetimeProducerSession(ProducerSession):
    """A producer that outlives every recording it logs across: threads start
    before the first ``start_recording`` and stop after the last, so it is
    mid-loop at every boundary."""

    def __init__(
        self, spec: ContextSpec, robot: object, plans: list[StreamPlan]
    ) -> None:
        super().__init__(spec, robot, plans)
        self._thread: threading.Thread | None = None
        self._frames: dict[str, list[EmittedFrame]] = {}
        self._error: BaseException | None = None

    def frame_code_recording_index(self, recording_ordinal: int) -> int:
        """Always ``0``: the frame index is session-wide and never resets, so
        codes stay unique on their own (see :func:`run_threaded_logging`)."""
        return 0

    def start(self) -> None:
        """Start logging now, before any recording exists."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._frames = run_threaded_logging(
                self._request(recording_ordinal=0, duration_sec=None)
            )
        except BaseException as exc:  # noqa: BLE001
            self._error = exc

    def run_recording(self, recording_ordinal: int) -> None:
        """Hold the window open: the producer is already running its own schedule."""
        time.sleep(self.spec.case.duration_sec)

    def finish(self) -> None:
        """Log past the last recording, then stop the threads and join them."""
        if self._thread is None:
            return
        try:
            # Logs past the stop so the daemon is shown rejecting post-stop frames.
            time.sleep(PER_THREAD_LOGGING_TAIL_S)
        finally:
            self.stop_event.set()
            self._thread.join()
            self._thread = None
        if self._error is not None:
            raise RuntimeError(f"Lifetime producer failed: {self._error}") from (
                self._error
            )

    def report(self) -> dict[str, list[EmittedFrame]]:
        """Return every frame logged over the whole context lifetime."""
        return self._frames


def make_producer_session(
    spec: ContextSpec, *, robot: object, marker_name: str
) -> ProducerSession:
    """Build the producer session a case asked for: the one place
    ``producer_channels`` decides which engine runs. *marker_name* is used only
    by the synchronous producer; per-thread producers ignore it.
    """
    case = spec.case
    plans = stream_plans_for_case(
        producer_channels=case.producer_channels,
        joint_names=joint_names_for_count(case.joint_count),
        camera_name_list=camera_names(case.video_count),
        depth_camera_name_list=depth_camera_names(case.depth_count),
        depth_mode=case.depth_mode,
        joint_fps=case.joint_fps,
        video_fps=case.video_fps,
        marker_name=marker_name,
    )
    if case.producer_channels == PRODUCER_PER_THREAD:
        return LifetimeProducerSession(spec, robot, plans)
    engine = (
        run_threaded_logging
        if case.producer_channels == PRODUCER_OLD_PER_THREAD
        else run_synchronous_logging
    )
    return BoundedProducerSession(spec, robot, plans, engine)
