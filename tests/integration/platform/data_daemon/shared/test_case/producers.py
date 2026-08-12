"""The producer engines, behind one session interface that hides which ran."""

from __future__ import annotations

import multiprocessing
import queue
import random
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

import neuracore as nc
from tests.integration.platform.data_daemon.shared.auth import ensure_login
from tests.integration.platform.data_daemon.shared.process_control import Timer
from tests.integration.platform.data_daemon.shared.test_case.boundaries import (
    EmittedFrame,
    RecordingControlBounds,
    TraceClassification,
    _classify_boundary_frames,
    classify_split_producer_frames,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    camera_names,
    depth_camera_names,
    joint_names_for_count,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    DETAIL_REALISTIC,
    MAX_TIME_TO_START_S,
    PACING_BURST_VIDEO,
    PACING_DEADLINE,
    PACING_SATURATE_WITH_BACKOFF,
    PER_THREAD_LOGGING_TAIL_S,
    PRODUCER_MULTI_PROCESS,
    PRODUCER_OLD_PER_THREAD,
    PRODUCER_PER_THREAD,
    PRODUCER_PROCESS_JOIN_TIMEOUT_S,
    PRODUCER_PROCESS_READY_POLL_S,
    PRODUCER_PROCESS_REPORT_TIMEOUT_S,
    PRODUCER_PROCESS_TERMINATE_TIMEOUT_S,
    random_phase_jitter_window,
)
from tests.integration.platform.data_daemon.shared.test_case.context_spec import (
    JOINT_STREAM,
    VIDEO_STREAM,
    ContextSpec,
    recording_timestamps,
    stream_phase_seed,
)
from tests.integration.platform.data_daemon.shared.test_case.frame_source import (
    prewarm_frame_bank,
)
from tests.integration.platform.data_daemon.shared.test_case.streams import (
    StreamEmitter,
    StreamPlan,
    _run_stream_threads,
    stream_plans_for_case,
)


def _should_pace(pacing: str, is_video: bool) -> bool:
    """Whether a stream sleeps to its wall-clock deadline under *pacing*.

    ``PACING_BURST_VIDEO`` un-paces depth cameras too, since both share a spool.
    """
    if pacing == PACING_DEADLINE:
        return True
    if pacing == PACING_BURST_VIDEO:
        return not is_video
    return False


def _should_backoff(pacing: str) -> bool:
    """Whether a rejected frame is retried rather than fatal under *pacing*."""
    return pacing == PACING_SATURATE_WITH_BACKOFF


def _await_frame_deadline(deadline: float, stop_event: threading.Event) -> bool:
    """Wait until *deadline*, unless *stop_event* fires first.

    Returns:
        ``True`` when *stop_event* fired, meaning the caller's loop should stop.
    """
    remaining = deadline - time.time()
    if remaining <= 0:
        return False
    return stop_event.wait(remaining)


@dataclass(frozen=True, slots=True)
class ProducerRequest:
    """Everything a producer needs to run, whatever that producer's lifetime.

    One shape for all three engines, so a case can swap producers without its
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
        pacing: How hard the streams may drive the SDK. Only the lifetime
            producer may carry a rate; every other producer is refused one when
            its case is built, so the value here is always one it honours.
        log_interval_s: Fixed sleep after each frame, imposed by the performance
            suites so latency is measured at a known offered rate.
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
    log_interval_s: float = 0.0

    def frame_budget(self, plan: StreamPlan) -> int | None:
        """Frames *plan* emits, or ``None`` when it runs until stopped."""
        if self.duration_sec is None:
            return None
        return plan.fps * self.duration_sec


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
            stop_event=request.stop_event,
            assert_deadline=request.assert_deadline,
            backoff=_should_backoff(request.pacing),
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
    emitter.emit(frame_index, timestamp)
    report.record(
        emitter,
        EmittedFrame(
            timestamp=timestamp,
            frame_index=frame_index,
            emitted_at=emitted_at,
            completed_at=time.time(),
            handle=handle,
        ),
    )


def run_synchronous_logging(request: ProducerRequest) -> dict[str, list[EmittedFrame]]:
    """Log every stream from one thread, in nominal schedule order.

    Scoped to one recording: returns before that recording stops.
    """
    emitters = _build_emitters(request)
    for emitter in emitters:
        emitter.prepare(request.image_width, request.image_height, request.video_detail)

    report = FrameReport()
    schedules = {
        emitter: stream_frame_schedule(emitter.plan, request) for emitter in emitters
    }
    pending = {
        emitter: frame
        for emitter, schedule in schedules.items()
        if (frame := next(schedule, None)) is not None
    }

    while pending and not request.stop_event.is_set():
        due = min(pending, key=lambda emitter: pending[emitter][0] / emitter.plan.fps)
        frame_index, timestamp = pending[due]
        _emit_and_record(due, frame_index, timestamp, request, report)
        next_frame = next(schedules[due], None)
        if next_frame is None:
            del pending[due]
        else:
            pending[due] = next_frame
        if request.log_interval_s:
            time.sleep(request.log_interval_s)

    return report.as_dict()


def run_old_per_thread_logging(
    request: ProducerRequest,
) -> dict[str, list[EmittedFrame]]:
    """Run one thread per stream, each bursting through its own frame budget.

    Streams race each other inside a recording, but every thread is joined
    before it stops, so no frame is in flight at a boundary.
    """
    emitters = _build_emitters(request)
    report = FrameReport()
    barrier = threading.Barrier(len(emitters))

    def worker(emitter: StreamEmitter) -> None:
        barrier.wait()
        emitter.prepare(request.image_width, request.image_height, request.video_detail)
        for frame_index, timestamp in stream_frame_schedule(emitter.plan, request):
            if request.stop_event.is_set():
                break
            _emit_and_record(emitter, frame_index, timestamp, request, report)
            if request.log_interval_s:
                time.sleep(request.log_interval_s)

    _run_stream_threads(emitters, worker, error_label="Old per-thread")

    return report.as_dict()


def run_per_thread_logging(request: ProducerRequest) -> dict[str, list[EmittedFrame]]:
    """Run one thread per stream, for as long as the caller keeps them running.

    Every frame is reported, even those logged while no recording was current,
    so the daemon's rejections are visible too (see
    :func:`_classify_boundary_frames`).

    Returns:
        Mapping of trace key -> every frame logged for it, in order.
    """
    emitters = _build_emitters(request)
    report = FrameReport()
    barrier = threading.Barrier(len(emitters))

    def worker(emitter: StreamEmitter) -> None:
        barrier.wait()
        emitter.prepare(request.image_width, request.image_height, request.video_detail)
        pace = _should_pace(request.pacing, emitter.plan.is_video)
        thread_wall_start = time.time()
        for frame_index, timestamp in stream_frame_schedule(emitter.plan, request):
            if request.stop_event.is_set():
                break
            frame_deadline = thread_wall_start + (frame_index / emitter.plan.fps)
            if pace and _await_frame_deadline(frame_deadline, request.stop_event):
                break
            _emit_and_record(emitter, frame_index, timestamp, request, report)
            if request.log_interval_s:
                time.sleep(request.log_interval_s)

    _run_stream_threads(emitters, worker, error_label="Per-thread")

    return report.as_dict()


class ProducerSession(ABC):
    """Runs a case's producer, whatever its lifetime, behind one interface.

    A caller states the start/run/stop protocol once; ``producer_channels``
    decides nothing about that code's shape. Every producer reports every
    frame with wall-clock brackets, so one classification decides which
    recording owns which (see :meth:`classify`).
    """

    needs_stop_gate_bracket: ClassVar[bool] = False
    """Whether :meth:`classify` needs ``stop_settled_at`` measured, not assumed
    — costs a polling thread inside every ``stop_recording`` call."""

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
            log_interval_s=self.spec.log_interval_s,
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

    def classify(
        self,
        trace_key: str,
        frames: list[EmittedFrame],
        bounds: RecordingControlBounds,
    ) -> TraceClassification:
        """Return what a recording requires of one trace's frames; overridden
        where the handle rule does not hold (see
        :func:`classify_split_producer_frames`)."""
        return _classify_boundary_frames(frames, bounds)


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
        codes stay unique on their own (see :func:`run_per_thread_logging`)."""
        return 0

    def start(self) -> None:
        """Start logging now, before any recording exists."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._frames = run_per_thread_logging(
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
            raise RuntimeError(f"Per-thread producer failed: {self._error}") from (
                self._error
            )

    def report(self) -> dict[str, list[EmittedFrame]]:
        """Return every frame logged over the whole context lifetime."""
        return self._frames


def partition_plans(
    plans: list[StreamPlan], process_streams: tuple[tuple[str, ...], ...]
) -> tuple[list[StreamPlan], list[list[StreamPlan]]]:
    """Split *plans* into the ones this process runs and one group per child;
    every plan lands in exactly one group, since the case already rejected
    duplicates."""
    groups = [frozenset(group) for group in process_streams]
    local: list[StreamPlan] = []
    children: list[list[StreamPlan]] = [[] for _ in process_streams]
    for plan in plans:
        for index, group in enumerate(groups):
            if plan.placement_tokens & group:
                children[index].append(plan)
                break
        else:
            local.append(plan)
    return local, children


@dataclass(frozen=True, slots=True)
class ProducerProcessSpec:
    """What one producer child needs, in picklable form — :class:`ProducerRequest`
    holds a live handle and ``threading.Event``, neither picklable."""

    robot_name: str
    dataset_name: str
    plans: tuple[StreamPlan, ...]
    image_width: int | None
    image_height: int | None
    video_detail: str
    timestamp_start_s: float
    random_phase: bool
    pacing: str
    context_index: int


def _producer_process(
    spec: ProducerProcessSpec,
    ready_event: Any,
    stop_event: Any,
    result_queue: Any,
) -> None:
    """Run the lifetime producer for *spec*'s streams in its own OS process.

    Never seals its own chunks — the daemon closes them on the recording
    owner's stop. ``ready_event``/``stop_event``/``result_queue`` are
    ``multiprocessing`` primitives from a ``"spawn"`` context.
    """
    robot = None
    try:
        ensure_login()
        nc.get_dataset(spec.dataset_name)
        robot = nc.connect_robot(spec.robot_name, overwrite=False)
        if (
            any(plan.is_video for plan in spec.plans)
            and spec.video_detail == DETAIL_REALISTIC
        ):
            # A lazy build inside the camera thread would cost seconds.
            prewarm_frame_bank(spec.video_detail, spec.image_width, spec.image_height)
        ready_event.set()
        report = run_per_thread_logging(
            ProducerRequest(
                robot=robot,
                robot_name=spec.robot_name,
                context_index=spec.context_index,
                recording_index=0,
                seed_ordinal=0,
                plans=spec.plans,
                image_width=spec.image_width,
                image_height=spec.image_height,
                video_detail=spec.video_detail,
                timestamp_start_s=spec.timestamp_start_s,
                random_phase=spec.random_phase,
                duration_sec=None,
                pacing=spec.pacing,
                stop_event=stop_event,
            )
        )
        result_queue.put({
            "ok": True,
            "report": report,
            "timer_stats": {label: dict(v) for label, v in Timer._stats.items()},
        })
    except BaseException:  # noqa: BLE001 - propagate full child traceback
        result_queue.put({"ok": False, "traceback": traceback.format_exc()})
    finally:
        if robot is not None:
            robot.close()


@dataclass(slots=True)
class _ChildProducer:
    """One spawned producer child and the primitives that drive it."""

    process: Any
    ready_event: Any
    stop_event: Any
    result_queue: Any


class MultiProcessProducerSession(ProducerSession):
    """A lifetime producer whose streams are split across OS processes:
    children never call ``start_recording``, learning of one only via the SSE
    notification — the topology a real deployment has when camera and
    recording owner are separate."""

    needs_stop_gate_bracket: ClassVar[bool] = True

    def __init__(
        self,
        spec: ContextSpec,
        robot: object,
        plans: list[StreamPlan],
        local_plans: list[StreamPlan],
        child_plan_groups: list[list[StreamPlan]],
    ) -> None:
        super().__init__(spec, robot, plans)
        self._child_plan_groups = child_plan_groups
        self._local = LifetimeProducerSession(spec, robot, local_plans)
        self._children: list[_ChildProducer] = []
        self._child_frames: dict[str, list[EmittedFrame]] = {}
        # Precomputed: only needed by classify(), which runs after finish().
        self._child_trace_keys = frozenset(
            key
            for group in child_plan_groups
            for plan in group
            for key in plan.trace_keys
        )

    def frame_code_recording_index(self, recording_ordinal: int) -> int:
        """Always ``0``: every producer here numbers frames session-wide."""
        return 0

    def _child_spec(self, child_plans: list[StreamPlan]) -> ProducerProcessSpec:
        case = self.spec.case
        return ProducerProcessSpec(
            robot_name=self.spec.robot_name,
            dataset_name=self.spec.dataset_name,
            plans=tuple(child_plans),
            image_width=case.image_width,
            image_height=case.image_height,
            video_detail=case.video_detail,
            timestamp_start_s=self.spec.timestamp_start_s,
            random_phase=case.random_phase,
            pacing=case.producer_pacing,
            context_index=self.spec.context_index,
        )

    def start(self) -> None:
        """Spawn every child and wait for it to connect, then start locally:
        children must already be logging before the first ``start_recording``,
        or the boundary proves nothing."""
        spawn_ctx = multiprocessing.get_context("spawn")
        for process_index, child_plans in enumerate(self._child_plan_groups):
            child = _ChildProducer(
                process=None,
                ready_event=spawn_ctx.Event(),
                stop_event=spawn_ctx.Event(),
                result_queue=spawn_ctx.Queue(),
            )
            child.process = spawn_ctx.Process(
                name=f"producer-{self.spec.context_index}-{process_index}",
                target=_producer_process,
                args=(
                    self._child_spec(child_plans),
                    child.ready_event,
                    child.stop_event,
                    child.result_queue,
                ),
            )
            child.process.start()
            self._children.append(child)

        for process_index, child in enumerate(self._children):
            if not self._await_child_ready(child):
                # finish() still reaps the other children already in self._children.
                failure = self._collect_child(child) or (
                    f"still running after {MAX_TIME_TO_START_S}s"
                )
                raise RuntimeError(
                    f"producer child {process_index} never started logging: "
                    f"{failure}"
                )
        self._local.start()

    def _await_child_ready(self, child: _ChildProducer) -> bool:
        """Whether *child* began logging within :data:`MAX_TIME_TO_START_S`,
        giving up early if the process dies so its own traceback can be
        reported."""
        deadline = time.time() + MAX_TIME_TO_START_S
        while time.time() < deadline:
            if child.ready_event.wait(timeout=PRODUCER_PROCESS_READY_POLL_S):
                return True
            if not child.process.is_alive():
                return False
        return False

    def run_recording(self, recording_ordinal: int) -> None:
        """Hold the window open: every producer is already running."""
        self._local.run_recording(recording_ordinal)

    def finish(self) -> None:
        """Stop the local streams, then every child: the local tail wait runs
        first, giving children the same window of post-stop logging."""
        try:
            self._local.finish()
        finally:
            for child in self._children:
                child.stop_event.set()
            failures = [self._collect_child(child) for child in self._children]
            self._children = []
        errors = [failure for failure in failures if failure]
        if errors:
            raise RuntimeError(
                f"{len(errors)} producer child process(es) failed:\n"
                + "\n".join(errors)
            )

    def _collect_child(self, child: _ChildProducer) -> str:
        """Fold *child*'s report in and reap it. Returns its failure, or ``""``.

        The queue is drained before the join — joining first can deadlock,
        since a process won't exit until its queued item clears the pipe.
        """
        try:
            outcome = child.result_queue.get(timeout=PRODUCER_PROCESS_REPORT_TIMEOUT_S)
        except queue.Empty:
            outcome = None
        child.process.join(timeout=PRODUCER_PROCESS_JOIN_TIMEOUT_S)
        if child.process.is_alive():
            child.process.terminate()
            child.process.join(timeout=PRODUCER_PROCESS_TERMINATE_TIMEOUT_S)
            return f"child {child.process.name} did not exit and was terminated"
        if outcome is None:
            return f"child {child.process.name} exited without reporting"
        if not outcome.get("ok"):
            return f"child {child.process.name} raised:\n{outcome.get('traceback')}"
        if child.process.exitcode != 0:
            return (
                f"child {child.process.name} exited with code "
                f"{child.process.exitcode}"
            )
        # Trace keys are disjoint across producers, so no trace is interleaved.
        self._child_frames.update(outcome["report"])
        Timer.merge_stats(outcome.get("timer_stats", {}))
        return ""

    def report(self) -> dict[str, list[EmittedFrame]]:
        """Every frame logged, by this process and by every child."""
        frames = dict(self._local.report())
        frames.update(self._child_frames)
        return frames

    def classify(
        self,
        trace_key: str,
        frames: list[EmittedFrame],
        bounds: RecordingControlBounds,
    ) -> TraceClassification:
        """Use the cross-process rule for traces a child wrote — a child's own
        handle value would make the in-process rule condemn frames the window
        accepted."""
        if trace_key not in self._child_trace_keys:
            return super().classify(trace_key, frames, bounds)
        return classify_split_producer_frames(frames, bounds)


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
    if case.producer_channels == PRODUCER_MULTI_PROCESS:
        local_plans, child_groups = partition_plans(
            plans, case.producer_process_streams
        )
        return MultiProcessProducerSession(
            spec, robot, plans, local_plans, child_groups
        )
    if case.producer_channels == PRODUCER_PER_THREAD:
        return LifetimeProducerSession(spec, robot, plans)
    engine = (
        run_old_per_thread_logging
        if case.producer_channels == PRODUCER_OLD_PER_THREAD
        else run_synchronous_logging
    )
    return BoundedProducerSession(spec, robot, plans, engine)
