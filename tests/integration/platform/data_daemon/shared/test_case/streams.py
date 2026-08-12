"""Stream decomposition, and the one path every ``nc.log_*`` call takes."""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

import neuracore as nc
from neuracore.data_daemon.bridge import LoggingStalledError
from tests.integration.platform.data_daemon.shared.process_control import (
    MAX_TIME_TO_LOG_S,
    Timer,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestCase,
    camera_names,
    depth_camera_names,
    generate_joint_values,
    joint_names_for_count,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    BACKLOG_BACKOFF_BASE_S,
    BACKLOG_BACKOFF_MAX_S,
    BACKLOG_STALL_BUDGET_S,
    JOINT_KINDS,
    PRODUCER_SYNCHRONOUS,
    DepthMode,
)
from tests.integration.platform.data_daemon.shared.test_case.frame_source import (
    encode_depth_frame,
    frame_code_base,
    make_camera_feed,
    preallocate_depth_buffer,
)

# Derived from JOINT_KINDS so the two cannot drift apart.
JOINT_DATA_TYPES = {kind: kind.upper() for kind in JOINT_KINDS}


def _log_with_backlog_backoff(
    call: Callable[[], None], stop_event: threading.Event
) -> None:
    """Run one ``nc.log_*`` call, retrying while the daemon is backlogged.

    Wraps the individual call so a retry can't duplicate a frame that was
    never admitted.

    Raises:
        LoggingStalledError: The stall persisted past
            :data:`BACKLOG_STALL_BUDGET_S` or *stop_event* fired; the frame
            never landed, so this surfaces rather than being swallowed.
    """
    delay = BACKLOG_BACKOFF_BASE_S
    stalled_for = 0.0
    while True:
        try:
            call()
            return
        except LoggingStalledError:
            stalled_for += delay
            if stalled_for >= BACKLOG_STALL_BUDGET_S or stop_event.wait(delay):
                raise
            delay = min(delay * 2, BACKLOG_BACKOFF_MAX_S)


def rgb_frame_code(
    *,
    context_index: int,
    recording_index: int,
    camera_index: int,
    frame_index: int,
) -> int:
    """Return the integer painted into a camera frame's pixels."""
    return (
        frame_code_base(
            context_index=context_index,
            recording_ordinal=recording_index,
            camera_index=camera_index,
        )
        + frame_index
    )


@dataclass(frozen=True, slots=True)
class StreamPlan:
    """One producer stream: which channels it logs, and under which marker.

    Producers consume it through :class:`StreamEmitter` rather than re-deriving
    channel names, log functions or frame codes for themselves.
    """

    name: str
    fps: int
    channel_names: tuple[str, ...]
    marker_name: str | None = None
    # Per-thread streams hold one kind; the synchronous producer bundles all.
    joint_kinds: tuple[str, ...] = ()
    camera_indexes: tuple[int, ...] = ()
    # Sample dtype for depth streams; ignored by every other kind.
    depth_mode: DepthMode = "float32"

    @property
    def is_rgb(self) -> bool:
        """Whether this stream logs RGB camera frames."""
        return self.name == "rgb"

    @property
    def is_depth(self) -> bool:
        """Whether this stream logs depth camera frames."""
        return self.name == "depth"

    @property
    def is_video(self) -> bool:
        """Whether this stream feeds the video pipeline rather than joint data.

        Both camera kinds share the video timestamp schedule and are paced
        together, so producers branch on this rather than :attr:`is_rgb`.
        """
        return self.is_rgb or self.is_depth

    @property
    def placement_tokens(self) -> frozenset[str]:
        """Names a producer placement may use to move this stream to a process.

        A camera answers to its channel and its kind; a joint stream answers
        only to its kind, since its channels log in one call and can't split.
        """
        if self.is_video:
            return frozenset({self.name, *self.channel_names})
        return frozenset({self.name})

    @property
    def trace_keys(self) -> list[str]:
        """Semantic trace keys one logged frame from this stream touches.

        Matches the ``data_type/data_type_name`` keys disk_helpers resolves from
        the DB, plus this stream's own ``CUSTOM_1D`` marker when it has one.
        """
        from neuracore_types.utils import validate_safe_name

        if self.is_rgb:
            data_types = ["RGB_IMAGES"]
        elif self.is_depth:
            data_types = ["DEPTH_IMAGES"]
        else:
            data_types = [JOINT_DATA_TYPES[kind] for kind in self.joint_kinds]
        keys = [
            f"{data_type}/{validate_safe_name(name)}"
            for data_type in data_types
            for name in self.channel_names
        ]
        if self.marker_name is not None:
            keys.append(f"CUSTOM_1D/{validate_safe_name(self.marker_name)}")
        return keys


def _per_camera_plans(
    kind: str,
    camera_name_list: list[str],
    video_fps: int,
    *,
    depth_mode: DepthMode = "float32",
) -> list[StreamPlan]:
    """One stream per camera of *kind*, each carrying its own marker.

    Every producer but the synchronous one gives a camera its own thread, so it
    gets its own marker series too.
    """
    return [
        StreamPlan(
            name=kind,
            marker_name=f"marker_{camera_name}",
            fps=video_fps,
            channel_names=(camera_name,),
            camera_indexes=(camera_index,),
            depth_mode=depth_mode,
        )
        for camera_index, camera_name in enumerate(camera_name_list)
    ]


def _bundled_camera_plan(
    kind: str,
    camera_name_list: list[str],
    video_fps: int,
    *,
    depth_mode: DepthMode = "float32",
) -> StreamPlan | None:
    """One stream covering every camera of *kind*, or ``None`` if there are none.

    The synchronous producer logs a kind's cameras from one thread and emits a
    single marker series of its own, so these plans carry no marker.
    """
    if not camera_name_list:
        return None
    return StreamPlan(
        name=kind,
        fps=video_fps,
        channel_names=tuple(camera_name_list),
        camera_indexes=tuple(range(len(camera_name_list))),
        depth_mode=depth_mode,
    )


def build_stream_plans(
    *,
    joint_names: list[str],
    camera_name_list: list[str],
    depth_camera_name_list: list[str],
    depth_mode: DepthMode,
    joint_fps: int,
    video_fps: int,
) -> list[StreamPlan]:
    """Decompose a workload into one stream per camera and per joint data type.

    An empty ``joint_names`` produces no joint-kind streams at all, which is how
    a camera-only producer process asks for cameras alone.
    """
    return [
        *_per_camera_plans("rgb", camera_name_list, video_fps),
        *_per_camera_plans(
            "depth", depth_camera_name_list, video_fps, depth_mode=depth_mode
        ),
        *(
            StreamPlan(
                name=kind,
                marker_name=f"marker_{kind}",
                fps=joint_fps,
                channel_names=tuple(joint_names),
                joint_kinds=(kind,),
            )
            for kind in JOINT_KINDS
            if joint_names
        ),
    ]


def build_synchronous_stream_plans(
    *,
    joint_names: list[str],
    camera_name_list: list[str],
    depth_camera_name_list: list[str],
    depth_mode: DepthMode,
    joint_fps: int,
    video_fps: int,
    marker_name: str,
) -> tuple[StreamPlan, StreamPlan | None, StreamPlan | None]:
    """Decompose a workload for the single-threaded producer.

    One stream per kind, not per camera, since a single thread logs every
    camera of a kind together; only the joint stream carries a marker.
    """
    joint_plan = StreamPlan(
        name="joints",
        marker_name=marker_name,
        fps=joint_fps,
        channel_names=tuple(joint_names),
        joint_kinds=JOINT_KINDS,
    )
    video_plan = _bundled_camera_plan("rgb", camera_name_list, video_fps)
    depth_plan = _bundled_camera_plan(
        "depth", depth_camera_name_list, video_fps, depth_mode=depth_mode
    )
    return joint_plan, video_plan, depth_plan


def stream_plans_for_case(
    *,
    producer_channels: str,
    joint_names: list[str],
    camera_name_list: list[str],
    depth_camera_name_list: list[str],
    depth_mode: DepthMode,
    joint_fps: int,
    video_fps: int,
    marker_name: str,
) -> list[StreamPlan]:
    """Every stream the producer for *producer_channels* will actually run."""
    if producer_channels == PRODUCER_SYNCHRONOUS:
        return [
            plan
            for plan in build_synchronous_stream_plans(
                joint_names=joint_names,
                camera_name_list=camera_name_list,
                depth_camera_name_list=depth_camera_name_list,
                depth_mode=depth_mode,
                joint_fps=joint_fps,
                video_fps=video_fps,
                marker_name=marker_name,
            )
            if plan is not None
        ]
    return build_stream_plans(
        joint_names=joint_names,
        camera_name_list=camera_name_list,
        depth_camera_name_list=depth_camera_name_list,
        depth_mode=depth_mode,
        joint_fps=joint_fps,
        video_fps=video_fps,
    )


def case_stream_plans(case: DataDaemonTestCase) -> list[StreamPlan]:
    """The per-stream decomposition *case*'s producer placement is spread over.

    One entry point so the placement guard, the late-start derivation and the
    session that partitions the plans all read the same decomposition.
    """
    return build_stream_plans(
        joint_names=joint_names_for_count(case.joint_count),
        camera_name_list=camera_names(case.video_count),
        depth_camera_name_list=depth_camera_names(case.depth_count),
        depth_mode=case.depth_mode,
        joint_fps=case.joint_fps,
        video_fps=case.video_fps,
    )


def late_starting_trace_keys(case: DataDaemonTestCase) -> frozenset[str]:
    """Trace keys *case* logs from a process that does not own the recording.

    Such a producer only learns a recording is active from the SSE
    notification, so its leading frames are legitimately missing; derived
    from the same plans the session partitions on, so the two agree by
    construction.
    """
    moved = {name for group in case.producer_process_streams for name in group}
    if not moved:
        return frozenset()
    return frozenset(
        key
        for plan in case_stream_plans(case)
        if plan.placement_tokens & moved
        for key in plan.trace_keys
    )


@dataclass(slots=True, eq=False)  # identity-keyed: one emitter per thread
class StreamEmitter:
    """Binds a :class:`StreamPlan` to one recording's logging calls.

    Owns every per-frame side effect, so producers cannot drift in what they
    log, how frames are identified, or how calls are labelled for reporting.
    """

    plan: StreamPlan
    robot_name: str
    context_index: int
    recording_index: int
    # Cuts a backlog retry short, so it is needed whether or not one can happen.
    stop_event: threading.Event
    assert_deadline: bool = False
    # Set only by PACING_SATURATE_WITH_BACKOFF: retry a refused frame.
    backoff: bool = False
    feed: object | None = None
    depth_buffer: np.ndarray | None = None
    image_width: int | None = None
    image_height: int | None = None
    # Resolved once, not per frame: building them walks validate_safe_name.
    trace_keys: tuple[str, ...] = ()

    def prepare(self, image_width: int | None, image_height: int | None, detail: str):
        """Build this stream's frame source. Called on the stream's own thread."""
        self.image_width = image_width
        self.image_height = image_height
        self.trace_keys = tuple(self.plan.trace_keys)
        self.feed = make_camera_feed(
            self.plan.is_rgb, image_width, image_height, detail
        )
        self.depth_buffer = preallocate_depth_buffer(
            self.plan.is_depth, image_width, image_height, self.plan.depth_mode
        )

    def _log_video_frame(self, label: str, call: Callable[[], None]) -> None:
        """Log one video frame, timed, retrying it if the daemon is backlogged.

        The ``Timer`` sits *inside* the retry so each attempt is measured on its
        own; it skips its deadline assertion for a refused attempt.
        """

        def timed() -> None:
            with Timer(
                MAX_TIME_TO_LOG_S, label=label, assert_deadline=self.assert_deadline
            ):
                call()

        if self.backoff:
            _log_with_backlog_backoff(timed, self.stop_event)
        else:
            timed()

    def emit(self, frame_index: int, timestamp: float | None) -> None:
        """Log one frame's payload plus this stream's marker."""
        if self.plan.is_rgb:
            self._emit_rgb(frame_index, timestamp)
        elif self.plan.is_depth:
            self._emit_depth(frame_index, timestamp)
        else:
            self._emit_joints(frame_index, timestamp)
        self._emit_marker(frame_index, timestamp)

    def _emit_rgb(self, frame_index: int, timestamp: float | None) -> None:
        for camera_name, camera_index in zip(
            self.plan.channel_names, self.plan.camera_indexes
        ):
            frame_code = rgb_frame_code(
                context_index=self.context_index,
                recording_index=self.recording_index,
                camera_index=camera_index,
                frame_index=frame_index,
            )
            rgb_image = self.feed.render(frame_index, frame_code)
            self._log_video_frame(
                "nc.log_rgb",
                functools.partial(
                    nc.log_rgb,
                    camera_name,
                    rgb_image,
                    robot_name=self.robot_name,
                    timestamp=timestamp,
                ),
            )

    def _emit_depth(self, frame_index: int, timestamp: float | None) -> None:
        for camera_name, camera_index in zip(
            self.plan.channel_names, self.plan.camera_indexes
        ):
            # Same identity the RGB path paints; here it seeds the depth pattern.
            frame_code = (
                frame_code_base(
                    context_index=self.context_index,
                    recording_ordinal=self.recording_index,
                    camera_index=camera_index,
                )
                + frame_index
            )
            depth_image = encode_depth_frame(
                frame_code,
                self.image_width,
                self.image_height,
                self.plan.depth_mode,
                out=self.depth_buffer,
            )
            self._log_video_frame(
                "nc.log_depth",
                functools.partial(
                    nc.log_depth,
                    camera_name,
                    depth_image,
                    robot_name=self.robot_name,
                    timestamp=timestamp,
                ),
            )

    def _emit_joints(self, frame_index: int, timestamp: float | None) -> None:
        joint_values = generate_joint_values(
            frame_index, self.plan.fps, list(self.plan.channel_names)
        )
        for kind in self.plan.joint_kinds:
            log_fn_name = f"log_{kind}"
            with Timer(
                MAX_TIME_TO_LOG_S,
                label=f"nc.{log_fn_name}",
                assert_deadline=self.assert_deadline,
            ):
                getattr(nc, log_fn_name)(
                    joint_values, robot_name=self.robot_name, timestamp=timestamp
                )

    def _emit_marker(self, frame_index: int, timestamp: float | None) -> None:
        if self.plan.marker_name is None:
            return
        with Timer(
            MAX_TIME_TO_LOG_S,
            label="nc.log_custom_1d",
            assert_deadline=self.assert_deadline,
        ):
            nc.log_custom_1d(
                self.plan.marker_name,
                np.array([float(frame_index)], dtype=np.float32),
                robot_name=self.robot_name,
                timestamp=timestamp,
            )


def _run_stream_threads(
    emitters: list[StreamEmitter],
    worker: Callable[[StreamEmitter], None],
    *,
    error_label: str,
) -> None:
    """Run *worker* on one daemon thread per stream, then join and surface errors."""
    thread_errors: list[BaseException] = []

    def guarded(emitter: StreamEmitter) -> None:
        try:
            worker(emitter)
        except BaseException as exc:  # noqa: BLE001
            thread_errors.append(exc)

    threads = [
        threading.Thread(target=guarded, args=(emitter,), daemon=True)
        for emitter in emitters
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if thread_errors:
        raise RuntimeError(
            f"{error_label} producer failed: {thread_errors[0]}"
        ) from thread_errors[0]
