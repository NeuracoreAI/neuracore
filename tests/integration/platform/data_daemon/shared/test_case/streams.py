"""Stream decomposition, and the one path every ``nc.log_*`` call takes."""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

import neuracore as nc
from tests.integration.platform.data_daemon.shared.process_control import (
    MAX_TIME_TO_LOG_S,
    Timer,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestCase,
    generate_joint_values,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    DATA_TYPE_BY_STREAM,
    DATA_TYPE_CUSTOM_1D,
    JOINT_KINDS,
    PRODUCER_SYNCHRONOUS,
    STREAM_DEPTH,
    STREAM_JOINTS,
    STREAM_RGB,
    DepthMode,
    camera_names,
    depth_camera_names,
    joint_group_name,
    joint_name_groups,
    joint_names_for_count,
    marker_name_for,
    trace_key_for,
)
from tests.integration.platform.data_daemon.shared.test_case.frame_source import (
    encode_depth_frame,
    frame_code_base,
    make_camera_feed,
    preallocate_depth_buffer,
)


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
    group_name: str | None = None
    # Sample dtype for depth streams; ignored by every other kind.
    depth_mode: DepthMode = "float32"

    @property
    def is_rgb(self) -> bool:
        """Whether this stream logs RGB camera frames."""
        return self.name == STREAM_RGB

    @property
    def is_depth(self) -> bool:
        """Whether this stream logs depth camera frames."""
        return self.name == STREAM_DEPTH

    @property
    def is_video(self) -> bool:
        """Whether this stream feeds the video pipeline rather than joint data.

        Both camera kinds share the video timestamp schedule and are paced
        together, so producers branch on this rather than :attr:`is_rgb`.
        """
        return self.is_rgb or self.is_depth

    @property
    def placement_tokens(self) -> frozenset[str]:
        """Names a producer placement may use to move this stream to a process:
        this stream's kind, and the device producing it — a camera for video, a
        joint group for joints."""
        devices = self.channel_names if self.is_video else (self.group_name,)
        return frozenset({self.name, *devices})

    @property
    def trace_keys(self) -> list[str]:
        """Semantic trace keys one logged frame from this stream touches.

        Matches the ``data_type/data_type_name`` keys disk_helpers resolves from
        the DB, plus this stream's own ``CUSTOM_1D`` marker when it has one.
        """
        if self.is_video:
            data_types = [DATA_TYPE_BY_STREAM[self.name]]
        else:
            data_types = [DATA_TYPE_BY_STREAM[kind] for kind in self.joint_kinds]
        keys = [
            trace_key_for(data_type, name)
            for data_type in data_types
            for name in self.channel_names
        ]
        if self.marker_name is not None:
            keys.append(trace_key_for(DATA_TYPE_CUSTOM_1D, self.marker_name))
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
            marker_name=marker_name_for(camera_name),
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
    joint_process_groups: int = 1,
) -> list[StreamPlan]:
    """Decompose a workload into one stream per camera and per joint data type.

    *joint_process_groups* splits the joints first, so one data type is written
    by several streams. Split, a marker is named after its group, keeping every
    stream's trace keys disjoint.
    """
    return [
        *_per_camera_plans(STREAM_RGB, camera_name_list, video_fps),
        *_per_camera_plans(
            STREAM_DEPTH, depth_camera_name_list, video_fps, depth_mode=depth_mode
        ),
        *(
            StreamPlan(
                name=kind,
                marker_name=marker_name_for(
                    kind if joint_process_groups < 2 else f"{group_name}_{kind}"
                ),
                fps=joint_fps,
                channel_names=tuple(group_joint_names),
                joint_kinds=(kind,),
                group_name=group_name,
            )
            for group_name, group_joint_names in joint_name_groups(
                joint_names, joint_process_groups
            )
            for kind in JOINT_KINDS
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
        name=STREAM_JOINTS,
        marker_name=marker_name,
        fps=joint_fps,
        channel_names=tuple(joint_names),
        joint_kinds=JOINT_KINDS,
        group_name=joint_group_name(0),
    )
    video_plan = _bundled_camera_plan(STREAM_RGB, camera_name_list, video_fps)
    depth_plan = _bundled_camera_plan(
        STREAM_DEPTH, depth_camera_name_list, video_fps, depth_mode=depth_mode
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
    joint_process_groups: int = 1,
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
        joint_process_groups=joint_process_groups,
    )


def case_stream_plans(case: DataDaemonTestCase) -> list[StreamPlan]:
    """The per-stream decomposition *case*'s producer placement is spread over."""
    return build_stream_plans(
        joint_names=joint_names_for_count(case.joint_count),
        camera_name_list=camera_names(case.video_count),
        depth_camera_name_list=depth_camera_names(case.depth_count),
        depth_mode=case.depth_mode,
        joint_fps=case.joint_fps,
        video_fps=case.video_fps,
        joint_process_groups=case.joint_process_groups,
    )


def late_starting_trace_keys(case: DataDaemonTestCase) -> frozenset[str]:
    """Trace keys *case* logs from a process that does not own the recording."""
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
    assert_deadline: bool = False
    feed: object | None = None
    depth_buffer: np.ndarray | None = None
    image_width: int | None = None
    image_height: int | None = None
    # Resolved once, not per frame: building them walks validate_safe_name.
    trace_keys: tuple[str, ...] = ()
    # Deadline breaches for the frame currently being emitted; drained by emit().
    _deadline_breaches: list[str] = field(default_factory=list)

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

    def _record_breach_if_slow(self, label: str, timer: Timer) -> None:
        """Record a deadline breach; the boundary classifier judges it later."""
        if self.assert_deadline and timer.interval >= MAX_TIME_TO_LOG_S:
            self._deadline_breaches.append(
                f"{label} took {timer.interval:.3f}s >= {MAX_TIME_TO_LOG_S:.3f}s"
            )

    def _timed_log(self, label: str, call: Callable[[], None]) -> None:
        """Make one ``nc.log_*`` call, timed under *label*."""
        with Timer(
            MAX_TIME_TO_LOG_S, label=label, assert_deadline=False, log_breaches=False
        ) as timer:
            call()
        self._record_breach_if_slow(label, timer)

    def emit(self, frame_index: int, timestamp: float | None) -> tuple[str, ...]:
        """Log one frame's payload plus this stream's marker.

        Returns:
            Labels of any ``nc.log_*`` calls that breached the deadline.
        """
        self._deadline_breaches = []
        if self.plan.is_rgb:
            self._emit_rgb(frame_index, timestamp)
        elif self.plan.is_depth:
            self._emit_depth(frame_index, timestamp)
        else:
            self._emit_joints(frame_index, timestamp)
        self._emit_marker(frame_index, timestamp)
        return tuple(self._deadline_breaches)

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
            self._timed_log(
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
            self._timed_log(
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
            self._timed_log(
                f"nc.{log_fn_name}",
                functools.partial(
                    getattr(nc, log_fn_name),
                    joint_values,
                    robot_name=self.robot_name,
                    timestamp=timestamp,
                ),
            )

    def _emit_marker(self, frame_index: int, timestamp: float | None) -> None:
        if self.plan.marker_name is None:
            return
        self._timed_log(
            "nc.log_custom_1d",
            functools.partial(
                nc.log_custom_1d,
                self.plan.marker_name,
                np.array([float(frame_index)], dtype=np.float32),
                robot_name=self.robot_name,
                timestamp=timestamp,
            ),
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
