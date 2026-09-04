"""Shared configuration: dataclasses, utilities, and reporting.

Defines the ``DataDaemonTestCase`` dataclass, ``DataDaemonTestBatch`` for
grouping cases with shared infrastructure parameters, utility functions, and
analysis/reporting helpers.  Per-suite case lists live in each suite's
``test_cases.py``;
context-spec interpretation and recording workers live in
``test_case.context_spec`` and ``test_case.context_worker``.
"""

# cspell:ignore vardur jointgroups
from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, fields
from itertools import zip_longest
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from tests.integration.platform.data_daemon.shared.test_case.context_spec import (
        ContextResult,
    )

from neuracore.core.config.config_manager import get_config_manager
from tests.integration.platform.data_daemon.shared.process_control import (
    BUCKET_KEYS,
    LATENCY_BUCKETS_S,
    Timer,
    percentile_upper_bound,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    BASE_DATASET_READY_TIMEOUT_S,
    CONTROL_LOCAL,
    CONTROL_REMOTE,
    CONTROL_SPLIT_PROCESS,
    DETAIL_REALISTIC,
    DURATION_MODE_FIXED,
    DURATION_MODE_VARIABLE,
    LOG_PRESERVE,
    MAX_DATASET_READY_TIMEOUT_S,
    MODE_SEQUENTIAL,
    PACING_BURST_VIDEO,
    PACING_DEADLINE,
    PRODUCER_MULTI_PROCESS,
    PRODUCER_PER_THREAD,
    PRODUCER_SYNCHRONOUS,
    STOP_METHOD_CLI,
    STORAGE_STATE_DELETE,
    STORAGE_STATE_EMPTY,
    DepthMode,
    LogAction,
    ProducerChannels,
    ProducerPacing,
    RecordingControl,
    StopMethod,
    StorageStateAction,
    VideoDetail,
    camera_names,
    depth_camera_names,
    joint_name_groups,
    joint_names_for_count,
)

logger = logging.getLogger(__name__)

SESSION_RUNS: list[dict[str, object]] = []

# Parameters that live at batch level and are propagated to each case.
_BATCH_PARAMS = frozenset({
    "kill_daemon_between_tests",
    "storage_state_action",
    "daemon_log_action",
    "preserve_artifacts_per_test",
    "stop_method",
})


def _unsupported_combination(case: DataDaemonTestCase) -> str | None:
    """Return why *case*'s parameters cannot run together, or None if they can."""
    if case.producer_pacing == PACING_BURST_VIDEO and not (
        case.has_video or case.has_depth
    ):
        return (
            f"producer_pacing={PACING_BURST_VIDEO!r} needs video_count > 0 or "
            "depth_count > 0: it clumps the video streams and there are none"
        )
    return _unsupported_recording_control(case) or _unsupported_process_placement(case)


def _unsupported_recording_control(case: DataDaemonTestCase) -> str | None:
    """Return why *case* cannot be driven as asked, or None if it can.

    A remotely-started window opens after the call returns, so the frames at
    the head of each recording are logged before any gate is open. Only a
    producer that runs across the boundary can report that. A split-control
    window is opened here, so what it needs instead is a process of its own to
    spawn the stopping peer in.
    """
    if case.recording_control == CONTROL_SPLIT_PROCESS:
        if case.parallel_contexts != 1:
            return (
                f"recording_control={CONTROL_SPLIT_PROCESS!r} needs contexts=1,"
                f"got {case.parallel_contexts}: parallel contexts run in pool "
                "workers, which cannot start the stopping peer"
            )
        return None
    if case.recording_control != CONTROL_REMOTE:
        return None
    if case.producer_channels not in (PRODUCER_PER_THREAD, PRODUCER_MULTI_PROCESS):
        return (
            f"recording_control={CONTROL_REMOTE!r} needs a producer that outlives "
            f"its recordings, not {case.producer_channels!r}: the window opens "
            "after the call returns, so a recording-scoped producer logs its "
            "leading frames into a window that is not open yet"
        )
    if case.wait:
        return (
            f"recording_control={CONTROL_REMOTE!r} cannot honour wait=True: the "
            "web's stop endpoint returns as soon as the backend accepts it and "
            "has no notion of waiting for the upload to finish"
        )
    return None


def _unsupported_process_placement(case: DataDaemonTestCase) -> str | None:
    """Return why *case*'s producer placement cannot run, or None if it can.

    A stream placed twice would paint the same frame codes from two processes,
    so it is refused rather than silently producing implausible data.
    """
    # Imported late to dodge a circular import with ``streams``.
    from tests.integration.platform.data_daemon.shared.test_case.streams import (
        case_stream_plans,
    )

    multi_process = case.producer_channels == PRODUCER_MULTI_PROCESS
    if case.joint_process_groups < 1:
        return (
            f"joint_process_groups must be at least 1, got "
            f"{case.joint_process_groups}"
        )
    if case.joint_process_groups > 1:
        if not multi_process:
            return (
                "joint_process_groups only applies to "
                f"producer_channels={PRODUCER_MULTI_PROCESS!r}, "
                f"not {case.producer_channels!r}"
            )
        if case.joint_process_groups > case.joint_count:
            return (
                f"joint_process_groups={case.joint_process_groups} exceeds "
                f"joint_count={case.joint_count}: a group would own no joints"
            )
    if case.producer_process_streams and not multi_process:
        return (
            "producer_process_streams only applies to "
            f"producer_channels={PRODUCER_MULTI_PROCESS!r}, "
            f"not {case.producer_channels!r}"
        )
    if not multi_process:
        return None

    if not case.producer_process_streams:
        return (
            f"producer_channels={PRODUCER_MULTI_PROCESS!r} needs "
            "producer_process_streams to name at least one stream to move: "
            f"with none it is just {PRODUCER_PER_THREAD!r}"
        )
    if case.parallel_contexts != 1:
        return (
            f"producer_channels={PRODUCER_MULTI_PROCESS!r} needs "
            f"parallel_contexts=1, got {case.parallel_contexts}: parallel "
            "contexts run in pool workers, which cannot start processes"
        )

    plans = case_stream_plans(case)
    placeable = {token for plan in plans for token in plan.placement_tokens}
    for group in case.producer_process_streams:
        if not group:
            return "producer_process_streams entries cannot be empty"
        for name in group:
            if name not in placeable:
                return (
                    f"producer_process_streams names {name!r}, which this case "
                    f"does not produce — it produces {sorted(placeable)}"
                )

    groups = [frozenset(group) for group in case.producer_process_streams]
    for plan in plans:
        claimants = sum(1 for group in groups if plan.placement_tokens & group)
        if claimants > 1:
            named = plan.channel_names[0] if plan.is_video else plan.name
            return (
                f"producer_process_streams places {named!r} in more than one "
                "process; every stream must run in exactly one"
            )
    # Moving every stream out is legitimate: the owner then only opens and
    # closes the window.
    return None


@dataclass(frozen=True)
class DataDaemonTestCase:
    """A single parametrised test case for the data-daemon integration suite.

    Each instance fully describes one combination of workload and daemon
    configuration parameters.  Hand-curated cases in each suite's ``test_cases.py``
    are constructed directly using shorthand defaults, typically grouped into a
    ``DataDaemonTestBatch`` that applies shared infrastructure parameters.

    Attributes:
        duration_sec: Capture duration per individual recording, in seconds. All
            frames are generated at ``fps`` Hz so the total expected frame count
            is ``fps * duration_sec``. Under the default ``producer_pacing``
            it is wall-clock time too; a ``"saturate"`` case spends the same
            frame budget as fast as the transport allows.
        parallel_contexts: Number of recording contexts that run concurrently.
            Each context owns an independent robot connection and cycles through
            its share of the total ``recording_count``.
        mode: Timestamp layout across parallel contexts — *not* an execution
            order; every context runs concurrently regardless. ``"staggered"``
            offsets each context so consecutive spans partly
            overlap. Single-context cases always use ``"sequential"``.
        CHANNELS: How producer threads are allocated and how long they live — a
            class attribute, not a field, chosen by constructing the matching
            subclass (:class:`Synchronous`, :class:`PerThread`,
            :class:`ProcessPerCamera`).
        producer_process_streams: One entry per extra producer process, naming
            the streams (a kind, a single camera's channel, or a joint group)
            that run there; streams not named stay with the recording-owning
            process. Only :class:`ProcessPerCamera` and its subclasses
            may set it; left empty they take that class's
            :meth:`default_process_streams`.
        joint_process_groups: How many groups the joints are split into, each
            addressed by its own placement name. Above ``1`` the same joint data
            type is written by more than one stream, so placing the groups apart
            gives one robot two joint-writing processes.
        video_count: Number of RGB camera streams to log per recording.  A
            value of ``0`` disables video entirely.
        image_width: Horizontal resolution of each camera frame in pixels.
            ``None`` when ``video_count`` is ``0``.
        image_height: Vertical resolution of each camera frame in pixels.
            ``None`` when ``video_count`` is ``0``.
        kill_daemon_between_tests: When ``True``, the daemon process is
            stopped and restarted before each test case so every case begins
            from a clean process state.  Set to ``False`` to keep the daemon
            alive across cases (faster but tests share daemon state).
        storage_state_action: Controls how the SQLite state database and
            recordings folder are handled between test cases.  ``"preserve"``
            leaves both untouched for post-mortem inspection; ``"empty"``
            truncates DB tables and clears recordings folder contents but keeps
            them on disk; ``"delete"`` removes both entirely.
        stop_method: Method used to stop the daemon process.  ``"cli"``
            (default) invokes the CLI stop command; ``"sigterm"`` sends SIGTERM
            directly; ``"sigkill"`` terminates immediately without giving the
            daemon a chance to flush buffers.
        preserve_artifacts_per_test: When ``True``, the recordings directory
            and DB file are copied to a timestamped artifact directory before
            cleanup so they can be inspected after the test run.  Implies
            ``storage_state_action == "preserve"`` for the active env paths.
        context_duration_mode: Controls per-recording duration within each
            context.  ``"fixed"`` makes every recording last exactly
            ``duration_sec`` seconds.  ``"variable"`` randomises the duration
            for each recording within a range around ``duration_sec``, which
            exercises the daemon's handling of recordings with unequal lengths.
        wait: When ``True``, recording contexts block until the daemon
            acknowledges the stop-recording call before returning.  When
            ``False`` the stop call is fire-and-forget, which exercises the
            daemon's ability to process uploads without an explicit client
            wait.  Cloud tests expand over both values; offline tests always
            use ``False``.
        joint_fps: Frame rate in Hz for joint data producers.  Determines the
            total expected joint frame count as ``joint_fps * duration_sec``.
        video_fps: Frame rate in Hz for video/camera producers.  Determines the
            total expected video frame count as ``video_fps * duration_sec``.
            Ignored when ``video_count`` is ``0``.
        random_phase: Controls *where* the explicitly-supplied timestamps sit,
            never *when* the producer runs — that is ``producer_pacing``'s
            business, and the two are independent.  ``False``
            (default) puts frames on an exact ``timestamp_start_s +
            frame_index / fps`` grid.  ``True`` offsets each frame by a
            deterministic pseudo-random amount within
            ``random_phase_jitter_window(fps)``, so the daemon sees
            non-uniformly spaced timestamps.
        skip: When ``True``, the case is skipped at collection time instead of
            being executed.  Lets unstable or not-yet-validated workloads stay
            in the suite (documented and discoverable) without running.  A
            batch with ``skip=True`` forces every case to skip regardless of
            this per-case value.
        video_codec: When set (e.g. ``"h264_medium"``), the case selects that
            global video codec via ``nc.set_video_encoding_options`` before
            recording, so RGB cameras upload a single lossy CRF-23 video and the
            lossless archive is dropped.  ``None`` keeps the default
            lossless+lossy behaviour.
        depth_count: Number of depth camera streams to log per recording. A
            value of ``0`` disables depth entirely. Depth cameras are
            independent streams from RGB cameras (distinct trace identities,
            see :func:`constants.depth_camera_names`) but reuse ``image_width``,
            ``image_height``, ``video_fps``, and the RGB video timestamp
            schedule — depth intentionally adds no separate resolution or
            frame-rate knobs.
        depth_mode: The NumPy dtype depth frames are logged as — ``"float16"``
            or ``"float32"``. Ignored when ``depth_count`` is ``0``.
        video_detail: Pixel content of the synthetic camera frames — realistic
            costs full compression/encode, flat is a cheap solid fill; frame
            identity is embedded either way.
        producer_pacing: When each stream offers its next frame. Every value
            works under every producer lifetime; none of them decide what
            timestamp a frame carries (see ``random_phase``).
        recording_control: Who opens and closes each window. ``"local"``
            (default) calls the SDK from the test process; ``"remote"`` calls
            the backend's own endpoints, so every process learns about the
            window over the notification stream. Needs the network, and a
            producer that outlives its recordings. ``"split"`` keeps both calls
            in the SDK but puts them in different processes: this process starts
            every window and a peer makes the stop call, which it can only do
            because the start was announced to it. The stop has to come back
            round the same way before this process drains, so a recording ends
            only if the notification stream works in both directions. Needs the
            network, and ``parallel_contexts=1`` to spawn the peer in.

    Note:
        ``mode="staggered"`` and ``context_duration_mode="variable"``:
        Both are computed from the base ``duration_sec`` separately (rather than
        stagger being a function of the calculated duration variation).
        With a 50 % stagger and a 75 % duration floor, context 1's timestamp
        start is guaranteed to fall before context 0's timestamp end.
    """

    CHANNELS: ClassVar[ProducerChannels] = PRODUCER_SYNCHRONOUS

    duration_sec: int = 5
    parallel_contexts: int = 1
    recording_count: int = 1
    mode: str = MODE_SEQUENTIAL
    joint_count: int = 10
    video_count: int = 0
    image_width: int | None = None
    image_height: int | None = None
    kill_daemon_between_tests: bool = True
    storage_state_action: StorageStateAction = STORAGE_STATE_EMPTY
    daemon_log_action: LogAction = LOG_PRESERVE
    stop_method: StopMethod = STOP_METHOD_CLI
    preserve_artifacts_per_test: bool = False
    context_duration_mode: str = DURATION_MODE_FIXED
    wait: bool = False
    joint_fps: int = 60
    video_fps: int = 60
    random_phase: bool = False
    skip: bool = False
    video_codec: str | None = None
    depth_count: int = 0
    depth_mode: DepthMode = "float32"
    video_detail: VideoDetail = DETAIL_REALISTIC
    producer_pacing: ProducerPacing = PACING_DEADLINE
    producer_process_streams: tuple[tuple[str, ...], ...] = ()
    joint_process_groups: int = 1
    recording_control: RecordingControl = CONTROL_LOCAL

    def __post_init__(self) -> None:
        """Fill in the class's default placement, then reject parameter
        combinations this case's shape cannot run."""
        if not self.producer_process_streams:
            object.__setattr__(
                self, "producer_process_streams", self.default_process_streams()
            )
        problem = _unsupported_combination(self)
        if problem is not None:
            raise ValueError(problem)

    def default_process_streams(self) -> tuple[tuple[str, ...], ...]:
        """The placement this case's class picks for its shape; empty keeps every
        stream in the recording-owning process."""
        return ()

    @property
    def producer_channels(self) -> ProducerChannels:
        """The producer model this case runs under, fixed by its class.

        Not a field: a variant *is* its producer model, so it cannot contradict itself.
        """
        return type(self).CHANNELS

    @property
    def has_video(self) -> bool:
        """Return True when this case logs at least one camera stream."""
        return self.video_count > 0

    @property
    def has_depth(self) -> bool:
        """Return True when this case logs at least one depth camera stream."""
        return self.depth_count > 0

    @property
    def lossy_only(self) -> bool:
        """Return True when the case drops the lossless archive for RGB video."""
        return self.video_codec == "h264_medium"

    @property
    def expected_joint_frames(self) -> int:
        """Return expected joint frames: ``joint_fps * duration_sec``."""
        return self.joint_fps * self.duration_sec

    @property
    def expected_video_frames(self) -> int:
        """Return expected video frames: ``video_fps * duration_sec``."""
        return self.video_fps * self.duration_sec

    @property
    def expected_depth_frames(self) -> int:
        """Return expected depth frames per camera.

        Depth cameras reuse the RGB video timestamp schedule (see
        :attr:`depth_count`), so this equals :attr:`expected_video_frames`.
        """
        return self.expected_video_frames

    @property
    def recordings_per_context(self) -> int:
        """Return the base recordings assigned per context.

        Computed as ``recording_count // parallel_contexts``.
        Any remainder is distributed to the first contexts when specs are built.
        """
        return self.recording_count // self.parallel_contexts


@dataclass(frozen=True)
class Synchronous(DataDaemonTestCase):
    """Logs every data type from one thread; no frame is in flight at a boundary."""

    CHANNELS: ClassVar[ProducerChannels] = PRODUCER_SYNCHRONOUS


@dataclass(frozen=True)
class PerThread(DataDaemonTestCase):
    """One thread per stream, running the whole context lifetime — mid-loop at
    every recording boundary."""

    CHANNELS: ClassVar[ProducerChannels] = PRODUCER_PER_THREAD


@dataclass(frozen=True)
class SeparateProcessRecordingControl(PerThread):
    """:class:`PerThread`, with every stream produced in one OS process of its
    own, so the original process only ever controls the recording — starting,
    stopping, or cancelling it — and never logs into it.

    A robot driver logging alongside an application that controls the windows:
    every trace is written by a process that learns its window second-hand.
    Subclasses keep that and split the producing further.
    """

    CHANNELS: ClassVar[ProducerChannels] = PRODUCER_MULTI_PROCESS

    def joint_process_streams(self) -> tuple[tuple[str, ...], ...]:
        """One entry per joint group, each owning every kind of its own joints.

        A group is a limb rather than a data type. A case with no joints yields
        none.
        """
        return tuple(
            (group_name,)
            for group_name, _ in joint_name_groups(
                joint_names_for_count(self.joint_count), self.joint_process_groups
            )
        )

    def camera_process_streams(self) -> tuple[tuple[str, ...], ...]:
        """One entry per camera device, named by channel so no two children share.

        A camera here is a *device*: index ``i``'s RGB and depth streams are the
        two outputs of one RGBD sensor, so they belong together the way one
        driver reads both. Indexes past the shorter count are a device with only
        that one output, so an unequal ``video_count``/``depth_count`` still
        places every stream.
        """
        return tuple(
            tuple(name for name in pair if name is not None)
            for pair in zip_longest(
                camera_names(self.video_count),
                depth_camera_names(self.depth_count),
            )
        )

    def default_process_streams(self) -> tuple[tuple[str, ...], ...]:
        """Every device in one child: the split is between bracketing a
        recording and producing into it, not between producers."""
        together = tuple(
            name
            for entry in (*self.joint_process_streams(), *self.camera_process_streams())
            for name in entry
        )
        return (together,) if together else ()


@dataclass(frozen=True)
class ProcessPerCamera(SeparateProcessRecordingControl):
    """:class:`SeparateProcessRecordingControl`, with the producing split per camera,
    so ``max(video_count, depth_count)`` decides how many camera children there
    are and the joints keep a child of their own."""

    def default_process_streams(self) -> tuple[tuple[str, ...], ...]:
        """A child per joint group, then a child per camera device."""
        return (*self.joint_process_streams(), *self.camera_process_streams())


@dataclass(frozen=True)
class ProcessPerLimbPerCamera(ProcessPerCamera):
    """:class:`ProcessPerCamera`, with the joints split across a child per limb
    instead of one child holding all of them.

    Each of the ``joint_process_groups`` limbs writes every kind of the joints it
    owns, so the same data type reaches the daemon from two processes at once —
    which no camera split does, a camera child owning a channel of its own.
    """

    joint_process_groups: int = 2


@dataclass(frozen=True)
class DataDaemonTestBatch:
    """A named collection of test cases sharing common infrastructure parameters.

    Groups ``DataDaemonTestCase`` instances that should run under the same
    daemon lifecycle, storage, and artifact settings.  The batch-level params
    (``kill_daemon_between_tests``, ``storage_state_action``,
    ``daemon_log_action``, ``preserve_artifacts_per_test``, ``stop_method``)
    are propagated to every case via :meth:`as_cases`.

    Attributes:
        cases: The individual test case workload definitions.
        kill_daemon_between_tests: Propagated to every case; see
            ``DataDaemonTestCase.kill_daemon_between_tests``.
        storage_state_action: Propagated to every case; see
            ``DataDaemonTestCase.storage_state_action``.
        preserve_artifacts_per_test: Propagated to every case; see
            ``DataDaemonTestCase.preserve_artifacts_per_test``.
        stop_method: Propagated to every case; see
            ``DataDaemonTestCase.stop_method``.
        skip: When ``True``, every case in the batch is skipped at collection
            time.  When ``False`` (default), each case keeps its own per-case
            ``skip`` value, so individual cases can still opt out.
        producer_variant: Subclass applied to every case when set; ``None``
            (default) keeps each case's own class.
        producer_pacing: Override applied to every case when set; a case whose
            shape cannot honour it raises ``ValueError`` at construction.
    """

    cases: tuple[DataDaemonTestCase, ...]
    kill_daemon_between_tests: bool = True
    storage_state_action: StorageStateAction = STORAGE_STATE_DELETE
    daemon_log_action: LogAction = LOG_PRESERVE
    preserve_artifacts_per_test: bool = False
    stop_method: StopMethod = STOP_METHOD_CLI
    skip: bool = False
    producer_variant: type[DataDaemonTestCase] | None = None
    producer_pacing: ProducerPacing | None = None

    def as_cases(self) -> list[DataDaemonTestCase]:
        """Return cases with batch-level infrastructure params applied."""
        batch_overrides = {
            "kill_daemon_between_tests": self.kill_daemon_between_tests,
            "storage_state_action": self.storage_state_action,
            "daemon_log_action": self.daemon_log_action,
            "preserve_artifacts_per_test": self.preserve_artifacts_per_test,
            "stop_method": self.stop_method,
        }
        if self.skip:
            batch_overrides["skip"] = True
        if self.producer_pacing is not None:
            batch_overrides["producer_pacing"] = self.producer_pacing
        return [
            (self.producer_variant or type(c))(**{
                **{
                    f.name: getattr(c, f.name)
                    for f in fields(c)
                    if f.name not in _BATCH_PARAMS
                },
                **batch_overrides,
            })
            for c in self.cases
        ]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _placement_id(case: DataDaemonTestCase) -> str:
    """Name which streams *case* moved out, or "" when it took its class's default."""
    if case.producer_process_streams == case.default_process_streams():
        return ""
    moved = "-".join(
        name.replace("_", "")
        for group in case.producer_process_streams
        for name in group
    )
    return f"{len(case.producer_process_streams)}proc-{moved}"


def case_id(case: DataDaemonTestCase) -> str:
    """Generate a short human-readable ID for a test case.

    The variant class leads; every other value is only named when it differs
    from *this case's own class* default.
    """
    default = type(case)
    mode_short = "seq" if case.mode == MODE_SEQUENTIAL else "stag"
    parts = [
        *([] if default is DataDaemonTestCase else [default.__name__]),
        f"{case.duration_sec}s",
        f"{case.recording_count}recs",
        *(["variable"] if case.context_duration_mode == DURATION_MODE_VARIABLE else []),
    ]
    if case.parallel_contexts > default.parallel_contexts:
        parts.append(f"{case.parallel_contexts}ctx")
        parts.append(mode_short)
    parts.append(f"{case.joint_count}joints")
    if case.joint_process_groups != default.joint_process_groups:
        parts.append(f"{case.joint_process_groups}jointgroups")
    if case.joint_fps != default.joint_fps:
        parts.append(f"{case.joint_fps}hz")
    if case.has_video:
        parts.append(f"{case.video_count}cam")
        parts.append(f"{case.image_width}x{case.image_height}")
        if case.video_fps != default.video_fps:
            parts.append(f"{case.video_fps}hz")
        if case.video_codec is not None:
            parts.append(case.video_codec)
        if case.video_detail != default.video_detail:
            parts.append(f"{case.video_detail}frames")
    if case.has_depth:
        parts.append(f"{case.depth_count}depth")
        parts.append(case.depth_mode)
    if placement := _placement_id(case):
        parts.append(placement)
    if case.producer_pacing != default.producer_pacing:
        parts.append(case.producer_pacing)
    if case.recording_control != default.recording_control:
        parts.append(f"{case.recording_control}-control")
    if case.random_phase:
        parts.append("random-phase")
    if case.wait:
        parts.append("wait")
    return "-".join(parts)


def case_ids(cases: Sequence[DataDaemonTestCase]) -> list[str]:
    """Generate stable pytest IDs and hyphen-suffix duplicates.

    Pytest auto-suffixes duplicate IDs without a separator. This helper keeps
    IDs readable by explicitly generating ``base-0``, ``base-1``, ... for
    duplicate base IDs while leaving unique IDs unchanged.
    """
    base_ids = [case_id(case) for case in cases]

    totals: dict[str, int] = {}
    for base in base_ids:
        totals[base] = totals.get(base, 0) + 1

    seen: dict[str, int] = {}
    resolved_ids: list[str] = []
    for base in base_ids:
        if totals[base] == 1:
            resolved_ids.append(base)
            continue
        suffix = seen.get(base, 0)
        resolved_ids.append(f"{base}-{suffix}")
        seen[base] = suffix + 1

    return resolved_ids


def has_configured_org() -> bool:
    """Check whether an organization is configured via env or saved config."""
    if os.environ.get("NEURACORE_ORG_ID"):
        return True
    try:
        return bool(get_config_manager().config.current_org_id)
    except Exception:  # noqa: BLE001
        return False


def generate_joint_values(
    frame_index: int,
    fps: int,
    joint_names: list[str],
) -> dict[str, float]:
    """Generate deterministic sinusoidal joint values for a frame."""
    timestamp = frame_index / fps
    return {
        joint_name: math.sin(timestamp * (0.5 + (index * 0.25)))
        for index, joint_name in enumerate(joint_names)
    }


def case_timeout_seconds(case: DataDaemonTestCase) -> float:
    """Compute a reasonable timeout for waiting on a case to complete."""
    image_pixels = 0
    if (
        (case.has_video or case.has_depth)
        and case.image_width is not None
        and case.image_height is not None
    ):
        image_stream_count = case.video_count + case.depth_count
        image_pixels = image_stream_count * case.image_width * case.image_height
    workload_units = (
        case.recording_count
        * case.duration_sec
        * (case.joint_count + max(1, image_pixels // 4096))
    )
    timeout_s = BASE_DATASET_READY_TIMEOUT_S + (workload_units * 0.2)
    if case.context_duration_mode == DURATION_MODE_VARIABLE:
        timeout_s *= 1.25
    return min(MAX_DATASET_READY_TIMEOUT_S, timeout_s)


# ---------------------------------------------------------------------------
# Analysis / reporting
# ---------------------------------------------------------------------------


MIN_SAMPLES_FOR_PERCENTILE = 100
"""Below this sample count a percentile is not reported — it would just be the max."""


def _percentile_text(stats: dict[str, float], quantile: float) -> str:
    """Render one percentile column, or empty when it would say nothing."""
    count = int(stats["count"])
    label = f"p{int(quantile * 100)}"
    if count < MIN_SAMPLES_FOR_PERCENTILE:
        return ""
    if not any(stats.get(key, 0.0) for key in BUCKET_KEYS):
        return f"{label}=n/a"
    bound = percentile_upper_bound(stats, quantile)
    if bound is None:
        return f"{label}>{LATENCY_BUCKETS_S[-1]:.3f}s"
    # A bucket edge, not an interpolated value — "<=" keeps that honest.
    return f"{label}<={min(bound, stats['max']):.3f}s"


def _format_timer_stats_line(label: str, stats: dict[str, float]) -> str:
    """Format a timer stats line for analysis output."""
    count = int(stats["count"])
    avg = stats["total"] / count if count > 0 else 0.0
    p95_text = _percentile_text(stats, 0.95)
    p99_text = _percentile_text(stats, 0.99)
    return (
        f"    {label:<42}  {count:3}x"
        f"  avg={avg:.3f}s  {p95_text:<12}  {p99_text:<12}"
        f"  max={stats['max']:.3f}s"
    )


def log_run_analysis(
    *,
    case: DataDaemonTestCase,
    results: list[ContextResult],
    title: str | None = None,
    status: str | None = None,
    note: str | None = None,
    extra_sections: list[str] | None = None,
    include_in_session_summary: bool = True,
    disk_durations: dict[str, float] | None = None,
    label_prefix: str | None = None,
    test_wall_s: float | None = None,
) -> str:
    """Log a detailed analysis of a test run for diagnostics."""

    display_case_id = (
        f"{label_prefix}-{case_id(case)}" if label_prefix else case_id(case)
    )
    separator = "=" * 64
    report_title = title or f"Run analysis: {display_case_id}"
    lines = [separator, report_title, separator]

    if status is not None:
        lines.append(f"  Analysis status: {status}")
    if note is not None:
        lines.append(f"  {note}")

    lines += [
        f"  Case:          {case.recording_count} recordings x"
        f" {case.duration_sec}s  joints@{case.joint_fps}Hz",
        f"                 {case.joint_count} joints,"
        f" {case.producer_channels} channels",
    ]
    if case.has_video:
        lines.append(
            f"                 {case.video_count} camera(s)"
            f" @ {case.image_width}x{case.image_height}  video@{case.video_fps}Hz"
        )
    if case.has_depth:
        lines.append(
            f"                 {case.depth_count} depth camera(s)"
            f" @ {case.image_width}x{case.image_height} ({case.depth_mode})"
        )
    lines.append(
        f"  Total joint frames:  {case.recording_count * case.expected_joint_frames}"
    )
    if case.has_video:
        total_video_frames = case.recording_count * case.expected_video_frames
        lines.append(f"  Total video frames:  {total_video_frames}")
    if case.has_depth:
        total_depth_frames = case.recording_count * case.expected_depth_frames
        lines.append(f"  Total depth frames:  {total_depth_frames}")

    if test_wall_s is not None:
        lines.append(f"\n  Test wall time:  {test_wall_s:.1f}s")

    if results:
        lines.append(f"\n  Dataset: {results[0].dataset_name!r}")
        lines.append("\n  Context wall times:")
        for result in sorted(results, key=lambda result: result.context_index):
            wall_s = result.wall_stopped_at - (result.wall_started_at or 0.0)
            recordings_per_context = len(result.recording_ids)
            avg_per_recording = (
                wall_s / recordings_per_context if recordings_per_context else 0.0
            )
            lines.append(
                f"    ctx[{result.context_index}]: {wall_s:.1f}s total,"
                f" {avg_per_recording:.1f}s avg per recording"
            )
    else:
        lines.append(
            "\n  Context wall times: unavailable "
            "(run aborted before contexts completed)"
        )

    session_labels = sorted(Timer._stats.keys())
    if session_labels:
        lines.append("\n  Timer stats  (n / avg / max):")
        for label in session_labels:
            stats = Timer._stats[label]
            count = int(stats["count"])
            stats["total"] / count if count > 0 else 0.0
            lines.append(_format_timer_stats_line(label, Timer._stats[label]))

    if disk_durations:
        avg_duration_s = sum(disk_durations.values()) / len(disk_durations)
        lines.append(
            f"\n  Disk recording durations ({len(disk_durations)} recording(s)):"
        )
        for rec_id, dur_s in sorted(disk_durations.items()):
            lines.append(f"    {rec_id}: {dur_s:.3f}s")
        lines.append(f"    avg: {avg_duration_s:.3f}s")

    if extra_sections:
        lines.extend(extra_sections)

    if include_in_session_summary:
        SESSION_RUNS.append({
            "case_id": display_case_id,
            "dataset_name": results[0].dataset_name if results else None,
            "test_wall_s": test_wall_s,
            "timer_stats": {
                label: dict(Timer._stats[label])
                for label in session_labels
                if not label.startswith("stop_daemon")
            },
            "context_results": [
                {
                    "context_index": result.context_index,
                    "wall_s": result.wall_stopped_at - (result.wall_started_at or 0.0),
                }
                for result in results
            ],
            **(
                {
                    "disk_durations": dict(disk_durations),
                    "avg_disk_duration_s": sum(disk_durations.values())
                    / len(disk_durations),
                }
                if disk_durations
                else {}
            ),
        })

    lines.append(separator)
    report = "\n".join(lines)
    logger.info("\n%s", report)
    return report
