"""Context-spec interpretation and recording worker logic.

Translates a ``DataDaemonTestCase`` into per-context worker specs, executes
the recording workload, and provides the context-mode assertion.
Configuration dataclasses and the matrix builder live in
``matrix_test_configs.py``; per-suite case lists live in ``test_cases.py``.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
import psutil

import neuracore as nc
from neuracore.core.streaming.recording_state_manager import RecordingStateManager
from tests.integration.platform.data_daemon.shared.assertions import assert_context_mode
from tests.integration.platform.data_daemon.shared.auth import ensure_login
from tests.integration.platform.data_daemon.shared.process_control import (
    MAX_TIME_TO_LOG_S,
    Timer,
    assert_on_schedule,
    init_worker_logging,
    relayed_worker_logs,
    surface_worker_errors,
)
from tests.integration.platform.data_daemon.shared.producer_diagnostics import (
    ProducerDiagnosticHistory,
    ProducerHeartbeatRegistry,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestCase,
    camera_names,
    case_id,
    generate_joint_values,
    joint_names_for_count,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    DATASET_POLL_INTERVAL_S,
    DURATION_MODE_VARIABLE,
    DURATION_VARIABLE_MAX_FACTOR,
    DURATION_VARIABLE_MIN_FACTOR,
    FRAME_BYTE_LENGTH,
    FRAME_COLOR_CHANNELS,
    FRAME_DEFAULT_FILL_VALUE,
    FRAME_GRID_SIZE,
    FRAME_HALF_DIVISOR,
    FRAME_MAX_COLOR_VALUE,
    MAX_TIME_TO_START_S,
    MODE_STAGGERED,
    PRODUCER_PER_THREAD,
    SCHEDULER_TOLERANCE_S,
    STOP_RECORDING_NO_WAIT_SLA_S,
    STOP_RECORDING_OVERHEAD_PER_SEC,
    STOP_RECORDING_UPLOAD_SLA_PER_JOINT_SAMPLE_S,
    STOP_RECORDING_UPLOAD_SLA_PER_VIDEO_PIXEL_S,
    TIMESTAMP_MODE_REAL,
    TIMESTAMP_MODE_STOCHASTIC,
    stochastic_jitter_window,
)

logger = logging.getLogger(__name__)

CONTEXT_DURATION_RANDOM = random.Random(0)
STOCHASTIC_TIMESTAMP_RANDOM = random.Random(1)


def encode_frame_number(frame_num: int, width: int, height: int) -> np.ndarray:
    """Encode a frame number into the pixel data of a synthetic video frame.

    The 16-byte big-endian representation of ``frame_num`` is written into the
    top-left 4x4 grid of the image. For each pixel at ``(row, col)`` in that
    grid the byte value is mapped to the RGB channels as follows:

    - Red channel = ``byte_value``
    - Green channel = ``FRAME_MAX_COLOR_VALUE - byte_value``
    - Blue channel = ``byte_value // FRAME_HALF_DIVISOR``

    The remaining pixels are filled with :data:`FRAME_DEFAULT_FILL_VALUE`.

    Args:
        frame_num: The frame number to embed. Must fit in 16 bytes (i.e.
            less than ``2 ** 128``).
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        A NumPy array with shape ``(height, width, 3)`` and dtype ``uint8``.
    """
    img = np.zeros((height, width, FRAME_COLOR_CHANNELS), dtype=np.uint8)
    img.fill(FRAME_DEFAULT_FILL_VALUE)

    frame_bytes = frame_num.to_bytes(FRAME_BYTE_LENGTH, byteorder="big")

    for row in range(FRAME_GRID_SIZE):
        for col in range(FRAME_GRID_SIZE):
            idx = row * FRAME_GRID_SIZE + col
            if idx < len(frame_bytes):
                pixel_value = frame_bytes[idx]
                img[row, col, 0] = pixel_value
                img[row, col, 1] = FRAME_MAX_COLOR_VALUE - pixel_value
                img[row, col, 2] = pixel_value // FRAME_HALF_DIVISOR

    return img


@dataclass(frozen=True, slots=True)
class RecordingExpectedTimestamps:
    """Expected timestamps per trace for one recording, keyed by semantic trace name.

    Produced during the recording loop (once the recording key is known) and
    consumed by :func:`~disk_helpers.assert_disk_recording_properties`
    to verify on-disk trace.json files match the manually-supplied timestamps
    that were logged.

    Attributes:
        by_trace: Maps semantic trace key (e.g. ``"JOINT_POSITIONS"``,
            ``"camera_0"``) to the ordered list of expected timestamps for
            that trace within this recording.
        by_trace_fps: Maps the same semantic trace key to the producer fps for
            that trace, so the stochastic assertion can size its jitter window
            from the case's frame rate.
    """

    by_trace: dict[str, list[float]]
    by_trace_fps: dict[str, int]


@dataclass(frozen=True, slots=True)
class ContextExpectedTimestamps:
    """Expected timestamps for all recordings produced by one context worker.

    Attributes:
        by_recording: Maps the on-disk recording directory name to its
            :class:`RecordingExpectedTimestamps`. The directory name is the
            integer ``recording_index`` (as a string) under the Rust daemon, or
            the cloud ``recording_id`` under the legacy daemon.
    """

    by_recording: dict[str, RecordingExpectedTimestamps]


@dataclass(frozen=True, slots=True)
class ContextCaseSpec:
    duration_sec: int
    joint_count: int
    producer_channels: str
    video_count: int
    image_width: int | None
    image_height: int | None
    joint_fps: int
    video_fps: int
    wait: bool
    timestamp_mode: str

    @property
    def stop_recording_sla_s(self) -> float:
        """Seconds allowed for the ``nc.stop_recording`` call.

        ``wait=False`` is fire-and-forget — the call never blocks on the
        upload pipeline — so it gets a flat constant. ``wait=True`` blocks
        until every trace has uploaded, so its budget is the sum of the
        joint-data and video-data upload costs: total joint samples
        (``duration_sec * joint_count * joint_fps``) and total video pixels
        (``duration_sec * video_fps * video_count * image_width *
        image_height``), each times an observed per-unit upload cost. The
        budget is floored at the duration-based overhead so short or
        low-volume recordings keep a sane minimum.
        """
        if not self.wait:
            return STOP_RECORDING_NO_WAIT_SLA_S
        duration_floor = self.duration_sec * STOP_RECORDING_OVERHEAD_PER_SEC
        joint_budget = (
            self.duration_sec
            * self.joint_count
            * self.joint_fps
            * STOP_RECORDING_UPLOAD_SLA_PER_JOINT_SAMPLE_S
        )
        video_budget = 0.0
        if self.video_count and self.image_width and self.image_height:
            video_budget = (
                self.duration_sec
                * self.video_fps
                * self.video_count
                * self.image_width
                * self.image_height
                * STOP_RECORDING_UPLOAD_SLA_PER_VIDEO_PIXEL_S
            )
        return max(duration_floor, joint_budget + video_budget)


@dataclass(frozen=True, slots=True)
class ContextResult:
    """Per-context result from a completed recording workload.

    Produced by :func:`context_worker` and consumed by assertion helpers
    and verification functions throughout the test suite.

    A recording is addressed by:

    - ``recording_ids`` — the cloud ``recording_id`` (TEXT) for each recording.
      These are what cloud verification (``verify_cloud_results``) matches
      against the dataset's ``recording.id``. Under the legacy daemon
      ``nc.start_recording()`` returns this directly. Under the Rust daemon the
      daemon mints it asynchronously, so an entry may be an empty string until
      the test resolves it (via ``resolve_cloud_recording_ids``) once online.

    The remaining fields apply only under the Rust daemon (the daemon owns
    recording identity); they are left empty under the legacy daemon, which uses
    ``recording_ids`` for every correlation:

    - ``recording_indexes`` — the daemon-assigned local INTEGER
      ``recording_index`` for each recording, resolved from the source DB.
      These are the on-disk directory names and the daemon-DB join key.
    - ``source`` is the ``(robot_id, robot_instance)`` identity used to correlate
      a worker's recordings to daemon-minted ``recording_index`` values without
      relying on the local handle.
    """

    dataset_name: str
    recording_ids: list[str]
    robot_name: str
    joint_names: list[str]
    camera_names: list[str]
    joint_frame_count: int
    video_frame_count: int
    joint_fps: int
    video_fps: int
    duration_sec: int
    timestamp_start_s: float
    timestamp_end_s: float
    marker_names: list[str]
    has_video: bool
    context_index: int
    wall_started_at: float | None
    wall_stopped_at: float
    timestamp_mode: str
    expected_timestamps: ContextExpectedTimestamps | None = None
    timer_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    recording_indexes: list[int] = field(default_factory=list)
    source: tuple[str, int] = ("", 0)


@dataclass(frozen=True, slots=True)
class ContextSpec:
    case: ContextCaseSpec
    context_index: int
    robot_name: str
    dataset_name: str
    recordings_per_context: int
    expected_joint_frames: int
    expected_video_frames: int
    timestamp_start_s: float
    timestamp_end_s: float
    start_delay_s: float
    assert_deadline: bool = False


def build_context_specs(
    case: DataDaemonTestCase,
    dataset_name: str | None = None,
    assert_deadline: bool = False,
) -> list[ContextSpec]:
    """Build per-context worker specs for a matrix case."""
    specs: list[ContextSpec] = []
    timestamp_stagger_s = case.duration_sec / 2.0
    wall_stagger_s = 0.5
    base_recordings_per_context = case.recording_count // case.parallel_contexts
    recording_remainder = case.recording_count % case.parallel_contexts
    shared_dataset_name = (
        dataset_name or f"testing_dataset_{case_id(case)}_{uuid.uuid4().hex[:6]}"
    )

    for context_index in range(case.parallel_contexts):
        timestamp_start_s = 0.0
        start_delay_s = 0.0
        if context_index > 0 and case.mode == MODE_STAGGERED:
            timestamp_start_s = float(timestamp_stagger_s * context_index)
            start_delay_s = wall_stagger_s * context_index

        if case.context_duration_mode == DURATION_MODE_VARIABLE:
            context_duration_sec = max(
                1,
                min(
                    int(
                        case.duration_sec
                        * CONTEXT_DURATION_RANDOM.uniform(
                            DURATION_VARIABLE_MIN_FACTOR, DURATION_VARIABLE_MAX_FACTOR
                        )
                    ),
                    RecordingStateManager.MAX_RECORDING_DURATION_S,
                ),
            )
        else:
            context_duration_sec = case.duration_sec

        recordings_for_context = base_recordings_per_context + (
            1 if context_index < recording_remainder else 0
        )

        specs.append(
            ContextSpec(
                case=ContextCaseSpec(
                    duration_sec=context_duration_sec,
                    joint_count=case.joint_count,
                    producer_channels=case.producer_channels,
                    video_count=case.video_count,
                    image_width=case.image_width,
                    image_height=case.image_height,
                    joint_fps=case.joint_fps,
                    video_fps=case.video_fps,
                    wait=case.wait,
                    timestamp_mode=case.timestamp_mode,
                ),
                context_index=context_index,
                robot_name=f"matrix_robot_{uuid.uuid4().hex[:10]}",
                dataset_name=shared_dataset_name,
                recordings_per_context=recordings_for_context,
                expected_joint_frames=case.joint_fps * context_duration_sec,
                expected_video_frames=case.video_fps * context_duration_sec,
                timestamp_start_s=timestamp_start_s,
                timestamp_end_s=(
                    timestamp_start_s + context_duration_sec * recordings_for_context
                ),
                start_delay_s=start_delay_s,
                assert_deadline=assert_deadline,
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Recording worker functions
# ---------------------------------------------------------------------------


def _cleanup_test_worker_robot(robot: object | None) -> None:
    """Clean up temp dirs and recording context on a worker robot."""
    if robot is None:
        return

    temp_dir = getattr(robot, "_temp_dir", None)
    if temp_dir is not None:
        try:
            temp_dir.cleanup()
        except Exception:  # noqa: BLE001
            logger.warning("Failed to cleanup worker robot temp dir", exc_info=True)
        finally:
            robot._temp_dir = None

    if hasattr(robot, "_daemon_recording_context"):
        robot._daemon_recording_context = None


def get_jitter(use_stochastic_timestamps: bool, fps: int) -> float:
    if use_stochastic_timestamps:
        window = stochastic_jitter_window(fps)
        return STOCHASTIC_TIMESTAMP_RANDOM.uniform(-window, window)
    return 0.0


PRODUCER_TIMING_REPORT_INTERVAL_S = 60.0
PRODUCER_TIMING_NEAR_LIMIT_FACTOR = 0.8
PRODUCER_RESOURCE_SAMPLE_INTERVAL_S = 1.0
PRODUCER_TIMING_METRICS = (
    "joint_lateness",
    "video_lateness",
    "pre_wait_lateness",
    "sleep_overshoot",
    "joint_work",
    "frame_build",
    "rgb_log",
)


@dataclass(slots=True)
class ProducerResourceSample:
    """One non-blocking resource snapshot for a producer worker process."""

    interval_s: float = 0.0
    process_cpu_pct: float = 0.0
    system_cpu_pct: float = 0.0
    load_per_cpu_pct: float = 0.0
    rss_mb: float = 0.0
    thread_count: int = 0
    voluntary_ctx_switches: int = 0
    involuntary_ctx_switches: int = 0
    disk_read_mbps: float = 0.0
    disk_write_mbps: float = 0.0


@dataclass(slots=True)
class ProducerTimingDiagnostics:
    """Rate-limited timing telemetry for a synchronous producer recording."""

    context_index: int
    recording_index: int
    started_at: float = field(default_factory=time.perf_counter)
    window_started_at: float = field(default_factory=time.perf_counter)
    window_events: int = 0
    near_limit_logged: bool = False
    window_max: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(PRODUCER_TIMING_METRICS, 0.0)
    )
    overall_max: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(PRODUCER_TIMING_METRICS, 0.0)
    )
    last_duration: dict[str, float] = field(
        default_factory=lambda: {
            "joint_work": 0.0,
            "frame_build": 0.0,
            "rgb_log": 0.0,
        }
    )

    process: psutil.Process = field(default_factory=psutil.Process, repr=False)
    resource_last_at: float = field(default_factory=time.perf_counter)
    resource_last_cpu_s: float = 0.0
    resource_last_ctx_voluntary: int = 0
    resource_last_ctx_involuntary: int = 0
    resource_last_disk_read_bytes: int = 0
    resource_last_disk_write_bytes: int = 0
    resource_latest: ProducerResourceSample = field(
        default_factory=ProducerResourceSample
    )
    resource_window_max: dict[str, float] = field(
        default_factory=lambda: {
            "process_cpu_pct": 0.0,
            "system_cpu_pct": 0.0,
            "load_per_cpu_pct": 0.0,
            "rss_mb": 0.0,
            "sample_gap_s": 0.0,
            "disk_read_mbps": 0.0,
            "disk_write_mbps": 0.0,
        }
    )

    def __post_init__(self) -> None:
        cpu_times = self.process.cpu_times()
        self.resource_last_cpu_s = cpu_times.user + cpu_times.system
        ctx_switches = self.process.num_ctx_switches()
        self.resource_last_ctx_voluntary = ctx_switches.voluntary
        self.resource_last_ctx_involuntary = ctx_switches.involuntary
        disk = psutil.disk_io_counters()
        if disk is not None:
            self.resource_last_disk_read_bytes = disk.read_bytes
            self.resource_last_disk_write_bytes = disk.write_bytes
        # Prime the non-blocking system-CPU baseline. Subsequent calls
        # report utilization over the interval since this one.
        psutil.cpu_percent(interval=None)

    def _sample_resources(self, *, force: bool = False) -> None:
        """Refresh resource telemetry without sleeping."""
        now = time.perf_counter()
        interval_s = now - self.resource_last_at
        if not force and interval_s < PRODUCER_RESOURCE_SAMPLE_INTERVAL_S:
            return
        if interval_s <= 0.0:
            return

        try:
            cpu_times = self.process.cpu_times()
            process_cpu_s = cpu_times.user + cpu_times.system
            ctx_switches = self.process.num_ctx_switches()
            disk = psutil.disk_io_counters()
            cpu_count = psutil.cpu_count() or 1
            try:
                load_per_cpu_pct = (os.getloadavg()[0] / cpu_count) * 100
            except (AttributeError, OSError):
                load_per_cpu_pct = 0.0

            disk_read_bytes = (
                disk.read_bytes
                if disk is not None
                else self.resource_last_disk_read_bytes
            )
            disk_write_bytes = (
                disk.write_bytes
                if disk is not None
                else self.resource_last_disk_write_bytes
            )
            sample = ProducerResourceSample(
                interval_s=interval_s,
                process_cpu_pct=(
                    (process_cpu_s - self.resource_last_cpu_s) / interval_s * 100
                ),
                system_cpu_pct=psutil.cpu_percent(interval=None),
                load_per_cpu_pct=load_per_cpu_pct,
                rss_mb=self.process.memory_info().rss / (1024 * 1024),
                thread_count=self.process.num_threads(),
                voluntary_ctx_switches=(
                    ctx_switches.voluntary - self.resource_last_ctx_voluntary
                ),
                involuntary_ctx_switches=(
                    ctx_switches.involuntary - self.resource_last_ctx_involuntary
                ),
                disk_read_mbps=(
                    (disk_read_bytes - self.resource_last_disk_read_bytes)
                    / interval_s
                    / (1024 * 1024)
                ),
                disk_write_mbps=(
                    (disk_write_bytes - self.resource_last_disk_write_bytes)
                    / interval_s
                    / (1024 * 1024)
                ),
            )
        except (OSError, psutil.Error):
            logger.debug("Failed to sample producer resources", exc_info=True)
            self.resource_last_at = now
            return

        self.resource_latest = sample
        self.resource_last_at = now
        self.resource_last_cpu_s = process_cpu_s
        self.resource_last_ctx_voluntary = ctx_switches.voluntary
        self.resource_last_ctx_involuntary = ctx_switches.involuntary
        self.resource_last_disk_read_bytes = disk_read_bytes
        self.resource_last_disk_write_bytes = disk_write_bytes
        for metric in self.resource_window_max:
            value = (
                sample.interval_s
                if metric == "sample_gap_s"
                else getattr(sample, metric)
            )
            self.resource_window_max[metric] = max(
                self.resource_window_max[metric], value
            )

    def _observe_max(self, metric: str, value: float) -> None:
        value = max(0.0, value)
        self.window_max[metric] = max(self.window_max[metric], value)
        self.overall_max[metric] = max(self.overall_max[metric], value)

    def observe_duration(self, metric: str, duration_s: float) -> None:
        """Record one producer work duration."""
        self.last_duration[metric] = duration_s
        self._observe_max(metric, duration_s)

    def observe_schedule(
        self,
        *,
        stream: str,
        frame_index: int,
        deadline: float,
        observed_before_wait: float,
        observed_after_wait: float,
        slept: bool,
        tolerance_s: float,
    ) -> None:
        """Record schedule timing and emit a rate-limited near-limit warning."""
        lateness = observed_after_wait - deadline
        absolute_lateness = abs(lateness)
        pre_wait_lateness = max(0.0, observed_before_wait - deadline)
        sleep_overshoot = max(0.0, lateness) if slept else 0.0
        lateness_metric = f"{stream}_lateness"

        self.window_events += 1
        previous_window_max = self.window_max[lateness_metric]
        self._observe_max(lateness_metric, absolute_lateness)
        self._observe_max("pre_wait_lateness", pre_wait_lateness)
        self._observe_max("sleep_overshoot", sleep_overshoot)

        exceeded = absolute_lateness > tolerance_s
        near_limit = absolute_lateness >= (
            tolerance_s * PRODUCER_TIMING_NEAR_LIMIT_FACTOR
        )
        self._sample_resources(force=near_limit)
        should_log = exceeded or (
            near_limit
            and not self.near_limit_logged
            and absolute_lateness >= previous_window_max
        )
        if should_log:
            event_monotonic = time.perf_counter()
            log = logger.error if exceeded else logger.warning
            log(
                "Producer timing near deadline ctx=%d rec_idx=%d stream=%s "
                "frame=%d elapsed=%.3fs monotonic=%.6f "
                "lateness=%+.1fms pre_wait_late=%.1fms "
                "sleep_overshoot=%.1fms last_joint_work=%.1fms "
                "last_frame_build=%.1fms last_rgb_log=%.1fms "
                "resource_interval=%.2fs proc_cpu=%.0f%% system_cpu=%.0f%% "
                "load_per_cpu=%.0f%% rss=%.1fMiB threads=%d "
                "ctx_switches[v=%d iv=%d] disk[r=%.1fMiB/s w=%.1fMiB/s]",
                self.context_index,
                self.recording_index,
                stream,
                frame_index,
                event_monotonic - self.started_at,
                event_monotonic,
                lateness * 1_000,
                pre_wait_lateness * 1_000,
                sleep_overshoot * 1_000,
                self.last_duration["joint_work"] * 1_000,
                self.last_duration["frame_build"] * 1_000,
                self.last_duration["rgb_log"] * 1_000,
                self.resource_latest.interval_s,
                self.resource_latest.process_cpu_pct,
                self.resource_latest.system_cpu_pct,
                self.resource_latest.load_per_cpu_pct,
                self.resource_latest.rss_mb,
                self.resource_latest.thread_count,
                self.resource_latest.voluntary_ctx_switches,
                self.resource_latest.involuntary_ctx_switches,
                self.resource_latest.disk_read_mbps,
                self.resource_latest.disk_write_mbps,
            )
            self.near_limit_logged = True

    def maybe_report(self, *, force: bool = False) -> None:
        """Print one compact roll-up per interval, or a final recording roll-up."""
        now = time.perf_counter()
        window_elapsed = now - self.window_started_at
        if not force and window_elapsed < PRODUCER_TIMING_REPORT_INTERVAL_S:
            return
        if self.window_events == 0:
            return
        self._sample_resources(force=True)

        logger.info(
            "Producer timing summary ctx=%d rec_idx=%d elapsed=%.1fs "
            "window=%.1fs events=%d max_late[joint=%.1fms video=%.1fms] "
            "max_wait[already_late=%.1fms sleep_overshoot=%.1fms] "
            "max_work[joint=%.1fms frame_build=%.1fms rgb_log=%.1fms] "
            "max_resource[proc_cpu=%.0f%% system_cpu=%.0f%% "
            "load_per_cpu=%.0f%% rss=%.1fMiB sample_gap=%.2fs "
            "disk_r=%.1fMiB/s disk_w=%.1fMiB/s]",
            self.context_index,
            self.recording_index,
            now - self.started_at,
            window_elapsed,
            self.window_events,
            self.window_max["joint_lateness"] * 1_000,
            self.window_max["video_lateness"] * 1_000,
            self.window_max["pre_wait_lateness"] * 1_000,
            self.window_max["sleep_overshoot"] * 1_000,
            self.window_max["joint_work"] * 1_000,
            self.window_max["frame_build"] * 1_000,
            self.window_max["rgb_log"] * 1_000,
            self.resource_window_max["process_cpu_pct"],
            self.resource_window_max["system_cpu_pct"],
            self.resource_window_max["load_per_cpu_pct"],
            self.resource_window_max["rss_mb"],
            self.resource_window_max["sample_gap_s"],
            self.resource_window_max["disk_read_mbps"],
            self.resource_window_max["disk_write_mbps"],
        )
        self.window_started_at = now
        self.window_events = 0
        self.near_limit_logged = False
        self.window_max = dict.fromkeys(PRODUCER_TIMING_METRICS, 0.0)
        self.resource_window_max = dict.fromkeys(self.resource_window_max, 0.0)


def log_synchronous_frames(
    *,
    robot_name: str,
    joint_frame_count: int,
    video_frame_count: int,
    recording_index: int,
    timestamp_start_s: float,
    joint_names: list[str],
    camera_name_list: list[str],
    image_width: int | None,
    image_height: int | None,
    joint_fps: int,
    video_fps: int,
    marker_name: str,
    context_index: int,
    use_real_timestamps: bool = False,
    use_stochastic_timestamps: bool = False,
    assert_deadline: bool = False,  # only set by performance tests
) -> None:
    """Log all joint and video frames for one recording synchronously.

    Joint and video frames are interleaved in a single loop using a wall-clock
    deadline scheduler, so both streams advance together in time order.
    """
    recording_wall_start = time.time()
    joint_index = 0
    video_index = 0
    diagnostics = (
        ProducerTimingDiagnostics(context_index, recording_index)
        if assert_deadline
        else None
    )
    history = ProducerDiagnosticHistory(
        context_index=context_index,
        recording_index=recording_index,
        enabled=assert_deadline,
    )
    if diagnostics is not None:
        logger.info(
            "Producer timing start ctx=%d rec_idx=%d joint_frames=%d@%dhz "
            "video_frames=%d@%dhz cameras=%d image=%sx%s stochastic=%s",
            context_index,
            recording_index,
            joint_frame_count,
            joint_fps,
            video_frame_count,
            video_fps,
            len(camera_name_list),
            image_width,
            image_height,
            use_stochastic_timestamps,
        )

    while joint_index < joint_frame_count or video_index < (
        video_frame_count if camera_name_list else 0
    ):
        iteration_started_ns = time.perf_counter_ns()
        joint_due = joint_index < joint_frame_count
        video_due = camera_name_list and video_index < video_frame_count
        # One jitter is shared by both deadlines/timestamps this iteration, so
        # size it to the tighter (higher-fps) window to stay within both.
        jitter = get_jitter(
            use_stochastic_timestamps,
            max(joint_fps, video_fps) if camera_name_list else joint_fps,
        )

        joint_deadline = (
            recording_wall_start + (joint_index / joint_fps) + jitter
            if joint_due
            else float("inf")
        )
        video_deadline = (
            recording_wall_start + (video_index / video_fps) + jitter
            if video_due
            else float("inf")
        )

        if joint_deadline <= video_deadline:
            role_name = "joint"
            history.record_gap(
                role_name=role_name,
                frame_index=joint_index,
                deadline=joint_deadline,
            )
            observed_before_wait = time.time()
            remaining = joint_deadline - observed_before_wait
            if remaining > 0:
                history.sleep(
                    remaining,
                    role_name=role_name,
                    frame_index=joint_index,
                    deadline=joint_deadline,
                )
            observed_after_wait = time.time()
            if diagnostics is not None:
                diagnostics.observe_schedule(
                    stream="joint",
                    frame_index=joint_index,
                    deadline=joint_deadline,
                    observed_before_wait=observed_before_wait,
                    observed_after_wait=observed_after_wait,
                    slept=remaining > 0,
                    tolerance_s=SCHEDULER_TOLERANCE_S,
                )
            history.record(
                "deadline_lateness",
                role_name=role_name,
                frame_index=joint_index,
                deadline=joint_deadline,
                details={"lateness_ms": (observed_after_wait - joint_deadline) * 1_000},
                statistic_value_ms=(observed_after_wait - joint_deadline) * 1_000,
            )
            if assert_deadline and use_stochastic_timestamps:
                assert_on_schedule(
                    joint_deadline,
                    SCHEDULER_TOLERANCE_S,
                    label=(
                        f"joint frame ctx={context_index} "
                        f"rec_idx={recording_index} frame={joint_index}"
                    ),
                    observed_at=observed_after_wait,
                    diagnostic_history=history,
                    role_name=role_name,
                    frame_index=joint_index,
                )
            joint_work_started = time.perf_counter()
            if use_real_timestamps:
                timestamp = None
            else:
                intended = timestamp_start_s + (joint_index / joint_fps)
                timestamp = intended + jitter
            with history.measure(
                "generate_joint_values",
                role_name=role_name,
                frame_index=joint_index,
                deadline=joint_deadline,
            ):
                joint_values = generate_joint_values(
                    joint_index, joint_fps, joint_names
                )
            with history.measure(
                "nc.log_joint_positions",
                role_name=role_name,
                frame_index=joint_index,
                deadline=joint_deadline,
            ):
                with Timer(
                    MAX_TIME_TO_LOG_S,
                    label="nc.log_joint_positions",
                    assert_deadline=assert_deadline,
                ):
                    nc.log_joint_positions(
                        joint_values, robot_name=robot_name, timestamp=timestamp
                    )
            with history.measure(
                "nc.log_joint_velocities",
                role_name=role_name,
                frame_index=joint_index,
                deadline=joint_deadline,
            ):
                with Timer(
                    MAX_TIME_TO_LOG_S,
                    label="nc.log_joint_velocities",
                    assert_deadline=assert_deadline,
                ):
                    nc.log_joint_velocities(
                        joint_values, robot_name=robot_name, timestamp=timestamp
                    )
            with history.measure(
                "nc.log_joint_torques",
                role_name=role_name,
                frame_index=joint_index,
                deadline=joint_deadline,
            ):
                with Timer(
                    MAX_TIME_TO_LOG_S,
                    label="nc.log_joint_torques",
                    assert_deadline=assert_deadline,
                ):
                    nc.log_joint_torques(
                        joint_values, robot_name=robot_name, timestamp=timestamp
                    )
            with history.measure(
                "nc.log_custom_1d",
                role_name=role_name,
                frame_index=joint_index,
                deadline=joint_deadline,
            ):
                with Timer(
                    MAX_TIME_TO_LOG_S,
                    label="nc.log_custom_1d",
                    assert_deadline=assert_deadline,
                ):
                    nc.log_custom_1d(
                        marker_name,
                        np.array([float(joint_index)], dtype=np.float32),
                        robot_name=robot_name,
                        timestamp=timestamp,
                    )
            history.record(
                "loop_iteration",
                role_name=role_name,
                frame_index=joint_index,
                started_ns=iteration_started_ns,
                deadline=joint_deadline,
            )
            joint_index += 1
            if diagnostics is not None:
                diagnostics.observe_duration(
                    "joint_work", time.perf_counter() - joint_work_started
                )
                diagnostics.maybe_report()
        else:
            role_name = "rgb"
            history.record_gap(
                role_name=role_name,
                frame_index=video_index,
                deadline=video_deadline,
            )
            observed_before_wait = time.time()
            remaining = video_deadline - observed_before_wait
            if remaining > 0:
                history.sleep(
                    remaining,
                    role_name=role_name,
                    frame_index=video_index,
                    deadline=video_deadline,
                )
            observed_after_wait = time.time()
            if diagnostics is not None:
                diagnostics.observe_schedule(
                    stream="video",
                    frame_index=video_index,
                    deadline=video_deadline,
                    observed_before_wait=observed_before_wait,
                    observed_after_wait=observed_after_wait,
                    slept=remaining > 0,
                    tolerance_s=SCHEDULER_TOLERANCE_S,
                )
            history.record(
                "deadline_lateness",
                role_name=role_name,
                frame_index=video_index,
                deadline=video_deadline,
                details={"lateness_ms": (observed_after_wait - video_deadline) * 1_000},
                statistic_value_ms=(observed_after_wait - video_deadline) * 1_000,
            )
            if assert_deadline and use_stochastic_timestamps:
                assert_on_schedule(
                    video_deadline,
                    SCHEDULER_TOLERANCE_S,
                    label=(
                        f"video frame ctx={context_index} "
                        f"rec_idx={recording_index} frame={video_index}"
                    ),
                    observed_at=observed_after_wait,
                    diagnostic_history=history,
                    role_name=role_name,
                    frame_index=video_index,
                )
            if use_real_timestamps:
                timestamp = None
            else:
                intended = timestamp_start_s + (video_index / video_fps)
                timestamp = intended + jitter

            for camera_index, camera_name in enumerate(camera_name_list):
                frame_code = (
                    (context_index * 1_000_000_000)
                    + (recording_index * 10_000_000)
                    + (camera_index * 100_000)
                    + video_index
                )
                frame_build_started = time.perf_counter()
                with history.measure(
                    "encode_frame_number",
                    role_name=role_name,
                    frame_index=video_index,
                    deadline=video_deadline,
                    details={"camera_name": camera_name},
                ):
                    rgb_image = encode_frame_number(
                        frame_code, image_width, image_height
                    )
                if diagnostics is not None:
                    diagnostics.observe_duration(
                        "frame_build", time.perf_counter() - frame_build_started
                    )
                rgb_log_started = time.perf_counter()
                with history.measure(
                    "nc.log_rgb",
                    role_name=role_name,
                    frame_index=video_index,
                    deadline=video_deadline,
                    details={"camera_name": camera_name},
                ):
                    with Timer(
                        MAX_TIME_TO_LOG_S,
                        label="nc.log_rgb",
                        assert_deadline=assert_deadline,
                    ):
                        nc.log_rgb(
                            camera_name,
                            rgb_image,
                            robot_name=robot_name,
                            timestamp=timestamp,
                        )
                if diagnostics is not None:
                    diagnostics.observe_duration(
                        "rgb_log", time.perf_counter() - rgb_log_started
                    )
            history.record(
                "loop_iteration",
                role_name=role_name,
                frame_index=video_index,
                started_ns=iteration_started_ns,
                deadline=video_deadline,
            )
            video_index += 1
            if diagnostics is not None:
                diagnostics.maybe_report()

    history.log_summary()
    if diagnostics is not None:
        diagnostics.maybe_report(force=True)
        logger.info(
            "Producer timing complete ctx=%d rec_idx=%d overall_max_late"
            "[joint=%.1fms video=%.1fms] overall_max_work"
            "[joint=%.1fms frame_build=%.1fms rgb_log=%.1fms]",
            context_index,
            recording_index,
            diagnostics.overall_max["joint_lateness"] * 1_000,
            diagnostics.overall_max["video_lateness"] * 1_000,
            diagnostics.overall_max["joint_work"] * 1_000,
            diagnostics.overall_max["frame_build"] * 1_000,
            diagnostics.overall_max["rgb_log"] * 1_000,
        )


def build_thread_roles(
    *,
    joint_names: list[str],
    camera_name_list: list[str],
) -> list[dict[str, object]]:
    """Build role specs for per-thread logging."""
    roles: list[dict[str, object]] = []
    for camera_name in camera_name_list:
        roles.append({
            "role": "rgb",
            "camera_names": [camera_name],
            "marker_name": f"marker_{camera_name}",
        })
    for role_name in ("joint_positions", "joint_velocities", "joint_torques"):
        roles.append({
            "role": role_name,
            "joint_names": list(joint_names),
            "marker_name": f"marker_{role_name}",
        })
    return roles


def run_threaded_logging(
    *,
    robot_name: str,
    joint_frame_count: int,
    video_frame_count: int,
    recording_index: int,
    timestamp_start_s: float,
    joint_fps: int,
    video_fps: int,
    context_index: int,
    joint_names: list[str],
    camera_name_list: list[str],
    image_width: int | None,
    image_height: int | None,
    use_real_timestamps: bool = False,
    use_stochastic_timestamps: bool = False,
    assert_deadline: bool = False,  # only set by performance tests
) -> list[str]:
    """Run logging across multiple threads, one per data role."""
    roles = build_thread_roles(
        joint_names=joint_names, camera_name_list=camera_name_list
    )
    barrier = threading.Barrier(len(roles))
    thread_errors: list[BaseException] = []
    heartbeat_registry = ProducerHeartbeatRegistry()

    def worker(role_spec: dict[str, object]) -> None:
        """Execute logging for a single thread role."""
        try:
            barrier.wait()
            role_name = str(role_spec["role"])
            marker_name = str(role_spec["marker_name"])
            is_rgb = role_name == "rgb"
            diagnostic_role = (
                f"rgb:{role_spec['camera_names'][0]}" if is_rgb else role_name
            )
            history = ProducerDiagnosticHistory(
                context_index=context_index,
                recording_index=recording_index,
                heartbeat_registry=heartbeat_registry,
                enabled=assert_deadline,
            )
            frame_count = video_frame_count if is_rgb else joint_frame_count
            fps = video_fps if is_rgb else joint_fps
            thread_wall_start = time.time()
            for frame_index in range(frame_count):
                iteration_started_ns = time.perf_counter_ns()
                jitter = get_jitter(use_stochastic_timestamps, fps)
                frame_deadline = thread_wall_start + (frame_index / fps) + jitter
                history.record_gap(
                    role_name=diagnostic_role,
                    frame_index=frame_index,
                    deadline=frame_deadline,
                )
                remaining = frame_deadline - time.time()
                if remaining > 0:
                    history.sleep(
                        remaining,
                        role_name=diagnostic_role,
                        frame_index=frame_index,
                        deadline=frame_deadline,
                    )
                observed_after_wait = time.time()
                history.record(
                    "deadline_lateness",
                    role_name=diagnostic_role,
                    frame_index=frame_index,
                    deadline=frame_deadline,
                    details={
                        "lateness_ms": (observed_after_wait - frame_deadline) * 1_000
                    },
                    statistic_value_ms=(observed_after_wait - frame_deadline) * 1_000,
                )
                if assert_deadline and use_stochastic_timestamps:
                    assert_on_schedule(
                        frame_deadline,
                        SCHEDULER_TOLERANCE_S,
                        label=f"{role_name} frame",
                        observed_at=observed_after_wait,
                        diagnostic_history=history,
                        role_name=diagnostic_role,
                        frame_index=frame_index,
                    )
                if use_real_timestamps:
                    timestamp = None
                else:
                    intended = timestamp_start_s + (frame_index / fps)
                    timestamp = intended + jitter
                if is_rgb:
                    for camera_offset, camera_name in enumerate(
                        role_spec["camera_names"]
                    ):
                        camera_id = str(camera_name)
                        camera_index = camera_name_list.index(camera_id) + camera_offset
                        frame_code = (
                            (context_index * 1_000_000_000)
                            + (recording_index * 10_000_000)
                            + (camera_index * 100_000)
                            + frame_index
                        )
                        with history.measure(
                            "encode_frame_number",
                            role_name=diagnostic_role,
                            frame_index=frame_index,
                            deadline=frame_deadline,
                            details={"camera_name": camera_id},
                        ):
                            rgb_image = encode_frame_number(
                                frame_code, image_width, image_height
                            )
                        with history.measure(
                            "nc.log_rgb",
                            role_name=diagnostic_role,
                            frame_index=frame_index,
                            deadline=frame_deadline,
                            details={"camera_name": camera_id},
                        ):
                            with Timer(
                                MAX_TIME_TO_LOG_S,
                                label="nc.log_rgb",
                                assert_deadline=assert_deadline,
                            ):
                                nc.log_rgb(
                                    camera_id,
                                    rgb_image,
                                    robot_name=robot_name,
                                    timestamp=timestamp,
                                )
                else:
                    thread_joint_names = list(role_spec["joint_names"])
                    with history.measure(
                        "generate_joint_values",
                        role_name=diagnostic_role,
                        frame_index=frame_index,
                        deadline=frame_deadline,
                    ):
                        joint_values = generate_joint_values(
                            frame_index, joint_fps, thread_joint_names
                        )
                    if role_name == "joint_positions":
                        with history.measure(
                            "nc.log_joint_positions",
                            role_name=diagnostic_role,
                            frame_index=frame_index,
                            deadline=frame_deadline,
                        ):
                            with Timer(
                                MAX_TIME_TO_LOG_S,
                                label="nc.log_joint_positions",
                                assert_deadline=assert_deadline,
                            ):
                                nc.log_joint_positions(
                                    joint_values,
                                    robot_name=robot_name,
                                    timestamp=timestamp,
                                )
                    elif role_name == "joint_velocities":
                        with history.measure(
                            "nc.log_joint_velocities",
                            role_name=diagnostic_role,
                            frame_index=frame_index,
                            deadline=frame_deadline,
                        ):
                            with Timer(
                                MAX_TIME_TO_LOG_S,
                                label="nc.log_joint_velocities",
                                assert_deadline=assert_deadline,
                            ):
                                nc.log_joint_velocities(
                                    joint_values,
                                    robot_name=robot_name,
                                    timestamp=timestamp,
                                )
                    else:
                        with history.measure(
                            "nc.log_joint_torques",
                            role_name=diagnostic_role,
                            frame_index=frame_index,
                            deadline=frame_deadline,
                        ):
                            with Timer(
                                MAX_TIME_TO_LOG_S,
                                label="nc.log_joint_torques",
                                assert_deadline=assert_deadline,
                            ):
                                nc.log_joint_torques(
                                    joint_values,
                                    robot_name=robot_name,
                                    timestamp=timestamp,
                                )
                with history.measure(
                    "nc.log_custom_1d",
                    role_name=diagnostic_role,
                    frame_index=frame_index,
                    deadline=frame_deadline,
                ):
                    with Timer(
                        MAX_TIME_TO_LOG_S,
                        label="nc.log_custom_1d",
                        assert_deadline=assert_deadline,
                    ):
                        nc.log_custom_1d(
                            marker_name,
                            np.array([float(frame_index)], dtype=np.float32),
                            robot_name=robot_name,
                            timestamp=timestamp,
                        )
                history.record(
                    "loop_iteration",
                    role_name=diagnostic_role,
                    frame_index=frame_index,
                    started_ns=iteration_started_ns,
                    deadline=frame_deadline,
                )
            history.log_summary()
        except BaseException as exc:  # noqa: BLE001
            thread_errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(role,), daemon=True) for role in roles
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if thread_errors:
        raise RuntimeError(
            f"Threaded producer failed: {thread_errors[0]}"
        ) from thread_errors[0]

    return [str(role["marker_name"]) for role in roles]


def log_frames(
    spec: ContextSpec,
    *,
    recording_index: int,
    marker_name: str,
) -> list[str]:
    """Log all frames for one recording, dispatching based on producer_channels.

    Derives timestamp mode and all frame parameters from *spec*.
    """
    use_real_timestamps = spec.case.timestamp_mode == TIMESTAMP_MODE_REAL
    use_stochastic_timestamps = spec.case.timestamp_mode == TIMESTAMP_MODE_STOCHASTIC
    recording_timestamp_start_s = (
        spec.timestamp_start_s + recording_index * spec.case.duration_sec
    )
    joint_name_list = joint_names_for_count(spec.case.joint_count)
    camera_name_list = camera_names(spec.case.video_count)

    if spec.case.producer_channels == PRODUCER_PER_THREAD:
        return run_threaded_logging(
            robot_name=spec.robot_name,
            joint_frame_count=spec.expected_joint_frames,
            video_frame_count=spec.expected_video_frames,
            recording_index=recording_index,
            timestamp_start_s=recording_timestamp_start_s,
            joint_fps=spec.case.joint_fps,
            video_fps=spec.case.video_fps,
            context_index=spec.context_index,
            joint_names=joint_name_list,
            camera_name_list=camera_name_list,
            image_width=spec.case.image_width,
            image_height=spec.case.image_height,
            use_real_timestamps=use_real_timestamps,
            use_stochastic_timestamps=use_stochastic_timestamps,
            assert_deadline=spec.assert_deadline,
        )

    log_synchronous_frames(
        robot_name=spec.robot_name,
        joint_frame_count=spec.expected_joint_frames,
        video_frame_count=spec.expected_video_frames,
        recording_index=recording_index,
        timestamp_start_s=recording_timestamp_start_s,
        joint_names=joint_name_list,
        camera_name_list=camera_name_list,
        image_width=spec.case.image_width,
        image_height=spec.case.image_height,
        joint_fps=spec.case.joint_fps,
        video_fps=spec.case.video_fps,
        marker_name=marker_name,
        context_index=spec.context_index,
        use_real_timestamps=use_real_timestamps,
        use_stochastic_timestamps=use_stochastic_timestamps,
        assert_deadline=spec.assert_deadline,
    )
    return [marker_name]


def _bind_worker_dataset(spec: ContextSpec) -> None:
    """Poll until the worker pool-shared dataset is visible to this worker.

    The timeout RuntimeError is raised inside the Timer block on purpose:
    ``Timer.__exit__`` skips its deadline assertion when an exception is in
    flight, so the real ``nc.get_dataset`` error (chained via ``last_error``)
    propagates instead of being masked by the Timer's own AssertionError.
    """
    last_error: Exception | None = None
    deadline = time.time() + MAX_TIME_TO_START_S
    with Timer(
        MAX_TIME_TO_START_S,
        label="nc.get_dataset",
        always_log=True,
        assert_deadline=spec.assert_deadline,
    ):
        while time.time() < deadline:
            try:
                nc.get_dataset(spec.dataset_name)
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(DATASET_POLL_INTERVAL_S)

        raise RuntimeError(
            f"Timed out waiting for shared dataset '{spec.dataset_name}' to exist"
        ) from last_error


@surface_worker_errors
def _subprocess_context_worker(spec: ContextSpec) -> ContextResult:
    """Subprocess wrapper for context_worker used by multiprocessing.Pool.

    On Linux, Pool uses fork so workers inherit a copy of the parent's
    Timer._stats. Clearing it here ensures workers only capture their own
    timers and the parent's pre-fork timers (e.g. nc.login) are not
    double-counted when stats are merged back. The stochastic-timestamp RNG
    is reseeded per-context so parallel workers produce independent jitter
    sequences instead of replaying the parent's seed. Spawned workers
    (macOS) additionally re-authenticate, as they do not inherit the
    parent's in-process auth state.
    """
    multiprocessing.current_process().name = f"ctx-{spec.context_index}"
    Timer._stats.clear()
    STOCHASTIC_TIMESTAMP_RANDOM.seed(1 + spec.context_index)
    ensure_login()
    return context_worker(spec)


def context_worker(spec: ContextSpec) -> ContextResult:
    """Execute recordings for a single parallel context."""
    from neuracore.data_daemon.rust_selection import is_rust_daemon_enabled
    from tests.integration.platform.data_daemon.shared.db_helpers import (
        wait_for_recording_index_for_source,
    )

    use_rust = is_rust_daemon_enabled()
    case = spec.case
    use_real_timestamps = case.timestamp_mode == TIMESTAMP_MODE_REAL
    joint_name_list = joint_names_for_count(case.joint_count)
    camera_name_list = camera_names(case.video_count)
    marker_names: list[str] = []
    recording_ids: list[str] = []
    recording_indexes: list[int] = []
    robot = None

    if spec.start_delay_s > 0.0:
        time.sleep(spec.start_delay_s)

    wall_started_at: float | None = None
    wall_stopped_at: float = 0.0

    try:
        _bind_worker_dataset(spec)
        with Timer(
            MAX_TIME_TO_START_S,
            label="nc.connect_robot",
            always_log=True,
            assert_deadline=spec.assert_deadline,
        ):
            robot = nc.connect_robot(spec.robot_name, overwrite=False)

        source: tuple[str, int] = (str(robot.id), int(robot.instance))

        expected_by_recording: dict[str, RecordingExpectedTimestamps] | None = (
            {} if not use_real_timestamps else None
        )

        for recording_ordinal in range(spec.recordings_per_context):
            recording_timestamp_start_s = (
                spec.timestamp_start_s + recording_ordinal * case.duration_sec
            )
            recording_capture_start_s = None if use_real_timestamps else time.time()
            recording_capture_stop_s = (
                None
                if recording_capture_start_s is None
                else recording_capture_start_s + case.duration_sec
            )

            with Timer(
                MAX_TIME_TO_START_S,
                label="nc.start_recording",
                always_log=True,
                assert_deadline=spec.assert_deadline,
            ):
                nc.start_recording(
                    robot_name=spec.robot_name, timestamp=recording_capture_start_s
                )
            if wall_started_at is None:
                wall_started_at = time.time()

            if use_rust:
                previous_index = recording_indexes[-1] if recording_indexes else 0
                daemon_recording_index = wait_for_recording_index_for_source(
                    source[0],
                    source[1],
                    after_index=previous_index,
                    timeout_s=MAX_TIME_TO_START_S,
                )
                recording_indexes.append(daemon_recording_index)

                cloud_recording_id = robot.get_cloud_recording_id(timeout_s=0.0)
                recording_ids.append(str(cloud_recording_id or ""))

                disk_recording_key = str(daemon_recording_index)
            else:
                recording_id = str(robot.get_current_recording_id() or "")
                recording_ids.append(recording_id)
                disk_recording_key = recording_id

            # Build per-recording expected timestamps once the recording key is
            # known. Trace keys use "data_type/data_type_name" to match the
            # semantic keys resolved from the DB in disk_helpers. data_type_name is
            # the storage name produced by validate_safe_name (e.g.
            # "vx300s_left\waist" for joint names).
            if expected_by_recording is not None:
                from neuracore_types.utils import validate_safe_name

                joint_ts = [
                    recording_timestamp_start_s + i / case.joint_fps
                    for i in range(spec.expected_joint_frames)
                ]
                video_ts = [
                    recording_timestamp_start_s + i / case.video_fps
                    for i in range(spec.expected_video_frames)
                ]
                by_trace: dict[str, list[float]] = {}
                for joint_name in joint_name_list:
                    safe = validate_safe_name(joint_name)
                    by_trace[f"JOINT_POSITIONS/{safe}"] = joint_ts
                    by_trace[f"JOINT_VELOCITIES/{safe}"] = joint_ts
                    by_trace[f"JOINT_TORQUES/{safe}"] = joint_ts
                for camera in camera_name_list:
                    safe_cam = validate_safe_name(camera)
                    by_trace[f"RGB_IMAGES/{safe_cam}"] = video_ts
                # CUSTOM_1D marker — name depends on producer_channels mode
                if case.producer_channels == PRODUCER_PER_THREAD:
                    # One marker per joint data type thread
                    for role_name in (
                        "joint_positions",
                        "joint_velocities",
                        "joint_torques",
                    ):
                        safe_marker = validate_safe_name(f"marker_{role_name}")
                        by_trace[f"CUSTOM_1D/{safe_marker}"] = joint_ts
                    for camera in camera_name_list:
                        safe_marker = validate_safe_name(f"marker_{camera}")
                        by_trace[f"CUSTOM_1D/{safe_marker}"] = video_ts
                else:
                    safe_marker = validate_safe_name("marker_synchronous")
                    by_trace[f"CUSTOM_1D/{safe_marker}"] = joint_ts
                by_trace_fps = {
                    trace_key: (
                        case.video_fps if timestamps is video_ts else case.joint_fps
                    )
                    for trace_key, timestamps in by_trace.items()
                }
                expected_by_recording[disk_recording_key] = RecordingExpectedTimestamps(
                    by_trace=by_trace,
                    by_trace_fps=by_trace_fps,
                )

            current_marker_names = log_frames(
                spec,
                recording_index=recording_ordinal,
                marker_name="marker_synchronous",
            )
            if not marker_names:
                marker_names = current_marker_names

            with Timer(
                case.stop_recording_sla_s,
                label="nc.stop_recording",
                always_log=True,
                assert_deadline=spec.assert_deadline,
            ):
                nc.stop_recording(
                    robot_name=spec.robot_name,
                    wait=case.wait,
                    timestamp=recording_capture_stop_s,
                )
            wall_stopped_at = time.time()

        captured_timer_stats = {k: dict(v) for k, v in Timer._stats.items()}
        return ContextResult(
            dataset_name=spec.dataset_name,
            recording_ids=recording_ids,
            recording_indexes=recording_indexes,
            source=source,
            robot_name=spec.robot_name,
            joint_names=joint_name_list,
            camera_names=camera_name_list,
            joint_frame_count=spec.expected_joint_frames,
            video_frame_count=spec.expected_video_frames,
            joint_fps=case.joint_fps,
            video_fps=case.video_fps,
            duration_sec=case.duration_sec,
            timestamp_start_s=spec.timestamp_start_s,
            timestamp_end_s=spec.timestamp_end_s,
            marker_names=marker_names,
            has_video=bool(camera_name_list),
            context_index=spec.context_index,
            wall_started_at=wall_started_at,
            wall_stopped_at=wall_stopped_at,
            timestamp_mode=case.timestamp_mode,
            expected_timestamps=(
                ContextExpectedTimestamps(by_recording=expected_by_recording)
                if expected_by_recording is not None
                else None
            ),
            timer_stats=captured_timer_stats,
        )
    except Exception:
        if robot is not None:
            try:
                if robot.is_recording():
                    nc.cancel_recording(robot_name=spec.robot_name)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to cancel active matrix recording for %s",
                    spec.robot_name,
                    exc_info=True,
                )
        raise
    finally:
        _cleanup_test_worker_robot(robot)


def run_case_contexts(
    case: DataDaemonTestCase,
    *,
    specs: list[ContextSpec] | None = None,
    assert_mode: bool = True,
    wait_for_traces: bool = False,
) -> list[ContextResult]:
    """Run all parallel contexts for a matrix test case.

    Executes each context spec either sequentially (when parallel_contexts==1)
    or concurrently via a multiprocessing pool. Sequential execution avoids
    pool overhead and simplifies debugging for single-context cases.

    Args:
        case: The test case defining parallelism level and context matrix.
        specs: Pre-built context specs to run. If None, built from ``case``
            via :func:`build_context_specs`.
        assert_mode: When ``True`` (default), calls :func:`assert_context_mode`
            after running to verify expected parallelization behaviour.
        wait_for_traces: When ``True``, waits for all traces to be written to
            disk after running (implies ``assert_mode``).

    Returns:
        List of result dicts from each context worker, one per spec.
    """
    ensure_login()

    if specs is None:
        specs = build_context_specs(case)

    if case.has_video:
        nc.set_video_encoding_options(
            nc.Codec(case.video_codec) if case.video_codec else nc.Codec.H264_LOSSLESS
        )

    if specs:
        with Timer(MAX_TIME_TO_START_S, label="nc.create_dataset", always_log=True):
            nc.create_dataset(specs[0].dataset_name)

    if case.parallel_contexts == 1:
        results = [context_worker(specs[0])]
    else:
        with relayed_worker_logs() as log_queue:
            with multiprocessing.Pool(
                case.parallel_contexts,
                initializer=init_worker_logging,
                initargs=(log_queue, logging.getLogger().getEffectiveLevel()),
            ) as pool:
                results = list(  # type: ignore[return-value]
                    pool.map(_subprocess_context_worker, specs)
                )
        for result in results:
            Timer.merge_stats(result.timer_stats)

    if assert_mode or wait_for_traces:
        assert_context_mode(case, results)

    if wait_for_traces:
        from tests.integration.platform.data_daemon.shared.db_helpers import (
            wait_for_all_traces_written,
        )

        wait_for_all_traces_written(results=results)

    return results


def create_testing_dataset_name(case: DataDaemonTestCase) -> str:
    """Create a unique dataset name for a test case."""
    return f"testing_dataset_{case_id(case)}_{uuid.uuid4().hex[:6]}"
