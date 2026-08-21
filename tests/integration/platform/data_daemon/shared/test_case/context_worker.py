"""Runs a context's recordings and packs what happened into a ContextResult."""

from __future__ import annotations

import logging
import multiprocessing
import threading
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

import neuracore as nc
from tests.integration.platform.data_daemon.shared.auth import ensure_login
from tests.integration.platform.data_daemon.shared.process_control import (
    Timer,
    init_worker_logging,
    relayed_worker_logs,
    surface_worker_errors,
)
from tests.integration.platform.data_daemon.shared.test_case.boundaries import (
    EmittedFrame,
    ObservedFrameCodes,
    RecordingControlBounds,
    TraceClassification,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestCase,
    case_id,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    DATA_TYPE_RGB_IMAGES,
    DATASET_POLL_INTERVAL_S,
    GATE_CLOSE_POLL_INTERVAL_S,
    GATE_CLOSE_WATCHER_JOIN_TIMEOUT_S,
    MAX_TIME_TO_START_S,
    PRODUCER_MULTI_PROCESS,
    PRODUCER_PER_THREAD,
    camera_names,
    depth_camera_names,
    joint_names_for_count,
    trace_key_for,
)
from tests.integration.platform.data_daemon.shared.test_case.context_spec import (
    ContextExpectedTimestamps,
    ContextResult,
    ContextSpec,
    RecordingExpectedTimestamps,
    build_context_specs,
)
from tests.integration.platform.data_daemon.shared.test_case.frame_source import (
    prewarm_frame_bank,
)
from tests.integration.platform.data_daemon.shared.test_case.producers import (
    make_producer_session,
)
from tests.integration.platform.data_daemon.shared.test_case.streams import (
    rgb_frame_code,
)

logger = logging.getLogger(__name__)


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


@dataclass(slots=True)
class _GateCloseObservation:
    """Carries the gate-close wall clock out of the block that measured it."""

    result: float = 0.0


@contextmanager
def watch_local_gate_close(
    robot: object, *, enabled: bool
) -> Generator[_GateCloseObservation]:
    """Measure when this process's local recording gate closes, if asked to.

    A tighter upper bracket than ``stop_recording`` returning, which waits out
    the flush barrier. Falls back to the block's exit time when disabled.
    """
    observation = _GateCloseObservation()
    stop_polling = threading.Event()

    def poll() -> None:
        while not stop_polling.is_set():
            if robot.get_current_recording_id() is None:  # type: ignore[attr-defined]
                observation.result = time.time()
                return
            time.sleep(GATE_CLOSE_POLL_INTERVAL_S)

    watcher = (
        threading.Thread(target=poll, name="gate-close-watch", daemon=True)
        if enabled
        else None
    )
    if watcher is not None:
        watcher.start()
    try:
        yield observation
    finally:
        stop_polling.set()
        if watcher is not None:
            watcher.join(timeout=GATE_CLOSE_WATCHER_JOIN_TIMEOUT_S)
        if observation.result == 0.0:
            observation.result = time.time()


def log_frames(
    spec: ContextSpec,
    *,
    robot: object,
    recording_index: int,
    marker_name: str,
) -> list[str]:
    """Log one recording's frames, for a caller driving that recording by hand.

    Refuses a case whose producer outlives its recordings: it has no single
    recording's worth of frames to log.

    Returns:
        The marker series the producer wrote.
    """
    if spec.case.producer_channels in (PRODUCER_PER_THREAD, PRODUCER_MULTI_PROCESS):
        raise ValueError(
            f"log_frames needs a producer scoped to one recording, but "
            f"producer_channels={spec.case.producer_channels!r} runs for the "
            "whole context lifetime — drive it through make_producer_session "
            "instead"
        )
    session = make_producer_session(spec, robot=robot, marker_name=marker_name)
    session.start()
    session.run_recording(recording_index)
    session.finish()
    return session.marker_names


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
    double-counted when stats are merged back. Spawned workers (macOS)
    additionally re-authenticate, as they do not inherit the parent's
    in-process auth state.
    """
    multiprocessing.current_process().name = f"ctx-{spec.context_index}"
    Timer._stats.clear()
    ensure_login()
    return context_worker(spec)


def context_worker(spec: ContextSpec) -> ContextResult:
    """Execute recordings for a single parallel context."""
    from tests.integration.platform.data_daemon.shared.db_helpers import (
        wait_for_recording_index_for_source,
    )

    case = spec.case
    joint_name_list = joint_names_for_count(case.joint_count)
    camera_name_list = camera_names(case.video_count)
    depth_camera_name_list = depth_camera_names(case.depth_count)
    if camera_name_list:
        prewarm_frame_bank(case.video_detail, case.image_width, case.image_height)
    marker_names: list[str] = []
    recording_ids: list[str] = []
    recording_indexes: list[int] = []
    robot = None

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

        expected_by_recording: dict[str, RecordingExpectedTimestamps] = {}
        expected_video_stop_timestamp_by_recording: dict[str, float] = {}
        bounds_by_disk_key: dict[str, RecordingControlBounds] = {}
        observed_frame_codes: dict[str, ObservedFrameCodes] = {}
        ordinal_by_disk_key: dict[str, int] = {}

        session = make_producer_session(
            spec, robot=robot, marker_name="marker_synchronous"
        )
        marker_names = session.marker_names
        try:
            session.start()
            for recording_ordinal in range(spec.recordings_per_context):
                recording_capture_start_s = time.time()
                recording_capture_stop_s = recording_capture_start_s + case.duration_sec

                start_called_at = time.time()
                with Timer(
                    MAX_TIME_TO_START_S,
                    label="nc.start_recording",
                    always_log=True,
                    assert_deadline=spec.assert_deadline,
                ):
                    nc.start_recording(
                        robot_name=spec.robot_name, timestamp=recording_capture_start_s
                    )
                start_returned_at = time.time()
                if wall_started_at is None:
                    wall_started_at = start_returned_at

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
                recording_handle = robot.get_current_recording_id()

                session.run_recording(recording_ordinal)

                # Brackets the window's upper bound, the mirror of the start.
                stop_called_at = time.time()
                stop_handle = robot.get_current_recording_id()
                with (
                    watch_local_gate_close(
                        robot, enabled=session.needs_stop_gate_bracket
                    ) as gate_closed_at,
                    Timer(
                        case.stop_recording_sla_s,
                        label="nc.stop_recording",
                        always_log=True,
                        assert_deadline=spec.assert_deadline,
                    ),
                ):
                    nc.stop_recording(
                        robot_name=spec.robot_name,
                        wait=case.wait,
                        timestamp=recording_capture_stop_s,
                    )
                wall_stopped_at = time.time()
                expected_video_stop_timestamp_by_recording[disk_recording_key] = (
                    spec.timestamp_start_s + (recording_ordinal + 1) * case.duration_sec
                )

                bounds_by_disk_key[disk_recording_key] = RecordingControlBounds(
                    handles=frozenset(
                        handle
                        for handle in (recording_handle, stop_handle)
                        if handle is not None
                    ),
                    start_called_at=start_called_at,
                    start_returned_at=start_returned_at,
                    stop_called_at=stop_called_at,
                    stop_settled_at=gate_closed_at.result,
                )
                ordinal_by_disk_key[disk_recording_key] = recording_ordinal
        finally:
            # A surviving producer thread breaks the cleanup assertions.
            session.finish()

        # Turns an RGB trace's classified frames back into painted codes.
        rgb_trace_cameras = {
            trace_key_for(DATA_TYPE_RGB_IMAGES, camera): (camera, camera_index)
            for camera_index, camera in enumerate(camera_name_list)
        }
        report = session.report()
        for disk_key, bounds in bounds_by_disk_key.items():
            code_recording_index = session.frame_code_recording_index(
                ordinal_by_disk_key[disk_key]
            )
            by_trace: dict[str, TraceClassification] = {}
            codes_inside: dict[str, list[int]] = {}
            codes_unknowable: dict[str, set[int]] = {}
            for trace_key, frames in report.items():
                classification = session.classify(trace_key, frames, bounds)
                by_trace[trace_key] = classification

                breaching = [
                    frame for frame in classification.owed if frame.deadline_breaches
                ]
                assert not breaching, (
                    f"{trace_key} logged {len(breaching)} frame(s) inside "
                    f"recording {disk_key} that breached the logging deadline: "
                    f"{breaching[0].deadline_breaches}"
                )

                camera = rgb_trace_cameras.get(trace_key)
                if camera is None:
                    continue
                camera_name, camera_index = camera

                def _codes(
                    frames: list[EmittedFrame],
                    index: int = camera_index,
                    recording_index: int = code_recording_index,
                ):
                    return [
                        rgb_frame_code(
                            context_index=spec.context_index,
                            recording_index=recording_index,
                            camera_index=index,
                            frame_index=frame.frame_index,
                        )
                        for frame in frames
                    ]

                codes_inside[camera_name] = _codes(classification.owed)
                codes_unknowable[camera_name] = set(_codes(classification.unknowable))

            expected_by_recording[disk_key] = RecordingExpectedTimestamps(
                by_trace=by_trace
            )
            observed_frame_codes[disk_key] = ObservedFrameCodes(
                inside=codes_inside, unknowable=codes_unknowable
            )

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
            random_phase=case.random_phase,
            expected_timestamps=ContextExpectedTimestamps(
                by_recording=expected_by_recording
            ),
            timer_stats=captured_timer_stats,
            depth_camera_names=depth_camera_name_list,
            depth_frame_count=(
                spec.expected_video_frames if depth_camera_name_list else 0
            ),
            depth_mode=case.depth_mode,
            has_depth=bool(depth_camera_name_list),
            observed_frame_codes=observed_frame_codes,
            expected_video_stop_timestamp_by_recording=(
                expected_video_stop_timestamp_by_recording
            ),
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
    wait_for_traces: bool = False,
) -> list[ContextResult]:
    """Run all parallel contexts for a matrix test case.

    Executes each context spec in-process (when parallel_contexts==1) or
    concurrently via a multiprocessing pool. The in-process path avoids pool
    overhead and simplifies debugging for single-context cases.

    Args:
        case: The test case defining parallelism level and context matrix.
        specs: Pre-built context specs to run. If None, built from ``case``
            via :func:`build_context_specs`.
        wait_for_traces: When ``True``, waits for all traces to be written to
            disk after running.

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

    if not wait_for_traces:
        return _run_context_specs(case, specs)

    from tests.integration.platform.data_daemon.shared.db_helpers import (
        latching_trace_write_observer,
        wait_for_all_traces_written,
    )

    # A late writer is reaped during the wait below, so observe across it.
    with latching_trace_write_observer() as observed:
        results = _run_context_specs(case, specs)
        wait_for_all_traces_written(results=results, observed=observed)
    return results


def _run_context_specs(
    case: DataDaemonTestCase,
    specs: list[ContextSpec],
) -> list[ContextResult]:
    """Run *specs* in-process or across a pool, per the case's parallelism."""
    if case.parallel_contexts == 1:
        return [context_worker(specs[0])]

    with relayed_worker_logs() as log_queue:
        with multiprocessing.Pool(
            case.parallel_contexts,
            initializer=init_worker_logging,
            initargs=(log_queue, logging.getLogger().getEffectiveLevel()),
        ) as pool:
            results: list[ContextResult] = []
            for result in pool.imap_unordered(_subprocess_context_worker, specs):
                Timer.merge_stats(result.timer_stats)
                results.append(result)
    return results


def create_testing_dataset_name(case: DataDaemonTestCase) -> str:
    """Create a unique dataset name for a test case."""
    return f"testing_dataset_{case_id(case)}_{uuid.uuid4().hex[:6]}"
