from __future__ import annotations

import os
import threading
import time
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass

import pytest

import neuracore as nc
from tests.integration.platform.data_daemon.daemon_test_cases import (
    PRE_NETWORK_INTEGRITY_CASES,
)
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
    verify_cloud_results,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    ObservedRecordingUploads,
    latching_upload_observer,
    resolve_cloud_recording_ids,
    wait_for_recording_index_for_source,
    wait_for_upload_complete_in_db,
)
from tests.integration.platform.data_daemon.shared.disk_helpers import (
    assert_rgb_trace_respects_the_recording_boundary,
)
from tests.integration.platform.data_daemon.shared.runners import (
    online_daemon_running,
    split_video_process_running,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestBatch,
    DataDaemonTestCase,
    case_ids,
    has_configured_org,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case_context import (  # noqa: E501
    ContextResult,
    RecordingControlBounds,
    build_context_specs,
    classify_split_producer_frames,
    create_testing_dataset_name,
    run_case_contexts,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    DETAIL_REALISTIC,
    MAX_TIME_TO_START_S,
    PACING_BURST_VIDEO,
    PRODUCER_CONTINUOUS,
    STOP_METHOD_CLI,
    STORAGE_STATE_DELETE,
)
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    scoped_storage_state,
    set_case_analysis_report,
)

_CASES = DataDaemonTestBatch(
    cases=PRE_NETWORK_INTEGRITY_CASES,
    storage_state_action=STORAGE_STATE_DELETE,
    stop_method=STOP_METHOD_CLI,
    producer_pacing=PACING_BURST_VIDEO,
    producer_channels=PRODUCER_CONTINUOUS,
).as_cases()


# Holdback pinned for the split-process boundary test only. The dispatcher
# retains a stopped window for 2x the holdback, and a video-only process's flush
# barrier takes roughly a second to drain its writer backlog — the same order as
# the 500ms default's 1s retention, which makes "does the tail chunk beat the
# eviction" a coin flip and a passing run meaningless. At 50ms the eviction
# deadline lands 100ms after the stop, firmly before the barrier, so the video
# survives only if the daemon waits for that process's own flush marker.
_BOUNDARY_RACE_HOLDBACK_MS = "50"


@contextmanager
def _pinned_holdback(holdback_ms: str) -> Generator[None]:
    """Set ``NCD_HOLDBACK_MS`` for the block, restoring whatever was there.

    Must wrap the daemon start: the daemon reads this once, at startup.

    Yields:
        ``None`` — the holdback override is in place while the body runs.
    """
    previous = os.environ.get("NCD_HOLDBACK_MS")
    os.environ["NCD_HOLDBACK_MS"] = holdback_ms
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("NCD_HOLDBACK_MS", None)
        else:
            os.environ["NCD_HOLDBACK_MS"] = previous


@dataclass(slots=True)
class _GateCloseObservation:
    """When the recording owner's local ``log_*`` gate closed."""

    result: float = 0.0


@contextmanager
def _watch_local_gate_close(robot: object) -> Generator[_GateCloseObservation]:
    """Record the instant this process's local recording gate closes.

    A tighter upper bracket on the window's real upper bound than "when
    ``stop_recording`` returned". The bound is stamped inside the bridge's
    ``stop_recording`` publish, and ``Robot._drain_streams_and_notify_daemon``
    clears the local handle immediately afterwards — before the flush barrier,
    which is what makes the public call take hundreds of milliseconds. So the
    handle going ``None`` is within a poll interval of the boundary, while the
    call returning is a barrier away from it.

    Polls rather than hooks anything: the gate is the recording-state manager's
    handle, and reading it is exactly what ``log_*`` does.

    Yields:
        An observation whose ``result`` is filled in with the gate-close wall
        clock by the time the block exits. Falls back to the block's own exit
        time if the gate never closed, which keeps the bracket valid (the bound
        is still somewhere before it) rather than reporting a zero.
    """
    observation = _GateCloseObservation()
    stop_polling = threading.Event()

    def poll() -> None:
        while not stop_polling.is_set():
            if robot.get_current_recording_id() is None:  # type: ignore[attr-defined]
                observation.result = time.time()
                return
            time.sleep(0.001)

    watcher = threading.Thread(target=poll, name="gate-close-watch", daemon=True)
    watcher.start()
    try:
        yield observation
    finally:
        stop_polling.set()
        watcher.join(timeout=5.0)
        if observation.result == 0.0:
            observation.result = time.time()


def _assert_online_verification_invariants(
    results: list[ContextResult],
    *,
    observed: ObservedRecordingUploads,
    timeout_seconds: float = 30.0,
) -> None:
    """Block until every recording in *results* has reached ``upload_complete``
    in the platform DB.  Must be called before cloud frame verification so
    that downloaded data reflects the fully-committed upload state.

    Upload completion is tracked in the daemon DB by the local
    ``recording_index`` correlation key.

    Recordings whose completion *observed* already latched during the record
    phase are satisfied: their rows have since been reclaimed by the recording
    reaper, so there is nothing left here to poll. Everything else is waited on
    exactly as before — not being complete yet is precisely what makes a
    recording ineligible for reclamation, so its rows are still there.
    """
    for result in results:
        for recording_index in result.recording_indexes:
            if observed.is_complete(recording_index):
                continue
            wait_for_upload_complete_in_db(recording_index, timeout_s=timeout_seconds)


@pytest.mark.parametrize("case", _CASES, ids=case_ids(_CASES))
def test_cloud_data_integrity(
    case: DataDaemonTestCase,
    clear_daemon_timer_stats,
    request: pytest.FixtureRequest,
    test_wall_timer: Callable[[], float],
) -> None:
    """Record data in online mode and verify cloud-side data integrity.

    Extends pre-network integrity (local disk timestamps + SQLite write status)
    by confirming the upload is correct on the platform side.

    - asserts no leftover daemon state before starting (isolation pre-condition)
    - records all context specs against the live platform
    - waits for every recording to reach ``upload_complete`` in the daemon DB
    - asserts exactly one daemon PID throughout
    - structural pass: verifies recording duration, byte size, and robot ID
      from the cloud (no sync required)
    - data pass: synchronises the dataset and validates per-episode frame
      counts and joint values against what was recorded
    - asserts no residual processes, files, sockets, or DB artefacts remain
      (isolation post-condition)
    """
    if not has_configured_org():
        pytest.skip(
            "Recording/playback matrix tests require NEURACORE_ORG_ID"
            " or a saved current organization."
        )

    dataset_name = create_testing_dataset_name(case)
    specs = build_context_specs(case, dataset_name=dataset_name)
    results: list[ContextResult] = []

    with scoped_storage_state(case, dataset_name=dataset_name):
        try:
            with online_daemon_running():
                assert_exactly_one_daemon_pid()
                with latching_upload_observer() as observed:
                    results = run_case_contexts(case, specs=specs)
                _assert_online_verification_invariants(results, observed=observed)
                results = resolve_cloud_recording_ids(results, observed=observed)
                verify_cloud_results(results=results, case=case)

        finally:
            set_case_analysis_report(
                request=request,
                case=case,
                results=results,
                test_wall_s=test_wall_timer(),
            )


def test_split_process_video_survives_recording_boundary() -> None:
    """A video producer in a different OS process than the recording owner
    must not lose its tail chunk at the recording boundary.

    Reproduces the real-world topology behind the RGB-tail-chunk-orphaned bug:
    one process owns ``start_recording``/``stop_recording`` for a robot while
    a separate process — sharing the same source, connected independently via
    :func:`~runners.split_video_process_running` — owns that robot's RGB
    video. The daemon's per-producer flush-marker tracking (see
    ``rust/data_daemon/src/pipeline/dispatcher.rs``) exists so the
    video-less lifecycle process's flush marker cannot vouch for the video
    process's still-open writer barrier.

    Requires ``online_daemon_running()``: the video-only process never calls
    ``start_recording`` itself, so it only learns a recording is active via
    the recording-state manager's SSE notification stream, which offline
    daemons never emit — see ``behavioural_correctness/test_multi_process_producers.py``
    for the same online-only constraint on a cross-process producer.

    Duration and resolution are pinned to the original bug report's shape:
    640x480x3 bytes/frame crosses the writer's 256 MiB chunk-flush threshold
    at frame 292 of 300 (30fps x 10s), so a chunk seals mid-recording — the
    daemon has already attributed this window's video to the producer process
    before its tail chunk (frames 292-300) is announced after the stop. The
    window retention is pinned too, so the boundary is crossed the losing way
    every run rather than on unlucky timing — see
    :data:`_BOUNDARY_RACE_HOLDBACK_MS`.

    Asserts the RGB trace reaches ``write_status='written'`` and then guards the
    boundary from both sides: every frame the video process logged strictly
    inside the recording is on disk, and no frame it logged after
    ``stop_recording`` returned is. The second half is the same bug from the
    other direction — the video process keeps logging until its stop
    notification arrives a whole SSE round trip later, and those frames share a
    chunk with the recording's own, so a daemon that takes the chunk whole ends
    up with video published after the window closed.

    Which frames those are is measured, not assumed: the wall-clock brackets
    around this process's control calls decide it, and the video process reports
    every frame it logged. A nominal ``fps * duration`` count could not do the
    job — it moves by tens of frames with the child's render throughput and with
    how long the SSE notification takes to open its logging gate, neither of
    which says anything about whether a chunk was mishandled.
    """
    if not has_configured_org():
        pytest.skip(
            "Recording/playback matrix tests require NEURACORE_ORG_ID"
            " or a saved current organization."
        )

    run_id = uuid.uuid4().hex[:10]
    dataset_name = f"split_video_dataset_{run_id}"
    robot_name = f"split_video_robot_{run_id}"
    case = DataDaemonTestCase(
        duration_sec=10,
        joint_count=0,
        video_count=1,
        image_width=640,
        image_height=480,
        video_fps=30,
        video_detail=DETAIL_REALISTIC,
        storage_state_action=STORAGE_STATE_DELETE,
    )

    robot = None
    with scoped_storage_state(case, dataset_name=dataset_name):
        try:
            with _pinned_holdback(_BOUNDARY_RACE_HOLDBACK_MS), online_daemon_running():
                assert_exactly_one_daemon_pid()
                nc.create_dataset(dataset_name)

                with split_video_process_running(
                    robot_name=robot_name,
                    dataset_name=dataset_name,
                    camera_name="camera_0",
                    case=case,
                ) as logged_frames:
                    robot = nc.connect_robot(robot_name, overwrite=False)
                    # Both control calls are bracketed on the wall clock: the
                    # window's real bounds are stamped inside them, so this is
                    # as tightly as the video process's frames can be attributed
                    # to the recording (see RecordingControlBounds).
                    start_called_at = time.time()
                    nc.start_recording(robot_name=robot_name, timestamp=0.0)
                    start_returned_at = time.time()
                    handle = robot.get_current_recording_id()
                    recording_index = wait_for_recording_index_for_source(
                        str(robot.id),
                        int(robot.instance),
                        timeout_s=MAX_TIME_TO_START_S,
                    )
                    time.sleep(case.duration_sec)
                    stop_called_at = time.time()
                    with _watch_local_gate_close(robot) as gate_closed_at:
                        # Deliberately not `wait=True`: this test asserts what
                        # reached local disk, and the barrier it needs is the
                        # video process's own flush (below) plus the daemon DB
                        # poll in the assertion — not the cloud upload. Waiting
                        # also polls `/recording/{id}/traces/complete`, which
                        # 404s (raising, not returning False) until the daemon
                        # has registered this recording's traces, so asking for
                        # it here trades a real assertion for a race.
                        nc.stop_recording(
                            robot_name=robot_name,
                            timestamp=float(case.duration_sec),
                        )
                    # Keep the video process logging past the stop, as a real
                    # camera loop does — its tail chunk is sealed by the flush
                    # barrier it runs on the way out.
                    time.sleep(2.0)

                bounds = RecordingControlBounds(
                    handle=handle,
                    start_called_at=start_called_at,
                    start_returned_at=start_returned_at,
                    stop_called_at=stop_called_at,
                    # The gate-close instant, not the instant `stop_recording`
                    # returned: both are valid upper brackets on the window's
                    # bound, and this one is ~2ms wide instead of ~200ms, which
                    # is the difference between policing the video process's
                    # post-stop frames and calling them unknowable.
                    stop_returned_at=gate_closed_at.result,
                )
                owed, forbidden_before_start, forbidden_after_stop = (
                    classify_split_producer_frames(
                        logged_frames["RGB_IMAGES/camera_0"], bounds
                    )
                )
                assert_rgb_trace_respects_the_recording_boundary(
                    recording_index,
                    owed_timestamps=[frame.timestamp for frame in owed],
                    forbidden_before_start_timestamps=[
                        frame.timestamp for frame in forbidden_before_start
                    ],
                    forbidden_after_stop_timestamps=[
                        frame.timestamp for frame in forbidden_after_stop
                    ],
                )
        finally:
            if robot is not None:
                robot.close()


def test_split_process_video_survives_back_to_back_recording_boundaries() -> None:
    """A video producer in a different OS process must not lose a recording's
    leading frames when it starts immediately after the previous one stops.

    The mirror, at the START boundary, of
    :func:`test_split_process_video_survives_recording_boundary`. That test
    proves a chunk straddling a recording's STOP is cut rather than taken
    whole (``ef3cf6c7``); this one polices the START side.

    A chunk routes by its *open* stamp (the ``publish_timestamp_ns`` of its
    first frame), so a chunk opened during the first recording and still open
    when the second starts routes whole into the *first* recording's window.
    The dispatcher's per-frame stop cut (``frames_inside_window``, see
    ``rust/data_daemon/src/pipeline/dispatcher.rs``) drops the frames published
    after the first recording stopped, and nothing reroutes them — so the
    second recording simply loses its leading video. The video-only process
    never calls ``start_recording``, so the writer's boundary split armed
    inside that call (``arm_boundary_split``, see
    ``rust/data_daemon_bridge/src/lib.rs``) never fires for it; instead
    ``Robot.arm_video_boundary_if_new_recording`` arms it from the video log
    path the first time this process forwards a frame under a new handle.

    Two defects had to be closed for this to hold, both of which this test
    reproduces on the first run of a second recording. The other is that
    ``StopRecording`` names a source, not a recording: the video process's
    remote-stop drain used to publish one a whole SSE round trip late, closing
    the *next* recording's window ~800 ms in — see
    ``Robot._drain_streams_and_notify_daemon``.

    Runs one continuous :func:`~runners.split_video_process_running` child
    across ``case.recording_count`` recordings started back-to-back, and
    classifies each recording's frames separately from the wall-clock
    brackets around its own control calls — mirroring how ``context_worker``
    builds ``bounds_by_disk_key`` per recording for a continuous producer.

    Duration and resolution are pinned against the same 256 MiB writer
    chunk-flush threshold the single-recording test pins against: at
    640x480x3 bytes/frame the threshold trips at a cumulative 292 frames. The
    video-only process never rolls its chunk on a lifecycle event (that split
    is armed only in the process that owns ``start_recording``), so its chunk
    keeps accumulating across both recordings regardless of the gap between
    them. At 30fps, 6s (~180 frames) alone stays under the threshold, so the
    chunk is still open when the second recording starts; the remaining
    ~112-frame headroom is then used up partway through the second recording,
    which is what leaves it missing a leading run of frames if the
    start-boundary defect is present, while the frames after the natural roll
    land in a fresh, correctly-routed chunk — a partial-loss signature rather
    than a total one.

    The start bracket (``[start_called_at, start_returned_at]``) needs no
    ``_watch_local_gate_close``-style tightening the way the stop bracket
    does: ``Robot.start_recording`` opens the local gate and publishes the
    window in one call with no flush barrier (unlike ``stop_recording``,
    which drains streams and runs a writer barrier before notifying the
    daemon), so the two timestamps taken immediately around the call already
    bracket the real boundary tightly.
    """
    if not has_configured_org():
        pytest.skip(
            "Recording/playback matrix tests require NEURACORE_ORG_ID"
            " or a saved current organization."
        )

    run_id = uuid.uuid4().hex[:10]
    dataset_name = f"split_video_dataset_{run_id}"
    robot_name = f"split_video_robot_{run_id}"
    case = DataDaemonTestCase(
        duration_sec=6,
        recording_count=2,
        joint_count=0,
        video_count=1,
        image_width=640,
        image_height=480,
        video_fps=30,
        video_detail=DETAIL_REALISTIC,
        storage_state_action=STORAGE_STATE_DELETE,
    )

    robot = None
    with scoped_storage_state(case, dataset_name=dataset_name):
        try:
            with _pinned_holdback(_BOUNDARY_RACE_HOLDBACK_MS), online_daemon_running():
                assert_exactly_one_daemon_pid()
                nc.create_dataset(dataset_name)

                bounds_by_recording: dict[int, RecordingControlBounds] = {}
                with split_video_process_running(
                    robot_name=robot_name,
                    dataset_name=dataset_name,
                    camera_name="camera_0",
                    case=case,
                ) as logged_frames:
                    robot = nc.connect_robot(robot_name, overwrite=False)
                    previous_index = 0
                    for _ in range(case.recording_count):
                        # Both control calls are bracketed on the wall clock,
                        # exactly as in the single-recording test — see
                        # RecordingControlBounds.
                        start_called_at = time.time()
                        nc.start_recording(robot_name=robot_name, timestamp=0.0)
                        start_returned_at = time.time()
                        handle = robot.get_current_recording_id()
                        recording_index = wait_for_recording_index_for_source(
                            str(robot.id),
                            int(robot.instance),
                            after_index=previous_index,
                            timeout_s=MAX_TIME_TO_START_S,
                        )
                        previous_index = recording_index
                        time.sleep(case.duration_sec)
                        stop_called_at = time.time()
                        with _watch_local_gate_close(robot) as gate_closed_at:
                            # Deliberately not `wait=True` — see the
                            # single-recording test for why, doubly so here:
                            # waiting on the upload pipeline would also widen
                            # the gap before the next recording's start,
                            # undermining "back-to-back".
                            nc.stop_recording(
                                robot_name=robot_name,
                                timestamp=float(case.duration_sec),
                            )
                        bounds_by_recording[recording_index] = RecordingControlBounds(
                            handle=handle,
                            start_called_at=start_called_at,
                            start_returned_at=start_returned_at,
                            stop_called_at=stop_called_at,
                            stop_returned_at=gate_closed_at.result,
                        )
                    # As in the single-recording test: give the still-running
                    # video process time to seal and announce its tail chunk
                    # on the way out, via its own flush barrier.
                    time.sleep(2.0)

                for recording_index, bounds in bounds_by_recording.items():
                    owed, forbidden_before_start, forbidden_after_stop = (
                        classify_split_producer_frames(
                            logged_frames["RGB_IMAGES/camera_0"], bounds
                        )
                    )
                    assert_rgb_trace_respects_the_recording_boundary(
                        recording_index,
                        owed_timestamps=[frame.timestamp for frame in owed],
                        forbidden_before_start_timestamps=[
                            frame.timestamp for frame in forbidden_before_start
                        ],
                        forbidden_after_stop_timestamps=[
                            frame.timestamp for frame in forbidden_after_stop
                        ],
                    )
        finally:
            if robot is not None:
                robot.close()
