"""The shared case matrix for the data-daemon integration suites.

Every case owns an axis combination no other case covers. A case that only
re-runs a covered combination at a different size belongs in a performance
workload, not here.
"""

from dataclasses import replace

from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestCase,
    PerThread,
    ProcessPerCamera,
    ProcessPerLimbPerCamera,
    SeparateProcessRecordingControl,
    Synchronous,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    CONTROL_REMOTE,
    CONTROL_SPLIT_PROCESS,
    DETAIL_FLAT,
    DETAIL_REALISTIC,
    DURATION_MODE_FIXED,
    DURATION_MODE_VARIABLE,
    MODE_STAGGERED,
    PACING_BURST_VIDEO,
    PACING_SATURATE,
    STREAM_JOINT_POSITIONS,
    STREAM_JOINT_TORQUES,
    STREAM_JOINT_VELOCITIES,
)

# Runs offline against disk and the DB, and online against the cloud. Local
# control only — the offline daemon has no backend to take a remote call from.
PRE_NETWORK_INTEGRITY_CASES = (
    # The joint path with nothing else in the way: one thread, one window.
    # Separates a trace write or upload defect from an encode one.
    Synchronous(
        duration_sec=10,
        joint_count=7,
        parallel_contexts=1,
        recording_count=1,
        producer_pacing=PACING_SATURATE,
    ),
    # RGB, depth and joints from one thread — the `float32` depth round trip.
    # Frame identity resolves by sync-iteration order here, the easy direction.
    Synchronous(
        duration_sec=10,
        joint_count=7,
        recording_count=1,
        video_count=1,
        image_height=64,
        image_width=64,
        depth_count=1,
        depth_mode="float32",
        producer_pacing=PACING_SATURATE,
        video_detail=DETAIL_FLAT,
    ),
    # Video and joints at different rates inside a variable-length window.
    # Catches a per-stream rate assumption in how sync points are built.
    Synchronous(
        duration_sec=10,
        joint_count=7,
        recording_count=1,
        video_count=1,
        image_height=64,
        image_width=64,
        context_duration_mode=DURATION_MODE_VARIABLE,
        video_fps=30,
        joint_fps=15,
        producer_pacing=PACING_SATURATE,
        video_detail=DETAIL_FLAT,
    ),
    # `float16` depth under staggered parallel contexts and live boundaries.
    # Frame identity cannot lean on capture order, which is the point of it.
    PerThread(
        duration_sec=10,
        joint_count=7,
        recording_count=4,
        video_count=1,
        image_height=64,
        image_width=64,
        context_duration_mode=DURATION_MODE_VARIABLE,
        video_fps=30,
        joint_fps=15,
        parallel_contexts=2,
        mode=MODE_STAGGERED,
        depth_count=1,
        depth_mode="float16",
        producer_pacing=PACING_SATURATE,
        video_detail=DETAIL_FLAT,
    ),
    # The transport's high-rate case, four boundaries deep with phase jitter.
    # 250Hz is its de-rated ceiling; above that a run measures scheduling noise.
    PerThread(
        duration_sec=10,
        recording_count=4,
        video_count=1,
        image_height=120,
        image_width=120,
        video_fps=120,
        joint_fps=250,
        random_phase=True,
        wait=False,
        producer_pacing=PACING_SATURATE,
        video_detail=DETAIL_FLAT,
    ),
    # The only lossy-codec case: asserts the lossless archive is *absent*.
    # Deliberately the plainest RGB shape, so the codec is the sole variable.
    Synchronous(
        duration_sec=10,
        joint_count=7,
        recording_count=1,
        video_count=1,
        image_height=64,
        image_width=64,
        video_codec="h264_medium",
        producer_pacing=PACING_SATURATE,
        video_detail=DETAIL_FLAT,
    ),
    # A camera child outliving fifteen windows; early ones must retire first.
    # Long-lived children are where chunk retention leaks become visible.
    ProcessPerCamera(
        duration_sec=6,
        recording_count=15,
        joint_count=7,
        video_count=1,
        image_width=640,
        image_height=480,
        video_fps=30,
        producer_pacing=PACING_SATURATE,
        video_detail=DETAIL_FLAT,
    ),
    # Two camera children: frames, chunks and windows stay isolated per child.
    # Cross-talk between children needs more than one child to show at all.
    ProcessPerCamera(
        duration_sec=6,
        recording_count=8,
        joint_count=7,
        video_count=2,
        image_width=256,
        image_height=256,
        video_fps=30,
        producer_pacing=PACING_SATURATE,
        video_detail=DETAIL_FLAT,
    ),
    # RGB and depth of one device share a child, as an RGBD driver's do.
    # Carries the depth path across a process boundary as well.
    ProcessPerCamera(
        duration_sec=6,
        recording_count=8,
        joint_count=7,
        video_count=1,
        image_width=64,
        image_height=64,
        video_fps=30,
        depth_count=1,
        depth_mode="float16",
        producer_pacing=PACING_SATURATE,
        video_detail=DETAIL_FLAT,
    ),
    # Every stream in one child; the owner process only opens and closes.
    # The baseline for a producer that learns its window second-hand.
    SeparateProcessRecordingControl(
        duration_sec=6,
        recording_count=8,
        joint_count=7,
        producer_pacing=PACING_SATURATE,
    ),
    # An explicit placement splitting one limb's joints by data type.
    # The only case that overrides the default stream-to-process mapping.
    SeparateProcessRecordingControl(
        duration_sec=6,
        recording_count=8,
        joint_count=7,
        producer_pacing=PACING_SATURATE,
        producer_process_streams=(
            (STREAM_JOINT_POSITIONS,),
            (STREAM_JOINT_VELOCITIES, STREAM_JOINT_TORQUES),
        ),
    ),
    # Two limb children writing the same data type at the same time.
    # No camera split does this — a camera child owns its channel alone.
    ProcessPerLimbPerCamera(
        duration_sec=6,
        recording_count=8,
        joint_count=8,
        producer_pacing=PACING_SATURATE,
    ),
    # The heaviest cross-process shape: four cameras, two limbs, 15 windows.
    # Local control, because the offline suite runs this list too.
    ProcessPerLimbPerCamera(
        duration_sec=15,
        recording_count=15,
        joint_count=7,
        video_count=4,
        image_width=640,
        image_height=480,
        video_fps=30,
        producer_pacing=PACING_SATURATE,
        video_detail=DETAIL_FLAT,
    ),
)

# Control paths that need a backend to drive them, so they cannot run offline.
NETWORK_ONLY_INTEGRITY_CASES = (
    # Web-initiated start and stop: the daemon follows with no SDK call at all.
    # Threads log through the announcement, so a late window loses their data.
    PerThread(
        duration_sec=6,
        recording_count=4,
        joint_count=7,
        recording_control=CONTROL_REMOTE,
        producer_pacing=PACING_SATURATE,
    ),
    # Remote control with the producing in a child that never brackets a window.
    # Two levels of indirection between the start and the process writing.
    SeparateProcessRecordingControl(
        duration_sec=6,
        recording_count=4,
        joint_count=7,
        recording_control=CONTROL_REMOTE,
    ),
    # Split control with video still in flight when the peer's stop lands.
    # Frames mid-encode at the boundary are what this split exposed.
    PerThread(
        duration_sec=6,
        recording_count=4,
        joint_count=7,
        video_count=1,
        image_width=64,
        image_height=64,
        video_fps=30,
        video_detail=DETAIL_FLAT,
        recording_control=CONTROL_SPLIT_PROCESS,
    ),
    # Remote control at the heaviest cross-process shape, fifteen windows deep.
    # Propagation has to hold under encode backlog, not only when idle.
    ProcessPerLimbPerCamera(
        duration_sec=15,
        recording_count=15,
        joint_count=7,
        video_count=4,
        image_width=640,
        image_height=480,
        video_fps=30,
        producer_pacing=PACING_SATURATE,
        video_detail=DETAIL_FLAT,
        recording_control=CONTROL_REMOTE,
    ),
    # Split control where the producing is a third process again.
    # Nobody in the writing path ever calls start or stop.
    SeparateProcessRecordingControl(
        duration_sec=6,
        recording_count=4,
        joint_count=7,
        recording_control=CONTROL_SPLIT_PROCESS,
    ),
)

PRE_NETWORK_PERFORMANCE_CASES = (
    # Joint-only throughput at control rate, with no encode in the way.
    # 100Hz is the de-rated CI value; the VM ceiling measured out at 250Hz.
    Synchronous(
        duration_sec=20,
        joint_count=7,
        video_count=0,
        parallel_contexts=1,
        recording_count=8,
        context_duration_mode=DURATION_MODE_FIXED,
        joint_fps=100,
    ),
    # The same joints at the same rate over five times the boundaries. Window
    # length is this one's axis, so it keeps the capture time and spends it in
    # forty windows: against the case above, per-recording cost is the only
    # thing that can differ.
    Synchronous(
        duration_sec=4,
        joint_count=7,
        video_count=0,
        parallel_contexts=1,
        recording_count=40,
        context_duration_mode=DURATION_MODE_FIXED,
        joint_fps=100,
    ),
    # Eight robots contending over four windows each, of varying length.
    # The one case whose load is per-context contention, not per-stream rate.
    PerThread(
        duration_sec=20,
        joint_count=7,
        video_count=1,
        image_width=256,
        image_height=256,
        parallel_contexts=8,
        recording_count=32,
        joint_fps=80,
        context_duration_mode=DURATION_MODE_VARIABLE,
        video_fps=30,
        producer_pacing=PACING_BURST_VIDEO,
        video_detail=DETAIL_FLAT,
    ),
    # Joint dimensionality: a trace per joint per type, each a fixed upload cost.
    # TODO: restore joint_count=1000 once that per-trace overhead is addressed —
    # at 1000 these dominate the suite's runtime and put enough load on the
    # backend to destabilise it.
    Synchronous(
        duration_sec=20,
        joint_count=100,
        video_count=0,
        parallel_contexts=1,
        recording_count=8,
    ),
    # Five-minute 1080p windows: memory growth and spool backlog over a long
    # run, which is why this one is exempt from the shared window length. Takes
    # variable durations and per-frame phase jitter as the harder of the two —
    # the offsets are deterministic, so a failure is drift, not scheduling noise.
    # 15fps is the CI encode-throughput ceiling, not a product limit.
    Synchronous(
        duration_sec=300,
        joint_count=10,
        video_count=1,
        image_width=1920,
        image_height=1080,
        parallel_contexts=2,
        recording_count=8,
        context_duration_mode=DURATION_MODE_VARIABLE,
        video_fps=15,
        joint_fps=15,
        random_phase=True,
        producer_pacing=PACING_BURST_VIDEO,
        video_detail=DETAIL_FLAT,
    ),
    # The only workload the encoder does real work in: every other video case
    # feeds it a solid fill, so their numbers are a floor rather than a cost.
    Synchronous(
        duration_sec=20,
        joint_count=10,
        video_count=1,
        image_width=1920,
        image_height=1080,
        parallel_contexts=1,
        recording_count=8,
        video_fps=15,
        joint_fps=15,
        producer_pacing=PACING_BURST_VIDEO,
        video_detail=DETAIL_REALISTIC,
    ),
    # Two RGBD devices: the depth path costs a `float32` array per frame and an
    # encode of its own. No other workload logs depth at all.
    PerThread(
        duration_sec=20,
        joint_count=7,
        video_count=2,
        image_width=640,
        image_height=480,
        recording_count=8,
        video_fps=15,
        joint_fps=15,
        depth_count=2,
        depth_mode="float32",
        producer_pacing=PACING_BURST_VIDEO,
        video_detail=DETAIL_FLAT,
    ),
    # Peak encode and transport inside a single window, no boundary to cross —
    # the recording count is this one's axis, so it keeps the shared window
    # length and takes one of them.
    PerThread(
        duration_sec=20,
        recording_count=1,
        video_count=1,
        image_height=120,
        image_width=120,
        video_fps=120,
        joint_fps=250,
        wait=False,
        producer_pacing=PACING_BURST_VIDEO,
        video_detail=DETAIL_FLAT,
    ),
    # The same peak rate over eight windows with phase jitter — the only
    # workload paying boundary cost at rate, and the shape the stop SLA is
    # drawn from. Against the case above, boundary cost is the only difference.
    PerThread(
        duration_sec=20,
        recording_count=8,
        video_count=1,
        image_height=120,
        image_width=120,
        video_fps=120,
        joint_fps=250,
        random_phase=True,
        wait=False,
        producer_pacing=PACING_BURST_VIDEO,
        video_detail=DETAIL_FLAT,
    ),
    # Two camera children at moderate resolution, windows of varying length.
    # Measures transport cost per child rather than per stream.
    ProcessPerCamera(
        duration_sec=20,
        joint_count=7,
        video_count=2,
        image_width=256,
        image_height=256,
        recording_count=8,
        joint_fps=80,
        video_fps=30,
        context_duration_mode=DURATION_MODE_VARIABLE,
        producer_pacing=PACING_BURST_VIDEO,
        video_detail=DETAIL_FLAT,
    ),
    # One 1080p child: the largest frame the transport has to carry across.
    # Encode backlog, not frame count, is what fails this one.
    ProcessPerCamera(
        duration_sec=20,
        joint_count=10,
        video_count=1,
        image_width=1920,
        image_height=1080,
        recording_count=8,
        joint_fps=15,
        video_fps=15,
        producer_pacing=PACING_BURST_VIDEO,
        video_detail=DETAIL_FLAT,
    ),
    # The same pixels per second as the case above, arriving as four streams
    # instead of one: what the daemon pays per stream rather than per pixel.
    ProcessPerCamera(
        duration_sec=20,
        joint_count=10,
        video_count=4,
        image_width=960,
        image_height=540,
        recording_count=8,
        joint_fps=15,
        video_fps=15,
        producer_pacing=PACING_BURST_VIDEO,
        video_detail=DETAIL_FLAT,
    ),
    # Two limb children and two camera children against the one daemon.
    # The only workload where a data type arrives from two processes at once.
    ProcessPerLimbPerCamera(
        duration_sec=20,
        joint_count=8,
        video_count=2,
        image_width=256,
        image_height=256,
        recording_count=8,
        joint_fps=80,
        video_fps=30,
        producer_pacing=PACING_BURST_VIDEO,
        video_detail=DETAIL_FLAT,
    ),
)

# Timing workloads whose control path needs a backend, so they cannot run
# offline. The pair differs only in who closes the window, so the cost of the
# control path is the one thing the two numbers can disagree about.
NETWORK_ONLY_PERFORMANCE_CASES = (
    # Web-initiated start and stop under load: propagation timed, not just
    # proven. A remote stop cannot block on the upload, so it has no wait twin.
    PerThread(
        duration_sec=20,
        joint_count=7,
        video_count=1,
        image_width=256,
        image_height=256,
        recording_count=8,
        joint_fps=80,
        video_fps=30,
        recording_control=CONTROL_REMOTE,
        producer_pacing=PACING_BURST_VIDEO,
        video_detail=DETAIL_FLAT,
    ),
    # This process starts every window and a peer stops it, both over the
    # notification stream. Windows close no faster than that round trip.
    PerThread(
        duration_sec=20,
        joint_count=7,
        video_count=1,
        image_width=256,
        image_height=256,
        recording_count=8,
        joint_fps=80,
        video_fps=30,
        recording_control=CONTROL_SPLIT_PROCESS,
        producer_pacing=PACING_BURST_VIDEO,
        video_detail=DETAIL_FLAT,
    ),
)


def _with_wait_variants(
    cases: tuple[DataDaemonTestCase, ...],
) -> tuple[DataDaemonTestCase, ...]:
    """Return each case twice: returning from stop at once, and blocking on it.

    Only the network suite cares — ``wait=True`` measures the upload, which the
    offline daemon never performs.
    """
    return tuple(
        variant for case in cases for variant in (case, replace(case, wait=True))
    )


NETWORK_PERFORMANCE_CASES = (
    *_with_wait_variants(PRE_NETWORK_PERFORMANCE_CASES),
    *NETWORK_ONLY_PERFORMANCE_CASES,
)
