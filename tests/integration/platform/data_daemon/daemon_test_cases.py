from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestCase,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    DETAIL_REALISTIC,
    DURATION_MODE_FIXED,
    DURATION_MODE_VARIABLE,
    MODE_STAGGERED,
    PACING_BURST_ALL,
    PRODUCER_PER_THREAD,
)

PRE_NETWORK_INTEGRITY_CASES = (
    DataDaemonTestCase(
        duration_sec=10,
        joint_count=7,
        parallel_contexts=1,
        recording_count=1,
    ),
    DataDaemonTestCase(
        duration_sec=10,
        joint_count=7,
        recording_count=1,
        video_count=1,
        image_height=64,
        image_width=64,
        # Depth alongside RGB and joint data, logged via the synchronous
        # producer (the default `producer_channels`). Covers `float32` —
        # `float16` is covered below via the threaded-producer case, so
        # together the two selected cases exercise both dtypes and both
        # producer modes.
        depth_count=1,
        depth_mode="float32",
    ),
    DataDaemonTestCase(
        duration_sec=10,
        joint_count=7,
        recording_count=1,
        video_count=1,
        image_height=64,
        image_width=64,
        context_duration_mode=DURATION_MODE_VARIABLE,
        video_fps=30,
        joint_fps=15,
    ),
    DataDaemonTestCase(
        duration_sec=10,
        joint_count=7,
        recording_count=4,
        video_count=1,
        image_height=64,
        image_width=64,
        context_duration_mode=DURATION_MODE_VARIABLE,
        video_fps=30,
        joint_fps=15,
        producer_channels=PRODUCER_PER_THREAD,
        parallel_contexts=2,
        mode=MODE_STAGGERED,
        # Depth via the threaded producer, covering `float16` under a more
        # complex scenario (multiple recordings, staggered contexts) — a
        # good stress case for the synchronized-frame-identity mapping used
        # in the depth round-trip assertion (see assertions.py), since it
        # can't rely on sync-iteration order lining up with capture order.
        depth_count=1,
        depth_mode="float16",
    ),
    DataDaemonTestCase(
        duration_sec=10,
        recording_count=1,
        video_count=1,
        image_height=120,
        image_width=120,
        video_fps=120,
        joint_fps=250,  # previously 1000 but flaky due to sync
        producer_channels=PRODUCER_PER_THREAD,
        wait=False,
    ),
    DataDaemonTestCase(
        duration_sec=10,
        recording_count=4,
        video_count=1,
        image_height=120,
        image_width=120,
        video_fps=120,
        joint_fps=250,  # previously 500 but flaky due to sync
        producer_channels=PRODUCER_PER_THREAD,
        random_phase=True,
        wait=False,
    ),
    DataDaemonTestCase(
        duration_sec=10,
        joint_count=15,
        recording_count=30,
        video_count=2,
        image_height=480,
        image_width=640,
        video_codec="h264_medium",
        video_detail=DETAIL_REALISTIC,
        producer_pacing=PACING_BURST_ALL,
    ),
)

PRE_NETWORK_PERFORMANCE_CASES = (
    # High frequency robot control at 100Hz joint data
    # Tests: high-frequency sampling, temporal jitter, joint-only streaming
    DataDaemonTestCase(
        duration_sec=60,
        joint_count=7,
        video_count=0,
        parallel_contexts=1,
        recording_count=5,
        context_duration_mode=DURATION_MODE_FIXED,
        joint_fps=100,
    ),
    # High number of medium-throughput robots with synchronized
    # recordings. Tests: multi-robot contention, mixed data types,
    # moderate-res cameras (256x256),
    # one producer thread per robot, back-to-back recordings
    DataDaemonTestCase(
        duration_sec=20,
        joint_count=7,
        video_count=1,
        image_width=256,
        image_height=256,
        parallel_contexts=8,
        recording_count=16,
        joint_fps=80,
        producer_channels=PRODUCER_PER_THREAD,
        context_duration_mode=DURATION_MODE_VARIABLE,
        video_fps=30,
    ),
    # Large number of joints without cameras
    # Tests: high joint dimensionality, memory efficiency, sensor-only workload
    #
    # TODO: restore joint_count=1000 once per-trace upload overhead is fixed.
    # One trace is produced per joint per data type and each pays a fixed
    # upload cost regardless of size, so 1000 joints dominates the suite's
    # runtime and puts enough load on the backend to destabilise it.
    DataDaemonTestCase(
        duration_sec=30,
        joint_count=100,
        video_count=0,
        parallel_contexts=1,
        recording_count=3,
    ),
    # 3x longer duration recordings
    # Tests: long-running stability, memory leak detection, large dataset
    # accumulation
    DataDaemonTestCase(
        duration_sec=300,
        joint_count=10,
        video_count=1,
        image_width=1920,
        image_height=1080,
        parallel_contexts=2,
        recording_count=16,
        context_duration_mode=DURATION_MODE_FIXED,
        video_fps=15,
        joint_fps=15,
    ),
    DataDaemonTestCase(
        duration_sec=300,
        joint_count=10,
        video_count=1,
        image_width=1920,
        image_height=1080,
        parallel_contexts=2,
        recording_count=16,
        context_duration_mode=DURATION_MODE_VARIABLE,
        video_fps=15,
        joint_fps=15,
        random_phase=True,
    ),
    DataDaemonTestCase(
        duration_sec=10,
        recording_count=1,
        video_count=1,
        image_height=120,
        image_width=120,
        video_fps=120,
        joint_fps=250,  # previously 1000 but flaky due to sync
        producer_channels=PRODUCER_PER_THREAD,
        wait=False,
    ),
)

NETWORK_PERFORMANCE_CASES = (
    # High frequency robot control at 100Hz joint data
    # Tests: high-frequency sampling, temporal jitter, joint-only streaming
    DataDaemonTestCase(
        duration_sec=60,
        joint_count=7,
        video_count=0,
        parallel_contexts=1,
        recording_count=5,
        context_duration_mode=DURATION_MODE_FIXED,
        joint_fps=100,
    ),
    DataDaemonTestCase(
        duration_sec=60,
        joint_count=7,
        video_count=0,
        parallel_contexts=1,
        recording_count=5,
        context_duration_mode=DURATION_MODE_FIXED,
        joint_fps=100,
        wait=True,
    ),
    # High number of medium-throughput robots with synchronized
    # recordings. Tests: multi-robot contention, mixed data types,
    # moderate-res cameras (256x256),
    # one producer thread per robot, back-to-back recordings
    DataDaemonTestCase(
        duration_sec=20,
        joint_count=7,
        video_count=1,
        image_width=256,
        image_height=256,
        parallel_contexts=8,
        recording_count=16,
        joint_fps=80,
        producer_channels=PRODUCER_PER_THREAD,
        context_duration_mode=DURATION_MODE_VARIABLE,
        video_fps=30,
    ),
    DataDaemonTestCase(
        duration_sec=20,
        joint_count=7,
        video_count=1,
        image_width=256,
        image_height=256,
        parallel_contexts=8,
        recording_count=16,
        joint_fps=80,
        producer_channels=PRODUCER_PER_THREAD,
        context_duration_mode=DURATION_MODE_VARIABLE,
        video_fps=30,
        wait=True,
    ),
    # Large number of joints without cameras
    # Tests: high joint dimensionality, memory efficiency, sensor-only workload
    #
    # TODO: restore joint_count=1000 once per-trace upload overhead is fixed.
    # One trace is produced per joint per data type and each pays a fixed
    # upload cost regardless of size, so 1000 joints dominates the suite's
    # runtime and puts enough load on the backend to destabilise it.
    DataDaemonTestCase(
        duration_sec=30,
        joint_count=100,
        video_count=0,
        parallel_contexts=1,
        recording_count=3,
    ),
    DataDaemonTestCase(
        duration_sec=30,
        joint_count=100,
        video_count=0,
        parallel_contexts=1,
        recording_count=3,
        wait=True,
    ),
    # 3x longer duration recordings
    # Tests: long-running stability, memory leak detection, large dataset
    # accumulation
    DataDaemonTestCase(
        duration_sec=300,
        joint_count=10,
        video_count=1,
        image_width=1920,
        image_height=1080,
        parallel_contexts=2,
        recording_count=16,
        context_duration_mode=DURATION_MODE_FIXED,
        video_fps=15,
        joint_fps=15,
    ),
    DataDaemonTestCase(
        duration_sec=300,
        joint_count=10,
        video_count=1,
        image_width=1920,
        image_height=1080,
        parallel_contexts=2,
        recording_count=16,
        context_duration_mode=DURATION_MODE_FIXED,
        video_fps=15,
        joint_fps=15,
        wait=True,
    ),
    DataDaemonTestCase(
        duration_sec=300,
        joint_count=10,
        video_count=1,
        image_width=1920,
        image_height=1080,
        parallel_contexts=2,
        recording_count=16,
        context_duration_mode=DURATION_MODE_VARIABLE,
        video_fps=15,
        joint_fps=15,
        random_phase=True,
    ),
    DataDaemonTestCase(
        duration_sec=300,
        joint_count=10,
        video_count=1,
        image_width=1920,
        image_height=1080,
        parallel_contexts=2,
        recording_count=16,
        context_duration_mode=DURATION_MODE_VARIABLE,
        video_fps=15,
        joint_fps=15,
        random_phase=True,
        wait=True,
    ),
    DataDaemonTestCase(
        duration_sec=10,
        recording_count=1,
        video_count=1,
        image_height=120,
        image_width=120,
        video_fps=120,
        joint_fps=250,
        producer_channels=PRODUCER_PER_THREAD,
    ),
    DataDaemonTestCase(
        duration_sec=10,
        recording_count=1,
        video_count=1,
        image_height=120,
        image_width=120,
        video_fps=120,
        joint_fps=250,
        producer_channels=PRODUCER_PER_THREAD,
        wait=True,
    ),
    DataDaemonTestCase(
        duration_sec=10,
        recording_count=4,
        video_count=1,
        image_height=120,
        image_width=120,
        video_fps=120,
        joint_fps=250,
        producer_channels=PRODUCER_PER_THREAD,
        random_phase=True,
        wait=False,
    ),
)
