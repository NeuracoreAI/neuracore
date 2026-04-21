from dataclasses import replace

from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestCase,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    DURATION_MODE_FIXED,
    DURATION_MODE_VARIABLE,
    MODE_STAGGERED,
    PRODUCER_PER_THREAD,
    TIMESTAMP_MODE_REAL,
)

BASE_CASES = (
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
        timestamp_mode=TIMESTAMP_MODE_REAL,
    ),
)

PRE_NETWORK_INTEGRITY_CASES = BASE_CASES
NETWORK_INTEGRITY_CASES = BASE_CASES + (
    DataDaemonTestCase(
        duration_sec=10,
        joint_count=7,
        parallel_contexts=1,
        recording_count=1,
        wait=True,
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
        timestamp_mode=TIMESTAMP_MODE_REAL,
        wait=True,
    ),
)

PRE_NETWORK_PERFORMANCE_CASES = (
    # High frequency robot control at 210Hz joint data
    # Tests: high-frequency sampling, temporal jitter, joint-only streaming
    DataDaemonTestCase(
        duration_sec=60,
        joint_count=7,
        video_count=0,
        parallel_contexts=1,
        recording_count=5,
        context_duration_mode=DURATION_MODE_FIXED,
        joint_fps=210,
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
    # Large number of joints without cameras (1000 joints)
    # Tests: high joint dimensionality, memory efficiency, sensor-only workload
    DataDaemonTestCase(
        duration_sec=30,
        joint_count=1000,
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
    ),
)

NETWORK_PERFORMANCE_CASES = (
    *[
        c
        for pair in PRE_NETWORK_PERFORMANCE_CASES
        for c in (pair, replace(pair, wait=True))
    ],
)
