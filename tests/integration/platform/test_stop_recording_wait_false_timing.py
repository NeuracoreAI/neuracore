import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
from neuracore_types import DataType

import neuracore as nc
from neuracore.core.robot import Robot
from neuracore.data_daemon.communications_management.producer_channel import ProducerChannel
from neuracore.data_daemon.helpers import get_daemon_db_path, get_daemon_pid_path
from neuracore.data_daemon.lifecycle.daemon_os_control import (
    launch_new_daemon_subprocess,
)
from recording_playback_shared import daemon_cleanup

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
sys.path.append(str(REPO_ROOT / "examples"))

# ruff: noqa: E402
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH
from recording_playback_shared import encode_frame_number

logger = logging.getLogger(__name__)

MAX_TIME_TO_START_S = 20
MAX_TIME_TO_STOP_S = 3
_TEST_DAEMON_ENV_OVERRIDES = {
    "NCD_BANDWIDTH_LIMIT": str(200 * 1024 * 1024),
    "NCD_MAX_CONCURRENT_UPLOADS": "30",
    # This test is transport-focused and stops at bridge cutoff observation,
    # so keep the daemon offline to avoid upload-side resource noise.
    "NCD_OFFLINE": "1",
}


class Timer:
    _stats: dict[str, dict[str, float]] = {}

    def __init__(
        self,
        *,
        label: str,
        max_time: float,
        always_log: bool = False,
        log_threshold: float | None = None,
    ) -> None:
        self.label = label
        self.max_time = max_time
        self.always_log = always_log
        self.log_threshold = log_threshold

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, *_args) -> bool:
        elapsed = time.perf_counter() - self.start
        stats = self._stats.setdefault(
            self.label,
            {"count": 0.0, "total": 0.0, "max": 0.0},
        )
        stats["count"] += 1
        stats["total"] += elapsed
        stats["max"] = max(stats["max"], elapsed)

        should_log = self.always_log
        if self.log_threshold is not None and elapsed >= self.log_threshold:
            should_log = True
        if elapsed >= self.max_time:
            should_log = True

        if should_log:
            logger.info(
                "Timer %-36s %.3fs (limit=%.3fs)",
                self.label,
                elapsed,
                self.max_time,
            )

        if exc_type is not None:
            return False

        assert elapsed < self.max_time, (
            f"{self.label} took too long: {elapsed:.3f}s >= {self.max_time:.3f}s"
        )
        return False


def _reset_timer_stats() -> None:
    Timer._stats.clear()


def _log_timer_stats() -> None:
    logger.info("Timer summary for wait=False timing test")
    for label in sorted(Timer._stats):
        stats = Timer._stats[label]
        count = int(stats["count"])
        avg = stats["total"] / max(1, count)
        logger.info(
            "  %-36s count=%d avg=%.3fs max=%.3fs total=%.3fs",
            label,
            count,
            avg,
            stats["max"],
            stats["total"],
        )


def _log_rgb_stream_transport_stats(robot: Robot, *, camera_name: str, phase: str) -> None:
    stream_id = f"{DataType.RGB_IMAGES.value}:{camera_name}"
    stream = robot.get_data_stream(stream_id)
    if stream is None:
        logger.info(
            "RGB transport stats unavailable phase=%s stream_id=%s",
            phase,
            stream_id,
        )
        return

    producer_channel: ProducerChannel = getattr(stream, "_producer_channel", None)
    if producer_channel is None:
        logger.info(
            "RGB transport stats unavailable phase=%s stream_id=%s producer_channel=None",
            phase,
            stream_id,
        )
        return

    stats = producer_channel.get_transport_stats()
    logger.info(
        "RGB transport stats phase=%s queue=%s/%s pending_seq=%s "
        "last_enqueued=%s last_sent=%s sender_alive=%s heartbeat_alive=%s "
        "trace_id=%s recording_id=%s",
        phase,
        stats["send_queue_qsize"],
        stats["send_queue_maxsize"],
        stats["pending_sequence_count"],
        stats["last_enqueued_sequence_number"],
        stats["last_socket_sent_sequence_number"],
        stats["sender_thread_alive"],
        stats["heartbeat_thread_alive"],
        stats["trace_id"],
        stats["recording_id"],
    )


@pytest.fixture(autouse=True)
def cleanup_repo_state_with_script(daemon_setup_teardown):
    """Run repo-local cleanup around each isolated timing test.

    `daemon_setup_teardown` comes from platform `conftest.py` and ensures the
    daemon pid/socket/db state is cleaned before and after each test.
    """
    if os.getenv("NCD_SKIP_DAEMON_CLEANUP_FOR_DEBUG") == "1":
        yield
        return
    cleanup_script = REPO_ROOT / "cleanup.sh"
    subprocess.run(["bash", str(cleanup_script)], cwd=REPO_ROOT, check=True)
    yield
    daemon_cleanup()
    subprocess.run(["bash", str(cleanup_script)], cwd=REPO_ROOT, check=True)


@pytest.fixture(autouse=True)
def launch_clean_daemon_for_test(cleanup_repo_state_with_script):
    """Launch a fresh daemon for this test before any client calls occur.

    This avoids relying on lazy background startup during `nc.start_recording()`,
    which made timing results differ from the explicit clean daemon runs.
    """
    if os.getenv("NCD_SKIP_DAEMON_CLEANUP_FOR_DEBUG") == "1":
        yield
        return

    previous_env = {
        key: os.environ.get(key) for key in _TEST_DAEMON_ENV_OVERRIDES
    }
    os.environ.update(_TEST_DAEMON_ENV_OVERRIDES)

    pid_path = get_daemon_pid_path()
    db_path = get_daemon_db_path()
    daemon_process = launch_new_daemon_subprocess(
        pid_path=pid_path,
        db_path=db_path,
        background=False,
        timeout_s=10.0,
    )
    logger.info("Launched clean data daemon for test pid=%s", daemon_process.pid)
    try:
        yield
    finally:
        daemon_cleanup()
        for key, previous_value in previous_env.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value


def test_stop_recording_wait_false_4k_timing(dataset_cleanup):
    _reset_timer_stats()

    robot_name = f"test_robot_4k_{uuid.uuid4().hex[:8]}"
    dataset_name = f"test_dataset_4k_{uuid.uuid4().hex[:8]}"
    dataset_cleanup(dataset_name)

    frame_width = 3840
    frame_height = 2160
    frame_rate_hz = 60
    duration_s = 30
    frame_count = frame_rate_hz * duration_s
    camera_name = "camera_0"
    bytes_per_frame = frame_width * frame_height * 3
    total_payload_mib = (frame_count * bytes_per_frame) / (1024 * 1024)
    log_checkpoints = {
        0,
        frame_rate_hz - 1,
        (frame_count // 2) - 1,
        frame_count - 1,
    }

    logger.info(
        "Starting isolated 4K wait=False integration test fps=%d duration_s=%d frames=%d resolution=%dx%d bytes_per_frame=%.2f MiB total=%.2f MiB",
        frame_rate_hz,
        duration_s,
        frame_count,
        frame_width,
        frame_height,
        bytes_per_frame / (1024 * 1024),
        total_payload_mib,
    )

    with Timer(label="nc.login", max_time=MAX_TIME_TO_START_S, always_log=True):
        nc.login()
    with Timer(
        label="nc.connect_robot",
        max_time=MAX_TIME_TO_START_S,
        always_log=True,
    ):
        robot = nc.connect_robot(
            robot_name,
            urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
            overwrite=False,
        )
    with Timer(
        label="nc.create_dataset",
        max_time=MAX_TIME_TO_START_S,
        always_log=True,
    ):
        nc.create_dataset(
            dataset_name,
            description="Isolated 4K RGB stop_recording(wait=False) timing test",
        )
    with Timer(
        label="nc.start_recording",
        max_time=MAX_TIME_TO_START_S,
        always_log=True,
    ):
        nc.start_recording()

    _log_rgb_stream_transport_stats(robot, camera_name=camera_name, phase="after_start")

    for frame_num in range(frame_count):
        timestamp = frame_num / frame_rate_hz
        with Timer(
            label="generate_frame",
            max_time=1.5,
            log_threshold=0.05,
        ):
            frame = encode_frame_number(frame_num, frame_width, frame_height)
        with Timer(
            label="nc.log_rgb",
            max_time=20.0,
            log_threshold=0.05,
        ):
            nc.log_rgb(camera_name, frame, timestamp=timestamp)

        if frame_num in log_checkpoints:
            _log_rgb_stream_transport_stats(
                robot,
                camera_name=camera_name,
                phase=f"after_log_frame_{frame_num}",
            )

    _log_rgb_stream_transport_stats(robot, camera_name=camera_name, phase="before_stop")

    with Timer(
        label="nc.stop_recording(wait=False)",
        max_time=MAX_TIME_TO_STOP_S,
        always_log=True,
        log_threshold=0.01,
    ):
        nc.stop_recording(wait=False)

    _log_rgb_stream_transport_stats(robot, camera_name=camera_name, phase="after_stop")
    logger.info(
        "Isolated 4K wait=False transport test completed after bridge cutoff acknowledgement frames_sent=%d total_payload_mib=%.2f",
        frame_count,
        total_payload_mib,
    )
    _log_timer_stats()
