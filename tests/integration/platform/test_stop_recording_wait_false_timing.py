import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
from neuracore_types import DataType

import neuracore as nc

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
sys.path.append(str(REPO_ROOT / "examples"))

# ruff: noqa: E402
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH
from recording_playback_shared import (
    decode_frame_number,
    encode_frame_number,
    wait_for_dataset_ready,
)

logger = logging.getLogger(__name__)

MAX_TIME_TO_START_S = 20
MAX_TIME_TO_STOP_S = 3
MAX_TIME_TO_DATASET_READY_S = 500

os.environ.setdefault("NCD_BANDWIDTH_LIMIT", str(200 * 1024 * 1024))
os.environ.setdefault("NCD_MAX_CONCURRENT_UPLOADS", "30")


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


def _log_rgb_stream_transport_stats(robot, *, camera_name: str, phase: str) -> None:
    stream_id = f"{DataType.RGB_IMAGES.value}:{camera_name}"
    stream = robot.get_data_stream(stream_id)
    if stream is None:
        logger.info(
            "RGB transport stats unavailable phase=%s stream_id=%s",
            phase,
            stream_id,
        )
        return

    producer_channel = getattr(stream, "_producer_channel", None)
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
    cleanup_script = REPO_ROOT / "cleanup.sh"
    subprocess.run(["bash", str(cleanup_script)], cwd=REPO_ROOT, check=True)
    yield
    subprocess.run(["bash", str(cleanup_script)], cwd=REPO_ROOT, check=True)


def test_stop_recording_wait_false_4k_timing(dataset_cleanup):
    _reset_timer_stats()

    robot_name = f"test_robot_4k_{uuid.uuid4().hex[:8]}"
    dataset_name = f"test_dataset_4k_{uuid.uuid4().hex[:8]}"
    dataset_cleanup(dataset_name)

    frame_width = 3840
    frame_height = 2160
    frame_count = 4
    camera_name = "camera_0"
    expected_frame_numbers = set(range(frame_count))
    bytes_per_frame = frame_width * frame_height * 3

    logger.info(
        "Starting isolated 4K wait=False integration test frames=%d resolution=%dx%d bytes_per_frame=%.2f MiB total=%.2f MiB",
        frame_count,
        frame_width,
        frame_height,
        bytes_per_frame / (1024 * 1024),
        (frame_count * bytes_per_frame) / (1024 * 1024),
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
        timestamp = frame_num * 0.25
        with Timer(
            label="generate_frame",
            max_time=1.5,
            log_threshold=0.05,
        ):
            frame = encode_frame_number(frame_num, frame_width, frame_height)
        with Timer(
            label="nc.log_rgb",
            max_time=5.0,
            log_threshold=0.05,
        ):
            nc.log_rgb(camera_name, frame, timestamp=timestamp)

        if frame_num in (0, frame_count - 1):
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

    with Timer(
        label="wait_for_dataset_ready",
        max_time=MAX_TIME_TO_DATASET_READY_S,
        always_log=True,
    ):
        wait_for_dataset_ready(
            dataset_name,
            expected_recording_count=1,
            poll_interval_s=0.5,
        )

    _log_rgb_stream_transport_stats(
        robot,
        camera_name=camera_name,
        phase="after_dataset_ready",
    )

    with Timer(label="nc.get_dataset", max_time=120.0, always_log=True):
        dataset = nc.get_dataset(dataset_name)
    with Timer(
        label="dataset.synchronize",
        max_time=MAX_TIME_TO_DATASET_READY_S,
        always_log=True,
    ):
        synced_dataset = dataset.synchronize()

    decoded_frame_numbers: set[int] = set()
    for synced_episode in synced_dataset:
        for sync_point in synced_episode:
            if DataType.RGB_IMAGES not in sync_point.data:
                continue
            for _, cam_data in sync_point[DataType.RGB_IMAGES].items():
                decoded_frame_numbers.add(
                    decode_frame_number(np.array(cam_data.frame))
                )

    logger.info(
        "Isolated 4K wait=False result retrieved_frames=%d expected_frames=%d",
        len(decoded_frame_numbers),
        frame_count,
    )
    _log_timer_stats()

    assert decoded_frame_numbers == expected_frame_numbers
