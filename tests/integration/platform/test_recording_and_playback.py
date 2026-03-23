import logging
import math
import multiprocessing
import os
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psutil
import pytest
from neuracore_types import DataType
from PIL import Image

import neuracore as nc
from neuracore.data_daemon.const import SOCKET_PATH
from neuracore.data_daemon.helpers import get_daemon_db_path, get_daemon_pid_path
from neuracore.data_daemon.lifecycle.daemon_lifecycle import (
    ensure_daemon_running,
    pid_is_running,
)

# Add examples dir to path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "..", "..", "..", "examples"))
# ruff: noqa: E402
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH

MAX_TIME_TO_LOG_S = 0.5
LEAST_TIME_TO_STOP_S = 10
MAX_TIME_TO_STOP_S = 20
HIGH_TIME_TO_STOP_S = 30
HIGH_TIME_TO_DATASET_READY_S = 500
MAX_TIME_TO_START_S = 20
MAX_TIME_TO_DATASET_READY_S = 120
JOINT_NAMES = [
    "vx300s_left/waist",
    "vx300s_left/shoulder",
    "vx300s_left/elbow",
    "vx300s_left/forearm_roll",
    "vx300s_left/wrist_angle",
    "vx300s_left/wrist_rotate",
    "vx300s_left/left_finger",
    "vx300s_left/right_finger",
    "vx300s_right/waist",
    "vx300s_right/shoulder",
    "vx300s_right/elbow",
    "vx300s_right/forearm_roll",
    "vx300s_right/wrist_angle",
    "vx300s_right/wrist_rotate",
    "vx300s_right/left_finger",
    "vx300s_right/right_finger",
]


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Keep integration timing deterministic across machines by avoiding
# low default daemon upload bandwidth derived from local disk size.
os.environ.setdefault("NCD_BANDWIDTH_LIMIT", str(200 * 1024 * 1024))  # 200 MiB/s
os.environ.setdefault("NCD_MAX_CONCURRENT_UPLOADS", "30")

# Track daemon profile files created by these integration tests so they
# can be cleaned even when a test fails midway.
_TEST_PROFILE_PATHS: set[Path] = set()


class Timer:
    _stats = {}

    def __init__(
        self,
        max_time=MAX_TIME_TO_LOG_S,
        label=None,
        always_log=False,
        log_threshold=None,
    ):
        self.max_time = max_time
        self.label = label
        self.always_log = always_log
        self.log_threshold = log_threshold

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        had_exception = len(args) > 0 and args[0] is not None
        if self.label:
            stats = self._stats.setdefault(
                self.label, {"count": 0, "total": 0.0, "max": 0.0}
            )
            stats["count"] += 1
            stats["total"] += self.interval
            stats["max"] = max(stats["max"], self.interval)

            should_log = self.always_log
            if self.log_threshold is not None and self.interval >= self.log_threshold:
                should_log = True
            if self.interval >= self.max_time:
                should_log = True

            if should_log:
                level = (
                    logging.WARNING if self.interval >= self.max_time else logging.INFO
                )
                logger.log(
                    level,
                    "Timer %-32s %.3fs (limit=%.3fs)",
                    self.label,
                    self.interval,
                    self.max_time,
                )

        # Don't mask real exceptions raised inside the timed block.
        if had_exception:
            return False

        assert self.interval < self.max_time, (
            f"{self.label or 'Function'} took too long: "
            f"{self.interval:.3f}s >= {self.max_time:.3f}s"
        )


class TestConfig:
    """Configuration for streaming tests"""

    def __init__(
        self,
        fps=30,
        duration_sec=20,
        image_width=640,
        image_height=480,
        num_cameras=1,
        use_depth=False,
        num_joints=16,
        synched_time=False,
        stop_wait_timeout_s=MAX_TIME_TO_STOP_S,
        start_wait_timeout_s=MAX_TIME_TO_START_S,
        log_wait_timeout_s=MAX_TIME_TO_LOG_S,
        dataset_wait_timeout_s=MAX_TIME_TO_DATASET_READY_S,
    ):
        self.fps = fps
        self.duration_sec = duration_sec
        self.image_width = image_width
        self.image_height = image_height
        self.expected_frames = int(fps * duration_sec)
        self.num_cameras = num_cameras
        self.use_depth = use_depth
        self.num_joints = num_joints
        self.robot_name = f"test_robot_{uuid.uuid4().hex[:8]}"
        self.dataset_name = f"test_dataset_{uuid.uuid4().hex[:8]}"
        self.synched_time = synched_time
        self.max_frames = fps * duration_sec
        self.stop_wait_timeout_s = stop_wait_timeout_s
        self.start_wait_timeout_s = start_wait_timeout_s
        self.log_wait_timeout_s = log_wait_timeout_s
        self.dataset_wait_timeout_s = dataset_wait_timeout_s

    def __str__(self):
        return (
            f"TestConfig(fps={self.fps}, duration={self.duration_sec}s, "
            f"size={self.image_width}x{self.image_height}, cameras={self.num_cameras}, "
            f"depth={self.use_depth}, joints={self.num_joints})"
        )


def daemon_cleanup():
    pid_path = get_daemon_pid_path()
    db_path = get_daemon_db_path()
    socket_path = Path(SOCKET_PATH)
    daemon_pids = set(get_runner_pids())

    if pid_path.exists():
        try:
            daemon_pids.add(int(pid_path.read_text(encoding="utf-8").strip()))
        except (OSError, ValueError):
            pass

    for pid in daemon_pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            subprocess.run(["kill", "-9", str(pid)], check=False)

    for path in (pid_path, db_path, socket_path):
        try:
            path.unlink(missing_ok=True)
        except IsADirectoryError:
            pass

    for suffix in (".shm", ".wal"):
        try:
            db_path.with_suffix(db_path.suffix + suffix).unlink(missing_ok=True)
        except OSError:
            pass


def get_runner_pids() -> set[int]:
    output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    runner_pids: set[int] = set()
    for line in output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid_text, args = parts
        if "neuracore.data_daemon.runner_entry" in args:
            runner_pids.add(int(pid_text))
    return runner_pids


def encode_frame_number(frame_num: int, width: int, height: int) -> np.ndarray:
    """Create an image with the frame number encoded in the top-left pixels

    This creates an image where:
    - The frame number is encoded in binary in the top-left 4x4 pixel block
    - We use the red channel (0) for the actual data
    - Other channels can be used for visual debugging
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img.fill(100)  # Gray background

    # Encode frame number in top-left pixels (16 bytes for frame number)
    frame_bytes = frame_num.to_bytes(16, byteorder="big")
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            if idx < len(frame_bytes):
                pixel_value = frame_bytes[idx]
                img[i, j, 0] = pixel_value
                img[i, j, 1] = (
                    255 - pixel_value
                )  # Use other channels for visual debugging
                img[i, j, 2] = pixel_value // 2

    return img


def get_daemon_process_pids():
    return get_runner_pids()


@pytest.fixture(autouse=True)
def daemon_setup_teardown():
    daemon_cleanup()
    yield
    daemon_cleanup()


def decode_frame_number(img: np.ndarray) -> int:
    """Decode a frame number from an encoded image"""
    frame_bytes = bytearray()
    for i in range(4):
        for j in range(4):
            frame_bytes.append(img[i, j, 0])

    return int.from_bytes(frame_bytes[:16], byteorder="big")


def generate_joint_positions(frame_num: int, fps: int, num_joints: int) -> dict:
    """Generate deterministic joint positions based on frame number

    Creates a sine wave with different frequencies for each joint
    so we can later verify the exact values expected
    """
    t = frame_num / fps
    joint_positions = {}
    for i in range(num_joints):
        # Use different frequencies for each joint for variety
        freq = 0.5 + i * 0.25
        joint_positions[JOINT_NAMES[i]] = math.sin(t * freq)

    return joint_positions


def generate_depth_image(frame_num: int, width: int, height: int) -> np.ndarray:
    """Generate a depth image"""
    return np.ones((height, width), dtype=np.float32) * 2.0  # Base 2m depth


def stream_data(config: TestConfig) -> int:
    """Stream test data according to configuration"""
    frame_count = 0

    # Start recording
    # TODO: Note this now takes a long time to start since adding in P2P
    with Timer(
        max_time=config.start_wait_timeout_s,
        label="nc.start_recording",
        always_log=True,
    ):
        nc.start_recording()

    # Generate and stream test data
    time_step = 1.0 / float(config.fps)
    t = 0.0
    try:
        while t < config.duration_sec:
            frame_code = frame_count
            # Create and stream camera images
            for cam_idx in range(config.num_cameras):
                camera_id = f"camera_{cam_idx}"

                frame_code = frame_count + (
                    config.num_cameras * cam_idx * config.max_frames
                )

                # RGB image
                rgb_img = encode_frame_number(
                    frame_code, config.image_width, config.image_height
                )
                with Timer(label="nc.log_rgb", max_time=config.log_wait_timeout_s):
                    nc.log_rgb(camera_id, rgb_img, timestamp=t)

                # Depth image if needed
                if config.use_depth:
                    depth_img = generate_depth_image(
                        frame_code, config.image_width, config.image_height
                    )
                    with Timer(
                        label="nc.log_depth", max_time=config.log_wait_timeout_s
                    ):
                        nc.log_depth(camera_id, depth_img, timestamp=t)

            # Stream joint positions
            joint_positions = generate_joint_positions(
                frame_count, config.fps, config.num_joints
            )
            with Timer(
                label="nc.log_joint_positions", max_time=config.log_wait_timeout_s
            ):
                nc.log_joint_positions(joint_positions, timestamp=t)

            with Timer(
                label="nc.log_joint_velocities", max_time=config.log_wait_timeout_s
            ):
                # use the same joint positions for velocities and torques
                nc.log_joint_velocities(joint_positions, timestamp=t)
            with Timer(
                label="nc.log_joint_torques", max_time=config.log_wait_timeout_s
            ):
                # use the same joint positions for velocities and torques
                nc.log_joint_torques(joint_positions, timestamp=t)

            with Timer(
                label="nc.log_parallel_gripper_open_amount",
                max_time=config.log_wait_timeout_s,
            ):
                nc.log_parallel_gripper_open_amount(
                    name="gripper", value=0.5, timestamp=t
                )

            # points = np.zeros((10, 3), dtype=np.float16)
            # rgb_points = np.zeros((10, 3), dtype=np.uint8)
            # with Timer(max_time=0.5, label="nc.log_point_cloud"):
            #     nc.log_point_cloud(
            #         "point_cloud_camera_0",
            #         points=points,
            #         rgb_points=rgb_points,
            #         extrinsics=np.eye(4),
            #         intrinsics=np.eye(3),
            #         timestamp=t,
            #     )

            with Timer(label="nc.log_custom_1d", max_time=config.log_wait_timeout_s):
                nc.log_custom_1d(
                    "test_custom_data",
                    np.array([frame_code], dtype=np.float32),
                    timestamp=t,
                )

            # Stream a test action occasionally
            if frame_count % 5 == 0:
                action = {
                    f"joint{i+1}": 0.1 * math.sin(frame_count * 0.1 * (i + 1))
                    for i in range(config.num_joints)
                }
                with Timer(label="nc.log_joint_target_positions"):
                    nc.log_joint_target_positions(action, timestamp=t)

            frame_count += 1
            t += time_step

        with Timer(
            config.stop_wait_timeout_s,
            label="nc.stop_recording(wait=True)",
            always_log=True,
        ):
            nc.stop_recording(wait=True)

    except Exception as exc:
        logger.error("Failed to stream data: %s", exc)
    return frame_count


def wait_for_dataset_ready(
    dataset_name: str,
    expected_recording_count: int = 1,
    timeout_s: float = MAX_TIME_TO_DATASET_READY_S,
    poll_interval_s: float = 1.5,
) -> None:
    """Wait for the dataset to be ready.

    Catches transient errors (404 while the dataset propagates, 429 rate-limit) and
    retries until the timeout is reached.
    """
    wait_start = time.perf_counter()
    poll_count = 0
    last_error = None

    while True:
        poll_count += 1
        elapsed_s = time.perf_counter() - wait_start
        try:
            dataset = nc.get_dataset(dataset_name)

            recording_count = len(dataset)
            if recording_count >= expected_recording_count:
                return
        except Exception as exc:
            last_error = exc
            logger.warning(
                "wait_for_dataset_ready poll=%d elapsed=%.1fs transient error: %s",
                poll_count,
                elapsed_s,
                exc,
            )

        if elapsed_s >= timeout_s:
            raise TimeoutError(
                f"Timed out waiting for dataset '{dataset_name}' to have "
                f"{expected_recording_count} recording(s) after {timeout_s}s"
            ) from last_error

        time.sleep(min(poll_interval_s, max(0.0, timeout_s - elapsed_s)))


def verify_dataset(config: TestConfig, expected_frame_count):
    """Verify the dataset integrity"""

    # Retrieve the dataset
    with Timer(
        max_time=MAX_TIME_TO_DATASET_READY_S, label="nc.get_dataset", always_log=True
    ):
        dataset = nc.get_dataset(config.dataset_name)
    with Timer(
        max_time=MAX_TIME_TO_DATASET_READY_S,
        label="dataset.synchronize",
        always_log=True,
    ):
        synced_dataset = dataset.synchronize()

    # Results tracking
    results = {
        "retrieved_frames": 0,
        "unique_frames": set(),
        "missing_frames": [],
        "duplicate_frames": [],
        "joint_mismatches": [],
    }

    with Timer(
        max_time=MAX_TIME_TO_DATASET_READY_S,
        label="verify_dataset.iterate_frames",
        always_log=True,
    ):
        # Iterate through the dataset episodes
        for synced_episode in synced_dataset:
            for frame_idx, sync_point in enumerate(synced_episode):
                results["retrieved_frames"] += 1

                # Check for camera images
                if DataType.RGB_IMAGES in sync_point.data:
                    for _, cam_data in sync_point[DataType.RGB_IMAGES].items():
                        img = cam_data.frame
                        assert isinstance(
                            img, Image.Image
                        ), "RGB image data should be a PIL Image"
                        np_img = np.array(img)
                        decoded_frame_num = decode_frame_number(np_img)
                        if decoded_frame_num in results["unique_frames"]:
                            results["duplicate_frames"].append(decoded_frame_num)
                        results["unique_frames"].add(decoded_frame_num)

                if DataType.JOINT_POSITIONS in sync_point.data:
                    expected_joints = generate_joint_positions(
                        frame_idx, config.fps, config.num_joints
                    )

                    for joint_name, expected_value in expected_joints.items():
                        if joint_name in sync_point[DataType.JOINT_POSITIONS]:
                            actual_value = sync_point[DataType.JOINT_POSITIONS][
                                joint_name
                            ].value
                            if abs(expected_value - actual_value) > 1e-5:
                                results["joint_mismatches"].append((
                                    decoded_frame_num,
                                    joint_name,
                                    expected_value,
                                    actual_value,
                                ))

    # Check for missing frames
    expected_frames = set(range(expected_frame_count))
    results["missing_frames"] = list(expected_frames - results["unique_frames"])

    return results


def run_streaming_test(config: TestConfig):
    """Run a complete streaming test with the given configuration"""
    with Timer(max_time=MAX_TIME_TO_START_S, label="nc.login", always_log=True):
        nc.login()
    with Timer(max_time=MAX_TIME_TO_START_S, label="nc.connect_robot", always_log=True):
        nc.connect_robot(
            config.robot_name,
            urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
            overwrite=False,
        )
    with Timer(
        max_time=MAX_TIME_TO_START_S, label="nc.create_dataset", always_log=True
    ):
        nc.create_dataset(
            config.dataset_name,
            description=(
                f"Test dataset with {config.fps}fps, "
                f"{config.image_width}x{config.image_height}"
            ),
        )

    actual_frame_count = stream_data(config)

    wait_for_dataset_ready(config.dataset_name, timeout_s=config.dataset_wait_timeout_s)

    results = verify_dataset(config, actual_frame_count)

    assert len(results["missing_frames"]) == 0
    assert len(results["duplicate_frames"]) == 0
    assert len(results["joint_mismatches"]) == 0
    return results


def _mp_stream_robot_data(config):
    nc.login()
    nc.connect_robot(config.robot_name)
    nc.create_dataset(config.dataset_name)
    return stream_data(config)


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    yield


@pytest.fixture(autouse=True)
def cleanup_test_profiles():
    yield

    for profile_path in list(_TEST_PROFILE_PATHS):
        try:
            profile_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to remove test profile: %s", profile_path)
    _TEST_PROFILE_PATHS.clear()


def test_basic_streaming():
    config = TestConfig(
        fps=10,
        duration_sec=2,
        image_width=640,
        image_height=480,
        synched_time=True,
        stop_wait_timeout_s=LEAST_TIME_TO_STOP_S,
    )
    run_streaming_test(config)


def test_basic_streaming_fast_mode():
    """Fast smoke test for daemon recording flow with minimal payload."""
    config = TestConfig(
        fps=1,
        duration_sec=1,
        image_width=64,
        image_height=64,
        num_cameras=1,
        use_depth=False,
        num_joints=1,
        synched_time=True,
        stop_wait_timeout_s=LEAST_TIME_TO_STOP_S,
    )

    with Timer(
        max_time=config.start_wait_timeout_s, label="fast.nc.login", always_log=True
    ):
        nc.login()
    with Timer(
        max_time=config.start_wait_timeout_s,
        label="fast.nc.connect_robot",
        always_log=True,
    ):
        nc.connect_robot(
            config.robot_name,
            urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
            overwrite=False,
        )
    with Timer(
        max_time=config.start_wait_timeout_s,
        label="fast.nc.create_dataset",
        always_log=True,
    ):
        nc.create_dataset(
            config.dataset_name,
            description="Fast integration smoke test",
        )

    with Timer(
        max_time=config.start_wait_timeout_s,
        label="fast.nc.start_recording",
        always_log=True,
    ):
        nc.start_recording()

    frame_code = 0
    rgb_img = encode_frame_number(frame_code, config.image_width, config.image_height)
    with Timer(max_time=config.log_wait_timeout_s, label="fast.nc.log_rgb"):
        nc.log_rgb("camera_0", rgb_img, timestamp=0.0)

    with Timer(
        max_time=config.stop_wait_timeout_s,
        label="fast.nc.stop_recording(wait=False)",
        always_log=True,
    ):
        nc.stop_recording(wait=True)

    with Timer(
        max_time=MAX_TIME_TO_DATASET_READY_S,
        label="fast.wait_for_dataset_ready",
        always_log=True,
    ):
        wait_for_dataset_ready(
            config.dataset_name,
            expected_recording_count=1,
            poll_interval_s=0.5,
        )

    with Timer(
        max_time=config.stop_wait_timeout_s,
        label="fast.nc.get_dataset",
        always_log=True,
    ):
        dataset = nc.get_dataset(config.dataset_name)
    with Timer(
        max_time=MAX_TIME_TO_DATASET_READY_S,
        label="fast.dataset.synchronize",
        always_log=True,
    ):
        synced_dataset = dataset.synchronize()

    found_rgb = False
    for synced_episode in synced_dataset:
        for sync_point in synced_episode:
            if DataType.RGB_IMAGES not in sync_point.data:
                continue
            for _, cam_data in sync_point[DataType.RGB_IMAGES].items():
                np_img = np.array(cam_data.frame)
                assert decode_frame_number(np_img) == frame_code
                found_rgb = True
                break
            if found_rgb:
                break
        if found_rgb:
            break

    assert found_rgb, "Expected at least one RGB frame in synchronized dataset"


def test_high_framerate():
    config = TestConfig(
        fps=200,
        duration_sec=2,
        image_width=640,
        image_height=480,
        synched_time=True,
        stop_wait_timeout_s=MAX_TIME_TO_STOP_S,
    )
    run_streaming_test(config)


def test_multiple_cameras():
    config = TestConfig(
        fps=30,
        duration_sec=2,
        image_width=640,
        image_height=480,
        num_cameras=3,
        synched_time=True,
        stop_wait_timeout_s=MAX_TIME_TO_STOP_S,
        dataset_wait_timeout_s=HIGH_TIME_TO_DATASET_READY_S,
    )
    run_streaming_test(config)


def test_with_depth():
    config = TestConfig(
        fps=30,
        duration_sec=2,
        image_width=640,
        image_height=480,
        use_depth=True,
        synched_time=True,
        stop_wait_timeout_s=MAX_TIME_TO_STOP_S,
    )
    run_streaming_test(config)


def test_multiple_concurrent_robots():
    num_robots = 3
    configs = [
        TestConfig(
            fps=30,
            duration_sec=2,
            synched_time=True,
            stop_wait_timeout_s=40,
            dataset_wait_timeout_s=700,
        )
        for _ in range(num_robots)
    ]
    nc.login()

    with multiprocessing.Pool(num_robots) as executor:
        frame_counts = list(executor.map(_mp_stream_robot_data, configs))

    for idx, frame_count in enumerate(frame_counts):
        logger.info(f"Verifying robot {idx+1}/{num_robots}")
        wait_for_dataset_ready(
            configs[idx].dataset_name, timeout_s=configs[idx].dataset_wait_timeout_s
        )
        results = verify_dataset(configs[idx], frame_count)
        assert len(results["missing_frames"]) == 0
        assert len(results["duplicate_frames"]) == 0
        assert len(results["joint_mismatches"]) == 0


def test_stop_start_sequences():
    config = TestConfig(
        fps=30,
        duration_sec=2,
        synched_time=True,
        stop_wait_timeout_s=HIGH_TIME_TO_STOP_S,
        dataset_wait_timeout_s=HIGH_TIME_TO_DATASET_READY_S,
    )

    nc.login()
    robot = nc.connect_robot(config.robot_name)
    nc.create_dataset(config.dataset_name)

    segments = 3
    total_frames = 0
    previous_recording_id: str | None = None

    # Ensure no stale active recording is reused for segment 1.
    stale_recording_id = robot.get_current_recording_id()
    if stale_recording_id is not None:
        logger.warning(
            "Stopping stale active recording before test start: %s",
            stale_recording_id,
        )
        with Timer(max_time=config.stop_wait_timeout_s):
            nc.stop_recording(wait=True)

    for segment in range(segments):
        logger.info(f"Starting recording segment {segment+1}/{segments}")
        started_recording_id: str | None = None
        for attempt in range(3):
            recording_id_before_start = robot.get_current_recording_id()
            with Timer(max_time=config.start_wait_timeout_s):
                nc.start_recording()
            started_recording_id = robot.get_current_recording_id()
            if started_recording_id is None:
                pytest.fail("No active recording ID after nc.start_recording()")
            is_new_id = (
                started_recording_id != previous_recording_id
                and started_recording_id != recording_id_before_start
            )
            if is_new_id:
                break
            logger.warning(
                "Segment %d reused active recording_id=%s on attempt %d; "
                "stopping and retrying start",
                segment + 1,
                started_recording_id,
                attempt + 1,
            )
            with Timer(max_time=config.stop_wait_timeout_s):
                nc.stop_recording(wait=True)
        else:
            pytest.fail(
                f"Unable to start a new recording for segment {segment+1}; "
                f"still using recording_id={started_recording_id}"
            )

        local_t = 0.0
        segment_frames = 0
        segment_time_offset = segment * (config.duration_sec + (1.0 / config.fps))

        while local_t < config.duration_sec:
            frame_num = total_frames + segment_frames
            img = encode_frame_number(
                frame_num, config.image_width, config.image_height
            )
            timestamp = segment_time_offset + local_t
            with Timer():
                nc.log_rgb("camera_0", img, timestamp=timestamp)

            joint_positions = generate_joint_positions(
                segment_frames, config.fps, config.num_joints
            )
            with Timer():
                nc.log_joint_positions(joint_positions, timestamp=timestamp)

            segment_frames += 1
            local_t += 1 / config.fps

        with Timer(max_time=config.stop_wait_timeout_s):
            nc.stop_recording(wait=True)
        logger.info(f"Completed segment {segment+1} with {segment_frames} frames")

        previous_recording_id = started_recording_id
        total_frames += segment_frames

    wait_for_dataset_ready(config.dataset_name, timeout_s=config.dataset_wait_timeout_s)

    results = verify_dataset(config, total_frames)
    assert len(results["missing_frames"]) == 0
    assert len(results["duplicate_frames"]) == 0
    assert len(results["joint_mismatches"]) == 0


def test_high_bandwidth():
    """Test high-throughput streaming with elevated daemon upload bandwidth."""
    previous_bandwidth_limit = os.environ.get("NCD_BANDWIDTH_LIMIT")
    previous_num_threads = os.environ.get("NCD_NUM_THREADS")
    config = TestConfig(
        fps=60,
        duration_sec=2,
        image_width=1920,
        image_height=1080,
        num_cameras=2,
        use_depth=True,
        synched_time=True,
        stop_wait_timeout_s=HIGH_TIME_TO_STOP_S,
    )

    # Estimate bandwidth
    fps = config.fps
    rgb_bytes = config.image_width * config.image_height * 3
    depth_bytes = config.image_width * config.image_height * 4  # float32
    total_image_bytes = config.num_cameras * (
        rgb_bytes + (depth_bytes if config.use_depth else 0)
    )
    joint_bytes = config.num_joints * 8  # float64
    bytes_per_second = (total_image_bytes + joint_bytes) * fps

    daemon_bandwidth_limit = int(bytes_per_second * 1.25)
    os.environ["NCD_BANDWIDTH_LIMIT"] = str(daemon_bandwidth_limit)
    os.environ["NCD_NUM_THREADS"] = "8"

    try:
        # Set up
        with Timer(max_time=MAX_TIME_TO_START_S, label="hb.nc.login", always_log=True):
            nc.login()
        with Timer(
            max_time=MAX_TIME_TO_START_S, label="hb.nc.connect_robot", always_log=True
        ):
            nc.connect_robot(config.robot_name)
        with Timer(
            max_time=MAX_TIME_TO_START_S, label="hb.nc.create_dataset", always_log=True
        ):
            nc.create_dataset(config.dataset_name)

        # Stream data
        actual_frame_count = stream_data(config)
        wait_for_dataset_ready(
            config.dataset_name,
            timeout_s=500,
            poll_interval_s=1.0,
        )

        # Verify
        results = verify_dataset(config, actual_frame_count)
        assert len(results["missing_frames"]) == 0
        assert len(results["duplicate_frames"]) == 0
        assert len(results["joint_mismatches"]) == 0

    except Exception as e:
        pytest.fail(f"High bandwidth test failed: {str(e)}")
    finally:
        if previous_bandwidth_limit is None:
            os.environ.pop("NCD_BANDWIDTH_LIMIT", None)
        else:
            os.environ["NCD_BANDWIDTH_LIMIT"] = previous_bandwidth_limit
        if previous_num_threads is None:
            os.environ.pop("NCD_NUM_THREADS", None)
        else:
            os.environ["NCD_NUM_THREADS"] = previous_num_threads


def test_ensure_single_daemon_process():
    """Ensure repeated startup calls do not create multiple daemon processes."""
    nc.login()

    def worker(barrier, results, i):
        barrier.wait()
        logger.info("worker %d", i)
        pid = ensure_daemon_running()
        results[i] = pid

    core_count = psutil.cpu_count(logical=False) or 4
    barrier = multiprocessing.Barrier(core_count)
    manager = multiprocessing.Manager()
    results = manager.dict()
    processes = []

    for i in range(core_count):
        p = multiprocessing.Process(
            target=worker,
            args=(barrier, results, i),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join(timeout=25)
        assert not p.is_alive(), f"worker process {p.pid} did not finish before timeout"
        assert p.exitcode == 0, f"worker process {p.pid} exited with code {p.exitcode}"

    pids = list(results.values())
    assert len(pids) == core_count
    assert len(set(pids)) == 1
    pid = pids[0]

    pid_path = get_daemon_pid_path()

    assert pid_path.exists()
    assert pid_is_running(pid)
    assert pid_path.read_text(encoding="utf-8").strip() == str(pid)

    runner_pids = get_runner_pids()
    assert pid in runner_pids
    assert (
        len(runner_pids) == 1
    ), f"Expected exactly one daemon runner process, found pids={sorted(runner_pids)}"


def _stop_data_daemon() -> None:
    subprocess.run(
        [sys.executable, "-m", "neuracore.data_daemon", "stop"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@contextmanager
def _use_offline_daemon_profile():
    profile_name = f"offline_profile_{uuid.uuid4().hex[:8]}"
    profile_path = (
        Path.home() / ".neuracore" / "data_daemon" / "profiles" / f"{profile_name}.yaml"
    )
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text("offline: true\n", encoding="utf-8")
    _TEST_PROFILE_PATHS.add(profile_path)

    previous_profile = os.environ.get("NEURACORE_DAEMON_PROFILE")
    os.environ["NEURACORE_DAEMON_PROFILE"] = profile_name
    _stop_data_daemon()

    try:
        yield
    finally:
        _stop_data_daemon()
        if previous_profile is None:
            os.environ.pop("NEURACORE_DAEMON_PROFILE", None)
        else:
            os.environ["NEURACORE_DAEMON_PROFILE"] = previous_profile


def _run_minimal_recording_flow(label_prefix: str = "offline") -> str:
    config = TestConfig(
        fps=2,
        duration_sec=1,
        image_width=64,
        image_height=64,
        num_cameras=1,
        use_depth=False,
        num_joints=1,
        synched_time=True,
        stop_wait_timeout_s=LEAST_TIME_TO_STOP_S,
    )

    with Timer(
        max_time=config.start_wait_timeout_s,
        label=f"{label_prefix}.nc.login",
        always_log=True,
    ):
        nc.login()
    with Timer(
        max_time=config.start_wait_timeout_s,
        label=f"{label_prefix}.nc.connect_robot",
        always_log=True,
    ):
        robot = nc.connect_robot(
            config.robot_name,
            urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
            overwrite=False,
        )
    with Timer(
        max_time=config.start_wait_timeout_s,
        label=f"{label_prefix}.nc.create_dataset",
        always_log=True,
    ):
        nc.create_dataset(config.dataset_name)
    with Timer(
        max_time=config.start_wait_timeout_s,
        label=f"{label_prefix}.nc.start_recording",
        always_log=True,
    ):
        nc.start_recording()

    recording_id = robot.get_current_recording_id()
    assert recording_id is not None

    frame = encode_frame_number(0, config.image_width, config.image_height)
    with Timer(max_time=config.log_wait_timeout_s, label=f"{label_prefix}.nc.log_rgb"):
        nc.log_rgb("camera_0", frame, timestamp=0.0)

    with Timer(
        max_time=config.log_wait_timeout_s,
        label=f"{label_prefix}.nc.log_joint_positions",
    ):
        nc.log_joint_positions(
            generate_joint_positions(0, config.fps, config.num_joints),
            timestamp=0.0,
        )

    with Timer(
        max_time=config.stop_wait_timeout_s,
        label=f"{label_prefix}.nc.stop_recording(wait=False)",
        always_log=True,
    ):
        nc.stop_recording(wait=False)

    return recording_id


def _fetch_trace_registration_stats(recording_id: str) -> tuple[int, int]:
    db_path = get_daemon_db_path()
    with sqlite3.connect(db_path) as conn:
        total_traces = conn.execute(
            "SELECT COUNT(*) FROM traces WHERE recording_id = ?",
            (recording_id,),
        ).fetchone()[0]
        non_pending = conn.execute(
            "SELECT COUNT(*) FROM traces "
            "WHERE recording_id = ? AND registration_status != 'pending'",
            (recording_id,),
        ).fetchone()[0]
    return int(total_traces), int(non_pending)


def _fetch_expected_trace_count_reported(recording_id: str) -> int | None:
    db_path = get_daemon_db_path()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT expected_trace_count_reported FROM "
            "recordings WHERE recording_id = ?",
            (recording_id,),
        ).fetchone()
    if row is None:
        return None
    return int(row[0])


def _wait_for_recording_to_exist_in_db(
    recording_id: str, timeout_s: float = 15.0
) -> None:
    deadline = time.time() + timeout_s
    poll = 0
    while time.time() < deadline:
        poll += 1
        total_traces, _ = _fetch_trace_registration_stats(recording_id)
        reported = _fetch_expected_trace_count_reported(recording_id)
        if total_traces > 0 and reported is not None:
            return
        time.sleep(0.2)

    pytest.fail(f"Recording {recording_id} did not appear in daemon DB before timeout")


def _fetch_recording_recovery_stats(recording_id: str) -> dict[str, int | str | None]:
    db_path = get_daemon_db_path()
    with sqlite3.connect(db_path) as conn:
        recording_row = conn.execute(
            "SELECT expected_trace_count, expected_trace_count_reported, "
            "progress_reported "
            "FROM recordings WHERE recording_id = ?",
            (recording_id,),
        ).fetchone()

        trace_row = conn.execute(
            "SELECT "
            "COUNT(*) AS total_traces, "
            "SUM(CASE WHEN registration_status != 'pending' THEN 1 ELSE 0 END) "
            "AS non_pending_registration_traces, "
            "SUM(CASE WHEN registration_status = 'registered' THEN 1 ELSE 0 END) "
            "AS registered_traces, "
            "SUM(CASE WHEN upload_status IN ('queued', 'uploading', 'uploaded') "
            "THEN 1 ELSE 0 END) AS upload_progress_traces, "
            "SUM(CASE WHEN upload_status = 'uploaded' THEN 1 ELSE 0 END) "
            "AS uploaded_traces "
            "FROM traces WHERE recording_id = ?",
            (recording_id,),
        ).fetchone()

    expected_trace_count = None
    expected_trace_count_reported = None
    progress_reported = None
    if recording_row is not None:
        expected_trace_count = int(recording_row[0])
        expected_trace_count_reported = int(recording_row[1])
        progress_reported = recording_row[2]

    return {
        "expected_trace_count": expected_trace_count,
        "expected_trace_count_reported": expected_trace_count_reported,
        "progress_reported": progress_reported,
        "total_traces": int(trace_row[0]),
        "non_pending_registration_traces": int(trace_row[1] or 0),
        "registered_traces": int(trace_row[2] or 0),
        "upload_progress_traces": int(trace_row[3] or 0),
        "uploaded_traces": int(trace_row[4] or 0),
    }


def _wait_for_online_recovery(recording_id: str, timeout_s: float = 90.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        stats = _fetch_recording_recovery_stats(recording_id)

        fully_uploaded = (
            stats["total_traces"] == 0 and stats["progress_reported"] == "reported"
        )
        registration_attempted_or_done = stats["non_pending_registration_traces"] > 0
        expected_count_locally_set = (stats["expected_trace_count"] or 0) > 0

        if (
            fully_uploaded
            or registration_attempted_or_done
            or expected_count_locally_set
        ):
            return

        time.sleep(0.5)

    stats = _fetch_recording_recovery_stats(recording_id)
    pytest.fail(
        "Online recovery did not progress for recording "
        f"{recording_id}; stats={stats}"
    )


@pytest.mark.usefixtures("run_before_and_after_tests")
class TestOfflineProfileBehavior:
    def test_offline_profile_does_not_register_traces(self):
        with _use_offline_daemon_profile():
            recording_id = _run_minimal_recording_flow(
                label_prefix="offline.registration"
            )
            _wait_for_recording_to_exist_in_db(recording_id)

            total_traces, non_pending = _fetch_trace_registration_stats(recording_id)
            assert total_traces > 0
            assert non_pending == 0

    def test_offline_profile_does_not_report_expected_trace_count(self):
        with _use_offline_daemon_profile():
            recording_id = _run_minimal_recording_flow(
                label_prefix="offline.expected_trace_count"
            )
            _wait_for_recording_to_exist_in_db(recording_id)

            expected_trace_count_reported = _fetch_expected_trace_count_reported(
                recording_id
            )
            assert expected_trace_count_reported == 0

    def test_offline_pending_data_recovers_when_online(self):
        with _use_offline_daemon_profile():
            recording_id = _run_minimal_recording_flow(
                label_prefix="offline.recovery_seed"
            )
            _wait_for_recording_to_exist_in_db(recording_id)

            total_traces, non_pending = _fetch_trace_registration_stats(recording_id)
            assert total_traces > 0
            assert non_pending == 0

            expected_trace_count_reported = _fetch_expected_trace_count_reported(
                recording_id
            )
            assert expected_trace_count_reported == 0

        previous_profile = os.environ.get("NEURACORE_DAEMON_PROFILE")
        try:
            os.environ.pop("NEURACORE_DAEMON_PROFILE", None)
            _stop_data_daemon()

            with Timer(max_time=MAX_TIME_TO_START_S, label="recovery.nc.login"):
                nc.login()
            with Timer(max_time=MAX_TIME_TO_START_S, label="recovery.nc.connect_robot"):
                nc.connect_robot(
                    f"recovery_robot_{uuid.uuid4().hex[:8]}",
                    urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
                    overwrite=False,
                )

            _wait_for_online_recovery(recording_id)
        finally:
            if previous_profile is None:
                os.environ.pop("NEURACORE_DAEMON_PROFILE", None)
            else:
                os.environ["NEURACORE_DAEMON_PROFILE"] = previous_profile
