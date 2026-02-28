import logging
import math
import multiprocessing
import os
import sys
import time
import uuid

import numpy as np
import pytest
from neuracore_types import DataType
from PIL import Image

import neuracore as nc

# Add examples dir to path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "..", "..", "..", "examples"))
# ruff: noqa: E402
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH

MAX_TIME_TO_LOG_S = 0.5
MAX_TIME_TO_SAVE_S = 40
MAX_TIME_TO_START = 40
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
os.environ.setdefault("NCD_NUM_THREADS", "4")


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

    @classmethod
    def reset_stats(cls):
        cls._stats = {}

    @classmethod
    def log_stats(cls, prefix="Timer stats"):
        if not cls._stats:
            return

        logger.info(prefix)
        for label, stats in sorted(
            cls._stats.items(), key=lambda item: item[1]["total"], reverse=True
        ):
            avg = stats["total"] / stats["count"]
            logger.info(
                "  %-32s count=%4d total=%7.3fs avg=%6.3fs max=%6.3fs",
                label,
                stats["count"],
                stats["total"],
                avg,
                stats["max"],
            )

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
        stop_wait_timeout_s=MAX_TIME_TO_SAVE_S,
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

    def __str__(self):
        return (
            f"TestConfig(fps={self.fps}, duration={self.duration_sec}s, "
            f"size={self.image_width}x{self.image_height}, cameras={self.num_cameras}, "
            f"depth={self.use_depth}, joints={self.num_joints})"
        )


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
    Timer.reset_stats()
    stream_start = time.perf_counter()
    frame_count = 0

    # Start recording
    # TODO: Note this now takes a long time to start since adding in P2P
    with Timer(max_time=MAX_TIME_TO_START, label="nc.start_recording", always_log=True):
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
                with Timer(label="nc.log_rgb"):
                    nc.log_rgb(camera_id, rgb_img, timestamp=t)

                # Depth image if needed
                if config.use_depth:
                    depth_img = generate_depth_image(
                        frame_code, config.image_width, config.image_height
                    )
                    with Timer(label="nc.log_depth"):
                        nc.log_depth(camera_id, depth_img, timestamp=t)

            # Stream joint positions
            joint_positions = generate_joint_positions(
                frame_count, config.fps, config.num_joints
            )
            with Timer(label="nc.log_joint_positions"):
                nc.log_joint_positions(joint_positions, timestamp=t)

            with Timer(label="nc.log_joint_velocities"):
                # use the same joint positions for velocities and torques
                nc.log_joint_velocities(joint_positions, timestamp=t)
            with Timer(label="nc.log_joint_torques"):
                # use the same joint positions for velocities and torques
                nc.log_joint_torques(joint_positions, timestamp=t)

            with Timer(label="nc.log_parallel_gripper_open_amount"):
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

            with Timer(label="nc.log_custom_1d"):
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

        logger.info(
            "Finished enqueueing %d frames in %.3fs; "
            "calling nc.stop_recording(wait=True)",
            frame_count,
            time.perf_counter() - stream_start,
        )
        with Timer(
            config.stop_wait_timeout_s,
            label="nc.stop_recording(wait=True)",
            always_log=True,
        ):
            nc.stop_recording(wait=True)

        # Allow some time for data to be fully saved and available for synchronization
        with Timer(
            max_time=MAX_TIME_TO_SAVE_S,
            label="post-stop sleep",
            always_log=True,
        ):
            time.sleep(2)
    finally:
        logger.info(
            "stream_data summary: frames=%d total_elapsed=%.3fs",
            frame_count,
            time.perf_counter() - stream_start,
        )
        Timer.log_stats(prefix="stream_data timing breakdown")

    return frame_count


def wait_for_dataset_ready(
    dataset_name: str,
    expected_recording_count: int = 1,
    timeout_s: float = MAX_TIME_TO_SAVE_S,
    poll_interval_s: float = 2.0,
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

        if elapsed_s >= timeout_s:
            raise TimeoutError(
                f"Timed out waiting for dataset '{dataset_name}' to have "
                f"{expected_recording_count} recording(s) after {timeout_s}s"
            ) from last_error

        time.sleep(min(poll_interval_s, max(0.0, timeout_s - elapsed_s)))


def verify_dataset(config, expected_frame_count):
    """Verify the dataset integrity"""

    # Retrieve the dataset
    with Timer(
        max_time=config.stop_wait_timeout_s, label="nc.get_dataset", always_log=True
    ):
        dataset = nc.get_dataset(config.dataset_name)
    with Timer(
        max_time=MAX_TIME_TO_SAVE_S, label="dataset.synchronize", always_log=True
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
        max_time=MAX_TIME_TO_SAVE_S,
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

    # Log summary
    logger.info("Verification results:")
    logger.info(f"  Retrieved frames: {results['retrieved_frames']}")
    logger.info(f"  Unique frames: {len(results['unique_frames'])}")
    logger.info(f"  Missing frames: {len(results['missing_frames'])}")
    logger.info(f"  Duplicate frames: {len(results['duplicate_frames'])}")
    logger.info(f"  Joint mismatches: {len(results['joint_mismatches'])}")

    if results["missing_frames"]:
        missing_count = len(results["missing_frames"])
        missing_percent = missing_count / expected_frame_count * 100
        logger.warning(f"Missing {missing_count} frames ({missing_percent:.2f}%)")
        logger.warning(
            f"First few missing frames: {sorted(results['missing_frames'])[:20]}"
        )

    if results["duplicate_frames"]:
        logger.warning(f"Duplicate frames: {results['duplicate_frames'][:20]}...")

    if results["joint_mismatches"]:
        logger.warning(f"Joint mismatches: {results['joint_mismatches'][:5]}...")

    return results


def run_streaming_test(config: TestConfig):
    """Run a complete streaming test with the given configuration"""
    # Set up
    logger.info(f"Starting test with config: {config}")
    with Timer(
        max_time=config.stop_wait_timeout_s,
        label="run_streaming_test.total",
        always_log=True,
    ):
        with Timer(max_time=MAX_TIME_TO_START, label="nc.login", always_log=True):
            nc.login()
        with Timer(
            max_time=MAX_TIME_TO_START, label="nc.connect_robot", always_log=True
        ):
            nc.connect_robot(
                config.robot_name,
                urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
                overwrite=False,
            )
        with Timer(
            max_time=MAX_TIME_TO_START, label="nc.create_dataset", always_log=True
        ):
            nc.create_dataset(
                config.dataset_name,
                description=(
                    f"Test dataset with {config.fps}fps, "
                    f"{config.image_width}x{config.image_height}"
                ),
            )

        actual_frame_count = stream_data(config)

        wait_for_dataset_ready(config.dataset_name)

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


def test_basic_streaming():
    config = TestConfig(
        fps=10, duration_sec=2, image_width=640, image_height=480, synched_time=True
    )
    run_streaming_test(config)


def test_high_framerate():
    config = TestConfig(
        fps=200,
        duration_sec=2,
        image_width=640,
        image_height=480,
        synched_time=True,
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
        stop_wait_timeout_s=math.inf,
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
        stop_wait_timeout_s=60,
    )
    run_streaming_test(config)


def test_multiple_concurrent_robots():
    num_robots = 3
    configs = [
        TestConfig(fps=30, duration_sec=2, synched_time=True) for _ in range(num_robots)
    ]
    nc.login()

    with multiprocessing.Pool(num_robots) as executor:
        frame_counts = list(executor.map(_mp_stream_robot_data, configs))

    for idx, frame_count in enumerate(frame_counts):
        logger.info(f"Verifying robot {idx+1}/{num_robots}")
        results = verify_dataset(configs[idx], frame_count)
        assert len(results["missing_frames"]) == 0
        assert len(results["duplicate_frames"]) == 0
        assert len(results["joint_mismatches"]) == 0


def test_stop_start_sequences():
    config = TestConfig(fps=30, duration_sec=2, synched_time=True)

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
        with Timer(max_time=MAX_TIME_TO_SAVE_S):
            nc.stop_recording(wait=True)
        time.sleep(2.0)

    for segment in range(segments):
        logger.info(f"Starting recording segment {segment+1}/{segments}")
        started_recording_id: str | None = None
        for attempt in range(3):
            recording_id_before_start = robot.get_current_recording_id()
            with Timer(max_time=MAX_TIME_TO_START):
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
            with Timer(max_time=MAX_TIME_TO_SAVE_S):
                nc.stop_recording(wait=True)
            time.sleep(2.0)
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

    wait_for_dataset_ready(config.dataset_name, timeout_s=180)

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
        stop_wait_timeout_s=180,
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
    mb_per_second = bytes_per_second / (1024 * 1024)

    daemon_bandwidth_limit = int(bytes_per_second * 1.25)
    os.environ["NCD_BANDWIDTH_LIMIT"] = str(daemon_bandwidth_limit)
    os.environ["NCD_NUM_THREADS"] = "8"

    logger.info(f"Estimated bandwidth: {mb_per_second:.2f} MB/s")
    logger.info(
        "Configured daemon bandwidth limit: %.2f MB/s",
        daemon_bandwidth_limit / (1024 * 1024),
    )

    try:
        # Set up
        nc.login()
        nc.connect_robot(config.robot_name)
        nc.create_dataset(config.dataset_name)

        # Stream data
        actual_frame_count = stream_data(config)
        wait_for_dataset_ready(config.dataset_name, timeout_s=180)
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
