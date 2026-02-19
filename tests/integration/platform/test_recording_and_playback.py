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

import neuracore as nc

# Add examples dir to path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "..", "..", "..", "examples"))
# ruff: noqa: E402
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH

# How much time we allow for nc.calls
TIME_GRACE_S = 0.05

TEST_ROBOT = "integration_test_robot"
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


class Timer:

    def __init__(self, max_time=TIME_GRACE_S):
        self.max_time = max_time

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        assert self.interval < self.max_time, "Function took too long"


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

    def __str__(self):
        return (
            f"TestConfig(fps={self.fps}, duration={self.duration_sec}s, "
            f"size={self.image_width}x{self.image_height}, cameras={self.num_cameras}, "
            f"depth={self.use_depth}, joints={self.num_joints})"
        )


def encode_frame_number(frame_num, width, height):
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


def decode_frame_number(img):
    """Decode a frame number from an encoded image"""
    frame_bytes = bytearray()
    for i in range(4):
        for j in range(4):
            frame_bytes.append(img[i, j, 0])

    return int.from_bytes(frame_bytes[:16], byteorder="big")


def generate_joint_positions(frame_num, fps, num_joints):
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


def generate_depth_image(frame_num, width, height):
    """Generate a depth image"""
    return np.ones((height, width), dtype=np.float32) * 2.0  # Base 2m depth


def stream_data(config):
    """Stream test data according to configuration"""
    # Start recording
    # TODO: Note this now takes a long time to start since adding in P2P
    with Timer(max_time=10):
        nc.start_recording()

    # Generate and stream test data
    start_time = time.time()
    frame_count = 0

    # Sleep time to maintain target frame rate
    sleep_time = 1.0 / float(config.fps)

    while time.time() - start_time < config.duration_sec:
        t = time.time() if config.synched_time else None
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
            with Timer():
                nc.log_rgb(camera_id, rgb_img, timestamp=t)

            # Depth image if needed
            if config.use_depth:
                depth_img = generate_depth_image(
                    frame_code, config.image_width, config.image_height
                )
                nc.log_depth(camera_id, depth_img, timestamp=t)

        # Stream joint positions
        joint_positions = generate_joint_positions(
            frame_count, config.fps, config.num_joints
        )
        with Timer():
            nc.log_joint_positions(joint_positions, timestamp=t)

        with Timer():
            # use the same joint positions for velocities and torques
            nc.log_joint_velocities(joint_positions, timestamp=t)
        with Timer():
            # use the same joint positions for velocities and torques
            nc.log_joint_torques(joint_positions, timestamp=t)

        with Timer():
            nc.log_parallel_gripper_open_amount(name="gripper", value=0.5, timestamp=t)

        points = np.zeros((1000, 3), dtype=np.float16)
        rgb_points = np.zeros((1000, 3), dtype=np.uint8)
        with Timer(max_time=0.5):
            # TODO: Speed up this call
            nc.log_point_cloud(
                "point_cloud_camera_0",
                points=points,
                rgb_points=rgb_points,
                extrinsics=np.eye(4),
                intrinsics=np.eye(3),
                timestamp=t,
            )

        with Timer():
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
            with Timer():
                nc.log_joint_target_positions(action, timestamp=t)

        frame_count += 1

        # Sleep to maintain target frame rate
        time.sleep(sleep_time)

    with Timer(max_time=10):
        nc.stop_recording(wait=True)

    return frame_count


def verify_dataset(config, expected_frame_count):
    """Verify the dataset integrity"""

    # Retrieve the dataset
    dataset = nc.get_dataset(config.dataset_name)

    # Results tracking
    results = {
        "retrieved_frames": 0,
        "unique_frames": set(),
        "missing_frames": [],
        "duplicate_frames": [],
        "joint_mismatches": [],
    }

    # Iterate through the dataset episodes
    for episode in dataset:
        logger.info(f"Verifying episode with {len(episode)} frames")
        for frame_idx, sync_point in enumerate(episode):
            results["retrieved_frames"] += 1

            # Check for camera images
            if DataType.RGB_IMAGES in sync_point.data:
                for _, cam_data in sync_point[DataType.RGB_IMAGES].items():
                    img = cam_data.frame
                    decoded_frame_num = decode_frame_number(img)
                    if decoded_frame_num in results["unique_frames"]:
                        results["duplicate_frames"].append(decoded_frame_num)
                    results["unique_frames"].add(decoded_frame_num)

            # Verify joint positions
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


def run_streaming_test(config):
    """Run a complete streaming test with the given configuration"""
    # Set up
    logger.info(f"Starting test with config: {config}")
    nc.login()
    nc.connect_robot(
        TEST_ROBOT, urdf_path=str(BIMANUAL_VIPERX_URDF_PATH), overwrite=False
    )
    nc.create_dataset(
        config.dataset_name,
        description=(
            f"Test dataset with {config.fps}fps, "
            f"{config.image_width}x{config.image_height}"
        ),
    )

    # Stream data
    actual_frame_count = stream_data(config)

    time.sleep(2)  # Allow some time for data to be processed

    results = verify_dataset(config, actual_frame_count)

    # Success criteria
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
        fps=200, duration_sec=2, image_width=640, image_height=480, synched_time=True
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
    )
    run_streaming_test(config)


def test_multiple_concurrent_robots():
    num_robots = 3
    configs = [
        TestConfig(fps=30, duration_sec=2, synched_time=True) for _ in range(num_robots)
    ]
    nc.login()

    # Run concurrent streams
    with multiprocessing.Pool(num_robots) as executor:
        frame_counts = list(executor.map(_mp_stream_robot_data, configs))

    # Verify each dataset
    for idx, frame_count in enumerate(frame_counts):
        logger.info(f"Verifying robot {idx+1}/{num_robots}")
        results = verify_dataset(configs[idx], frame_count)
        assert len(results["missing_frames"]) == 0
        assert len(results["duplicate_frames"]) == 0
        assert len(results["joint_mismatches"]) == 0


def test_stop_start_sequences():
    config = TestConfig(fps=30, duration_sec=2, synched_time=True)

    # Set up
    nc.login()
    nc.connect_robot(config.robot_name)
    nc.create_dataset(config.dataset_name)

    segments = 3
    total_frames = 0

    for segment in range(segments):
        logger.info(f"Starting recording segment {segment+1}/{segments}")
        with Timer():
            nc.start_recording()

        # Stream for a bit
        start_time = time.time()
        segment_frames = 0

        while time.time() - start_time < config.duration_sec:
            frame_num = total_frames + segment_frames
            img = encode_frame_number(
                frame_num, config.image_width, config.image_height
            )
            with Timer():
                nc.log_rgb("camera_0", img)

            joint_positions = generate_joint_positions(
                segment_frames, config.fps, config.num_joints
            )
            with Timer():
                nc.log_joint_positions(joint_positions)

            segment_frames += 1
            time.sleep(1 / config.fps)

        # Stop recording
        with Timer(max_time=5):
            nc.stop_recording(wait=True)
        logger.info(f"Completed segment {segment+1} with {segment_frames} frames")

        total_frames += segment_frames

    # Verify dataset
    results = verify_dataset(config, total_frames)
    assert len(results["missing_frames"]) == 0
    assert len(results["duplicate_frames"]) == 0
    assert len(results["joint_mismatches"]) == 0


def test_high_bandwidth():
    """Test streaming with high bandwidth requirements (high res, high fps)"""
    config = TestConfig(
        fps=60,
        duration_sec=2,
        image_width=1920,
        image_height=1080,
        num_cameras=2,
        use_depth=True,
        synched_time=True,
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

    logger.info(f"Estimated bandwidth: {mb_per_second:.2f} MB/s")

    try:
        # Set up
        nc.login()
        nc.connect_robot(config.robot_name)
        nc.create_dataset(config.dataset_name)

        # Stream data
        actual_frame_count = stream_data(config)

        # Verify
        results = verify_dataset(config, actual_frame_count)
        assert len(results["missing_frames"]) == 0
        assert len(results["duplicate_frames"]) == 0
        assert len(results["joint_mismatches"]) == 0

    except Exception as e:
        pytest.fail(f"High bandwidth test failed: {str(e)}")
