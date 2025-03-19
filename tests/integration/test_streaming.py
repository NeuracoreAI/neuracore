import logging
import math
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import neuracore as nc

# Add examples dir to path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "..", "..", "examples"))
# ruff: noqa: E402
from common.constants import BIMANUAL_VIPERX_URDF_PATH

TEST_ROBOT = "integration_test_robot"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
        joint_positions[f"joint{i+1}"] = math.sin(t * freq)

    return joint_positions


def generate_depth_image(frame_num, width, height):
    """Generate a depth image with encoded frame number"""
    # Create a depth image with a gradient
    depth = np.ones((height, width), dtype=np.float32) * 2.0  # Base 2m depth

    # Add a gradient
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    gradient = (xx + yy) * 1.5  # 0 to 3m gradient
    depth += gradient

    # Create a recognizable pattern in the middle based on the frame number
    radius = min(width, height) // 4
    center_x, center_y = width // 2, height // 2
    for i in range(height):
        for j in range(width):
            dist = ((i - center_y) ** 2 + (j - center_x) ** 2) ** 0.5
            if dist < radius:
                # Create a unique depth pattern based on frame number
                depth[i, j] = 1.0 + (frame_num % 100) / 100

    # Encode frame number in top left corner
    for i in range(4):
        for j in range(4):
            depth[i, j] = 0.5 + ((frame_num >> (i * 4 + j)) & 1) * 0.1

    return depth


def stream_data(config):
    """Stream test data according to configuration"""
    # Start recording
    nc.start_recording()
    time.sleep(5)  # Wait a bit for recording to start

    # Generate and stream test data
    start_time = time.time()
    frame_count = 0

    # Sleep time to maintain target frame rate
    sleep_time = 1.0 / float(config.fps)

    while time.time() - start_time < config.duration_sec:
        t = time.time() if config.synched_time else None

        # Create and stream camera images
        for cam_idx in range(config.num_cameras):
            camera_id = f"camera_{cam_idx}"

            # RGB image
            rgb_img = encode_frame_number(
                frame_count, config.image_width, config.image_height
            )
            nc.log_rgb(camera_id, rgb_img, timestamp=t)

            # Depth image if needed
            if config.use_depth:
                depth_img = generate_depth_image(
                    frame_count, config.image_width, config.image_height
                )
                nc.log_depth(camera_id, depth_img, timestamp=t)

        # Stream joint positions
        joint_positions = generate_joint_positions(
            frame_count, config.fps, config.num_joints
        )
        nc.log_joints(joint_positions, timestamp=t)

        # Stream a test action occasionally
        if frame_count % 5 == 0:
            action = {
                f"joint{i+1}": 0.1 * math.sin(frame_count * 0.1 * (i + 1))
                for i in range(config.num_joints)
            }
            nc.log_action(action, timestamp=t)

        frame_count += 1

        # Sleep to maintain target frame rate
        time.sleep(sleep_time)

    time.sleep(5)  # Wait a bit for recording to stop
    nc.stop_recording()

    return frame_count


def verify_dataset(config, expected_frame_count):
    """Verify the dataset integrity"""
    # Wait a bit for server processing to complete
    time.sleep(5)

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
        for frame in episode:
            results["retrieved_frames"] += 1

            # Check for camera images
            if "images" in frame:
                for cam_idx in range(config.num_cameras):
                    camera_id = f"rgb_camera_{cam_idx}"
                    if camera_id in frame["images"]:
                        img = frame["images"][camera_id]

                        decoded_frame_num = decode_frame_number(img)
                        logger.info(f"decoded_frame_num: {decoded_frame_num}")

                        # Track this frame
                        if decoded_frame_num in results["unique_frames"]:
                            results["duplicate_frames"].append(decoded_frame_num)
                        results["unique_frames"].add(decoded_frame_num)

                        # Verify joint positions
                        if "joint_positions" in frame:
                            expected_joints = generate_joint_positions(
                                decoded_frame_num, config.fps, config.num_joints
                            )

                            for joint_name, expected_value in expected_joints.items():
                                if joint_name in frame["joint_positions"]:
                                    actual_value = frame["joint_positions"][joint_name]
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
    nc.connect_robot(TEST_ROBOT, urdf_path=BIMANUAL_VIPERX_URDF_PATH, overwrite=False)
    nc.create_dataset(
        config.dataset_name,
        description=(
            f"Test dataset with {config.fps}fps, "
            f"{config.image_width}x{config.image_height}"
        ),
    )

    # Stream data
    actual_frame_count = stream_data(config)

    # Wait for data to save in the backend
    time.sleep(10)

    results = verify_dataset(config, actual_frame_count)

    # Success criteria
    assert len(results["missing_frames"]) == 0
    assert len(results["duplicate_frames"]) == 0
    assert len(results["joint_mismatches"]) == 0

    return results


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    yield


def test_basic_streaming():
    config = TestConfig(
        fps=10, duration_sec=2, image_width=640, image_height=480, synched_time=True
    )
    run_streaming_test(config)


def test_high_framerate():
    config = TestConfig(fps=200, duration_sec=2, image_width=640, image_height=480)
    run_streaming_test(config)


def test_multiple_cameras():
    config = TestConfig(
        fps=30, duration_sec=2, image_width=640, image_height=480, num_cameras=3
    )
    run_streaming_test(config)


def test_with_depth():
    config = TestConfig(
        fps=30, duration_sec=2, image_width=640, image_height=480, use_depth=True
    )
    run_streaming_test(config)


def test_multiple_concurrent_robots():
    num_robots = 3
    configs = [TestConfig(fps=30, duration_sec=4) for _ in range(num_robots)]

    # Set up all robots
    nc.login()
    for config in configs:
        nc.connect_robot(config.robot_name)
        nc.create_dataset(config.dataset_name)

    # Use threads to stream data concurrently
    def stream_robot_data(idx):
        robot_name = configs[idx].robot_name
        nc.connect_robot(robot_name)  # Reconnect in this thread
        return stream_data(configs[idx])

    # Run concurrent streams
    with ThreadPoolExecutor(max_workers=num_robots) as executor:
        frame_counts = list(executor.map(stream_robot_data, range(num_robots)))

    # Verify each dataset
    all_success = True
    for idx, frame_count in enumerate(frame_counts):
        logger.info(f"Verifying robot {idx+1}/{num_robots}")
        results = verify_dataset(configs[idx], frame_count)

        # Success criteria
        missing_percentage = len(results["missing_frames"]) / frame_count * 100
        if missing_percentage > 10:
            logger.error(f"Robot {idx+1} has {missing_percentage:.1f}% missing frames")
            all_success = False

    assert all_success, "One or more robots failed data verification"


def test_stop_start_sequences():
    config = TestConfig(fps=30, duration_sec=10)

    # Set up
    nc.login()
    nc.connect_robot(config.robot_name)
    nc.create_dataset(config.dataset_name)

    segments = 3
    total_frames = 0

    for segment in range(segments):
        logger.info(f"Starting recording segment {segment+1}/{segments}")
        nc.start_recording()

        # Stream for a bit
        start_time = time.time()
        segment_frames = 0

        while time.time() - start_time < config.duration_sec:
            frame_num = total_frames + segment_frames
            img = encode_frame_number(
                frame_num, config.image_width, config.image_height
            )
            nc.log_rgb("camera_0", img)

            joint_positions = generate_joint_positions(
                frame_num, config.fps, config.num_joints
            )
            nc.log_joints(joint_positions)

            segment_frames += 1
            time.sleep(1 / config.fps)

        # Stop recording
        nc.stop_recording()
        logger.info(f"Completed segment {segment+1} with {segment_frames} frames")

        total_frames += segment_frames

    # Verify dataset
    results = verify_dataset(config, total_frames)

    # Allow some missing frames for stop/start sequences but not too many
    missing_percentage = len(results["missing_frames"]) / total_frames * 100
    assert (
        missing_percentage == 0
    ), f"Too many missing frames: {missing_percentage:.1f}%"
    assert results["retrieved_frames"] > 0, "No frames were retrieved"


def test_high_bandwidth():
    """Test streaming with high bandwidth requirements (high res, high fps)"""
    config = TestConfig(
        fps=60,
        duration_sec=15,
        image_width=1920,
        image_height=1080,
        num_cameras=2,
        use_depth=True,
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

        # For high bandwidth, allow some missing frames
        missing_percentage = len(results["missing_frames"]) / actual_frame_count * 100
        assert (
            missing_percentage < 20
        ), f"Too many missing frames: {missing_percentage:.1f}%"
        assert results["retrieved_frames"] > 0, "No frames were retrieved"

    except Exception as e:
        logger.error(f"High bandwidth test error: {str(e)}")
        pytest.fail(f"High bandwidth test failed: {str(e)}")
    finally:
        nc.stop_all()
