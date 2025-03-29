import numpy as np
import pytest

import neuracore as nc
from neuracore.core.const import API_URL
from neuracore.core.exceptions import RobotError


def test_log_actions(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf, monkeypatch
):
    """Test logging actions and sensor data."""
    # Ensure login and robot connection
    nc.login("test_api_key")

    # Mock robot creation
    mock_auth_requests.post(f"{API_URL}/robots", json="mock_robot_id", status_code=200)

    # Connect robot
    nc.connect_robot("test_robot", mock_urdf)

    # Test logging functions
    try:
        nc.log_joint_positions({"vx300s_left/waist": 0.5, "vx300s_right/waist": -0.3})
        nc.log_action({"action1": 0.1, "action2": 0.2})

        # Uint8 image
        rgb_uint8 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        nc.log_rgb("front_camera", rgb_uint8)

        # Test depth logging
        depth = np.ones((100, 100), dtype=np.float32) * 1.0  # meters
        nc.log_depth("depth_camera", depth)

    except Exception as e:
        pytest.fail(f"Logging functions raised unexpected exception: {e}")


def test_log_with_extrinsics_intrinsics(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf
):
    """Test logging with extrinsics and intrinsics matrices."""
    # Ensure login and robot connection
    nc.login("test_api_key")
    mock_auth_requests.post(f"{API_URL}/robots", json="mock_robot_id", status_code=200)
    nc.connect_robot("test_robot", mock_urdf)

    # Create test data
    rgb_uint8 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.float32) * 1.0  # meters

    # Create extrinsics and intrinsics matrices
    extrinsics = np.eye(4, dtype=np.float32)
    intrinsics = np.array([[500, 0, 50], [0, 500, 50], [0, 0, 1]], dtype=np.float32)

    # Log with extrinsics and intrinsics
    nc.log_rgb("front_camera", rgb_uint8, extrinsics=extrinsics, intrinsics=intrinsics)
    nc.log_depth("depth_camera", depth, extrinsics=extrinsics, intrinsics=intrinsics)


def test_log_gripper_data(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf
):
    """Test logging gripper data."""
    # Ensure login and robot connection
    nc.login("test_api_key")
    mock_auth_requests.post(f"{API_URL}/robots", json="mock_robot_id", status_code=200)
    nc.connect_robot("test_robot", mock_urdf)

    # Log gripper data
    nc.log_gripper_data({"gripper1": 0.5, "gripper2": 0.7})


def test_log_joint_velocities_and_torques(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf
):
    """Test logging joint velocities and torques."""
    # Ensure login and robot connection
    nc.login("test_api_key")
    mock_auth_requests.post(f"{API_URL}/robots", json="mock_robot_id", status_code=200)
    nc.connect_robot("test_robot", mock_urdf)

    # Log joint velocities
    nc.log_joint_velocities({"joint1": 0.5, "joint2": -0.3})

    # Log joint torques
    nc.log_joint_torques({"joint1": 1.5, "joint2": 2.3})


def test_log_language(temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf):
    """Test logging language annotations."""
    # Ensure login and robot connection
    nc.login("test_api_key")
    mock_auth_requests.post(f"{API_URL}/robots", json="mock_robot_id", status_code=200)
    nc.connect_robot("test_robot", mock_urdf)

    # Log language
    nc.log_language("Pick up the red cube")


def test_log_custom_data(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf
):
    """Test logging custom data."""
    # Ensure login and robot connection
    nc.login("test_api_key")
    mock_auth_requests.post(f"{API_URL}/robots", json="mock_robot_id", status_code=200)
    nc.connect_robot("test_robot", mock_urdf)

    # Log custom data
    custom_data = {
        "object_detections": [
            {"label": "cube", "confidence": 0.95, "bbox": [10, 20, 50, 60]},
            {"label": "sphere", "confidence": 0.82, "bbox": [100, 120, 150, 180]},
        ]
    }
    nc.log_custom_data("vision_detections", custom_data)


def test_log_point_cloud(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf
):
    """Test logging point cloud data."""
    # Ensure login and robot connection
    nc.login("test_api_key")
    mock_auth_requests.post(f"{API_URL}/robots", json="mock_robot_id", status_code=200)
    nc.connect_robot("test_robot", mock_urdf)

    # Create a small point cloud (1000 points x 3 dimensions)
    points = np.random.rand(1000, 3).astype(np.float32)

    # Optional RGB data for each point
    rgb_points = np.random.randint(0, 256, (1000, 3), dtype=np.uint8)

    # Log point cloud
    nc.log_point_cloud("lidar", points, rgb_points=rgb_points)


def test_log_synced_data(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf
):
    """Test logging synchronized data from multiple sensors."""
    # Ensure login and robot connection
    nc.login("test_api_key")
    mock_auth_requests.post(f"{API_URL}/robots", json="mock_robot_id", status_code=200)
    nc.connect_robot("test_robot", mock_urdf)

    # Prepare test data
    joint_positions = {"joint1": 0.5, "joint2": -0.3}
    joint_velocities = {"joint1": 0.1, "joint2": -0.2}
    joint_torques = {"joint1": 1.0, "joint2": 2.0}
    gripper_open_amounts = {"gripper1": 0.5}
    action = {"action1": 0.1, "action2": 0.2}

    # RGB images
    rgb_data = {"cam1": np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)}

    # Depth images
    depth_data = {"cam1": np.ones((100, 100), dtype=np.float32) * 0.5}

    # Point clouds (empty for this test)
    point_cloud_data = {}

    # Log synced data
    nc.log_synced_data(
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
        joint_torques=joint_torques,
        gripper_open_amounts=gripper_open_amounts,
        action=action,
        rgb_data=rgb_data,
        depth_data=depth_data,
        point_cloud_data=point_cloud_data,
    )


def test_log_with_no_robot(temp_config_dir, mock_auth_requests, reset_neuracore):
    """Test that logging without an active robot raises an error."""
    # Ensure login but don't connect a robot
    nc.logout()
    nc.login("test_api_key")

    # Attempt to log data without an active robot should raise an error
    with pytest.raises(RobotError, match="No active robot"):
        nc.log_joint_positions({"joint1": 0.5})


def test_log_invalid_data_format(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf
):
    """Test validation of input data formats."""
    # Ensure login and robot connection
    nc.login("test_api_key")
    mock_auth_requests.post(f"{API_URL}/robots", json="mock_robot_id", status_code=200)
    nc.connect_robot("test_robot", mock_urdf)

    # Test invalid joint positions (not float)
    with pytest.raises(ValueError, match="Joint data must be floats"):
        nc.log_joint_positions({"joint1": "not_a_float"})

    # Test invalid image format (wrong dimensions)
    with pytest.raises(ValueError, match="Image must be uint8"):
        nc.log_rgb(
            "camera", np.ones((100, 100), dtype=np.float32)
        )  # Missing channel dimension

    # Test invalid depth format (wrong dtype)
    with pytest.raises(ValueError, match="Depth image must be float16 or float32"):
        nc.log_depth("camera", np.ones((100, 100), dtype=np.uint8))

    # Test depth values exceed max depth
    with pytest.raises(ValueError, match="Depth image should be in meters"):
        nc.log_depth(
            "camera", np.ones((100, 100), dtype=np.float32) * 1000
        )  # Too large
