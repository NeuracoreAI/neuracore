import numpy as np
import pytest
from neuracore_types import DataType

import neuracore as nc
from neuracore.core.const import API_URL
from neuracore.core.exceptions import RobotError


def test_log_joints_and_cams(
    temp_config_dir,
    mock_auth_requests,
    reset_neuracore,
    mock_urdf,
    monkeypatch,
    mocked_org_id,
):
    """Test logging actions and sensor data."""
    # Ensure login and robot connection
    nc.login("test_api_key")

    # Mock robot creation
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )

    # Connect robot
    nc.connect_robot("test_robot", urdf_path=mock_urdf)

    nc.log_joint_positions(
        positions={"vx300s_left/waist": 0.5, "vx300s_right/waist": -0.3},
    )

    # Uint8 image
    rgb_uint8 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    nc.log_rgb("front_camera", rgb_uint8)

    # Test depth logging
    depth = np.ones((100, 100), dtype=np.float32) * 1.0  # meters
    nc.log_depth("depth_camera", depth)


def test_log_with_extrinsics_intrinsics(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf, mocked_org_id
):
    """Test logging with extrinsics and intrinsics matrices."""
    # Ensure login and robot connection
    nc.login("test_api_key")
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )
    nc.connect_robot("test_robot", urdf_path=mock_urdf)

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
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf, mocked_org_id
):
    """Test logging gripper data."""
    # Ensure login and robot connection
    nc.login("test_api_key")
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )
    nc.connect_robot("test_robot", urdf_path=mock_urdf)

    nc.log_parallel_gripper_open_amount(name="gripper1", value=0.5)
    nc.log_parallel_gripper_open_amount(name="gripper2", value=0.7)
    nc.log_parallel_gripper_open_amounts({"gripper1": 0.5, "gripper2": 0.7})


def test_log_joint_velocities_and_torques(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf, mocked_org_id
):
    """Test logging joint velocities and torques."""
    # Ensure login and robot connection
    nc.login("test_api_key")
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )
    nc.connect_robot("test_robot", urdf_path=mock_urdf)

    nc.log_joint_velocities({"joint1": 0.5, "joint2": -0.3})
    nc.log_joint_torques({"joint1": 1.5, "joint2": 2.3})
    nc.log_joint_velocity(name="joint1", velocity=0.5)
    nc.log_joint_torque(name="joint2", torque=2.3)


def test_log_joint_group_cache_reuse_and_rebuild(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf, mocked_org_id
):
    """A stable joint set reuses the resolved group; a changed set rebuilds it."""
    nc.login("test_api_key")
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )
    robot = nc.connect_robot("test_robot", urdf_path=mock_urdf)

    # Two frames with the same joints reuse the same cached group object — the
    # steady-state hot path does no per-joint re-resolution.
    nc.log_joint_positions({"joint1": 0.5, "joint2": -0.3})
    first_group = robot._joint_group_cache[DataType.JOINT_POSITIONS]
    nc.log_joint_positions({"joint1": 0.6, "joint2": -0.4})
    assert robot._joint_group_cache[DataType.JOINT_POSITIONS] is first_group

    # A changed joint set is rebuilt — the caller may change which joints they
    # log and we must honour the live set, never a stale cached one.
    nc.log_joint_positions({"joint1": 0.6, "joint2": -0.4, "joint3": 0.1})
    rebuilt_group = robot._joint_group_cache[DataType.JOINT_POSITIONS]
    assert rebuilt_group is not first_group
    assert len(rebuilt_group.bindings) == 3


def test_log_joint_group_rebuilds_on_middle_joint_change(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf, mocked_org_id
):
    """A changed *middle* joint rebuilds the group even with stable boundaries.

    The group cache key is the exact joint-name tuple, not a sampled one, so a
    rename far from either end — which a boundary-only check would miss — is
    still detected and the data is logged against the correct joints.
    """
    nc.login("test_api_key")
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )
    robot = nc.connect_robot("test_robot", urdf_path=mock_urdf)

    # 20 joints — wider than the value-validation sample on each end, so the
    # changed joint below sits in the unsampled middle.
    frame = {f"joint{index}": float(index) for index in range(20)}
    nc.log_joint_positions(frame)
    first_group = robot._joint_group_cache[DataType.JOINT_POSITIONS]

    # Rebuild preserving order so only the middle entry differs; the first and
    # last entries (the only ones a boundary check would inspect) are unchanged.
    changed = {
        ("renamed_middle" if name == "joint10" else name): value
        for name, value in frame.items()
    }
    nc.log_joint_positions(changed)
    rebuilt_group = robot._joint_group_cache[DataType.JOINT_POSITIONS]

    assert rebuilt_group is not first_group
    assert "renamed_middle" in rebuilt_group.joint_names
    assert "joint10" not in rebuilt_group.joint_names


def test_pruned_joint_streams_reregister_on_next_log(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf, mocked_org_id
):
    """Pruning a joint stream evicts its cached binding so the next log call
    registers a live stream instead of reviving the pruned one."""
    nc.login("test_api_key")
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )
    robot = nc.connect_robot("test_robot", urdf_path=mock_urdf)

    nc.log_joint_positions({"joint1": 0.5, "joint2": -0.3})
    pruned_streams = list(robot._data_streams.values())
    assert pruned_streams
    for stream_id, stream in list(robot._data_streams.items()):
        robot._remove_data_stream(stream_id, stream)
    assert not robot._data_streams

    nc.log_joint_positions({"joint1": 0.6, "joint2": -0.4})

    bindings = robot._joint_stream_bindings[DataType.JOINT_POSITIONS]
    assert len(bindings) == 2
    for binding in bindings.values():
        assert robot.get_data_stream(binding.stream_id) is binding.stream
        assert binding.stream not in pruned_streams
    group = robot._joint_group_cache[DataType.JOINT_POSITIONS]
    for binding in group.bindings:
        assert robot.get_data_stream(binding.stream_id) is binding.stream


def test_log_joints_large_frame(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf, mocked_org_id
):
    """A frame larger than the smoke sample logs every joint without error."""
    nc.login("test_api_key")
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )
    nc.connect_robot("test_robot", urdf_path=mock_urdf)

    positions = {f"joint{index}": float(index) for index in range(100)}
    nc.log_joint_positions(positions)
    sync_point = nc.get_latest_sync_point(include_remote=False)
    logged = sync_point.data[DataType.JOINT_POSITIONS]
    assert len(logged) == 100


def test_log_visual_joint_positions(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf, mocked_org_id
):
    """Test logging visual joint positions."""
    # Ensure login and robot connection
    nc.login("test_api_key")
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )
    nc.connect_robot("test_robot", urdf_path=mock_urdf)

    # Test plural function with dict
    nc.log_visual_joint_positions({"finger1": 0.5, "finger2": -0.3})

    # Test singular function
    nc.log_visual_joint_position(name="finger1", position=0.5)
    nc.log_visual_joint_position(name="finger2", position=-0.3)


def test_log_language(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf, mocked_org_id
):
    """Test logging language annotations."""
    # Ensure login and robot connection
    nc.login("test_api_key")
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )
    nc.connect_robot("test_robot", urdf_path=mock_urdf)

    nc.log_language(name="instruction", language="Pick up the red cube")
    nc.log_language(name="sub_task", language="Move to the table")


def test_log_custom_data(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf, mocked_org_id
):
    """Test logging custom data."""
    # Ensure login and robot connection
    nc.login("test_api_key")
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )
    nc.connect_robot("test_robot", urdf_path=mock_urdf)
    nc.log_custom_1d("vision_detections", np.array([10, 20, 50, 60]))


def test_log_point_cloud(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf, mocked_org_id
):
    """Test logging point cloud data."""
    # Ensure login and robot connection
    nc.login("test_api_key")
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )
    nc.connect_robot("test_robot", urdf_path=mock_urdf)

    points = np.random.rand(1000, 3).astype(np.float16)

    # Optional RGB data for each point
    rgb_points = np.random.randint(0, 256, (1000, 3), dtype=np.uint8)

    # Log point cloud
    nc.log_point_cloud("lidar", points, rgb_points=rgb_points)


def test_log_with_no_robot(temp_config_dir, mock_auth_requests, reset_neuracore):
    """Test that logging without an active robot raises an error."""
    # Ensure login but don't connect a robot
    nc.logout()
    nc.login("test_api_key")

    # Attempt to log data without an active robot should raise an error
    with pytest.raises(RobotError, match="No active robot"):
        nc.log_joint_positions({"joint1": 0.5})


def test_log_invalid_data_format(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf, mocked_org_id
):
    """Test validation of input data formats."""
    # Ensure login and robot connection
    nc.login("test_api_key")
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )
    nc.connect_robot("test_robot", urdf_path=mock_urdf)

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

    with pytest.raises(ValueError, match="End effector pose must be a numpy array"):
        nc.log_end_effector_pose(name="right_ee", pose="not_a_list")

    with pytest.raises(ValueError, match="End effector names must be strings"):
        nc.log_end_effector_pose(
            name=123, pose=np.array([0.6, 0.4, 0.3, 0.0, 0.7071, 0.0, 0.7071])
        )

    # Test invalid end effector pose quaternions
    with pytest.raises(
        ValueError, match="End effector pose must be a valid unit quaternion"
    ):
        nc.log_end_effector_pose(
            name="right_ee", pose=np.array([0.6, 0.4, 0.3, 0.0, 1.0, 0.0, 1.0])
        )

    # Test invalid end effector pose length (not 7 elements)
    with pytest.raises(
        ValueError, match="End effector pose must be a 7-element numpy array"
    ):
        nc.log_end_effector_pose(
            name="left_ee", pose=np.array([0.5, 0.3, 0.2, 0.5, 0.5, 0.5])
        )

    # Test invalid pose type (not numpy array)
    with pytest.raises(ValueError, match="Pose must be a numpy array"):
        nc.log_pose(name="object_pose", pose="not_an_array")

    # Test invalid pose type (list instead of numpy array)
    with pytest.raises(ValueError, match="Pose must be a numpy array"):
        nc.log_pose(name="object_pose", pose=[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])

    # Test invalid pose length (not 7 elements)
    with pytest.raises(ValueError, match="Pose must be a numpy array of length 7"):
        nc.log_pose(name="object_pose", pose=np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]))

    with pytest.raises(
        ValueError, match="Parallel gripper open amounts must be floats"
    ):
        nc.log_parallel_gripper_open_amount(name="gripper1", value="not_a_float")

    with pytest.raises(
        ValueError, match="Parallel gripper open amounts must be between 0.0 and 1.0"
    ):
        nc.log_parallel_gripper_open_amount(name="gripper1", value=-0.5)

    with pytest.raises(
        ValueError, match="Parallel gripper open amounts must be between 0.0 and 1.0"
    ):
        nc.log_parallel_gripper_open_amount(name="gripper2", value=1.5)

    with pytest.raises(ValueError, match="Parallel gripper names must be strings"):
        nc.log_parallel_gripper_open_amount(name=123, value=0.5)


def test_log_end_effector_poses(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf, mocked_org_id
):
    """Test logging end effector pose data."""
    nc.login("test_api_key")
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )
    nc.connect_robot("test_robot", urdf_path=mock_urdf)

    nc.log_end_effector_pose(
        name="left_ee",
        pose=np.array([0.5, 0.3, 0.2, 0.5, 0.5, 0.5, 0.5]),
    )


def test_log_pose(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf, mocked_org_id
):
    """Test logging pose data."""
    nc.login("test_api_key")
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )
    nc.connect_robot("test_robot", urdf_path=mock_urdf)

    nc.log_pose(
        name="object_pose",
        pose=np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]),
    )


def test_log_parallel_gripper_open_amounts(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf, mocked_org_id
):
    """Test logging parallel gripper open amounts."""
    nc.login("test_api_key")
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )
    nc.connect_robot("test_robot", urdf_path=mock_urdf)

    nc.log_parallel_gripper_open_amount(
        name="gripper1",
        value=0.5,
    )
    nc.log_parallel_gripper_open_amount(
        name="gripper2",
        value=0.7,
    )


def test_log_parallel_gripper_target_open_amounts(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf, mocked_org_id
):
    """Test logging target parallel gripper open amounts."""
    nc.login("test_api_key")
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )
    nc.connect_robot("test_robot", urdf_path=mock_urdf)

    # Test singular function
    nc.log_parallel_gripper_target_open_amount(
        name="gripper1",
        value=0.5,
    )
    nc.log_parallel_gripper_target_open_amount(
        name="gripper2",
        value=0.7,
    )

    # Test plural function with dict
    nc.log_parallel_gripper_target_open_amounts({"gripper1": 0.5, "gripper2": 0.7})


def test_log_parallel_gripper_target_open_amount_validation(
    temp_config_dir, mock_auth_requests, reset_neuracore, mock_urdf, mocked_org_id
):
    """Test validation of target parallel gripper open amount inputs."""
    nc.login("test_api_key")
    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "mock_robot_id", "has_urdf": True},
        status_code=200,
    )
    nc.connect_robot("test_robot", urdf_path=mock_urdf)

    # Test invalid value type
    with pytest.raises(
        ValueError, match="Parallel gripper target open amounts must be floats"
    ):
        nc.log_parallel_gripper_target_open_amount(name="gripper1", value="not_a_float")

    # Test value below range
    with pytest.raises(
        ValueError,
        match="Parallel gripper target open amounts must be between 0.0 and 1.0",
    ):
        nc.log_parallel_gripper_target_open_amount(name="gripper1", value=-0.5)

    # Test value above range
    with pytest.raises(
        ValueError,
        match="Parallel gripper target open amounts must be between 0.0 and 1.0",
    ):
        nc.log_parallel_gripper_target_open_amount(name="gripper2", value=1.5)

    # Test invalid name type
    with pytest.raises(ValueError, match="Parallel gripper names must be strings"):
        nc.log_parallel_gripper_target_open_amount(name=123, value=0.5)
