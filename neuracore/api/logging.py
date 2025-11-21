"""Robot data logging utilities.

This module provides functions for logging various types of robot sensor data
including joint positions, camera images, point clouds, and custom data streams.
All logging functions support optional robot identification and timestamping.
"""

import json
import time
from typing import Any, Optional, Tuple
from warnings import filterwarnings, warn

import numpy as np
from neuracore_types import (
    CameraData,
    CustomData,
    DataType,
    EndEffectorPoseData,
    JointData,
    LanguageData,
    ParallelGripperOpenAmountData,
    PointCloudData,
    PoseData,
)

from neuracore.api.core import _get_robot
from neuracore.core.exceptions import RobotError
from neuracore.core.robot import Robot
from neuracore.core.streaming.data_stream import (
    DataStream,
    DepthDataStream,
    JsonDataStream,
    RGBDataStream,
    VideoDataStream,
)
from neuracore.core.streaming.p2p.stream_manager_orchestrator import (
    StreamManagerOrchestrator,
)
from neuracore.core.utils.depth_utils import MAX_DEPTH


class ExperimentalPointCloudWarning(UserWarning):
    """Warning for experimental point cloud features."""

    pass


filterwarnings("once", category=ExperimentalPointCloudWarning)


def start_stream(robot: Robot, data_stream: DataStream) -> None:
    """Start recording on a data stream if robot is currently recording.

    Args:
        robot: Robot instance
        data_stream: Data stream to start recording on
    """
    current_recording = robot.get_current_recording_id()
    if current_recording is not None and not data_stream.is_recording():
        data_stream.start_recording(current_recording)


def _log_joint_data(
    name: str,
    data_type: DataType,
    folder_name: str,
    joint_data: dict[str, float],
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log joint data for a robot.

    Args:
        name: Name of the joint group
        data_type: Type of joint data (e.g. DataType.JOINT_POSITIONS)
        folder_name: Folder name for the joint data type
        joint_data: Dictionary mapping joint names to joint data values
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If joint_data is not a dictionary of floats
    """
    timestamp = timestamp or time.time()
    if not isinstance(joint_data, dict):
        raise ValueError("Joint data must be a dictionary of floats")
    for key, value in joint_data.items():
        if not isinstance(value, float):
            raise ValueError(f"Joint data must be floats. {key} is not a float.")

    robot = _get_robot(robot_name, instance)
    joint_str_id = f"{folder_name}_{name}"
    joint_stream = robot.get_data_stream(joint_str_id)
    if joint_stream is None:
        joint_stream = JsonDataStream(f"{folder_name}/{name}.json")
        robot.add_data_stream(joint_str_id, joint_stream)

    start_stream(robot, joint_stream)

    data = JointData(
        timestamp=timestamp,
        values=joint_data,
    )
    assert isinstance(
        joint_stream, JsonDataStream
    ), "Expected stream to be instance of JSONDataStream"
    joint_stream.log(data=data)
    if robot.id is None:
        raise RobotError("Robot not initialized. Call init() first.")
    StreamManagerOrchestrator().get_provider_manager(
        robot.id, robot.instance
    ).get_json_source(folder_name, data_type, sensor_key=joint_str_id).publish(
        data.model_dump(mode="json")
    )


def _validate_extrinsics_intrinsics(
    extrinsics: Optional[np.ndarray], intrinsics: Optional[np.ndarray]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Validate and convert camera extrinsics and intrinsics matrices.

    Args:
        extrinsics: Optional extrinsics matrix as numpy array
        intrinsics: Optional intrinsics matrix as numpy array

    Returns:
        Tuple of validated (intrinsics, extrinsics) matrices

    Raises:
        ValueError: If matrices have incorrect shapes
    """
    if extrinsics is not None:
        if not isinstance(extrinsics, np.ndarray) or extrinsics.shape != (4, 4):
            raise ValueError("Extrinsics must be a numpy array of shape (4, 4)")
        extrinsics = extrinsics.astype(np.float16)
    if intrinsics is not None:
        if not isinstance(intrinsics, np.ndarray) or intrinsics.shape != (3, 3):
            raise ValueError("Intrinsics must be a numpy array of shape (3, 3)")
        intrinsics = intrinsics.astype(np.float16)
    return extrinsics, intrinsics


def _log_camera_data(
    camera_type: DataType,
    camera_id: str,
    image: np.ndarray,
    extrinsics: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log camera data for a robot.

    Args:
        camera_type: Type of camera (e.g. DataType.RGB or DataType.DEPTH)
        camera_id: Unique identifier for the camera
        image: Image data as numpy array
        extrinsics: Optional extrinsics matrix (4x4)
        intrinsics: Optional intrinsics matrix (3x3)
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If image format is invalid or camera type is unsupported
    """
    assert camera_type in (
        DataType.RGB_IMAGES,
        DataType.DEPTH_IMAGES,
    ), "Unsupported camera type"

    timestamp = timestamp or time.time()
    extrinsics, intrinsics = _validate_extrinsics_intrinsics(extrinsics, intrinsics)
    robot = _get_robot(robot_name, instance)
    full_cam_id = f"{camera_type.value}_{camera_id}"

    stream = robot.get_data_stream(full_cam_id)
    if stream is None:
        if camera_type == DataType.RGB_IMAGES:
            stream = RGBDataStream(full_cam_id, image.shape[1], image.shape[0])
        elif camera_type == DataType.DEPTH_IMAGES:
            stream = DepthDataStream(full_cam_id, image.shape[1], image.shape[0])
        else:
            raise ValueError(f"Invalid camera type: {camera_type}")
        robot.add_data_stream(full_cam_id, stream)

    start_stream(robot, stream)

    assert isinstance(
        stream, VideoDataStream
    ), "Expected stream as instance of VideoDataStream"

    if stream.width != image.shape[1] or stream.height != image.shape[0]:
        raise ValueError(
            f"Camera image dimensions {image.shape[1]}x{image.shape[0]} do not match "
            f"stream dimensions {stream.width}x{stream.height}"
        )

    if extrinsics is not None:
        extrinsics = extrinsics.tolist()
    if intrinsics is not None:
        intrinsics = intrinsics.tolist()

    camera_data = CameraData(
        timestamp=timestamp, extrinsics=extrinsics, intrinsics=intrinsics
    )
    stream.log(
        image,
        camera_data,
    )
    if robot.id is None:
        raise RobotError("Robot not initialized. Call init() first.")
    StreamManagerOrchestrator().get_provider_manager(
        robot.id, robot.instance
    ).get_video_source(camera_id, camera_type, f"{camera_id}_{camera_type}").add_frame(
        image, camera_data
    )


def log_custom_data(
    name: str,
    data: Any,
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log arbitrary data for a robot.

    Args:
        name: Name of the data stream
        data: Data to log (must be JSON serializable)
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If data is not JSON serializable
    """
    timestamp = timestamp or time.time()
    robot = _get_robot(robot_name, instance)
    str_id = f"{name}_custom"
    stream = robot.get_data_stream(str_id)
    if stream is None:
        stream = JsonDataStream(f"custom/{name}.json")
        robot.add_data_stream(str_id, stream)

    start_stream(robot, stream)

    try:
        json.dumps(data)
    except TypeError:
        raise ValueError(
            "Data is not serializable. Please ensure that all data is serializable."
        )
    assert isinstance(
        stream, JsonDataStream
    ), "Expected stream to be instance of JSONDataStream"

    custom_data = CustomData(timestamp=timestamp, data=data)
    stream.log(custom_data)

    if robot.id is None:
        raise RobotError("Robot not initialized. Call init() first.")

    StreamManagerOrchestrator().get_provider_manager(
        robot.id, robot.instance
    ).get_json_source(name, DataType.CUSTOM, sensor_key=str_id).publish(
        custom_data.model_dump(mode="json")
    )


def log_joint_positions(
    name: str,
    positions: dict[str, float],
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log joint positions for a robot.

    Args:
        name: Name of the joint group
        positions: Dictionary mapping joint names to positions (in radians)
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If positions is not a dictionary of floats
    """
    _log_joint_data(
        name,
        DataType.JOINT_POSITIONS,
        "joint_positions",
        positions,
        robot_name,
        instance,
        timestamp,
    )


def log_joint_target_positions(
    name: str,
    target_positions: dict[str, float],
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log joint target positions for a robot.

    Args:
        name: Name of the joint group
        target_positions: Dictionary mapping joint names to
            target positions (in radians)
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If target_positions is not a dictionary of floats
    """
    _log_joint_data(
        name,
        DataType.JOINT_TARGET_POSITIONS,
        "joint_target_positions",
        target_positions,
        robot_name,
        instance,
        timestamp,
    )


def log_joint_velocities(
    name: str,
    velocities: dict[str, float],
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log joint velocities for a robot.

    Args:
        name: Name of the joint group
        velocities: Dictionary mapping joint names to velocities (in radians/second)
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If velocities is not a dictionary of floats
    """
    _log_joint_data(
        name,
        DataType.JOINT_VELOCITIES,
        "joint_velocities",
        velocities,
        robot_name,
        instance,
        timestamp,
    )


def log_joint_torques(
    name: str,
    torques: dict[str, float],
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log joint torques for a robot.

    Args:
        name: Name of the joint group
        torques: Dictionary mapping joint names to torques (in Newton-meters)
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If torques is not a dictionary of floats
    """
    _log_joint_data(
        name,
        DataType.JOINT_TORQUES,
        "joint_torques",
        torques,
        robot_name,
        instance,
        timestamp,
    )


def log_pose(
    name: str,
    pose: np.ndarray,
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log pose data for a robot.

    Args:
        name: Name of the pose.
        pose: 7-element lists: [x, y, z, qx, qy, qz, qw]
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If poses is not a dictionary of 7-element lists
    """
    timestamp = timestamp or time.time()
    if not isinstance(pose, np.ndarray):
        raise ValueError(f"Poses must be lists. {name} is not a list.")
    if len(pose) != 7:
        raise ValueError(f"Poses must be lists of length 7. {name} is not length 7.")
    robot = _get_robot(robot_name, instance)
    str_id = f"{name}_pose_data"
    stream = robot.get_data_stream(str_id)
    if stream is None:
        stream = JsonDataStream(f"poses/{name}.json")
        robot.add_data_stream(str_id, stream)

    start_stream(robot, stream)
    assert isinstance(
        stream, JsonDataStream
    ), "Expected stream to be instance of JSONDataStream"

    pose_data = PoseData(timestamp=timestamp, pose=pose.tolist())
    stream.log(pose_data)

    if robot.id is None:
        raise RobotError("Robot not initialized. Call init() first.")

    StreamManagerOrchestrator().get_provider_manager(
        robot.id, robot.instance
    ).get_json_source(str_id, DataType.POSES, sensor_key=str_id).publish(
        pose_data.model_dump(mode="json")
    )


def log_end_effector_pose(
    name: str,
    pose: np.ndarray,
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log end-effector pose data for a robot.

    Args:
        name: Name of the end effector
        pose: 7-element lists: [x, y, z, qx, qy, qz, qw]
        robot_name: Optional robot ID
        instance: Optional instance number
        timestamp: Optional timestamp
    """
    timestamp = timestamp or time.time()

    if not isinstance(pose, np.ndarray):
        raise ValueError(
            f"End effector pose must be a list. " f"{pose} is of type {type(pose)}"
        )
    if len(pose) != 7:
        raise ValueError(
            f"End effector pose must be a 7-element list. "
            f"{name} is of length {len(pose)}."
        )
    if not isinstance(name, str):
        raise ValueError(
            f"End effector names must be strings. " f"{name} is of type {type(name)}"
        )
    # check if last 4 elements of pose are a valid quaternion
    orientation = pose[3:]
    if not np.isclose(np.linalg.norm(orientation), 1.0, atol=1e-4):
        raise ValueError(
            f"End effector pose must be a valid unit quaternion. "
            f"{orientation} is not a valid unit quaternion."
        )

    robot = _get_robot(robot_name, instance)
    str_id = f"{name}_end_effector_poses"
    stream = robot.get_data_stream(str_id)
    if stream is None:
        stream = JsonDataStream(f"end_effector_poses/{name}.json")
        robot.add_data_stream(str_id, stream)

    start_stream(robot, stream)
    assert isinstance(stream, JsonDataStream)

    ee_pose_data = EndEffectorPoseData(timestamp=timestamp, pose=pose.tolist())
    stream.log(ee_pose_data)

    if robot.id is None:
        raise RobotError("Robot not initialized. Call init() first.")

    StreamManagerOrchestrator().get_provider_manager(
        robot.id, robot.instance
    ).get_json_source(str_id, DataType.END_EFFECTOR_POSES, sensor_key=str_id).publish(
        ee_pose_data.model_dump(mode="json")
    )


def log_parallel_gripper_open_amount(
    name: str,
    value: float,
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log parallel gripper open amount data for a robot.

    Args:
        name: Name of the parallel gripper
        value: Open amount (0.0 = closed, 1.0 = fully open)
        robot_name: Optional robot ID
        instance: Optional instance number
        timestamp: Optional timestamp
    """
    timestamp = timestamp or time.time()
    if not isinstance(name, str):
        raise ValueError(
            f"Parallel gripper names must be strings. " f"{name} is not a string."
        )
    if not isinstance(value, float):
        raise ValueError(
            f"Parallel gripper open amounts must be floats. " f"{value} is not a float."
        )
    if value < 0.0 or value > 1.0:
        raise ValueError("Parallel gripper open amounts must be between 0.0 and 1.0.")

    robot = _get_robot(robot_name, instance)
    str_id = f"{name}_parallel_gripper_open_amounts"
    stream = robot.get_data_stream(str_id)
    if stream is None:
        stream = JsonDataStream(f"parallel_gripper_open_amounts/{name}.json")
        robot.add_data_stream(str_id, stream)

    start_stream(robot, stream)
    assert isinstance(stream, JsonDataStream)

    parallel_gripper_open_amount_data = ParallelGripperOpenAmountData(
        timestamp=timestamp, open_amounts=value
    )
    stream.log(parallel_gripper_open_amount_data)

    if robot.id is None:
        raise RobotError("Robot not initialized. Call init() first.")

    StreamManagerOrchestrator().get_provider_manager(
        robot.id, robot.instance
    ).get_json_source(str_id, DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS, str_id).publish(
        parallel_gripper_open_amount_data.model_dump(mode="json")
    )


def log_language(
    name: str,
    language: str,
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log language annotation for a robot.

    Args:
        name: Name of the language annotation
        language: A language string associated with this timestep
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If language is not a string
    """
    timestamp = timestamp or time.time()
    if not isinstance(language, str):
        raise ValueError("Language must be a string")
    robot = _get_robot(robot_name, instance)
    str_id = f"{name}_language"
    stream = robot.get_data_stream(str_id)
    if stream is None:
        stream = JsonDataStream(f"language/{name}.json")
        robot.add_data_stream(str_id, stream)
    start_stream(robot, stream)
    assert isinstance(
        stream, JsonDataStream
    ), "Expected stream to be instance of JSONDataStream"

    data = LanguageData(timestamp=timestamp, text=language)
    stream.log(data)

    if robot.id is None:
        raise RobotError("Robot not initialized. Call init() first.")

    StreamManagerOrchestrator().get_provider_manager(
        robot.id, robot.instance
    ).get_json_source(str_id, DataType.LANGUAGE, sensor_key=str_id).publish(
        data.model_dump(mode="json")
    )


def log_rgb(
    name: str,
    rgb: np.ndarray,
    extrinsics: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log RGB image from a camera.

    Args:
        name: Unique identifier for the camera
        rgb: RGB image as numpy array (HxWx3, dtype=uint8)
        extrinsics: Optional extrinsics matrix (4x4)
        intrinsics: Optional intrinsics matrix (3x3)
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If image format is invalid
    """
    if not isinstance(rgb, np.ndarray):
        raise ValueError("Image image must be a numpy array")
    if rgb.dtype != np.uint8:
        raise ValueError("Image must be uint8 with range 0-255")
    _log_camera_data(
        DataType.RGB_IMAGES,
        name,
        rgb,
        extrinsics,
        intrinsics,
        robot_name,
        instance,
        timestamp,
    )


def log_depth(
    name: str,
    depth: np.ndarray,
    extrinsics: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log depth image from a camera.

    Args:
        name: Unique identifier for the camera
        depth: Depth image as numpy array (HxW, dtype=float16 or float32, in meters)
        extrinsics: Optional extrinsics matrix (4x4)
        intrinsics: Optional intrinsics matrix (3x3)
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If depth format is invalid
    """
    if not isinstance(depth, np.ndarray):
        raise ValueError("Depth image must be a numpy array")
    if depth.dtype not in (np.float16, np.float32):
        raise ValueError(
            f"Depth image must be float16 or float32, but got {depth.dtype}"
        )
    if depth.max() > MAX_DEPTH:
        raise ValueError(
            "Depth image should be in meters. "
            f"You are attempting to log depth values > {MAX_DEPTH}. "
            "The values you are passing in are likely in millimeters."
        )
    _log_camera_data(
        DataType.DEPTH_IMAGES,
        name,
        depth,
        extrinsics,
        intrinsics,
        robot_name,
        instance,
        timestamp,
    )


def log_point_cloud(
    name: str,
    points: np.ndarray,
    rgb_points: Optional[np.ndarray] = None,
    extrinsics: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log point cloud data from a camera.

    Args:
        name: Unique identifier for the point cloud
        points: Point cloud as numpy array (Nx3, dtype=float32, in meters)
        rgb_points: Optional RGB values for each point (Nx3, dtype=uint8)
        extrinsics: Optional extrinsics matrix (4x4)
        intrinsics: Optional intrinsics matrix (3x3)
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If point cloud format is invalid
    """
    warn(
        "Point cloud logging is experimental and may change in future releases.",
        ExperimentalPointCloudWarning,
    )
    timestamp = timestamp or time.time()
    if not isinstance(points, np.ndarray):
        raise ValueError("Point cloud must be a numpy array")
    if points.dtype != np.float16:
        raise ValueError("Point cloud must be float16")
    if points.shape[1] != 3:
        raise ValueError("Point cloud must have 3 columns")
    if points.shape[0] > 307200:
        raise ValueError("Point cloud must have at most 307200 points")

    if rgb_points is not None:
        if not isinstance(rgb_points, np.ndarray):
            raise ValueError("RGB point cloud must be a numpy array")
        if rgb_points.dtype != np.uint8:
            raise ValueError("RGB point cloud must be uint8")
        if rgb_points.shape[0] != points.shape[0]:
            raise ValueError(
                "RGB point cloud must have the same number of points as the point cloud"
            )
        if rgb_points.shape[1] != 3:
            raise ValueError("RGB point cloud must have 3 columns")

    extrinsics, intrinsics = _validate_extrinsics_intrinsics(extrinsics, intrinsics)
    robot = _get_robot(robot_name, instance)
    str_id = f"point_cloud_{name}"
    stream = robot.get_data_stream(str_id)
    if stream is None:
        stream = JsonDataStream(f"point_clouds/{name}.json")
        robot.add_data_stream(str_id, stream)
    assert isinstance(
        stream, JsonDataStream
    ), "Expected stream to be instance of JSONDataStream"
    start_stream(robot, stream)
    point_data = PointCloudData(
        timestamp=timestamp,
        points=points,
        rgb_points=rgb_points,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
    )
    stream.log(point_data)
    if robot.id is None:
        raise RobotError("Robot not initialized. Call init() first.")

    json_data = point_data.model_dump(mode="json")
    src = (
        StreamManagerOrchestrator()
        .get_provider_manager(robot.id, robot.instance)
        .get_json_source(name, DataType.POINT_CLOUDS, sensor_key=str_id)
    )
    src.publish(json_data)
