"""Robot data logging utilities.

This module provides functions for logging various types of robot sensor data
including joint positions, camera images, point clouds, and custom data streams.
All logging functions support optional robot identification and timestamping.
"""

import base64
import hashlib
import json
import time
from typing import Any, Dict, List, Optional

import numpy as np

from neuracore.api.core import _get_robot
from neuracore.core.exceptions import RobotError
from neuracore.core.nc_types import (
    CameraData,
    CustomData,
    EndEffectorData,
    JointData,
    LanguageData,
    PointCloudData,
    PoseData,
    TrackKind,
)
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


def _create_group_id_from_dict(joint_names: Dict[str, Any]) -> str:
    """Create a unique group ID from joint names dictionary.

    Args:
        joint_names: Dictionary mapping joint names to values

    Returns:
        str: Base64 encoded hash of sorted joint names
    """
    joint_names_list: List[str] = list(joint_names.keys())
    joint_names_list.sort()
    return (
        base64.urlsafe_b64encode(
            hashlib.md5("".join(joint_names_list).encode()).digest()
        )
        .decode()
        .rstrip("=")
    )


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
    data_type: str,
    joint_data: dict[str, float],
    additional_urdf_data: Optional[dict[str, float]] = None,
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log joint data for a robot.

    Args:
        data_type: Type of joint data (e.g., "joint_positions", "joint_velocities")
        joint_data: Dictionary mapping joint names to joint data values
        additional_urdf_data: Dictionary mapping joint names to
            joint data. These won't ever be included for
            training, and instead used for visualization purposes
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
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
    if additional_urdf_data:
        if not isinstance(additional_urdf_data, dict):
            raise ValueError("Additional visual data must be a dictionary of floats")
        for key, value in additional_urdf_data.items():
            if not isinstance(value, float):
                raise ValueError(
                    f"Additional visual data must be floats. {key} is not a float."
                )

    robot = _get_robot(robot_name, instance)
    joint_group_id = _create_group_id_from_dict(joint_data)
    joint_str_id = f"{data_type}_{joint_group_id}"
    joint_stream = robot.get_data_stream(joint_str_id)
    if joint_stream is None:
        joint_stream = JsonDataStream(f"{data_type}/{joint_group_id}.json")
        robot.add_data_stream(joint_str_id, joint_stream)

    start_stream(robot, joint_stream)

    data = JointData(
        timestamp=timestamp,
        values=joint_data,
        additional_values=additional_urdf_data,
    )
    assert isinstance(
        joint_stream, JsonDataStream
    ), "Expected stream to be instance of JSONDataStream"
    joint_stream.log(data=data)
    if robot.id is None:
        raise RobotError("Robot not initialized. Call init() first.")
    StreamManagerOrchestrator().get_provider_manager(
        robot.id, robot.instance
    ).get_json_source(data_type, TrackKind.JOINTS, sensor_key=joint_str_id).publish(
        data.model_dump(mode="json")
    )


def _validate_extrinsics_intrinsics(
    extrinsics: Optional[np.ndarray], intrinsics: Optional[np.ndarray]
) -> tuple[Optional[list[list[float]]], Optional[list[list[float]]]]:
    """Validate and convert camera extrinsics and intrinsics matrices.

    Args:
        extrinsics: Optional extrinsics matrix as numpy array
        intrinsics: Optional intrinsics matrix as numpy array

    Returns:
        tuple: Converted extrinsics and intrinsics as lists of lists

    Raises:
        ValueError: If matrices have incorrect shapes
    """
    if extrinsics is not None:
        if not isinstance(extrinsics, np.ndarray) or extrinsics.shape != (4, 4):
            raise ValueError("Extrinsics must be a numpy array of shape (4, 4)")
        extrinsics = extrinsics.tolist()

    if intrinsics is not None:
        if not isinstance(intrinsics, np.ndarray) or intrinsics.shape != (3, 3):
            raise ValueError("Intrinsics must be a numpy array of shape (3, 3)")
        intrinsics = intrinsics.tolist()
    return extrinsics, intrinsics


def _log_camera_data(
    camera_type: TrackKind,
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
        camera_type: Type of camera (e.g. TrackKind.RGB or TrackKind.DEPTH)
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
    assert camera_type in (TrackKind.RGB, TrackKind.DEPTH), "Unsupported camera type"

    timestamp = timestamp or time.time()
    extrinsics, intrinsics = _validate_extrinsics_intrinsics(extrinsics, intrinsics)
    robot = _get_robot(robot_name, instance)
    full_cam_id = f"{camera_type.value}_{camera_id}"

    stream = robot.get_data_stream(full_cam_id)
    if stream is None:
        if camera_type == TrackKind.RGB:
            stream = RGBDataStream(full_cam_id, image.shape[1], image.shape[0])
        elif camera_type == TrackKind.DEPTH:
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


def log_synced_data(
    joint_positions: dict[str, float],
    joint_velocities: dict[str, float],
    joint_torques: dict[str, float],
    gripper_open_amounts: dict[str, float],
    joint_target_positions: dict[str, float],
    rgb_data: dict[str, np.ndarray],
    depth_data: dict[str, np.ndarray],
    point_cloud_data: dict[str, np.ndarray],
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log synchronized data from multiple sensors.

    Useful for simulated data, or when relying on ROS to sync the data.

    Args:
        joint_positions: Dictionary mapping joint names to positions
        joint_velocities: Dictionary mapping joint names to velocities
        joint_torques: Dictionary mapping joint names to torques
        gripper_open_amounts: Dictionary mapping gripper names to open amounts
        joint_target_positions: Dictionary mapping joint names to target positions
        rgb_data: Dictionary mapping camera IDs to RGB images
        depth_data: Dictionary mapping camera IDs to depth images
        point_cloud_data: Dictionary mapping camera IDs to point clouds
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp
    """
    timestamp = timestamp or time.time()
    log_joint_positions(
        joint_positions, robot_name=robot_name, instance=instance, timestamp=timestamp
    )
    log_joint_velocities(
        joint_velocities, robot_name=robot_name, instance=instance, timestamp=timestamp
    )
    log_joint_torques(
        joint_torques, robot_name=robot_name, instance=instance, timestamp=timestamp
    )
    log_joint_target_positions(
        joint_target_positions,
        robot_name=robot_name,
        instance=instance,
        timestamp=timestamp,
    )
    log_gripper_data(
        gripper_open_amounts,
        robot_name=robot_name,
        instance=instance,
        timestamp=timestamp,
    )
    for camera_id, image in rgb_data.items():
        log_rgb(
            camera_id,
            image,
            robot_name=robot_name,
            instance=instance,
            timestamp=timestamp,
        )
    for camera_id, depth in depth_data.items():
        log_depth(
            camera_id,
            depth,
            robot_name=robot_name,
            instance=instance,
            timestamp=timestamp,
        )
    for camera_id, point_cloud in point_cloud_data.items():
        log_point_cloud(
            camera_id,
            point_cloud,
            robot_name=robot_name,
            instance=instance,
            timestamp=timestamp,
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
    ).get_json_source(name, TrackKind.CUSTOM, sensor_key=str_id).publish(
        custom_data.model_dump(mode="json")
    )


def log_joint_positions(
    positions: dict[str, float],
    additional_urdf_positions: Optional[dict[str, float]] = None,
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log joint positions for a robot.

    Args:
        positions: Dictionary mapping joint names to positions (in radians)
        additional_urdf_positions: Dictionary mapping joint names to
            positions (in radians). These won't ever be included for
            training, and instead used for visualization purposes
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If positions is not a dictionary of floats
    """
    _log_joint_data(
        "joint_positions",
        positions,
        additional_urdf_positions,
        robot_name,
        instance,
        timestamp,
    )


def log_joint_target_positions(
    target_positions: dict[str, float],
    additional_urdf_positions: Optional[dict[str, float]] = None,
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log joint target positions for a robot.

    Args:
        target_positions: Dictionary mapping joint names to
            target positions (in radians)
        additional_urdf_positions: Dictionary mapping joint names to
            positions (in radians). These won't ever be included for
            training, and instead used for visualization purposes
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If target_positions is not a dictionary of floats
    """
    _log_joint_data(
        "joint_target_positions",
        target_positions,
        additional_urdf_positions,
        robot_name,
        instance,
        timestamp,
    )


def log_joint_velocities(
    velocities: dict[str, float],
    additional_urdf_velocities: Optional[dict[str, float]] = None,
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log joint velocities for a robot.

    Args:
        velocities: Dictionary mapping joint names to velocities (in radians/second)
        additional_urdf_velocities: Dictionary mapping joint names to
            velocities (in radians/second). These won't ever be included for
            training, and instead used for visualization purposes
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If velocities is not a dictionary of floats
    """
    _log_joint_data(
        "joint_velocities",
        velocities,
        additional_urdf_velocities,
        robot_name,
        instance,
        timestamp,
    )


def log_joint_torques(
    torques: dict[str, float],
    additional_urdf_torques: Optional[dict[str, float]] = None,
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log joint torques for a robot.

    Args:
        torques: Dictionary mapping joint names to torques (in Newton-meters)
        additional_urdf_torques: Dictionary mapping joint names to
            torques (in Newton-meters). These won't ever be included for
            training, and instead used for visualization purposes
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If torques is not a dictionary of floats
    """
    _log_joint_data(
        "joint_torques",
        torques,
        additional_urdf_torques,
        robot_name,
        instance,
        timestamp,
    )


def log_pose_data(
    poses: dict[str, list[float]],
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log pose data for a robot.

    Args:
        poses: Dictionary mapping pose names to pose data
            (7-element lists: [x, y, z, qx, qy, qz, qw])
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If poses is not a dictionary of 7-element lists
    """
    timestamp = timestamp or time.time()
    if not isinstance(poses, dict):
        raise ValueError("Poses must be a dictionary of lists")
    for key, value in poses.items():
        if not isinstance(value, list):
            raise ValueError(f"Poses must be lists. {key} is not a list.")
        if len(value) != 7:
            raise ValueError(f"Poses must be lists of length 7. {key} is not length 7.")
    robot = _get_robot(robot_name, instance)
    group_id = _create_group_id_from_dict(poses)
    str_id = f"{group_id}_pose_data"
    stream = robot.get_data_stream(str_id)
    if stream is None:
        stream = JsonDataStream(f"poses/{group_id}.json")
        robot.add_data_stream(str_id, stream)

    start_stream(robot, stream)
    assert isinstance(
        stream, JsonDataStream
    ), "Expected stream to be instance of JSONDataStream"

    pose_data = PoseData(timestamp=timestamp, pose=poses)
    stream.log(pose_data)

    if robot.id is None:
        raise RobotError("Robot not initialized. Call init() first.")

    StreamManagerOrchestrator().get_provider_manager(
        robot.id, robot.instance
    ).get_json_source(str_id, TrackKind.POSE, sensor_key=str_id).publish(
        pose_data.model_dump(mode="json")
    )


def log_gripper_data(
    open_amounts: dict[str, float],
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log gripper data for a robot.

    Args:
        open_amounts: Dictionary mapping gripper names to
            open amounts (0.0 = closed, 1.0 = fully open)
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If open_amounts is not a dictionary of floats
    """
    timestamp = timestamp or time.time()
    if not isinstance(open_amounts, dict):
        raise ValueError("Gripper open amounts must be a dictionary of floats")
    for key, value in open_amounts.items():
        if not isinstance(value, float):
            raise ValueError(
                f"Gripper open amounts must be floats. {key} is not a float."
            )
    robot = _get_robot(robot_name, instance)
    group_id = _create_group_id_from_dict(open_amounts)
    str_id = f"{group_id}_gripper_data"
    stream = robot.get_data_stream(str_id)
    if stream is None:
        stream = JsonDataStream(f"gripper_open_amounts/{group_id}.json")
        robot.add_data_stream(str_id, stream)

    start_stream(robot, stream)

    assert isinstance(
        stream, JsonDataStream
    ), "Expected stream to be instance of EndEffectorData"
    end_effector_data = EndEffectorData(timestamp=timestamp, open_amounts=open_amounts)
    stream.log(end_effector_data)

    if robot.id is None:
        raise RobotError("Robot not initialized. Call init() first.")

    StreamManagerOrchestrator().get_provider_manager(
        robot.id, robot.instance
    ).get_json_source(str_id, TrackKind.GRIPPER, str_id).publish(
        end_effector_data.model_dump(mode="json")
    )


def log_language(
    language: str,
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log language annotation for a robot.

    Args:
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
    str_id = "language"
    stream = robot.get_data_stream(str_id)
    if stream is None:
        stream = JsonDataStream("language_annotations.json")
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
    ).get_json_source(str_id, TrackKind.LANGUAGE, sensor_key=str_id).publish(
        data.model_dump(mode="json")
    )


def log_rgb(
    camera_id: str,
    image: np.ndarray,
    extrinsics: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log RGB image from a camera.

    Args:
        camera_id: Unique identifier for the camera
        image: RGB image as numpy array (HxWx3, dtype=uint8)
        extrinsics: Optional extrinsics matrix (4x4)
        intrinsics: Optional intrinsics matrix (3x3)
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If image format is invalid
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Image image must be a numpy array")
    if image.dtype != np.uint8:
        raise ValueError("Image must be uint8 with range 0-255")
    _log_camera_data(
        TrackKind.RGB,
        camera_id,
        image,
        extrinsics,
        intrinsics,
        robot_name,
        instance,
        timestamp,
    )


def log_depth(
    camera_id: str,
    depth: np.ndarray,
    extrinsics: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    robot_name: Optional[str] = None,
    instance: int = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log depth image from a camera.

    Args:
        camera_id: Unique identifier for the camera
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
        TrackKind.DEPTH,
        camera_id,
        depth,
        extrinsics,
        intrinsics,
        robot_name,
        instance,
        timestamp,
    )


def log_point_cloud(
    camera_id: str,
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
        camera_id: Unique identifier for the camera
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
    timestamp = timestamp or time.time()
    if not isinstance(points, np.ndarray):
        raise ValueError("Point cloud must be a numpy array")
    if points.dtype != np.float32:
        raise ValueError("Point cloud must be float32")
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
        rgb_points = rgb_points.tolist()

    extrinsics, intrinsics = _validate_extrinsics_intrinsics(extrinsics, intrinsics)
    robot = _get_robot(robot_name, instance)
    str_id = f"point_cloud_{camera_id}"
    stream = robot.get_data_stream(str_id)
    if stream is None:
        stream = JsonDataStream(f"point_clouds/{camera_id}.json")
        robot.add_data_stream(str_id, stream)
    assert isinstance(
        stream, JsonDataStream
    ), "Expected stream to be instance of JSONDataStream"
    start_stream(robot, stream)

    point_data = PointCloudData(
        timestamp=timestamp,
        points=points.tolist(),
        rgb_points=rgb_points,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
    )

    stream.log(point_data)

    if robot.id is None:
        raise RobotError("Robot not initialized. Call init() first.")

    StreamManagerOrchestrator().get_provider_manager(
        robot.id, robot.instance
    ).get_json_source(camera_id, TrackKind.POINT_CLOUD, sensor_key=str_id).publish(
        point_data.model_dump(mode="json")
    )
