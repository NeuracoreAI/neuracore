import base64
import hashlib
import json
import time
from typing import Any, Optional

import numpy as np

from neuracore.api.core import _get_robot
from neuracore.core.robot import Robot
from neuracore.core.streaming.client_stream.client_stream_manager import (
    get_robot_streaming_manager,
)

from ..core.nc_types import (
    CameraData,
    CustomData,
    EndEffectorData,
    JointData,
    LanguageData,
    PointCloudData,
    PoseData,
)
from ..core.streaming.data_stream import (
    DataStream,
    DepthDataStream,
    JsonDataStream,
    RGBDataStream,
)
from ..core.utils.depth_utils import MAX_DEPTH


def _create_group_id_from_dict(joint_names: dict[str, float]) -> str:
    joint_names = list(joint_names.keys())
    joint_names.sort()
    return (
        base64.urlsafe_b64encode(hashlib.md5("".join(joint_names).encode()).digest())
        .decode()
        .rstrip("=")
    )


def start_stream(robot: Robot, data_stream: DataStream):
    current_recording = robot.get_current_recording_id()
    if current_recording is not None and not data_stream.is_recording():
        data_stream.start_recording(current_recording)


def _log_joint_data(
    data_type: str,
    joint_data: dict[str, float],
    additional_urdf_data: Optional[dict[str, float]] = None,
    robot_name: Optional[str] = None,
    instance: Optional[int] = 0,
    timestamp: Optional[float] = None,
) -> None:
    """
    Log joint positions for a robot.

    Args:
        joint_data: Dictionary mapping joint names to joint dat
        additional_urdf_data: Dictionary mapping joint names to
            joint data. These wont ever be included for
            training, and instead used for visualization purposes
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
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

    joint_stream.log(data=data)
    get_robot_streaming_manager(robot.id, robot.instance).get_json_source(
        data_type, "joints", sensor_key=joint_str_id
    ).publish(data.model_dump(mode="json"))


def _validate_extrinsics_intrinsics(
    extrinsics: Optional[np.ndarray], intrinsics: Optional[np.ndarray]
) -> tuple[Optional[list[list[float]]], Optional[list[list[float]]]]:
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
    camera_type: str,
    camera_id: str,
    image: np.ndarray,
    extrinsics: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    robot_name: Optional[str] = None,
    instance: Optional[int] = 0,
    timestamp: Optional[float] = None,
) -> None:
    """
    Log camera data

    Args:
        camera_type: Type of camera (e.g. "rgb", "depth")
        camera_id: Unique identifier for the camera
        image: RGB image as numpy array (HxWx3, dtype=uint8 or float32)
        extrinsics: Optional extrinsics matrix (4x4)
        intrinsics: Optional intrinsics matrix (3x3)
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If image format is invalid
    """
    timestamp = timestamp or time.time()
    extrinsics, intrinsics = _validate_extrinsics_intrinsics(extrinsics, intrinsics)
    robot = _get_robot(robot_name, instance)
    camera_id = f"{camera_type}_{camera_id}"

    stream = robot.get_data_stream(camera_id)
    if stream is None:
        if camera_type == "rgb":
            stream = RGBDataStream(camera_id, image.shape[1], image.shape[0])
        elif camera_type == "depth":
            stream = DepthDataStream(camera_id, image.shape[1], image.shape[0])
        else:
            raise ValueError(f"Invalid camera type: {camera_type}")
        robot.add_data_stream(camera_id, stream)

    start_stream(robot, stream)

    if stream.width != image.shape[1] or stream.height != image.shape[0]:
        raise ValueError(
            f"Camera image dimensions {image.shape[1]}x{image.shape[0]} do not match "
            f"stream dimensions {stream.width}x{stream.height}"
        )
    stream.log(
        image,
        CameraData(timestamp=timestamp, extrinsics=extrinsics, intrinsics=intrinsics),
    )
    get_robot_streaming_manager(robot.id, robot.instance).get_video_source(
        camera_id, camera_type
    ).add_frame(image)


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
    instance: Optional[int] = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Useful for simulated data, or you are relying on ROS to sync the data"""
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
    instance: Optional[int] = 0,
    timestamp: Optional[float] = None,
) -> None:
    """Log arbitrary data for a robot.

    Args:
        name: Name of the data stream
        data: Data to log (numpy array of arbitrary shape and dtype)
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp
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
    stream.log(CustomData(timestamp=timestamp, data=data))


def log_joint_positions(
    positions: dict[str, float],
    additional_urdf_positions: Optional[dict[str, float]] = None,
    robot_name: Optional[str] = None,
    instance: Optional[int] = 0,
    timestamp: Optional[float] = None,
) -> None:
    """
    Log joint positions for a robot.

    Args:
        positions: Dictionary mapping joint names to positions (in radians)
        additional_urdf_positions: Dictionary mapping joint names to
            positions (in radians). These wont ever be included for
            training, and instead used for visualization purposes
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
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
    instance: Optional[int] = 0,
    timestamp: Optional[float] = None,
) -> None:
    """
    Log joint target positions for a robot.

    Args:
        target_positions: Dictionary mapping joint names to positions (in radians)
        additional_urdf_positions: Dictionary mapping joint names to
            positions (in radians). These wont ever be included for
            training, and instead used for visualization purposes
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
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
    instance: Optional[int] = 0,
    timestamp: Optional[float] = None,
) -> None:
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
    instance: Optional[int] = 0,
    timestamp: Optional[float] = None,
) -> None:
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
    instance: Optional[int] = 0,
    timestamp: Optional[float] = None,
) -> None:
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

    stream.log(PoseData(timestamp=timestamp, pose=poses))


def log_gripper_data(
    open_amounts: dict[str, float],
    robot_name: Optional[str] = None,
    instance: Optional[int] = 0,
    timestamp: Optional[float] = None,
) -> None:
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

    stream.log(EndEffectorData(timestamp=timestamp, open_amounts=open_amounts))


def log_language(
    language: str,
    robot_name: Optional[str] = None,
    instance: Optional[int] = 0,
    timestamp: Optional[float] = None,
) -> None:
    """
    Log language for a robot.

    Args:
        language: A language string associated with this timestep
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
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
    stream.log(LanguageData(timestamp=timestamp, text=language))


def log_rgb(
    camera_id: str,
    image: np.ndarray,
    extrinsics: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    robot_name: Optional[str] = None,
    instance: Optional[int] = 0,
    timestamp: Optional[float] = None,
) -> None:
    """
    Log RGB image from a camera.

    Args:
        camera_id: Unique identifier for the camera
        image: RGB image as numpy array (HxWx3, dtype=uint8 or float32)
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
        raise ValueError("Image must be uint8 wth range 0-255")
    _log_camera_data(
        "rgb", camera_id, image, extrinsics, intrinsics, robot_name, instance, timestamp
    )


def log_depth(
    camera_id: str,
    depth: np.ndarray,
    extrinsics: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    robot_name: Optional[str] = None,
    instance: Optional[int] = 0,
    timestamp: Optional[float] = None,
) -> None:
    """
    Log depth image from a camera.

    Args:
        camera_id: Unique identifier for the camera
        depth: Depth image as numpy array (HxW, dtype=float32, in meters)
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
        "depth",
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
    instance: Optional[int] = 0,
    timestamp: Optional[float] = None,
) -> None:
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

    start_stream(robot, stream)

    stream.log(
        PointCloudData(
            timestamp=timestamp,
            points=points.tolist(),
            rgb_points=rgb_points,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )
    )
