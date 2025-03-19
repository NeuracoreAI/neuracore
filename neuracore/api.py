from typing import Optional

import numpy as np

from .core.auth import login as _login
from .core.auth import logout as _logout
from .core.dataset import Dataset
from .core.endpoint import EndpointPolicy
from .core.endpoint import connect_endpoint as _connect_endpoint
from .core.endpoint import connect_local_endpoint as _connect_local_endpoint
from .core.exceptions import RobotError
from .core.robot import Robot, get_robot
from .core.robot import init as _init_robot
from .core.streaming.data_stream import (
    MAX_DEPTH,
    ActionDataStream,
    DataStream,
    DepthDataStream,
    JointDataStream,
    RGBDataStream,
)

# Global active robot ID - allows us to avoid passing robot_name to every call
_active_robot: Optional[Robot] = None
_active_dataset_id: Optional[str] = None
_active_recording_id: Optional[str] = None
_data_streams: dict[str, DataStream] = {}


def login(api_key: Optional[str] = None) -> None:
    """
    Authenticate with NeuraCore server.

    Args:
        api_key: Optional API key. If not provided, will look for NEURACORE_API_KEY
                environment variable or previously saved configuration.

    Raises:
        AuthenticationError: If authentication fails
    """
    _login(api_key)


def logout() -> None:
    """Clear authentication state."""
    _logout()


def connect_robot(
    robot_name: str,
    urdf_path: Optional[str] = None,
    mjcf_path: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    Initialize a robot connection.

    Args:
        robot_name: Unique identifier for the robot
        urdf_path: Optional path to robot's URDF file
        mjcf_path: Optional path to robot's MJCF file
        overwrite: Whether to overwrite an existing robot with the same name
    """
    global _active_robot
    _active_robot = _init_robot(robot_name, urdf_path, mjcf_path, overwrite)


def _get_robot(robot_name: str) -> Robot:
    """Get a robot by name."""
    robot: Robot = _active_robot
    if robot_name is None:
        if _active_robot is None:
            raise RobotError(
                "No active robot. Call init() first or provide robot_name."
            )
    else:
        robot = get_robot(robot_name)
    return robot


def log_joints(
    positions: dict[str, float],
    robot_name: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> None:
    """
    Log joint positions for a robot.

    Args:
        positions: Dictionary mapping joint names to positions (in radians)
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
    """
    if not isinstance(positions, dict):
        raise ValueError("Joint positions must be a dictionary of floats")
    for key, value in positions.items():
        if not isinstance(value, float):
            raise ValueError(f"Joint positions must be floats. {key} is not a float.")
    robot = _get_robot(robot_name)
    str_id = f"{robot.name}_joints"
    stream = _data_streams.get(str_id)
    if stream is None:
        stream = JointDataStream()
        _data_streams[str_id] = stream
        if _active_recording_id is not None:
            stream.start_recording(_active_recording_id)
    stream.log(positions, timestamp)


def log_action(
    action: dict[str, float],
    robot_name: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> None:
    """
    Log action for a robot.

    Args:
        action: Dictionary mapping joint names to positions (in radians)
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
    """
    if not isinstance(action, dict):
        raise ValueError("Actions must be a dictionary of floats")
    for key, value in action.items():
        if not isinstance(value, float):
            raise ValueError(f"Actions must be floats. {key} is not a float.")
    robot = _get_robot(robot_name)
    str_id = f"{robot.name}_action"
    stream = _data_streams.get(str_id)
    if stream is None:
        stream = ActionDataStream()
        _data_streams[str_id] = stream
        if _active_recording_id is not None:
            stream.start_recording(_active_recording_id)
    stream.log(action, timestamp)


def log_rgb(
    camera_id: str,
    image: np.ndarray,
    robot_name: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> None:
    """
    Log RGB image from a camera.

    Args:
        camera_id: Unique identifier for the camera
        image: RGB image as numpy array (HxWx3, dtype=uint8 or float32)
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If image format is invalid
    """
    # Validate image is numpy array of type uint8
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")
    if image.dtype != np.uint8:
        raise ValueError("Image must be uint8 wth range 0-255")
    robot = _get_robot(robot_name)
    str_id = f"{robot.name}_rgb_{camera_id}"
    stream = _data_streams.get(str_id)
    if stream is None:
        stream = RGBDataStream(camera_id, image.shape[1], image.shape[0])
        _data_streams[str_id] = stream
        if _active_recording_id is not None:
            stream.start_recording(_active_recording_id)
    if stream.width != image.shape[1] or stream.height != image.shape[0]:
        raise ValueError(
            f"RGB image dimensions {image.shape[1]}x{image.shape[0]} do not match "
            f"stream dimensions {stream.width}x{stream.height}"
        )
    stream.log(image, timestamp)


def log_depth(
    camera_id: str,
    depth: np.ndarray,
    robot_name: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> None:
    """
    Log depth image from a camera.

    Args:
        camera_id: Unique identifier for the camera
        depth: Depth image as numpy array (HxW, dtype=float32, in meters)
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
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
    robot = _get_robot(robot_name)
    str_id = f"{robot.name}_depth_{camera_id}"
    stream = _data_streams.get(str_id)
    if stream is None:
        stream = DepthDataStream(camera_id, depth.shape[1], depth.shape[0])
        _data_streams[str_id] = stream
        if _active_recording_id is not None:
            stream.start_recording(_active_recording_id)
    if stream.width != depth.shape[1] or stream.height != depth.shape[0]:
        raise ValueError(
            f"Depth image dimensions {depth.shape[1]}x{depth.shape[0]} do not match "
            f"stream dimensions {stream.width}x{stream.height}"
        )
    stream.log(depth, timestamp)


def start_recording(robot_name: Optional[str] = None) -> None:
    """
    Start recording data for a specific robot.

    Args:
        robot_name: Optional robot ID. If not provided, uses the last initialized robot

    Raises:
        RobotError: If no robot is active and no robot_name provided
    """
    global _active_recording_id
    if _active_recording_id is not None:
        raise RobotError("Recording already in progress. Call stop_recording() first.")
    robot = _get_robot(robot_name)
    if _active_dataset_id is None:
        raise RobotError("No active dataset. Call create_dataset() first.")
    _active_recording_id = robot.start_recording(_active_dataset_id)
    for stream in _data_streams.values():
        stream.start_recording(_active_recording_id)


def stop_recording(robot_name: Optional[str] = None) -> None:
    """
    Stop recording data for a specific robot.

    Args:
        robot_name: Optional robot ID. If not provided, uses the last initialized robot

    Raises:
        RobotError: If no robot is active and no robot_name provided
    """
    global _active_recording_id
    robot = _get_robot(robot_name)
    if _active_recording_id is None:
        raise RobotError("No active recording. Call start_recording() first.")
    robot.stop_recording(_active_recording_id)
    for stream in _data_streams.values():
        stream.stop_recording()
    _active_recording_id = None


def get_dataset(name: str) -> Dataset:
    """Get a dataset by name.

    Args:
        name: Dataset name

    """
    global _active_dataset_id
    _active_dataset = Dataset.get(name)
    _active_dataset_id = _active_dataset.id
    return _active_dataset


def create_dataset(
    name: str, description: Optional[str] = None, tags: Optional[list[str]] = None
) -> Dataset:
    """
    Create a new dataset for robot demonstrations.

    Args:
        name: Dataset name
        description: Optional description
        tags: Optional list of tags

    Raises:
        DatasetError: If dataset creation fails
    """
    global _active_dataset_id
    _active_dataset = Dataset.create(name, description, tags)
    _active_dataset_id = _active_dataset.id
    return _active_dataset


def connect_endpoint(name: str) -> EndpointPolicy:
    """
    Connect to a deployed model endpoint.

    Args:
        name: Name of the deployed endpoint

    Returns:
        EndpointPolicy: Policy object that can be used for predictions

    Raises:
        EndpointError: If endpoint connection fails
    """
    return _connect_endpoint(name)


def connect_local_endpoint(
    path_to_model: Optional[str] = None, train_run_name: Optional[str] = None
) -> EndpointPolicy:
    """
    Connect to a local model endpoint.

    Can supply either path_to_model or train_run_name, but not both.

    Args:
        path_to_model: Path to the local .mar model
        train_run_name: Optional train run name

    Returns:
        EndpointPolicy: Policy object that can be used for predictions

    Raises:
        EndpointError: If endpoint connection fails
    """
    return _connect_local_endpoint(path_to_model, train_run_name)
