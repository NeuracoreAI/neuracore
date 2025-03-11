import atexit

import numpy as np

from .auth import login as _login
from .auth import logout as _logout
from .dataset import Dataset
from .endpoint import EndpointPolicy
from .endpoint import connect_endpoint as _connect_endpoint
from .endpoint import connect_local_endpoint as _connect_local_endpoint
from .exceptions import RobotError
from .robot import Robot, get_robot
from .robot import init as _init_robot
from .streaming import MAX_DEPTH
from .streaming import log_action as _log_action
from .streaming import log_depth as _log_depth
from .streaming import log_joints as _log_joints
from .streaming import log_rgb as _log_rgb
from .streaming import stop_all_streams as _stop_all_streams
from .streaming import stop_streaming as _stop_streaming
from .streaming import wait_until_stream_empty

# Global active robot ID - allows us to avoid passing robot_name to every call
_active_robot: Robot | None = None
_active_dataset_id: str | None = None
_active_recording_id: str | None = None


def login(api_key: str | None = None) -> None:
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
    urdf_path: str | None = None,
    mjcf_path: str | None = None,
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


def log_joints(positions: dict[str, float], robot_name: str | None = None) -> None:
    """
    Log joint positions for a robot.

    Args:
        positions: Dictionary mapping joint names to positions (in radians)
        robot_name: Optional robot ID. If not provided, uses the last initialized robot

    Raises:
        RobotError: If no robot is active and no robot_name provided
        StreamingError: If logging fails
    """
    if not isinstance(positions, dict):
        raise ValueError("Joint positions must be a dictionary of floats")
    for key, value in positions.items():
        if not isinstance(value, float):
            raise ValueError(f"Joint positions must be floats. {key} is not a float.")
    _log_joints(_get_robot(robot_name), positions)


def log_action(action: dict[str, float], robot_name: str | None = None) -> None:
    """
    Log action for a robot.

    Args:
        action: Dictionary mapping joint names to positions (in radians)
        robot_name: Optional robot ID. If not provided, uses the last initialized robot

    Raises:
        RobotError: If no robot is active and no robot_name provided
        StreamingError: If logging fails
    """
    if not isinstance(action, dict):
        raise ValueError("Actions must be a dictionary of floats")
    for key, value in action.items():
        if not isinstance(value, float):
            raise ValueError(f"Actions must be floats. {key} is not a float.")
    _log_action(_get_robot(robot_name), action)


def log_rgb(camera_id: str, image: np.ndarray, robot_name: str | None = None) -> None:
    """
    Log RGB image from a camera.

    Args:
        camera_id: Unique identifier for the camera
        image: RGB image as numpy array (HxWx3, dtype=uint8 or float32)
        robot_name: Optional robot ID. If not provided, uses the last initialized robot

    Raises:
        RobotError: If no robot is active and no robot_name provided
        StreamingError: If logging fails
        ValueError: If image format is invalid
    """
    # Validate image is numpy array of type uint8
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")
    if image.dtype != np.uint8:
        raise ValueError("Image must be uint8 wth range 0-255")
    _log_rgb(_get_robot(robot_name), camera_id, image)


def log_depth(camera_id: str, depth: np.ndarray, robot_name: str | None = None) -> None:
    """
    Log depth image from a camera.

    Args:
        camera_id: Unique identifier for the camera
        depth: Depth image as numpy array (HxW, dtype=float32, in meters)
        robot_name: Optional robot ID. If not provided, uses the last initialized robot

    Raises:
        RobotError: If no robot is active and no robot_name provided
        StreamingError: If logging fails
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
    _log_depth(_get_robot(robot_name), camera_id, depth)


def start_recording(robot_name: str | None = None) -> None:
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


def stop_recording(robot_name: str | None = None) -> None:
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
    wait_until_stream_empty(robot)
    robot.stop_recording(_active_recording_id)
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
    name: str, description: str | None = None, tags: list[str] | None = None
) -> None:
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
    path_to_model: str | None = None, train_run_name: str | None = None
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


def stop(robot_name: str | None = None) -> None:
    """
    Stop streaming for a specific robot.

    Args:
        robot_name: Optional robot ID. If not provided, uses the last initialized robot

    Raises:
        RobotError: If no robot is active and no robot_name provided
    """
    global _active_robot
    _stop_streaming(_get_robot(robot_name))
    if robot_name == _active_robot.name:
        _active_robot = None


def stop_all() -> None:
    """Stop all active data streams."""
    global _active_robot
    _stop_all_streams()
    _active_robot = None


atexit.register(stop_all)
