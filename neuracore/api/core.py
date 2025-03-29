from threading import Thread
from typing import Optional

from ..core.auth import get_auth
from ..core.exceptions import RobotError
from ..core.robot import Robot, get_robot
from ..core.robot import init as _init_robot
from .globals import GlobalSingleton


def _get_robot(robot_name: str) -> Robot:
    """Get a robot by name."""
    robot: Robot = GlobalSingleton()._active_robot
    if robot_name is None:
        if GlobalSingleton()._active_robot is None:
            raise RobotError(
                "No active robot. Call init() first or provide robot_name."
            )
    else:
        robot = get_robot(robot_name)
    return robot


def _stop_recording_wait_for_threads(
    robot: Robot, recording_id: str, threads: list[Thread]
) -> None:
    for thread in threads:
        thread.join()
    robot.stop_recording(recording_id)


def validate_version() -> None:
    """
    Validate the NeuraCore version.

    Raises:
        RobotError: If the NeuraCore version is not compatible
    """
    if not GlobalSingleton()._has_validated_version:
        get_auth().validate_version()
        GlobalSingleton()._has_validated_version = True


def login(api_key: Optional[str] = None) -> None:
    """
    Authenticate with NeuraCore server.

    Args:
        api_key: Optional API key. If not provided, will look for NEURACORE_API_KEY
                environment variable or previously saved configuration.

    Raises:
        AuthenticationError: If authentication fails
    """
    get_auth().login(api_key)


def logout() -> None:
    """Clear authentication state."""
    get_auth().logout()
    GlobalSingleton()._active_robot = None
    GlobalSingleton()._active_recording_ids = {}
    GlobalSingleton()._active_dataset_id = None
    GlobalSingleton()._has_validated_version = False


def connect_robot(
    robot_name: str,
    urdf_path: Optional[str] = None,
    mjcf_path: Optional[str] = None,
    overwrite: bool = False,
    shared: bool = False,
) -> Robot:
    """
    Initialize a robot connection.

    Args:
        robot_name: Unique identifier for the robot
        urdf_path: Optional path to robot's URDF file
        mjcf_path: Optional path to robot's MJCF file
        overwrite: Whether to overwrite an existing robot with the same name
        shared: Whether the robot is shared
    """
    validate_version()
    robot = _init_robot(robot_name, urdf_path, mjcf_path, overwrite, shared)
    GlobalSingleton()._active_robot = robot
    return robot


def start_recording(robot_name: Optional[str] = None) -> None:
    """
    Start recording data for a specific robot.

    Args:
        robot_name: Optional robot ID. If not provided, uses the last initialized robot

    Raises:
        RobotError: If no robot is active and no robot_name provided
    """
    robot = _get_robot(robot_name)
    if robot.name in GlobalSingleton()._active_recording_ids:
        raise RobotError("Recording already in progress. Call stop_recording() first.")
    if GlobalSingleton()._active_dataset_id is None:
        raise RobotError("No active dataset. Call create_dataset() first.")
    new_active_recording_id = robot.start_recording(
        GlobalSingleton()._active_dataset_id
    )
    for sname, stream in GlobalSingleton()._data_streams.items():
        if sname.startswith(robot.name):
            stream.start_recording(new_active_recording_id)
    GlobalSingleton()._active_recording_ids[robot.name] = new_active_recording_id


def stop_recording(robot_name: Optional[str] = None, wait: bool = False) -> None:
    """
    Stop recording data for a specific robot.

    Args:
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        wait: Whether to wait for the recording to finish

    Raises:
        RobotError: If no robot is active and no robot_name provided
    """
    robot = _get_robot(robot_name)
    if robot.name not in GlobalSingleton()._active_recording_ids:
        raise RobotError("No active recording. Call start_recording() first.")
    threads: Thread = []
    for sname, stream in GlobalSingleton()._data_streams.items():
        if sname.startswith(robot.name):
            threads.append(stream.stop_recording())
    stop_recording_thread = Thread(
        target=_stop_recording_wait_for_threads,
        args=(robot, GlobalSingleton()._active_recording_ids[robot.name], threads),
        daemon=False,
    )
    stop_recording_thread.start()
    GlobalSingleton()._active_recording_ids.pop(robot.name)
    if wait:
        stop_recording_thread.join()
