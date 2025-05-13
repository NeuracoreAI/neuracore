import logging
import time
from typing import Optional

from neuracore.core.streaming.client_stream.client_stream_manager import (
    get_robot_streaming_manager,
)
from neuracore.core.streaming.recording_state_manager import get_recording_state_manager
from neuracore.core.utils import backend_utils

from ..core.auth import get_auth
from ..core.exceptions import RobotError
from ..core.robot import Robot, get_robot
from ..core.robot import init as _init_robot
from .globals import GlobalSingleton

logger = logging.getLogger(__name__)


def _get_robot(robot_name: str, instance: int) -> Robot:
    """Get a robot by name and instance."""
    robot: Robot = GlobalSingleton()._active_robot
    if robot_name is None:
        if GlobalSingleton()._active_robot is None:
            raise RobotError(
                "No active robot. Call init() first or provide robot_name."
            )
    else:
        robot = get_robot(robot_name, instance)
    return robot


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
    instance: int = 0,
    urdf_path: Optional[str] = None,
    mjcf_path: Optional[str] = None,
    overwrite: bool = False,
    shared: bool = False,
) -> Robot:
    """
    Initialize a robot connection.

    Args:
        robot_name: Unique identifier for the robot
        instance: Instance number of the robot
        urdf_path: Optional path to robot's URDF file
        mjcf_path: Optional path to robot's MJCF file
        overwrite: Whether to overwrite an existing robot with the same name
        shared: Whether the robot is shared
    """
    validate_version()
    robot = _init_robot(robot_name, instance, urdf_path, mjcf_path, overwrite, shared)
    GlobalSingleton()._active_robot = robot
    # Initialize push update managers
    get_robot_streaming_manager(robot.id, robot.instance)
    get_recording_state_manager()
    return robot


def start_recording(
    robot_name: Optional[str] = None, instance: Optional[int] = 0
) -> None:
    """
    Start recording data for a specific robot.

    Args:
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot

    Raises:
        RobotError: If no robot is active and no robot_name provided
    """
    robot = _get_robot(robot_name, instance)
    if robot.is_recording():
        raise RobotError("Recording already in progress. Call stop_recording() first.")
    if GlobalSingleton()._active_dataset_id is None:
        raise RobotError("No active dataset. Call create_dataset() first.")
    robot.start_recording(GlobalSingleton()._active_dataset_id)


def stop_recording(
    robot_name: Optional[str] = None, instance: Optional[int] = 0, wait: bool = False
) -> None:
    """
    Stop recording data for a specific robot.

    Args:
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        wait: Whether to wait for the recording to finish

    Raises:
        RobotError: If no robot is active and no robot_name provided
    """
    robot = _get_robot(robot_name, instance)
    if not robot.is_recording():
        logger.warning("No active recordings to stop.")
        return
    recording_id = robot.get_current_recording_id()
    robot.stop_recording(recording_id)
    if wait:
        while backend_utils.get_num_active_streams(recording_id) > 0:
            time.sleep(2.0)


def stop_live_data(
    robot_name: Optional[str] = None, instance: Optional[int] = 0
) -> None:
    """
    Stop sharing live data for active monitoring from the neuracore platform.

    Args:
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
    """
    robot = _get_robot(robot_name, instance)
    get_robot_streaming_manager(robot.id, robot.instance).close()
