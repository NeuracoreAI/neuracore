"""High-level API for Neuracore robot management and data recording.

This module provides the main public interface for connecting to robots,
managing authentication, controlling data recording sessions, and handling
live data streaming. It maintains global state for active robots and
recording sessions.
"""

import logging
import time
from typing import Optional
from warnings import warn

from neuracore.core.config.config_manager import get_config_manager
from neuracore.core.organizations import list_my_orgs
from neuracore.core.streaming.p2p.provider.global_live_data_enabled import (
    global_provide_live_data_manager,
)
from neuracore.core.streaming.p2p.stream_manager_orchestrator import (
    StreamManagerOrchestrator,
)
from neuracore.core.streaming.recording_state_manager import get_recording_state_manager
from neuracore.core.utils import backend_utils

from ..core.auth import get_auth
from ..core.exceptions import RobotError
from ..core.robot import Robot, get_robot
from ..core.robot import init as _init_robot
from .globals import GlobalSingleton

logger = logging.getLogger(__name__)


def _get_robot(robot_name: Optional[str], instance: int) -> Robot:
    """Get a robot by name and instance.

    Retrieves either the active robot from global state or looks up a specific
    robot by name and instance. Falls back to the active robot if no name
    is provided.

    Args:
        robot_name: Name of the robot to retrieve.
        instance: Instance number of the robot.

    Returns:
        The requested robot instance.

    Raises:
        RobotError: If no active robot exists and no robot_name is provided.
    """
    if robot_name is None:
        robot = GlobalSingleton()._active_robot
        if robot is None:
            raise RobotError(
                "No active robot. Call init() first or provide robot_name."
            )
    else:
        robot = get_robot(robot_name, instance)
    return robot


def validate_version() -> None:
    """Validate the Neuracore version compatibility.

    Checks if the current Neuracore client version is compatible with the
    server. This validation is performed once per session and cached.

    Raises:
        RobotError: If the Neuracore version is not compatible with the server.
    """
    if not GlobalSingleton()._has_validated_version:
        get_auth().validate_version()
        GlobalSingleton()._has_validated_version = True


def login(api_key: Optional[str] = None) -> None:
    """Authenticate with the Neuracore server.

    Establishes authentication using an API key from the parameter, environment
    variable, or previously saved configuration. The authentication state is
    maintained for subsequent API calls.

    Args:
        api_key: API key for authentication. If not provided, will look for
            NEURACORE_API_KEY environment variable or previously saved configuration.

    Raises:
        AuthenticationError: If authentication fails due to invalid credentials
            or network issues.
        InputError: If there is an issue with the user's input.

    """
    get_auth().login(api_key)


def logout() -> None:
    """Clear authentication state and reset global session data.

    Logs out from the Neuracore server and resets all global state including
    active robots, recording IDs, dataset IDs, and version validation status.
    """
    get_auth().logout()
    GlobalSingleton()._active_robot = None
    GlobalSingleton()._active_dataset_id = None
    GlobalSingleton()._has_validated_version = False


def set_organization(id_or_name: str) -> None:
    """Set the current organization based upon it's name or or id.

    this value may be overridden by the `NEURACORE_ORG_ID` environment variable.

    Args:
        id_or_name: the uuid of the organization or its exact name

    Raises:
        AuthenticationError: If the user is not logged in
        OrganizationError: If there is an issue contacting the backend
        ValueError: If the id or name does not exist
    """
    orgs = list_my_orgs()

    org = next(
        (org for org in orgs if org.id == id_or_name or org.name == id_or_name), None
    )
    if not org:
        raise ValueError(f"No org found with id or name '{id_or_name}'")

    config_manager = get_config_manager()
    config_manager.config.current_org_id = org.id
    config_manager.save_config()


def connect_robot(
    robot_name: str,
    instance: int = 0,
    urdf_path: Optional[str] = None,
    mjcf_path: Optional[str] = None,
    overwrite: bool = False,
    shared: bool = False,
) -> Robot:
    """Initialize a robot connection and set it as the active robot.

    Creates or connects to a robot instance, validates version compatibility,
    and initializes streaming managers for live data and recording state updates.
    The robot becomes the active robot for subsequent operations.

    Upload of a robot description file (URDF or MJCF) is not required,
    but it is recommended for better visualization within Neuracore.

    Args:
        robot_name: Unique identifier for the robot.
        instance: Instance number of the robot for multi-instance deployments.
        urdf_path: Path to the robot's URDF file.
        mjcf_path: Path to the robot's MJCF file. This will be converted
            into URDF.
        overwrite: Whether to overwrite an existing robot configuration
            with the same name.
        shared: Whether you want to register the robot as shared/open-source.
            Note that setting shared=True is only available to specific
            members allocated by the Neuracore team.

    Returns:
        The initialized and connected robot instance.
    """
    validate_version()
    robot = _init_robot(robot_name, instance, urdf_path, mjcf_path, overwrite, shared)
    GlobalSingleton()._active_robot = robot
    if robot.archived is True:
        warn(
            f"This robot '{robot.name}' is archived. Was this intentional?",
        )
    # Initialize push update managers
    if robot.id is None:
        raise RobotError("Robot not initialized. Call init() first.")
    StreamManagerOrchestrator().get_provider_manager(robot.id, robot.instance)
    get_recording_state_manager()
    return robot


def is_recording(robot_name: Optional[str] = None, instance: int = 0) -> bool:
    """Check if a robot is currently recording.

    Args:
        robot_name: Robot identifier. If not provided, uses the currently
            active robot from the global state.
        instance: Instance number of the robot for multi-instance scenarios.

    Returns:
        bool: True if the robot is recording, False otherwise.
    """
    robot = _get_robot(robot_name, instance)
    return robot.is_recording()


def start_recording(robot_name: Optional[str] = None, instance: int = 0) -> None:
    """Start recording data for a specific robot.

    Begins a new recording session for the specified robot, capturing all
    configured data streams. Requires an active dataset to be set before
    starting the recording.

    Args:
        robot_name: Robot identifier. If not provided, uses the currently
            active robot from the global state.
        instance: Instance number of the robot for multi-instance scenarios.

    Raises:
        RobotError: If no robot is active and no robot_name is provided,
            if a recording is already in progress, or if no active dataset
            has been set.
    """
    robot = _get_robot(robot_name, instance)
    active_dataset_id = GlobalSingleton()._active_dataset_id
    if active_dataset_id is None:
        raise RobotError("No active dataset. Call create_dataset() first.")
    robot.start_recording(active_dataset_id)


def stop_recording(
    robot_name: Optional[str] = None, instance: int = 0, wait: bool = False
) -> None:
    """Stop recording data for a specific robot.

    Ends the current recording session for the specified robot. Optionally
    waits for all data streams to finish uploading before returning.

    Args:
        robot_name: Robot identifier. If not provided, uses the currently
            active robot from the global state.
        instance: Instance number of the robot for multi-instance scenarios.
        wait: Whether to block until all data streams have finished uploading
            to the backend storage.

    Raises:
        RobotError: If no robot is active and no robot_name is provided.
    """
    robot = _get_robot(robot_name, instance)
    if not robot.is_recording():
        warn(
            "No active recordings to stop. "
            "Your recording may have been stopped by another node."
        )
        return
    recording_id = robot.get_current_recording_id()
    if not recording_id:
        raise ValueError("Recording_id is None, no current recording")
    robot.stop_recording(recording_id)
    if wait:
        while backend_utils.get_num_active_streams(recording_id) > 0:
            time.sleep(2.0)


def stop_live_data(robot_name: Optional[str] = None, instance: int = 0) -> None:
    """Stop sharing live data for active monitoring from the Neuracore platform.

    Terminates the live data streaming connection that allows real-time
    monitoring and visualization of robot data through the Neuracore platform.
    This does not affect data recording, only the live streaming capability.

    Args:
        robot_name: Robot identifier. If not provided disables streaming for all robots
        instance: Instance number of the robot for multi-instance scenarios.

    """
    if not robot_name:
        global_provide_live_data_manager.disable()
        return

    robot = _get_robot(robot_name, instance)
    if not robot.id:
        raise RobotError("Robot not initialized. Call init() first.")
    StreamManagerOrchestrator().remove_manager(robot.id, robot.instance)
