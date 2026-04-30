"""High-level API for Neuracore robot management and data recording.

This module provides the main public interface for connecting to robots,
managing authentication, controlling data recording sessions, and handling
live data streaming. It maintains global state for active robots and
recording sessions.
"""

import logging
from warnings import warn

from neuracore.core.config.config_manager import get_config_manager
from neuracore.core.organizations import list_my_orgs
from neuracore.core.streaming.p2p.provider.global_live_data_enabled import (
    get_provide_live_data_enabled_manager,
)
from neuracore.core.streaming.p2p.stream_manager_orchestrator import (
    StreamManagerOrchestrator,
)
from neuracore.core.streaming.recording_state_manager import get_recording_state_manager

from ..core.auth import get_auth
from ..core.exceptions import RobotError
from ..core.robot import Robot, get_robot, get_robot_id_from_name
from ..core.robot import init as _init_robot
from ..core.robot import register_existing_robot
from ..core.robot import update_robot_name as _update_robot_name
from .globals import GlobalSingleton

logger = logging.getLogger(__name__)


def _get_robot(robot_name: str | None, instance: int) -> Robot:
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


def login(api_key: str | None = None) -> None:
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
    GlobalSingleton()._active_dataset = None
    GlobalSingleton()._active_dataset_id = None
    GlobalSingleton()._has_validated_version = False


def set_organization(id_or_name: str) -> None:
    """Set the current organization based upon its name or id.

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


def create_robot(
    robot_name: str,
    instance: int = 0,
    urdf_path: str | None = None,
    mjcf_path: str | None = None,
    overwrite: bool = False,
    shared: bool = False,
    exist_ok: bool = False,
) -> Robot:
    """Create a robot in Neuracore without setting it as the active robot.

    Creates a robot instance and uploads its robot description file when
    provided. Use `connect_robot()` when the robot should become the active robot
    for logging, recording, or live streaming.

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
        exist_ok: If True, return an existing matching robot instead of raising.

    Returns:
        The initialized robot instance.

    Raises:
        RobotError: If this robot has already been created locally or in
            Neuracore.
    """
    validate_version()

    try:
        robot = get_robot(robot_name, instance)
    except RobotError:
        pass
    else:
        if exist_ok:
            return robot
        raise RobotError(
            f"Robot '{robot_name}' with instance '{instance}' already exists. "
            "Call connect_robot() to connect to the existing robot."
        )

    try:
        robot_id = get_robot_id_from_name(robot_name)
    except RobotError:
        pass
    else:
        if exist_ok:
            return register_existing_robot(robot_name, robot_id, instance, shared)
        raise RobotError(
            f"Robot '{robot_name}' already exists in Neuracore. "
            "Call connect_robot() to connect to the existing robot."
        )

    robot = _init_robot(
        robot_name,
        instance,
        urdf_path,
        mjcf_path,
        overwrite,
        shared,
    )

    if robot.archived:
        warn(f"This robot '{robot.name}' is archived. Was this intentional?")

    return robot


def connect_robot(
    robot_name: str,
    instance: int = 0,
    shared: bool = False,
) -> Robot:
    """Connect to a created robot and set it as the active robot.

    Looks up a robot already created with `create_robot()` or already present
    in Neuracore, validates version compatibility, and initializes streaming
    managers for live data and recording state updates. The robot becomes the
    active robot for subsequent operations.

    Args:
        robot_name: Unique identifier for the robot.
        instance: Instance number of the robot for multi-instance deployments.
        shared: Whether you want to register the robot as shared/open-source.
            Note that setting shared=True is only available to specific
            members allocated by the Neuracore team.

    Returns:
        The connected robot instance.
    """
    validate_version()

    try:
        robot = get_robot(robot_name, instance)
    except RobotError:
        try:
            robot_id = get_robot_id_from_name(robot_name)
        except RobotError as exc:
            raise RobotError(
                f"Robot '{robot_name}' with instance '{instance}' has not been "
                "created. Call create_robot() first."
            ) from exc
        robot = register_existing_robot(
            robot_name=robot_name,
            robot_id=robot_id,
            instance=instance,
            shared=shared,
        )

    if robot.archived:
        warn(f"This robot '{robot.name}' is archived. Was this intentional?")

    GlobalSingleton()._active_robot = robot
    if robot.id is None:
        raise RobotError("Robot not initialized. Call create_robot() first.")
    StreamManagerOrchestrator().get_provider_manager(robot.id, robot.instance)
    get_recording_state_manager()
    return robot


def update_robot_name(
    robot_key: str,
    new_robot_name: str,
    instance: int = 0,
    shared: bool = False,
) -> str:
    """Update the robot name for a robot.

    Args:
        robot_key: Old robot name or ID of the robot to update.
        new_robot_name: New robot name to set for the robot.
        instance: Robot instance to associate with the update.
        shared: Whether the robot is shared/open-source.

    Returns:
        The resolved robot ID.
    """
    robot_id = _update_robot_name(
        robot_key,
        new_robot_name,
        instance=instance,
        shared=shared,
    )
    return robot_id


def get_training_job_logs(
    job_id: str, max_entries: int = 100, severity_filter: str | None = None
) -> dict:
    """Retrieve logs for a training job from neuracore backend.

    This convenience wrapper keeps the high-level API surface in `core.py`
    while delegating the backend call to the training API module.

    Args:
        job_id: The ID of the training job.
        max_entries: Maximum number of log entries to return.
        severity_filter: Optional log severity filter (for example: "ERROR").

    Returns:
        dict: Cloud compute logs payload.
    """
    from .training import get_training_job_logs as _get_training_job_logs

    return _get_training_job_logs(
        job_id=job_id,
        max_entries=max_entries,
        severity_filter=severity_filter,
    )


def is_recording(robot_name: str | None = None, instance: int = 0) -> bool:
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


def stop_live_data(robot_name: str | None = None, instance: int = 0) -> None:
    """Stop sharing live data for active monitoring from the Neuracore platform.

    Terminates the live data streaming connection that allows real-time
    monitoring and visualization of robot data through the Neuracore platform.
    This does not affect data recording, only the live streaming capability.

    Args:
        robot_name: Robot identifier. If not provided disables streaming for all robots
        instance: Instance number of the robot for multi-instance scenarios.

    """
    if not robot_name:
        get_provide_live_data_enabled_manager().disable()
        return

    robot = _get_robot(robot_name, instance)
    if not robot.id:
        raise RobotError("Robot not initialized. Call init() first.")
    StreamManagerOrchestrator().remove_manager(robot.id, robot.instance)


def cancel_recording(robot_name: str | None = None, instance: int = 0) -> None:
    """Cancel the current recording for a specific robot without saving any data.

    Args:
        robot_name: Robot identifier.
        instance: Instance number of the robot for multi-instance scenarios.

    """
    robot = _get_robot(robot_name, instance)
    if not robot.is_recording():
        return
    recording_id = robot.get_current_recording_id()
    if not recording_id:
        return
    robot.cancel_recording(recording_id)
