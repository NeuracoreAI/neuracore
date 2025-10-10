from typing import Any, Optional

import numpy as np

from neuracore_new_data_format.dataset import Dataset
from neuracore_new_data_format.recording import RecordingFactory, RecordingType


class Robot:
    pass


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
    # Just for compatibility
    pass


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
    # Just for compatibility
    return Robot()


_current_dataset: Dataset | None = None
recording_factory = RecordingFactory(RecordingType.MCAP)


def get_dataset() -> Dataset:
    assert _current_dataset is not None, "No active dataset"
    return _current_dataset


def create_dataset(
    name: str,
    description: Optional[str] = None,
    tags: Optional[list[str]] = None,
    shared: bool = False,
) -> Dataset:
    """Create a new dataset for robot demonstrations.

    Args:
        name: Dataset name
        description: Optional description
        tags: Optional list of tags
        shared: Whether the dataset should be shared/open-source.
            Note that setting shared=True is only available to specific
            members allocated by the Neuracore team.

    Returns:
        Dataset: The newly created dataset instance

    Raises:
        DatasetError: If dataset creation fails
    """
    global _current_dataset
    _current_dataset = Dataset(name=name, recording_factory=recording_factory)
    return _current_dataset


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

    return get_dataset().start_recording()


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
    get_dataset().stop_recording()


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
    get_dataset().log_custom_data(name=name, data=data, timestamp=timestamp)


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
    get_dataset().log_joint_data(
        data_type="joint_positions",
        joint_data=positions,
        additional_urdf_data=additional_urdf_positions,
        timestamp=timestamp,
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

    get_dataset().log_joint_data(
        data_type="joint_target_positions",
        joint_data=target_positions,
        additional_urdf_data=additional_urdf_positions,
        timestamp=timestamp,
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

    get_dataset().log_joint_data(
        data_type="joint_velocities",
        joint_data=velocities,
        additional_urdf_data=additional_urdf_velocities,
        timestamp=timestamp,
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
    get_dataset().log_language_data(
        language=language,
        timestamp=timestamp,
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
    get_dataset().log_rgb(
        camera_id=camera_id,
        image=image,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        timestamp=timestamp,
    )
