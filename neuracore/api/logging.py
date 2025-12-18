"""Robot data logging utilities.

This module provides functions for logging various types of robot sensor data
including joint positions, camera images, point clouds, and custom data streams.
All logging functions support optional robot identification and timestamping.
"""

import time
from warnings import filterwarnings, warn

import numpy as np
from neuracore_types import (
    CameraData,
    Custom1DData,
    DataType,
    DepthCameraData,
    EndEffectorPoseData,
    JointData,
    LanguageData,
    ParallelGripperOpenAmountData,
    PointCloudData,
    PoseData,
    RGBCameraData,
)
from neuracore_types.utils import validate_safe_name

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


def _log_single_joint_data(
    data_type: DataType, name: str, value: float, robot: Robot, timestamp: float
) -> None:
    """Log single joint data for a robot.

    Args:
        data_type: Type of joint data (e.g. DataType.JOINT_POSITIONS)
        name: Name of the joint
        value: Joint data value
        robot: Robot instance
        timestamp: Timestamp of the data
    """
    storage_name = validate_safe_name(name)
    str_id = f"{data_type.value}:{name}"
    joint_stream = robot.get_data_stream(str_id)
    if joint_stream is None:
        joint_stream = JsonDataStream(data_type=data_type, data_type_name=storage_name)
        robot.add_data_stream(str_id, joint_stream)

    start_stream(robot, joint_stream)

    data = JointData(
        timestamp=timestamp,
        value=value,
    )
    assert isinstance(
        joint_stream, JsonDataStream
    ), "Expected stream to be instance of JSONDataStream"
    joint_stream.log(data=data)
    if robot.id is None:
        raise RobotError("Robot not initialized. Call init() first.")
    StreamManagerOrchestrator().get_provider_manager(
        robot.id, robot.instance
    ).get_json_source(str_id, data_type, sensor_key=str_id).publish(
        data.model_dump(mode="json")
    )


def _log_group_of_joint_data(
    data_type: DataType,
    joint_data: dict[str, float],
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
) -> None:
    """Log joint data for a robot.

    Args:
        data_type: Type of joint data (e.g. DataType.JOINT_POSITIONS)
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
    for key, value in joint_data.items():
        _log_single_joint_data(data_type, key, value, robot, timestamp)


def _validate_extrinsics_intrinsics(
    extrinsics: np.ndarray | None, intrinsics: np.ndarray | None
) -> tuple[np.ndarray | None, np.ndarray | None]:
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
    camera_data_without_frame: CameraData,
    image: np.ndarray,
    name: str,
    robot_name: str | None = None,
    instance: int = 0,
) -> None:
    """Log camera data for a robot.

    Args:
        camera_type: Type of camera (e.g. DataType.RGB or DataType.DEPTH)
        camera_data_without_frame: Camera data to log without frame
            (e.g. RGBCameraData or DepthCameraData)
        image: Image data as numpy array
        name: Unique identifier for the camera
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If image format is invalid or camera type is unsupported
    """
    assert camera_type in (
        DataType.RGB_IMAGES,
        DataType.DEPTH_IMAGES,
    ), "Unsupported camera type"

    robot = _get_robot(robot_name, instance)
    storage_name = validate_safe_name(name)
    str_id = f"{camera_type.value}:{name}"

    if robot.id is None:
        raise RobotError("Robot not initialized. Call init() first.")

    # data streaming for bucket storage (lossless and lossy)
    stream = robot.get_data_stream(str_id)
    # create the stream if it doesn't exist
    if stream is None:
        if camera_type == DataType.RGB_IMAGES:
            stream = RGBDataStream(storage_name, image.shape[1], image.shape[0])
        elif camera_type == DataType.DEPTH_IMAGES:
            stream = DepthDataStream(storage_name, image.shape[1], image.shape[0])
        else:
            raise ValueError(f"Invalid camera type: {camera_type}")
        robot.add_data_stream(str_id, stream)
    assert isinstance(
        stream, VideoDataStream
    ), "Expected stream as instance of VideoDataStream"

    start_stream(robot, stream)
    if stream.width != image.shape[1] or stream.height != image.shape[0]:
        raise ValueError(
            f"Camera image dimensions {image.shape[1]}x{image.shape[0]} do not match "
            f"stream dimensions {stream.width}x{stream.height}"
        )

    # NOTE: we explicitly do not include the frame in the
    # camera_data_without_frame object to avoid serializing the frame to JSON
    # or having to make two copies for streaming and bucket storage.
    camera_data_copy = camera_data_without_frame.model_copy()
    stream.log(camera_data_without_frame, frame=image)

    # peer to peer (p2p) streaming
    # NOTE: to avoid serializing the frame, we make another copy of the
    # camera_data_without_frame object because stream.log modifies the object
    # and adds the frame to it.
    StreamManagerOrchestrator().get_provider_manager(
        robot.id, robot.instance
    ).get_video_source(name, camera_type, f"{name}_{camera_type}").add_frame(
        camera_data_copy, frame=image
    )


def log_custom_1d(
    name: str,
    data: np.ndarray,
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
) -> None:
    """Log arbitrary data for a robot.

    Args:
        name: Name of the data stream
        data: Data to log (must be a numpy ndarray)
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If data is not JSON serializable
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy ndarray")
    if data.ndim != 1:
        raise ValueError("Data must be a 1D numpy ndarray")
    timestamp = timestamp or time.time()
    robot = _get_robot(robot_name, instance)
    storage_name = validate_safe_name(name)
    str_id = f"{DataType.CUSTOM_1D.value}:{name}"
    stream = robot.get_data_stream(str_id)
    if stream is None:
        stream = JsonDataStream(
            data_type=DataType.CUSTOM_1D, data_type_name=storage_name
        )
        robot.add_data_stream(str_id, stream)

    start_stream(robot, stream)

    assert isinstance(
        stream, JsonDataStream
    ), "Expected stream to be instance of JSONDataStream"

    custom_data = Custom1DData(timestamp=timestamp, data=data)
    stream.log(custom_data)

    if robot.id is None:
        raise RobotError("Robot not initialized. Call init() first.")

    StreamManagerOrchestrator().get_provider_manager(
        robot.id, robot.instance
    ).get_json_source(str_id, DataType.CUSTOM_1D, sensor_key=str_id).publish(
        custom_data.model_dump(mode="json")
    )


def log_joint_positions(
    positions: dict[str, float],
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
) -> None:
    """Log joint positions for a robot.

    Args:
        positions: Dictionary mapping joint names to positions (in radians)
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If positions is not a dictionary of floats
    """
    _log_group_of_joint_data(
        DataType.JOINT_POSITIONS,
        positions,
        robot_name,
        instance,
        timestamp,
    )


def log_joint_position(
    name: str,
    position: float,
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
) -> None:
    """Log joint positions for a robot.

    Args:
        positions: Dictionary mapping joint names to positions (in radians)
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If positions is not a dictionary of floats
    """
    _log_group_of_joint_data(
        DataType.JOINT_POSITIONS,
        {name: position},
        robot_name,
        instance,
        timestamp,
    )


def log_joint_target_positions(
    target_positions: dict[str, float],
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
) -> None:
    """Log joint target positions for a robot.

    Args:
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
    _log_group_of_joint_data(
        DataType.JOINT_TARGET_POSITIONS,
        target_positions,
        robot_name,
        instance,
        timestamp,
    )


def log_joint_target_position(
    name: str,
    target_position: float,
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
) -> None:
    """Log joint target position for a robot.

    Args:
        name: Name of the joint
        target_position: Target position of the joint (in radians)
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If target_position is not a float
    """
    _log_group_of_joint_data(
        DataType.JOINT_TARGET_POSITIONS,
        {name: target_position},
        robot_name,
        instance,
        timestamp,
    )


def log_joint_velocities(
    velocities: dict[str, float],
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
) -> None:
    """Log joint velocities for a robot.

    Args:
        velocities: Dictionary mapping joint names to velocities (in radians/second)
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If velocities is not a dictionary of floats
    """
    _log_group_of_joint_data(
        DataType.JOINT_VELOCITIES,
        velocities,
        robot_name,
        instance,
        timestamp,
    )


def log_joint_velocity(
    name: str,
    velocity: float,
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
) -> None:
    """Log joint velocity for a robot.

    Args:
        name: Name of the joint
        velocity: Velocity of the joint (in radians/second)
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If velocity is not a float
    """
    _log_group_of_joint_data(
        DataType.JOINT_VELOCITIES,
        {name: velocity},
        robot_name,
        instance,
        timestamp,
    )


def log_joint_torques(
    torques: dict[str, float],
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
) -> None:
    """Log joint torques for a robot.

    Args:
        torques: Dictionary mapping joint names to torques (in Newton-meters)
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If torques is not a dictionary of floats
    """
    _log_group_of_joint_data(
        DataType.JOINT_TORQUES,
        torques,
        robot_name,
        instance,
        timestamp,
    )


def log_joint_torque(
    name: str,
    torque: float,
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
) -> None:
    """Log joint torque for a robot.

    Args:
        name: Name of the joint
        torque: Torque of the joint (in Newton-meters)
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If torque is not a float
    """
    _log_group_of_joint_data(
        DataType.JOINT_TORQUES,
        {name: torque},
        robot_name,
        instance,
        timestamp,
    )


def log_pose(
    name: str,
    pose: np.ndarray,
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
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
    storage_name = validate_safe_name(name)
    str_id = f"{DataType.POSES.value}:{name}"
    stream = robot.get_data_stream(str_id)
    if stream is None:
        stream = JsonDataStream(data_type=DataType.POSES, data_type_name=storage_name)
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
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
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
    storage_name = validate_safe_name(name)
    str_id = f"{DataType.END_EFFECTOR_POSES.value}:{name}"
    stream = robot.get_data_stream(str_id)
    if stream is None:
        stream = JsonDataStream(
            data_type=DataType.END_EFFECTOR_POSES, data_type_name=storage_name
        )
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
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
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
    storage_name = validate_safe_name(name)
    str_id = f"{DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS.value}:{name}"
    stream = robot.get_data_stream(str_id)
    if stream is None:
        stream = JsonDataStream(
            data_type=DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            data_type_name=storage_name,
        )
        robot.add_data_stream(str_id, stream)

    start_stream(robot, stream)
    assert isinstance(stream, JsonDataStream)

    parallel_gripper_open_amount_data = ParallelGripperOpenAmountData(
        timestamp=timestamp, open_amount=value
    )
    stream.log(parallel_gripper_open_amount_data)

    if robot.id is None:
        raise RobotError("Robot not initialized. Call init() first.")

    StreamManagerOrchestrator().get_provider_manager(
        robot.id, robot.instance
    ).get_json_source(str_id, DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS, str_id).publish(
        parallel_gripper_open_amount_data.model_dump(mode="json")
    )


def log_parallel_gripper_open_amounts(
    values: dict[str, float],
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
) -> None:
    """Log parallel gripper open amount data for a robot.

    Args:
        values: Dictionary mapping gripper names to open amounts
            (0.0 = closed, 1.0 = fully open)
        robot_name: Optional robot ID
        instance: Optional instance number
        timestamp: Optional timestamp
    """
    timestamp = timestamp or time.time()
    for name, value in values.items():
        log_parallel_gripper_open_amount(
            name=name,
            value=value,
            robot_name=robot_name,
            instance=instance,
            timestamp=timestamp,
        )


def log_language(
    name: str,
    language: str,
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
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
    storage_name = validate_safe_name(name)
    str_id = f"{DataType.LANGUAGE.value}:{name}"
    stream = robot.get_data_stream(str_id)
    if stream is None:
        stream = JsonDataStream(
            data_type=DataType.LANGUAGE, data_type_name=storage_name
        )
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
    extrinsics: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
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
    extrinsics, intrinsics = _validate_extrinsics_intrinsics(extrinsics, intrinsics)
    timestamp = timestamp or time.time()
    rgb_camera_data = RGBCameraData(
        timestamp=timestamp,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        frame=None,
    )
    _log_camera_data(
        DataType.RGB_IMAGES,
        rgb_camera_data,
        rgb,
        name,
        robot_name,
        instance,
    )


def log_depth(
    name: str,
    depth: np.ndarray,
    extrinsics: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
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
    extrinsics, intrinsics = _validate_extrinsics_intrinsics(extrinsics, intrinsics)
    timestamp = timestamp or time.time()
    depth_camera_data = DepthCameraData(
        timestamp=timestamp,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        frame=None,
    )
    _log_camera_data(
        DataType.DEPTH_IMAGES,
        depth_camera_data,
        depth,
        name,
        robot_name,
        instance,
    )


def log_point_cloud(
    name: str,
    points: np.ndarray,
    rgb_points: np.ndarray | None = None,
    extrinsics: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
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
    storage_name = validate_safe_name(name)
    str_id = f"{DataType.POINT_CLOUDS.value}:{name}"
    stream = robot.get_data_stream(str_id)
    if stream is None:
        stream = JsonDataStream(
            data_type=DataType.POINT_CLOUDS, data_type_name=storage_name
        )
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
        .get_json_source(str_id, DataType.POINT_CLOUDS, sensor_key=str_id)
    )
    src.publish(json_data)
