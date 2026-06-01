"""Robot data logging utilities.

This module provides functions for logging various types of robot sensor data
including joint positions, camera images, point clouds, and custom data streams.
All logging functions support optional robot identification and timestamping.
"""

import json
import logging
import time
from dataclasses import dataclass
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
    RobotInstanceIdentifier,
)
from neuracore_types.utils import validate_safe_name

from neuracore.api.core import _get_robot
from neuracore.api.globals import GlobalSingleton
from neuracore.core.exceptions import RobotError
from neuracore.core.robot import Robot
from neuracore.core.streaming.data_stream import (
    DataRecordingContext,
    DataStream,
    DepthDataStream,
    JsonDataStream,
    RGBDataStream,
    VideoDataStream,
)
from neuracore.core.streaming.p2p.provider.global_live_data_enabled import (
    get_provide_live_data_enabled_manager,
)
from neuracore.core.streaming.p2p.stream_manager_orchestrator import (
    StreamManagerOrchestrator,
)
from neuracore.core.streaming.recording_state_manager import get_recording_state_manager
from neuracore.core.utils.depth_utils import MAX_DEPTH
from neuracore.data_daemon.models import (
    BatchedJointDataItemPayload,
    BatchedJointDataPayload,
)
from neuracore.data_daemon.rust_selection import rust_daemon_enabled

logger = logging.getLogger(__name__)


class ExperimentalPointCloudWarning(UserWarning):
    """Warning for experimental point cloud features."""

    pass


filterwarnings("once", category=ExperimentalPointCloudWarning)


@dataclass(frozen=True)
class JointStreamBinding:
    """Resolved per-joint stream metadata and stream instance."""

    stream_id: str
    storage_name: str
    stream: JsonDataStream


@dataclass(frozen=True)
class LoggedJointData:
    """Result of logging one joint sample into the local stream graph."""

    binding: JointStreamBinding
    data: JointData
    trace_id: str | None


def _publish_json_to_p2p(
    robot: Robot,
    str_id: str,
    data_type: DataType,
    data: (
        JointData
        | Custom1DData
        | PoseData
        | EndEffectorPoseData
        | ParallelGripperOpenAmountData
        | LanguageData
        | PointCloudData
    ),
) -> None:
    """Publish JSON data to P2P streaming.

    Args:
        robot: Robot instance
        str_id: Stream identifier
        data_type: Type of data being published
        data: Data to publish
    """
    if robot.id is None:
        raise RobotError("Robot not initialized. Call init() first.")
    # Short-circuit before the Pydantic `model_dump` when live-data provision
    # is globally disabled (e.g. `NEURACORE_PROVIDE_LIVE_DATA=no`, or after a
    # runtime `disable()` on the manager). The downstream `JSONSource.publish`
    # already returns early in that case, but it does so *after* receiving the
    # serialized dict — and `model_dump(mode="json")` is the dominant cost at
    # 1 kHz+ logging rates. Checking the manager first turns the whole call
    # into a single attribute lookup when there is no consumer.
    if get_provide_live_data_enabled_manager().is_disabled():
        return
    StreamManagerOrchestrator().get_provider_manager(
        robot.id, robot.instance
    ).get_json_source(str_id, data_type, sensor_key=str_id).publish(
        data.model_dump(mode="json")
    )


def _record_json_to_daemon(
    robot: Robot,
    data_type: DataType,
    storage_name: str,
    data: (
        JointData
        | Custom1DData
        | PoseData
        | EndEffectorPoseData
        | ParallelGripperOpenAmountData
        | LanguageData
        | PointCloudData
    ),
    timestamp: float,
) -> None:
    """Forward one JSON sample to the Rust daemon's recording pipeline.

    The generic counterpart to :func:`_publish_json_to_p2p`: where that feeds
    live consumers, this persists the sample into the active recording. Every
    non-joint, non-video data type shares this single path — the daemon's
    ``log_json`` entry point is datatype-agnostic, so adding a new JSON type
    needs no daemon-side change.

    A no-op unless the Rust daemon is active and a recording is in progress
    (the legacy daemon is fed by :meth:`JsonDataStream.log` instead).

    Args:
        robot: Robot instance owning the daemon recording context.
        data_type: Wire label for the sample's trace.
        storage_name: Sensor name the trace is stored under.
        data: Data object to serialize and persist.
        timestamp: Capture timestamp in seconds.
    """
    if not (rust_daemon_enabled() and robot.get_current_recording_id() is not None):
        return
    payload = json.dumps(data.model_dump(mode="json")).encode("utf-8")
    robot._get_daemon_recording_context().log_json(
        data_type.value, storage_name, payload, timestamp
    )


def _publish_video_to_p2p(
    robot: Robot,
    name: str,
    camera_type: DataType,
    camera_data_without_frame: CameraData,
    image: np.ndarray,
) -> None:
    """Publish video frame to P2P streaming.

    Args:
        robot: Robot instance
        name: Camera name
        camera_type: Type of camera (RGB or DEPTH)
        camera_data_without_frame: Camera metadata
        image: Frame data
    """
    if robot.id is None:
        raise RobotError("Robot not initialized. Call init() first.")
    StreamManagerOrchestrator().get_provider_manager(
        robot.id, robot.instance
    ).get_video_source(name, camera_type, f"{name}_{camera_type}").add_frame(
        camera_data_without_frame, frame=image
    )


def start_stream(robot: Robot, data_stream: DataStream) -> None:
    """Start recording on a data stream if robot is currently recording.

    Args:
        robot: Robot instance
        data_stream: Data stream to start recording on
    """
    current_recording = robot.get_current_recording_id()
    if current_recording is not None and not data_stream.is_recording():
        instance_key = RobotInstanceIdentifier(
            robot_id=robot.id,
            robot_instance=robot.instance,
        )
        dataset_id = get_recording_state_manager().active_dataset_ids.get(instance_key)
        if dataset_id is None:
            dataset_id = GlobalSingleton()._active_dataset_id
            logger.debug(
                "start_stream: falling back to global dataset_id=%s recording_id=%s",
                dataset_id,
                current_recording,
            )
        context = DataRecordingContext(
            recording_id=current_recording,
            robot_id=robot.id,
            robot_name=robot.name,
            robot_instance=robot.instance,
            dataset_id=dataset_id,
            dataset_name=None,
        )
        data_stream.start_recording(context)


def _log_single_joint_data(
    data_type: DataType,
    name: str,
    value: float,
    robot: Robot,
    timestamp: float,
    dry_run: bool = False,
) -> None:
    """Log single joint data for a robot.

    Args:
        data_type: Type of joint data (e.g. DataType.JOINT_POSITIONS)
        name: Name of the joint
        value: Joint data value
        robot: Robot instance
        timestamp: Timestamp of the data
        dry_run: If True, skip actual logging (validation only)
    """
    if dry_run:
        return

    _log_joint_data_point(
        data_type=data_type,
        name=name,
        value=value,
        robot=robot,
        timestamp=timestamp,
        send_to_daemon=True,
    )


def _get_or_create_joint_stream(
    data_type: DataType,
    name: str,
    robot: Robot,
) -> JointStreamBinding:
    """Return the stream id, storage name, and JsonDataStream for one joint."""
    storage_name = validate_safe_name(name)
    str_id = f"{data_type.value}:{name}"
    joint_stream = robot.get_data_stream(str_id)
    if joint_stream is None:
        joint_stream = JsonDataStream(data_type=data_type, data_type_name=storage_name)
        robot.add_data_stream(str_id, joint_stream)
    assert isinstance(
        joint_stream, JsonDataStream
    ), "Expected stream to be instance of JSONDataStream"
    return JointStreamBinding(
        stream_id=str_id,
        storage_name=storage_name,
        stream=joint_stream,
    )


def _log_joint_data_point(
    *,
    data_type: DataType,
    name: str,
    value: float,
    robot: Robot,
    timestamp: float,
    send_to_daemon: bool,
) -> LoggedJointData:
    """Log one joint sample into the local stream graph and live publishers."""
    joint_stream_binding = _get_or_create_joint_stream(data_type, name, robot)
    joint_stream = joint_stream_binding.stream
    start_stream(robot, joint_stream)

    data = JointData(
        timestamp=timestamp,
        value=value,
    )
    joint_stream.log(data=data, send_to_daemon=send_to_daemon)

    if robot.id is None:
        raise RobotError("Robot not initialized. Call init() first.")

    _publish_json_to_p2p(robot, joint_stream_binding.stream_id, data_type, data)

    producer_channel = joint_stream.get_producer_channel()
    trace_id = (
        producer_channel.trace_id
        if producer_channel is not None
        and joint_stream.get_recording_context() is not None
        else None
    )
    return LoggedJointData(
        binding=joint_stream_binding,
        data=data,
        trace_id=trace_id,
    )


def _log_group_of_joint_data(
    data_type: DataType,
    joint_data: dict[str, float],
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
    dry_run: bool = False,
) -> None:
    """Log joint data for a robot.

    Args:
        data_type: Type of joint data (e.g. DataType.JOINT_POSITIONS)
        joint_data: Dictionary mapping joint names to joint data values
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp
        dry_run: If True, skip actual logging (validation only)

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If joint_data is not a dictionary of floats
    """
    if timestamp is None:
        timestamp = time.time()
    if not isinstance(joint_data, dict):
        raise ValueError("Joint data must be a dictionary of floats")
    for key, value in joint_data.items():
        if not isinstance(value, float):
            raise ValueError(f"Joint data must be floats. {key} is not a float.")

    if dry_run:
        return

    robot = _get_robot(robot_name, instance)
    rust_mode = rust_daemon_enabled()

    # Hot path: at 1 kHz × N joints the per-iteration Python overhead
    # dominates. Hoist the robot-wide state out of the loop, look up cached
    # per-joint bindings, and inline the small amount of work the legacy
    # _log_joint_data_point did so each iteration is roughly: dict lookup,
    # one Pydantic init, one attribute write, one list append.
    binding_cache = robot._joint_stream_bindings
    current_recording_id = robot.get_current_recording_id()
    live_data_disabled = get_provide_live_data_enabled_manager().is_disabled()
    robot_id = robot.id
    robot_instance = robot.instance
    live_data_orchestrator = (
        None if live_data_disabled or robot_id is None else StreamManagerOrchestrator()
    )

    # Under the Rust daemon the whole joint group is delivered in one call to
    # the recording-scoped native session, addressed by joint name; under the
    # legacy daemon each joint's per-stream channel contributes a trace_id to
    # a `BatchedJointDataPayload`.
    native_items: list[tuple[str, float]] = []
    batched_items: list[BatchedJointDataItemPayload] = []
    batch_transport_stream: JsonDataStream | None = None

    for joint_name, joint_value in joint_data.items():
        cache_key = (data_type, joint_name)
        binding = binding_cache.get(cache_key)
        if binding is None:
            binding = _get_or_create_joint_stream(data_type, joint_name, robot)
            binding_cache[cache_key] = binding

        joint_stream = binding.stream
        if current_recording_id is not None and not joint_stream.is_recording():
            start_stream(robot, joint_stream)

        data = JointData(timestamp=timestamp, value=joint_value)
        joint_stream.log(data=data, send_to_daemon=False)

        if live_data_orchestrator is not None and robot_id is not None:
            live_data_orchestrator.get_provider_manager(
                robot_id, robot_instance
            ).get_json_source(
                binding.stream_id, data_type, sensor_key=binding.stream_id
            ).publish(
                data.model_dump(mode="json")
            )

        if rust_mode:
            if current_recording_id is not None:
                native_items.append((binding.storage_name, joint_value))
            continue

        producer_channel = joint_stream.get_producer_channel()
        if producer_channel is None or joint_stream.get_recording_context() is None:
            continue
        trace_id = producer_channel.trace_id
        if trace_id is None:
            continue
        if batch_transport_stream is None:
            batch_transport_stream = joint_stream
        batched_items.append(
            BatchedJointDataItemPayload(
                trace_id=trace_id,
                data_type_name=binding.storage_name,
                value=joint_value,
            )
        )

    if rust_mode:
        if native_items:
            robot._get_daemon_recording_context().log_joints(
                data_type.value, timestamp, native_items
            )
        return

    if batch_transport_stream is None or not batched_items:
        return

    batch_context = batch_transport_stream.get_recording_context()
    batch_transport_channel = batch_transport_stream.get_producer_channel()
    if batch_context is None or batch_transport_channel is None:
        return

    batch_transport_channel.send_batched_joint_data(
        BatchedJointDataPayload(
            recording_id=batch_context.recording_id,
            timestamp=timestamp,
            dataset_id=batch_context.dataset_id,
            dataset_name=batch_context.dataset_name,
            robot_name=batch_context.robot_name,
            robot_id=batch_context.robot_id,
            robot_instance=batch_context.robot_instance,
            data_type=data_type,
            items=batched_items,
        )
    )


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
    dry_run: bool = False,
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
        dry_run: If True, skip actual logging (validation only)

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If image format is invalid or camera type is unsupported
    """
    assert camera_type in (
        DataType.RGB_IMAGES,
        DataType.DEPTH_IMAGES,
    ), "Unsupported camera type"

    storage_name = validate_safe_name(name)
    if dry_run:
        return

    robot = _get_robot(robot_name, instance)
    str_id = f"{camera_type.value}:{name}"

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
    stream.log(camera_data_without_frame, frame=image)

    if rust_daemon_enabled() and robot.get_current_recording_id() is not None:
        contiguous = image if image.flags.c_contiguous else np.ascontiguousarray(image)
        robot._get_daemon_recording_context().log_frame(
            camera_type.value,
            storage_name,
            int(image.shape[1]),
            int(image.shape[0]),
            memoryview(contiguous).cast("B"),
            camera_data_without_frame.timestamp,
        )

    _publish_video_to_p2p(robot, name, camera_type, camera_data_without_frame, image)


def log_custom_1d(
    name: str,
    data: np.ndarray,
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
    dry_run: bool = False,
) -> None:
    """Log arbitrary data for a robot.

    Args:
        name: Name of the data stream
        data: Data to log (must be a numpy ndarray)
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp
        dry_run: If True, skip actual logging (validation only)

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If data is not JSON serializable
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy ndarray")
    if data.ndim != 1:
        raise ValueError("Data must be a 1D numpy ndarray")
    if timestamp is None:
        timestamp = time.time()

    storage_name = validate_safe_name(name)
    if dry_run:
        return

    robot = _get_robot(robot_name, instance)
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
    _record_json_to_daemon(
        robot, DataType.CUSTOM_1D, storage_name, custom_data, timestamp
    )
    _publish_json_to_p2p(robot, str_id, DataType.CUSTOM_1D, custom_data)


def log_joint_positions(
    positions: dict[str, float],
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
    dry_run: bool = False,
) -> None:
    """Log joint positions for a robot.

    Args:
        positions: Dictionary mapping joint names to positions (in radians)
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp
        dry_run: If True, skip actual logging (validation only)

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
        dry_run,
    )


def log_joint_position(
    name: str,
    position: float,
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
    dry_run: bool = False,
) -> None:
    """Log joint positions for a robot.

    Args:
        positions: Dictionary mapping joint names to positions (in radians)
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp
        dry_run: If True, skip actual logging (validation only)

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
        dry_run,
    )


def log_joint_target_positions(
    target_positions: dict[str, float],
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
    dry_run: bool = False,
) -> None:
    """Log joint target positions for a robot.

    Args:
        target_positions: Dictionary mapping joint names to
            target positions (in radians)
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp
        dry_run: If True, skip actual logging (validation only)

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
        dry_run,
    )


def log_joint_target_position(
    name: str,
    target_position: float,
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
    dry_run: bool = False,
) -> None:
    """Log joint target position for a robot.

    Args:
        name: Name of the joint
        target_position: Target position of the joint (in radians)
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp
        dry_run: If True, skip actual logging (validation only)

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
        dry_run,
    )


def log_joint_velocities(
    velocities: dict[str, float],
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
    dry_run: bool = False,
) -> None:
    """Log joint velocities for a robot.

    Args:
        velocities: Dictionary mapping joint names to velocities (in radians/second)
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp
        dry_run: If True, skip actual logging (validation only)

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
        dry_run,
    )


def log_joint_velocity(
    name: str,
    velocity: float,
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
    dry_run: bool = False,
) -> None:
    """Log joint velocity for a robot.

    Args:
        name: Name of the joint
        velocity: Velocity of the joint (in radians/second)
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp
        dry_run: If True, skip actual logging (validation only)

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
        dry_run,
    )


def log_joint_torques(
    torques: dict[str, float],
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
    dry_run: bool = False,
) -> None:
    """Log joint torques for a robot.

    Args:
        torques: Dictionary mapping joint names to torques (in Newton-meters)
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp
        dry_run: If True, skip actual logging (validation only)

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
        dry_run,
    )


def log_joint_torque(
    name: str,
    torque: float,
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
    dry_run: bool = False,
) -> None:
    """Log joint torque for a robot.

    Args:
        name: Name of the joint
        torque: Torque of the joint (in Newton-meters)
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp
        dry_run: If True, skip actual logging (validation only)

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
        dry_run,
    )


def log_visual_joint_positions(
    positions: dict[str, float],
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
    dry_run: bool = False,
) -> None:
    """Log visual joint positions for a robot.

    Visual joint positions are joint positions that are required for URDF
    visualisation but not used during training (e.g. individual finger
    joints in a gripper).

    Args:
        positions: Dictionary mapping joint names to positions (in radians)
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp
        dry_run: If True, skip actual logging (validation only)

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If positions is not a dictionary of floats
    """
    _log_group_of_joint_data(
        DataType.VISUAL_JOINT_POSITIONS,
        positions,
        robot_name,
        instance,
        timestamp,
        dry_run,
    )


def log_visual_joint_position(
    name: str,
    position: float,
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
    dry_run: bool = False,
) -> None:
    """Log visual joint position for a robot.

    Visual joint positions are joint positions that are required for URDF
    visualisation but not used during training (e.g. individual finger
    joints in a gripper).

    Args:
        name: Name of the joint
        position: Position of the joint (in radians)
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp
        dry_run: If True, skip actual logging (validation only)

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If position is not a float
    """
    _log_group_of_joint_data(
        DataType.VISUAL_JOINT_POSITIONS,
        {name: position},
        robot_name,
        instance,
        timestamp,
        dry_run,
    )


def log_pose(
    name: str,
    pose: np.ndarray,
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
    dry_run: bool = False,
) -> None:
    """Log pose data for a robot.

    Args:
        name: Name of the pose.
        pose: 7-element numpy array: [x, y, z, qx, qy, qz, qw]
        robot_name: Optional robot name.
            If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp
        dry_run: If True, skip actual logging (validation only)

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If pose is not a 7-element numpy array
    """
    if timestamp is None:
        timestamp = time.time()
    if not isinstance(pose, np.ndarray):
        raise ValueError(
            f"Pose must be a numpy array, got {type(pose).__name__} for '{name}'."
        )
    if len(pose) != 7:
        raise ValueError(
            f"Pose must be a numpy array of length 7, got length {len(pose)} for "
            f"'{name}'."
        )

    storage_name = validate_safe_name(name)
    if dry_run:
        return

    robot = _get_robot(robot_name, instance)
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
    _record_json_to_daemon(robot, DataType.POSES, storage_name, pose_data, timestamp)
    _publish_json_to_p2p(robot, str_id, DataType.POSES, pose_data)


def log_end_effector_pose(
    name: str,
    pose: np.ndarray,
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
    dry_run: bool = False,
) -> None:
    """Log end-effector pose data for a robot.

    Args:
        name: Name of the end effector
        pose: 7-element numpy array: [x, y, z, qx, qy, qz, qw]
        robot_name: Optional robot ID
        instance: Optional instance number
        timestamp: Optional timestamp
        dry_run: If True, skip actual logging (validation only)
    """
    if timestamp is None:
        timestamp = time.time()

    if not isinstance(pose, np.ndarray):
        raise ValueError(
            f"End effector pose must be a numpy array, got {type(pose).__name__}."
        )
    if len(pose) != 7:
        raise ValueError(
            f"End effector pose must be a 7-element numpy array, got length "
            f"{len(pose)} for '{name}'."
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

    storage_name = validate_safe_name(name)
    if dry_run:
        return

    robot = _get_robot(robot_name, instance)
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
    _record_json_to_daemon(
        robot, DataType.END_EFFECTOR_POSES, storage_name, ee_pose_data, timestamp
    )
    _publish_json_to_p2p(robot, str_id, DataType.END_EFFECTOR_POSES, ee_pose_data)


def log_parallel_gripper_open_amount(
    name: str,
    value: float,
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
    dry_run: bool = False,
) -> None:
    """Log parallel gripper open amount data for a robot.

    Args:
        name: Name of the parallel gripper
        value: Open amount (0.0 = closed, 1.0 = fully open)
        robot_name: Optional robot ID
        instance: Optional instance number
        timestamp: Optional timestamp
        dry_run: If True, skip actual logging (validation only)
    """
    if timestamp is None:
        timestamp = time.time()
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

    storage_name = validate_safe_name(name)
    if dry_run:
        return

    robot = _get_robot(robot_name, instance)
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
    _record_json_to_daemon(
        robot,
        DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
        storage_name,
        parallel_gripper_open_amount_data,
        timestamp,
    )
    _publish_json_to_p2p(
        robot,
        str_id,
        DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
        parallel_gripper_open_amount_data,
    )


def log_parallel_gripper_open_amounts(
    values: dict[str, float],
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
    dry_run: bool = False,
) -> None:
    """Log parallel gripper open amount data for a robot.

    Args:
        values: Dictionary mapping gripper names to open amounts
            (0.0 = closed, 1.0 = fully open)
        robot_name: Optional robot ID
        instance: Optional instance number
        timestamp: Optional timestamp
        dry_run: If True, skip actual logging (validation only)
    """
    if timestamp is None:
        timestamp = time.time()
    for name, value in values.items():
        log_parallel_gripper_open_amount(
            name=name,
            value=value,
            robot_name=robot_name,
            instance=instance,
            timestamp=timestamp,
            dry_run=dry_run,
        )


def log_parallel_gripper_target_open_amount(
    name: str,
    value: float,
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
    dry_run: bool = False,
) -> None:
    """Log parallel gripper target open amount data for a robot.

    This logs the commanded/target gripper open amount, as opposed to
    log_parallel_gripper_open_amount which logs the actual gripper state.

    Args:
        name: Name of the parallel gripper
        value: Target open amount (0.0 = closed, 1.0 = fully open)
        robot_name: Optional robot ID
        instance: Optional instance number
        timestamp: Optional timestamp
        dry_run: If True, skip actual logging (validation only)
    """
    if timestamp is None:
        timestamp = time.time()
    if not isinstance(name, str):
        raise ValueError(
            f"Parallel gripper names must be strings. " f"{name} is not a string."
        )
    if not isinstance(value, float):
        raise ValueError(
            f"Parallel gripper target open amounts must be floats. "
            f"{value} is not a float."
        )
    if value < 0.0 or value > 1.0:
        raise ValueError(
            "Parallel gripper target open amounts must be between 0.0 and 1.0."
        )

    storage_name = validate_safe_name(name)
    if dry_run:
        return

    robot = _get_robot(robot_name, instance)
    str_id = f"{DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS.value}:{name}"
    stream = robot.get_data_stream(str_id)
    if stream is None:
        stream = JsonDataStream(
            data_type=DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
            data_type_name=storage_name,
        )
        robot.add_data_stream(str_id, stream)

    start_stream(robot, stream)
    assert isinstance(stream, JsonDataStream)

    parallel_gripper_target_open_amount_data = ParallelGripperOpenAmountData(
        timestamp=timestamp, open_amount=value
    )
    stream.log(parallel_gripper_target_open_amount_data)
    _record_json_to_daemon(
        robot,
        DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
        storage_name,
        parallel_gripper_target_open_amount_data,
        timestamp,
    )
    _publish_json_to_p2p(
        robot,
        str_id,
        DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
        parallel_gripper_target_open_amount_data,
    )


def log_parallel_gripper_target_open_amounts(
    values: dict[str, float],
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
    dry_run: bool = False,
) -> None:
    """Log parallel gripper target open amount data for a robot.

    This logs the commanded/target gripper open amounts, as opposed to
    log_parallel_gripper_open_amounts which logs the actual gripper states.

    Args:
        values: Dictionary mapping gripper names to target open amounts
            (0.0 = closed, 1.0 = fully open)
        robot_name: Optional robot ID
        instance: Optional instance number
        timestamp: Optional timestamp
        dry_run: If True, skip actual logging (validation only)
    """
    if timestamp is None:
        timestamp = time.time()
    for name, value in values.items():
        log_parallel_gripper_target_open_amount(
            name=name,
            value=value,
            robot_name=robot_name,
            instance=instance,
            timestamp=timestamp,
            dry_run=dry_run,
        )


def log_language(
    name: str,
    language: str,
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
    dry_run: bool = False,
) -> None:
    """Log language annotation for a robot.

    Args:
        name: Name of the language annotation
        language: A language string associated with this timestep
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        timestamp: Optional timestamp
        dry_run: If True, skip actual logging (validation only)

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If language is not a string
    """
    if timestamp is None:
        timestamp = time.time()
    if not isinstance(language, str):
        raise ValueError("Language must be a string")

    storage_name = validate_safe_name(name)
    if dry_run:
        return

    robot = _get_robot(robot_name, instance)
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

    language_data = LanguageData(timestamp=timestamp, text=language)
    stream.log(language_data)
    _record_json_to_daemon(
        robot, DataType.LANGUAGE, storage_name, language_data, timestamp
    )
    _publish_json_to_p2p(robot, str_id, DataType.LANGUAGE, language_data)


def log_rgb(
    name: str,
    rgb: np.ndarray,
    extrinsics: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
    dry_run: bool = False,
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
        dry_run: If True, skip actual logging (validation only)

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If image format is invalid
    """
    if not isinstance(rgb, np.ndarray):
        raise ValueError("Image must be a numpy array")
    if rgb.dtype != np.uint8:
        raise ValueError("Image must be uint8 with range 0-255")
    extrinsics, intrinsics = _validate_extrinsics_intrinsics(extrinsics, intrinsics)
    if timestamp is None:
        timestamp = time.time()
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
        dry_run,
    )


def log_depth(
    name: str,
    depth: np.ndarray,
    extrinsics: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
    dry_run: bool = False,
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
        dry_run: If True, skip actual logging (validation only)

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
    if timestamp is None:
        timestamp = time.time()
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
        dry_run,
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
    dry_run: bool = False,
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
        dry_run: If True, skip actual logging (validation only)

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If point cloud format is invalid
    """
    warn(
        "Point cloud logging is experimental and may change in future releases.",
        ExperimentalPointCloudWarning,
    )
    if timestamp is None:
        timestamp = time.time()
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

    storage_name = validate_safe_name(name)
    if dry_run:
        return

    extrinsics, intrinsics = _validate_extrinsics_intrinsics(extrinsics, intrinsics)
    robot = _get_robot(robot_name, instance)
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
    _record_json_to_daemon(
        robot, DataType.POINT_CLOUDS, storage_name, point_data, timestamp
    )
    _publish_json_to_p2p(robot, str_id, DataType.POINT_CLOUDS, point_data)
