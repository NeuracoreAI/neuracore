"""Neuracore data types and models for robot sensor data and ML operations.

This module defines the core data structures used throughout Neuracore for
representing robot sensor data, synchronized data points, dataset descriptions,
and model predictions. All data types include automatic timestamping and
support for serialization via Pydantic.
"""

import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, NamedTuple, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, NonNegativeInt


class NCData(BaseModel):
    """Base class for all Neuracore data with automatic timestamping.

    Provides a common base for all data types in the system with automatic
    timestamp generation for temporal synchronization and data ordering.
    """

    timestamp: float = Field(default_factory=lambda: time.time())


class JointData(NCData):
    """Robot joint state data including positions, velocities, or torques.

    Represents joint-space data for robotic systems with support for named
    joints and additional auxiliary values. Used for positions, velocities,
    torques, and target positions.
    """

    values: dict[str, float]
    additional_values: Optional[dict[str, float]] = None


class CameraData(NCData):
    """Camera sensor data including images and calibration information.

    Contains image data along with camera intrinsic and extrinsic parameters
    for 3D reconstruction and computer vision applications. The frame field
    is populated during dataset iteration for efficiency.
    """

    frame_idx: int = 0  # Needed so we can index video after sync
    extrinsics: Optional[list[list[float]]] = None
    intrinsics: Optional[list[list[float]]] = None
    frame: Optional[Union[Any, str]] = None  # Only filled in when using dataset iter


class PoseData(NCData):
    """6DOF pose data for objects, end-effectors, or coordinate frames.

    Represents position and orientation information for tracking objects
    or robot components in 3D space. Poses are stored as dictionaries
    mapping pose names to [x, y, z, rx, ry, rz] values.
    """

    pose: dict[str, list[float]]


class EndEffectorData(NCData):
    """End-effector state data including gripper and tool configurations.

    Contains the state of robot end-effectors such as gripper opening amounts,
    tool activations, or other end-effector specific parameters.
    """

    open_amounts: dict[str, float]


class PointCloudData(NCData):
    """3D point cloud data with optional RGB coloring and camera parameters.

    Represents 3D spatial data from depth sensors or LiDAR systems with
    optional color information and camera calibration for registration.
    """

    points: list[list[float]]
    rgb_points: Optional[list[list[int]]] = None
    extrinsics: Optional[list[list[float]]] = None
    intrinsics: Optional[list[list[float]]] = None


class LanguageData(NCData):
    """Natural language instruction or description data.

    Contains text-based information such as task descriptions, voice commands,
    or other linguistic data associated with robot demonstrations.
    """

    text: str


class CustomData(NCData):
    """Generic container for application-specific data types.

    Provides a flexible way to include custom sensor data or application-specific
    information that doesn't fit into the standard data categories.
    """

    data: Any


class SyncPoint(BaseModel):
    """Synchronized collection of all sensor data at a single time point.

    Represents a complete snapshot of robot state and sensor information
    at a specific timestamp. Used for creating temporally aligned datasets
    and ensuring consistent data relationships across different sensors.
    """

    timestamp: float = Field(default_factory=lambda: time.time())
    joint_positions: Optional[JointData] = None
    joint_velocities: Optional[JointData] = None
    joint_torques: Optional[JointData] = None
    joint_target_positions: Optional[JointData] = None
    end_effectors: Optional[EndEffectorData] = None
    poses: Optional[dict[str, PoseData]] = None
    rgb_images: Optional[dict[str, CameraData]] = None
    depth_images: Optional[dict[str, CameraData]] = None
    point_clouds: Optional[dict[str, PointCloudData]] = None
    language_data: Optional[LanguageData] = None
    custom_data: Optional[dict[str, CustomData]] = None


class SyncedData(BaseModel):
    """Complete synchronized dataset containing a sequence of data points.

    Represents an entire recording or demonstration as a time-ordered sequence
    of synchronized data points with start and end timestamps for temporal
    reference.
    """

    frames: list[SyncPoint]
    start_time: float
    end_time: float


class DataType(str, Enum):
    """Enumeration of supported data types in the Neuracore system.

    Defines the standard data categories used for dataset organization,
    model training, and data processing pipelines.
    """

    # Robot state
    JOINT_POSITIONS = "joint_positions"
    JOINT_VELOCITIES = "joint_velocities"
    JOINT_TORQUES = "joint_torques"
    JOINT_TARGET_POSITIONS = "joint_target_positions"
    END_EFFECTORS = "end_effectors"

    # Vision
    RGB_IMAGE = "rgb_image"
    DEPTH_IMAGE = "depth_image"
    POINT_CLOUD = "point_cloud"

    # Other
    POSES = "poses"
    LANGUAGE = "language"
    CUSTOM = "custom"


class DataItemStats(BaseModel):
    """Statistical summary of data dimensions and distributions.

    Contains statistical information about data arrays including means,
    standard deviations, counts, and maximum lengths for normalization
    and model configuration purposes.
    """

    mean: list[float] = Field(default_factory=list)
    std: list[float] = Field(default_factory=list)
    count: list[int] = Field(default_factory=list)
    max_len: int = Field(default_factory=lambda data: len(data["mean"]))


class DatasetDescription(BaseModel):
    """Comprehensive description of dataset contents and statistics.

    Provides metadata about a complete dataset including statistical summaries
    for all data types, maximum counts for variable-length data, and methods
    for determining which data types are present.
    """

    # Joint data statistics
    joint_positions: DataItemStats = Field(default_factory=DataItemStats)
    joint_velocities: DataItemStats = Field(default_factory=DataItemStats)
    joint_torques: DataItemStats = Field(default_factory=DataItemStats)
    joint_target_positions: DataItemStats = Field(default_factory=DataItemStats)

    # End-effector statistics
    end_effector_states: DataItemStats = Field(default_factory=DataItemStats)

    # Pose statistics
    poses: DataItemStats = Field(default_factory=DataItemStats)

    # Visual data counts
    max_num_rgb_images: int = 0
    max_num_depth_images: int = 0
    max_num_point_clouds: int = 0

    # Language data
    max_language_length: int = 0

    # Custom data statistics
    custom_data_stats: dict[str, DataItemStats] = Field(default_factory=dict)

    def get_data_types(self) -> list[DataType]:
        """Determine which data types are present in the dataset.

        Analyzes the dataset statistics to identify which data modalities
        contain actual data (non-zero lengths/counts).

        Returns:
            List of DataType enums representing the data modalities
            present in this dataset.
        """
        data_types = []

        # Joint data
        if self.joint_positions.max_len > 0:
            data_types.append(DataType.JOINT_POSITIONS)
        if self.joint_velocities.max_len > 0:
            data_types.append(DataType.JOINT_VELOCITIES)
        if self.joint_torques.max_len > 0:
            data_types.append(DataType.JOINT_TORQUES)
        if self.joint_target_positions.max_len > 0:
            data_types.append(DataType.JOINT_TARGET_POSITIONS)

        # End-effector data
        if self.end_effector_states.max_len > 0:
            data_types.append(DataType.END_EFFECTORS)

        # Pose data
        if self.poses.max_len > 0:
            data_types.append(DataType.POSES)

        # Visual data
        if self.max_num_rgb_images > 0:
            data_types.append(DataType.RGB_IMAGE)
        if self.max_num_depth_images > 0:
            data_types.append(DataType.DEPTH_IMAGE)
        if self.max_num_point_clouds > 0:
            data_types.append(DataType.POINT_CLOUD)

        # Language data
        if self.max_language_length > 0:
            data_types.append(DataType.LANGUAGE)

        # Custom data
        if self.custom_data_stats:
            data_types.append(DataType.CUSTOM)

        return data_types

    def add_custom_data_stats(
        self, key: str, stats: DataItemStats, max_length: int = 0
    ) -> None:
        """Add statistics for a custom data type.

        Args:
            key: Name of the custom data type
            stats: Statistical information for the custom data
            max_length: Maximum length of the custom data arrays
        """
        self.custom_data_stats[key] = stats


class RecordingDescription(BaseModel):
    """Description of a single recording episode with statistics and counts.

    Provides metadata about an individual recording including data statistics,
    sensor counts, and episode length for analysis and processing.
    """

    # Joint data statistics
    joint_positions: DataItemStats = Field(default_factory=DataItemStats)
    joint_velocities: DataItemStats = Field(default_factory=DataItemStats)
    joint_torques: DataItemStats = Field(default_factory=DataItemStats)
    joint_target_positions: DataItemStats = Field(default_factory=DataItemStats)

    # End-effector statistics
    end_effector_states: DataItemStats = Field(default_factory=DataItemStats)

    # Pose statistics
    poses: DataItemStats = Field(default_factory=DataItemStats)

    # Visual data counts
    num_rgb_images: int = 0
    num_depth_images: int = 0
    num_point_clouds: int = 0

    # Language data
    max_language_length: int = 0

    # Episode metadata
    episode_length: int = 0

    # Custom data statistics
    custom_data_stats: dict[str, DataItemStats] = Field(default_factory=dict)

    def get_data_types(self) -> list[DataType]:
        """Determine which data types are present in the recording.

        Analyzes the recording statistics to identify which data modalities
        contain actual data (non-zero lengths/counts).

        Returns:
            List of DataType enums representing the data modalities
            present in this recording.
        """
        data_types = []

        # Joint data
        if self.joint_positions.max_len > 0:
            data_types.append(DataType.JOINT_POSITIONS)
        if self.joint_velocities.max_len > 0:
            data_types.append(DataType.JOINT_VELOCITIES)
        if self.joint_torques.max_len > 0:
            data_types.append(DataType.JOINT_TORQUES)
        if self.joint_target_positions.max_len > 0:
            data_types.append(DataType.JOINT_TARGET_POSITIONS)

        # End-effector data
        if self.end_effector_states.max_len > 0:
            data_types.append(DataType.END_EFFECTORS)

        # Pose data
        if self.poses.max_len > 0:
            data_types.append(DataType.POSES)

        # Visual data
        if self.num_rgb_images > 0:
            data_types.append(DataType.RGB_IMAGE)
        if self.num_depth_images > 0:
            data_types.append(DataType.DEPTH_IMAGE)
        if self.num_point_clouds > 0:
            data_types.append(DataType.POINT_CLOUD)

        # Language data
        if self.max_language_length > 0:
            data_types.append(DataType.LANGUAGE)

        # Custom data
        if self.custom_data_stats:
            data_types.append(DataType.CUSTOM)

        return data_types


class ModelDevice(str, Enum):
    """Enumeration of device types for model training and inference."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


class ModelInitDescription(BaseModel):
    """Configuration specification for initializing Neuracore models.

    Defines the model architecture requirements including dataset characteristics,
    input/output data types, and prediction horizons for model initialization
    and training configuration.
    """

    dataset_description: DatasetDescription
    input_data_types: list[DataType]
    output_data_types: list[DataType]
    output_prediction_horizon: int = 1
    device: ModelDevice = ModelDevice.AUTO


class ModelPrediction(BaseModel):
    """Model inference output containing predictions and timing information.

    Represents the results of model inference including predicted outputs
    for each configured data type and optional timing information for
    performance monitoring.
    """

    outputs: dict[DataType, Any] = Field(default_factory=dict)
    prediction_time: Optional[float] = None


class SyncedDataset(BaseModel):
    """Represents a dataset of robot demonstrations.

    A Synchronized dataset groups related robot demonstrations together
    and maintains metadata about the collection as a whole.

    Attributes:
        id: Unique identifier for the synced dataset.
        parent_id: Unique identifier of the corresponding dataset.
        freq: Frequency at which dataset was proccessed.
        name: Human-readable name for the dataset.
        created_at: Unix timestamp of dataset creation.
        modified_at: Unix timestamp of last modification.
        description: Optional description of the dataset.
        recording_ids: List of recording IDs in this dataset
        num_demonstrations: Total number of demonstrations.
        total_duration_seconds: Total duration of all demonstrations.
        is_shared: Whether the dataset is shared with other users.
        metadata: Additional arbitrary metadata.
    """

    id: str
    parent_id: str
    freq: int
    name: str
    created_at: float
    modified_at: float
    description: Optional[str] = None
    recording_ids: list[str] = Field(default_factory=list)
    num_demonstrations: int = 0
    num_processed_demonstrations: int = 0
    total_duration_seconds: float = 0.0
    is_shared: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    dataset_description: DatasetDescription = Field(default_factory=DatasetDescription)
    all_data_types: dict[DataType, int] = Field(default_factory=dict)
    common_data_types: dict[DataType, int] = Field(default_factory=dict)


class Dataset(BaseModel):
    """Represents a dataset of robot demonstrations.

    A dataset groups related robot demonstrations together and maintains metadata
    about the collection as a whole.

    Attributes:
        id: Unique identifier for the dataset.
        name: Human-readable name for the dataset.
        created_at: Unix timestamp of dataset creation.
        modified_at: Unix timestamp of last modification.
        description: Optional description of the dataset.
        tags: List of tags for categorizing the dataset.
        recording_ids: List of recording IDs in this dataset
        demonstration_ids: List of demonstration IDs in this dataset.
        num_demonstrations: Total number of demonstrations.
        total_duration_seconds: Total duration of all demonstrations.
        size_bytes: Total size of all demonstrations.
        is_shared: Whether the dataset is shared with other users.
        metadata: Additional arbitrary metadata.
        synced_dataset_ids: List of synced dataset IDs in this dataset.
                            They point to synced datasets that synchronized
                            this dataset at a particular frequency.
    """

    id: str
    name: str
    created_at: float
    modified_at: float
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    recording_ids: list[str] = Field(default_factory=list)
    num_demonstrations: int = 0
    total_duration_seconds: float = 0.0
    size_bytes: int = 0
    is_shared: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    synced_dataset_ids: dict[str, Any] = Field(default_factory=dict)
    all_data_types: dict[DataType, int] = Field(default_factory=dict)
    common_data_types: dict[DataType, int] = Field(default_factory=dict)
    recording_ids_in_bucket: bool = False


class MessageType(str, Enum):
    """Enumerates the types of signaling messages for WebRTC handshakes.

    These types are used to identify the purpose of a message sent through
    the signaling server during connection establishment.
    """

    SDP_OFFER = "offer"  # Session Description Protocol (SDP) offer from the caller
    SDP_ANSWER = "answer"  # Session Description Protocol (SDP) answer from the callee
    ICE_CANDIDATE = "ice"  # Interactive Connectivity Establishment (ICE) candidate
    OPEN_CONNECTION = "open_connection"  # Request to open a new connection


class HandshakeMessage(BaseModel):
    """Represents a signaling message for the WebRTC handshake process.

    This message is exchanged between two peers via a signaling server to
    negotiate the connection details, such as SDP offers/answers and ICE
    candidates.

    Attributes:
        from_id: The unique identifier of the sender peer.
        to_id: The unique identifier of the recipient peer.
        data: The payload of the message, typically an SDP string or a JSON
              object with ICE candidate information.
        connection_id: The unique identifier for the connection session.
        type: The type of the handshake message, as defined by MessageType.
        id: A unique identifier for the message itself.
    """

    from_id: str
    to_id: str
    data: str
    connection_id: str
    type: MessageType
    id: str = Field(default_factory=lambda: uuid4().hex)


class VideoFormat(str, Enum):
    """Enumerates video format styles over a WebRTC connection."""

    # use a standard video track with negotiated codec this is more efficient
    WEB_RTC_NEGOTIATED = "WEB_RTC_NEGOTIATED"
    # uses neuracore's data URI format over a custom data channel
    NEURACORE_CUSTOM = "NEURACORE_CUSTOM"


class OpenConnectionRequest(BaseModel):
    """Represents a request to open a new WebRTC connection.

    Attributes:
        from_id: The unique identifier of the consumer peer.
        to_id: The unique identifier of the producer peer.
        robot_id: The unique identifier for the robot to be created.
        robot_instance: The identifier for the instance of the robot to connect to.
        video_format: The type of video the consumer expects to receive.
        id: the identifier for this connection request.
        created_at: when the request was created.
    """

    from_id: str
    to_id: str
    robot_id: str
    robot_instance: NonNegativeInt
    video_format: VideoFormat
    id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OpenConnectionDetails(BaseModel):
    """The details describing properties about the new connection.

    Attributes:
        connection_token: The token used for security to establish the connection.
        robot_id: The unique identifier for the robot to connect to
        robot_instance: The identifier for the instance of the robot to connect to.
        video_format: The type of video the consumer expects to receive.
    """

    connection_token: str
    robot_id: str
    robot_instance: NonNegativeInt
    video_format: VideoFormat


class StreamAliveResponse(BaseModel):
    """Represents the response from asserting a stream is alive.

    This is returned when a client pings a stream to keep it active.

    Attributes:
        resurrected: A boolean indicating if the stream was considered dead
                     and has been successfully resurrected by this request.
    """

    resurrected: bool


class RobotInstanceIdentifier(NamedTuple):
    """A tuple that uniquely identifies a robot instance.

    Attributes:
        robot_id: The unique identifier of the robot providing the stream.
        robot_instance: The specific instance number of the robot.
    """

    robot_id: str
    robot_instance: int


class TrackKind(str, Enum):
    """Enumerates the supported track kinds for streaming."""

    JOINTS = "joints"
    RGB = "rgb"
    DEPTH = "depth"
    LANGUAGE = "language"
    GRIPPER = "gripper"
    POINT_CLOUD = "point_cloud"
    POSE = "pose"
    CUSTOM = "custom"


class RobotStreamTrack(BaseModel):
    """Metadata for a robot's media stream track.

    This model holds all the necessary information to identify and manage
    a single media track (e.g., a video or audio feed) from a specific
    robot instance.

    Attributes:
        robot_id: The unique identifier of the robot providing the stream.
        robot_instance: The specific instance number of the robot.
        stream_id: The identifier for the overall media stream session.
        kind: The type of media track, typically 'audio' or 'video'.
        label: A human-readable label for the track (e.g., 'front_camera').
        mid: The media ID used in SDP, essential for WebRTC negotiation.
        id: A unique identifier for this track metadata object.
        created_at: The UTC timestamp when this track metadata was created.
    """

    robot_id: str
    robot_instance: NonNegativeInt
    stream_id: str
    kind: TrackKind
    label: str
    mid: str
    id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AvailableRobotInstance(BaseModel):
    """Represents a single, available instance of a robot.

    Attributes:
        robot_instance: The unique identifier for this robot instance.
        tracks: A dictionary of available media stream tracks for this instance.
        connections: The number of current connections to this instance.
    """

    robot_instance: NonNegativeInt
    # stream_id to list of tracks
    tracks: dict[str, list[RobotStreamTrack]]
    connections: int


class AvailableRobot(BaseModel):
    """Represents an available robot, including all its running instances.

    Attributes:
        robot_id: The unique identifier for the robot model/type.
        instances: A dictionary of all available instances for this robot,
                   keyed by instance ID.
    """

    robot_id: str
    instances: dict[int, AvailableRobotInstance]


class AvailableRobotCapacityUpdate(BaseModel):
    """Represents an update on the available capacity of all robots.

    This model is used to broadcast the current state of all available
    robots and their instances.

    Attributes:
        robots: A list of all available robots and their instances.
    """

    robots: list[AvailableRobot]


class BaseRecodingUpdatePayload(BaseModel):
    """Base payload for recording update notifications.

    Contains the minimum information needed to identify a recording
    and the robot instance it belongs to.
    """

    recording_id: str
    robot_id: str
    instance: NonNegativeInt


class RecodingRequestedPayload(BaseRecodingUpdatePayload):
    """Payload for recording request notifications.

    Contains information about who requested the recording and what
    data types should be captured.
    """

    created_by: str
    dataset_ids: list[str] = Field(default_factory=list)
    data_types: set[DataType] = Field(default_factory=set)


class RecordingStartPayload(RecodingRequestedPayload):
    """Payload for recording start notifications.

    Extends the request payload with the actual start timestamp
    when recording begins.
    """

    start_time: float


class RecordingNotificationType(str, Enum):
    """Types of recording lifecycle notifications."""

    INIT = "init"
    REQUESTED = "requested"
    START = "start"
    STOP = "stop"
    SAVED = "saved"
    DISCARDED = "discarded"
    EXPIRED = "expired"


class RecordingNotification(BaseModel):
    """Notification message for recording lifecycle events.

    Used to communicate recording state changes across the system,
    including initialization, start/stop events, and final disposition.
    """

    type: RecordingNotificationType
    payload: Union[
        RecordingStartPayload,
        RecodingRequestedPayload,
        list[Union[RecordingStartPayload, RecodingRequestedPayload]],
        BaseRecodingUpdatePayload,
    ]
    id: str = Field(default_factory=lambda: uuid4().hex)
