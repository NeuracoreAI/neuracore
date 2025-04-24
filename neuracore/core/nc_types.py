import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class NCData(BaseModel):
    timestamp: float = Field(default_factory=lambda: time.time())


class JointData(NCData):
    values: dict[str, float]
    additional_values: Optional[dict[str, float]] = None


class CameraData(NCData):
    frame_idx: int = 0  # Needed so we can index video after sync
    extrinsics: Optional[list[list[float]]] = None
    intrinsics: Optional[list[list[float]]] = None
    frame: Optional[Any] = None  # Only filled in when using dataset iter


class PoseData(NCData):
    pose: dict[str, list[float]]


class EndEffectorData(NCData):
    open_amounts: dict[str, float]


class PointCloudData(NCData):
    points: list[list[float]]
    rgb_points: Optional[list[list[int]]] = None
    extrinsics: Optional[list[list[float]]] = None
    intrinsics: Optional[list[list[float]]] = None


class LanguageData(NCData):
    text: str


class CustomData(NCData):
    data: Any


class SyncPoint(BaseModel):
    """Synchronized data point."""

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
    frames: list[SyncPoint]
    start_time: float
    end_time: float


class DataType(str, Enum):

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
    mean: list[float] = Field(default_factory=list)
    std: list[float] = Field(default_factory=list)
    count: list[int] = Field(default_factory=list)
    max_len: int = Field(default_factory=lambda data: len(data["mean"]))


class DatasetDescription(BaseModel):
    joint_positions: DataItemStats = Field(default_factory=lambda: DataItemStats())
    joint_velocities: DataItemStats = Field(default_factory=lambda: DataItemStats())
    joint_torques: DataItemStats = Field(default_factory=lambda: DataItemStats())
    joint_target_positions: DataItemStats = Field(
        default_factory=lambda: DataItemStats()
    )
    end_effector_states: DataItemStats = Field(default_factory=lambda: DataItemStats())
    poses: DataItemStats = Field(default_factory=lambda: DataItemStats())
    max_num_rgb_images: int = 0
    max_num_depth_images: int = 0
    max_num_point_clouds: int = 0
    max_language_length: int = 0

    def get_data_types(self) -> list[DataType]:
        data_types = []
        if self.joint_positions.max_len > 0:
            data_types.append(DataType.JOINT_POSITIONS)
        if self.joint_velocities.max_len > 0:
            data_types.append(DataType.JOINT_VELOCITIES)
        if self.joint_torques.max_len > 0:
            data_types.append(DataType.JOINT_TORQUES)
        if self.joint_target_positions.max_len > 0:
            data_types.append(DataType.JOINT_TARGET_POSITIONS)
        if self.max_num_rgb_images > 0:
            data_types.append(DataType.RGB_IMAGE)
        if self.max_num_depth_images > 0:
            data_types.append(DataType.DEPTH_IMAGE)
        if self.max_num_point_clouds > 0:
            data_types.append(DataType.POINT_CLOUD)
        if self.max_language_length > 0:
            data_types.append(DataType.LANGUAGE)
        return data_types


class RecordingDescription(BaseModel):
    joint_positions: DataItemStats = Field(default_factory=lambda: DataItemStats())
    joint_velocities: DataItemStats = Field(default_factory=lambda: DataItemStats())
    joint_torques: DataItemStats = Field(default_factory=lambda: DataItemStats())
    joint_target_positions: DataItemStats = Field(
        default_factory=lambda: DataItemStats()
    )
    end_effector_states: DataItemStats = Field(default_factory=lambda: DataItemStats())
    poses: DataItemStats = Field(default_factory=lambda: DataItemStats())
    num_rgb_images: int = 0
    num_depth_images: int = 0
    num_point_clouds: int = 0
    max_language_length: int = 0
    episode_length: int = 0

    def get_data_types(self) -> list[DataType]:
        data_types = []
        if self.joint_positions.max_len > 0:
            data_types.append(DataType.JOINT_POSITIONS)
        if self.joint_velocities.max_len > 0:
            data_types.append(DataType.JOINT_VELOCITIES)
        if self.joint_torques.max_len > 0:
            data_types.append(DataType.JOINT_TORQUES)
        if self.joint_target_positions.max_len > 0:
            data_types.append(DataType.JOINT_TARGET_POSITIONS)
        if self.num_rgb_images > 0:
            data_types.append(DataType.RGB_IMAGE)
        if self.num_depth_images > 0:
            data_types.append(DataType.DEPTH_IMAGE)
        if self.num_point_clouds > 0:
            data_types.append(DataType.POINT_CLOUD)
        if self.max_language_length > 0:
            data_types.append(DataType.LANGUAGE)
        return data_types


class ModelInitDescription(BaseModel):
    """Description of a Neuracore model."""

    dataset_description: DatasetDescription
    input_data_types: list[DataType]
    output_data_types: list[DataType]
    output_prediction_horizon: int = 1


class ModelPrediction(BaseModel):
    """Synchronized data point."""

    outputs: dict[DataType, Any] = Field(default_factory=dict)
    prediction_time: Optional[float] = None
