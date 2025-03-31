from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def _get_max_num_items(data: Optional[list[float]]) -> int:
    return len(data) if data is not None else 0


class NCData(BaseModel):
    timestamp: float


class JointData(NCData):
    values: dict[str, float]
    additional_values: Optional[dict[str, float]] = None


class ActionData(NCData):
    values: dict[str, float]


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


class LanguageData(NCData):
    text: str


class CustomData(NCData):
    data: Any


class SyncPoint(BaseModel):
    """Synchronized data point."""

    timestamp: float
    joint_positions: Optional[JointData] = None
    joint_velocities: Optional[JointData] = None
    joint_torques: Optional[JointData] = None
    end_effectors: Optional[EndEffectorData] = None
    poses: Optional[dict[str, PoseData]] = None
    actions: Optional[ActionData] = None
    rgb_images: Optional[dict[str, CameraData]] = None
    depth_images: Optional[dict[str, CameraData]] = None
    point_clouds: Optional[dict[str, PointCloudData]] = None
    language_data: Optional[LanguageData] = None
    custom_data: Optional[dict[str, CustomData]] = None


class SyncedData(BaseModel):
    frames: list[SyncPoint]
    start_time: float
    end_time: float


class DataType(Enum):

    # Robot state
    JOINT_POSITIONS = "joint_positions"
    JOINT_VELOCITIES = "joint_velocities"
    JOINT_TORQUES = "joint_torques"

    # Actions
    ACTIONS = "actions"

    # Vision
    RGB_IMAGE = "rgb_image"
    DEPTH_IMAGE = "depth_image"
    POINT_CLOUD = "point_cloud"


class DataItemStats(BaseModel):
    mean: list[float] = Field(default_factory=list)
    std: list[float] = Field(default_factory=list)
    max_len: int = Field(default_factory=lambda data: len(data["mean"]))


class DatasetDescription(BaseModel):
    actions: DataItemStats = Field(default_factory=lambda: DataItemStats())
    joint_positions: DataItemStats = Field(default_factory=lambda: DataItemStats())
    joint_velocitys: DataItemStats = Field(default_factory=lambda: DataItemStats())
    joint_torques: DataItemStats = Field(default_factory=lambda: DataItemStats())
    end_effector_states: DataItemStats = Field(default_factory=lambda: DataItemStats())
    poses: DataItemStats = Field(default_factory=lambda: DataItemStats())
    max_num_rgb_images: int = 0
    max_num_depth_images: int = 0
    max_num_point_clouds: int = 0

    def get_data_types(self) -> list[DataType]:
        data_types = []
        if self.joint_positions.max_len > 0:
            data_types.append(DataType.JOINT_POSITIONS)
        if self.joint_velocitys.max_len > 0:
            data_types.append(DataType.JOINT_VELOCITIES)
        if self.joint_torques.max_len > 0:
            data_types.append(DataType.JOINT_TORQUES)
        if self.actions.max_len > 0:
            data_types.append(DataType.ACTIONS)
        if self.max_num_rgb_images > 0:
            data_types.append(DataType.RGB_IMAGE)
        if self.max_num_depth_images > 0:
            data_types.append(DataType.DEPTH_IMAGE)
        if self.max_num_point_clouds > 0:
            data_types.append(DataType.POINT_CLOUD)
        return data_types


class ModelInitDescription(BaseModel):
    """Description of a Neuracore model."""

    dataset_description: DatasetDescription
    action_prediction_horizon: int = 1
