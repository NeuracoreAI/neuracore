from typing import Any, Optional

from pydantic import BaseModel


class NCData(BaseModel):
    timestamp: float


class JointData(NCData):
    values: dict[str, float]
    additional_values: Optional[dict[str, float]] = None


class ActionData(NCData):
    values: dict[str, float]


class CameraMetaData(NCData):
    frame_idx: int = 0  # Needed so we can index video after sync
    extrinsics: Optional[list[list[float]]] = None
    intrinsics: Optional[list[list[float]]] = None


class GripperData(NCData):
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
    grippers: Optional[GripperData] = None
    actions: Optional[ActionData] = None
    rgb_images: Optional[dict[str, CameraMetaData]] = None
    depth_images: Optional[dict[str, CameraMetaData]] = None
    point_clouds: Optional[dict[str, PointCloudData]] = None
    custom_data: Optional[dict[str, CustomData]] = None


class SyncedData(BaseModel):
    frames: list[SyncPoint]
    start_time: float
    end_time: float
