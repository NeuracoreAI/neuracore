from pydantic import BaseModel, Field


class NCData(BaseModel):
    timestamp: float


class JointData(NCData):
    values: dict[str, float]
    additional_values: dict[str, float] = Field(default_factory=dict)


class ActionData(NCData):
    values: dict[str, float]


class CameraData(NCData):
    frame_idx: int
    relative_timestamp: float


class GripperData(NCData):
    open_amounts: dict[str, float]


class SyncPoint(BaseModel):
    """Synchronized data point with joint, action, and image data."""

    timestamp: float
    joint_positions: JointData
    joint_velocities: JointData | None = None
    joint_torques: JointData | None = None
    grippers: GripperData | None = None
    actions: ActionData | None = None
    cameras: dict[str, CameraData] | None = None


class SyncedData(BaseModel):
    frames: list[SyncPoint]
    start_time: float
    end_time: float
    frame_count: int
