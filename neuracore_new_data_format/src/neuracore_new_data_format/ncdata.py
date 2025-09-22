from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(kw_only=True, slots=True)
class NCData:
    """Base class for all Neuracore data with automatic timestamping.

    Provides a common base for all data types in the system with automatic
    timestamp generation for temporal synchronization and data ordering.
    """

    timestamp: float


@dataclass(kw_only=True, slots=True)
class JointData(NCData):
    """Robot joint state data including positions, velocities, or torques.

    Represents joint-space data for robotic systems with support for named
    joints and additional auxiliary values. Used for positions, velocities,
    torques, and target positions.
    """

    values: dict[str, float]
    additional_values: Optional[dict[str, float]] = None


@dataclass(kw_only=True, slots=True)
class CameraMetadata(NCData):
    """Camera sensor data including images and calibration information.

    Contains image data along with camera intrinsic and extrinsic parameters
    for 3D reconstruction and computer vision applications. The frame field
    is populated during dataset iteration for efficiency.
    """

    camera_id: str
    frame_idx: int = 0  # Needed so we can index video after sync
    extrinsics: Optional[list[list[float]]] = None
    intrinsics: Optional[list[list[float]]] = None


@dataclass(kw_only=True, slots=True)
class CameraData(CameraMetadata):
    """Camera sensor data including images and calibration information.

    Contains image data along with camera intrinsic and extrinsic parameters
    for 3D reconstruction and computer vision applications. The frame field
    is populated during dataset iteration for efficiency.
    """

    frame: np.ndarray


@dataclass(kw_only=True, slots=True)
class LanguageData(NCData):
    """Natural language instruction or description data.

    Contains text-based information such as task descriptions, voice commands,
    or other linguistic data associated with robot demonstrations.
    """

    text: str


@dataclass(kw_only=True, slots=True)
class CustomData(NCData):
    """Generic container for application-specific data types.

    Provides a flexible way to include custom sensor data or application-specific
    information that doesn't fit into the standard data categories.
    """

    name: str
    data: Any
