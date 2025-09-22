import time
from collections import defaultdict
from typing import Any, Optional

import numpy as np

from neuracore_new_data_format.ncdata import (
    CameraData,
    CustomData,
    JointData,
    LanguageData,
)
from neuracore_new_data_format.recording import RecordingFactory, Recording


class Dataset:
    def __init__(self, name: str, recording_factory: RecordingFactory):
        self.name = name
        self.recording_factory = recording_factory
        self.recording: None | Recording = None
        self.recording_count = 0
        self.frame_index: defaultdict[str, int] = defaultdict(lambda: 0)

    def start_recording(self) -> None:
        self.recording = self.recording_factory.create_recording(
            name=f"{self.name}-{self.recording_count}"
        )
        self.recording_count += 1
        return self.recording

    def stop_recording(self):
        if self.recording:
            self.recording.stop()
            self.recording = None

    def log_custom_data(
        self, name: str, data: Any, timestamp: Optional[float] = None
    ) -> None:
        if not self.recording:
            return

        self.recording.log_data(
            data_type="custom",
            data=CustomData(name=name, timestamp=timestamp or time.time(), data=data),
        )

    def log_joint_data(
        self,
        data_type: str,
        joint_data: dict[str, float],
        additional_urdf_data: Optional[dict[str, float]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        if not self.recording:
            return

        self.recording.log_data(
            data_type=data_type,
            data=JointData(
                timestamp=timestamp or time.time(),
                values=joint_data,
                additional_values=additional_urdf_data,
            ),
        )

    def log_language_data(
        self, language: str, timestamp: Optional[float] = None
    ) -> None:
        if not self.recording:
            return

        self.recording.log_data(
            data_type="language",
            data=LanguageData(timestamp=timestamp or time.time(), text=language),
        )

    def log_rgb(
        self,
        camera_id: str,
        image: np.ndarray,
        extrinsics: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        if not self.recording:
            return

        index = self.frame_index[camera_id]
        self.frame_index[camera_id] += 1

        self.recording.log_data(
            data_type="rgb",
            data=CameraData(
                camera_id=camera_id,
                timestamp=timestamp or time.time(),
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                frame_idx=index,
                frame=image,
            ),
        )
