"""Shared robot / embodiment / hardware constants for ML integration tests.

Single source of truth for the bimanual-VX300s embodiment and the GPU/frequency
settings reused across the ml integration suite (training flow, inference,
resume, back-to-back, sync failure). Kept here so the tests share one definition
rather than each re-declaring it.
"""

import os
import sys

from neuracore_types import DataType, EmbodimentDescription

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_DIR = os.path.join(_THIS_DIR, "..", "..", "..", "..", "examples")
if _EXAMPLES_DIR not in sys.path:
    sys.path.append(_EXAMPLES_DIR)

# ruff: noqa: E402
from common.base_env import BimanualViperXTask

NC_CAM_NAME = "rgb_angle"
MJ_CAM_NAME = "angle"
JOINT_NAMES = (
    BimanualViperXTask.LEFT_ARM_JOINT_NAMES + BimanualViperXTask.RIGHT_ARM_JOINT_NAMES
)
GRIPPER_NAMES = ["left_gripper", "right_gripper"]
POSE_SENSOR_NAME = "tcp"
LANGUAGE_LABEL = "instruction"


def _indexed_names(names: list[str] | tuple[str, ...]) -> dict[int, str]:
    return {index: name for index, name in enumerate(names)}


# Training/Inference robot (VX300s) embodiment descriptions
INPUT_EMBODIMENT_DESCRIPTION: EmbodimentDescription = {
    DataType.RGB_IMAGES: {0: NC_CAM_NAME},
    DataType.JOINT_POSITIONS: _indexed_names(names=JOINT_NAMES),
    DataType.LANGUAGE: {0: LANGUAGE_LABEL},
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: _indexed_names(names=GRIPPER_NAMES),
}
OUTPUT_EMBODIMENT_DESCRIPTION: EmbodimentDescription = {
    DataType.JOINT_TARGET_POSITIONS: _indexed_names(names=JOINT_NAMES),
    DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS: _indexed_names(names=GRIPPER_NAMES),
}

INPUT_DATA_TYPES = list(INPUT_EMBODIMENT_DESCRIPTION.keys())
OUTPUT_DATA_TYPES = list(OUTPUT_EMBODIMENT_DESCRIPTION.keys())

ROBOT_NAME = "integration_test_robot"
GPU_TYPE = "NVIDIA_TESLA_V100"
NUM_GPUS = 1
FREQUENCY = 20
