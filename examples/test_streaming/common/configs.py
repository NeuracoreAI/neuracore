"""Minimal config for neuracore stress tests."""

import os
from pathlib import Path

# URDF for visualization (10, 11). Set STRESS_TEST_URDF env or place
# `robot.urdf` in `stress_tests/urdf/`.
_STRESS_ROOT = Path(__file__).parent.parent
URDF_PATH = os.environ.get(
    "STRESS_TEST_URDF",
    str(_STRESS_ROOT / "piper_description" / "urdf" / "piper_description.urdf"),
)

# Match schema used by streaming scripts and neuracore
GRIPPER_LOGGING_NAME = "gripper"
CAMERA_LOGGING_NAME = "rgb"
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
