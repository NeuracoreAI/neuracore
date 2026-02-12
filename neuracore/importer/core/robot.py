"""Utility functions for extracting and processing robot information."""

import logging
import tempfile
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from pathlib import Path

import numpy as np
import pink
import pinocchio
from pink.limits import ConfigurationLimit
from pink.tasks import FrameTask
from pinocchio.robot_wrapper import RobotWrapper
from robot_descriptions.loaders.pinocchio import load_robot_description
from scipy.spatial.transform import Rotation as R

from neuracore.core.mjcf_to_urdf import convert as mjcf_to_urdf

logger = logging.getLogger("RobotUtils")
TEMP_DIR = tempfile.TemporaryDirectory()


@dataclass
class RobotInfo:
    """Robot information for registration with Neuracore."""

    name: str
    urdf_path: str | None = None
    mjcf_path: str | None = None
    description: str = ""


class RobotType(Enum):
    """Common robots used, mapping to robot description modules.

    For more robot descriptions, see:
    https://github.com/robot-descriptions/robot_descriptions.py
    """

    FRANKA = ("Franka Panda", "panda_description")
    FRANKA_2F85 = ("Franka Panda with Robotiq 2f85", "panda_description")
    KUKA = ("Kuka iiwa 7", "iiwa7_description")
    WIDOW_X = ("Widow X", "widow_mj_description")
    GOOGLE = ("Google Robot", "google_robot_mj_description")
    KINOVA_GEN2 = ("Kinova Gen 2", "gen2_jaco_description")
    UR5_2F85 = ("UR5 with Robotiq 2f85", "ur5_2f85_description")
    XARM7 = ("UFACTORY xArm7", "xarm7_mj_description")
    PR2 = ("PR2", "pr2_description")
    SAWYER = ("Sawyer", "sawyer_mj_description")
    UNITREE_H1 = ("Unitree H1", "h1_description")


class Robot:
    """Robot utility class for processing robot information and kinematics."""

    def __init__(self, robot_type: RobotType):
        """Initialize robot with specified robot type.

        Args:
            robot_type: Type of robot to initialize.
        """
        self.robot_type = robot_type
        self.robot_info = self._get_robot_info()
        self._pinnochio_robot: RobotWrapper | None = (
            None  # Needs to be init in the processes
        )
        self._joint_names: list[str] | None = None
        self._joint_limits: dict[str, tuple[float, float]] | None = None

    def _get_robot_info(self) -> RobotInfo:
        """Get paths to common robot description files.

        Args:
            robot_type: Type of robot ("franka", "ur5", etc.)

        Returns:
            Dictionary with module, urdf_path and/or mjcf_path keys
        """
        robot_name, description_module = self.robot_type.value
        desc = import_module(f"robot_descriptions.{description_module}")
        urdf_path = None
        if hasattr(desc, "URDF_PATH"):
            urdf_path = Path(desc.URDF_PATH)
        else:
            mjcf_path = str(Path(desc.MJCF_PATH))
            urdf_path = Path(TEMP_DIR.name) / f"{description_module}_model.urdf"
            mjcf_to_urdf(mjcf_path, urdf_path, asset_file_prefix="meshes/")

        # Define robot information
        return RobotInfo(
            name=robot_name,
            urdf_path=str(urdf_path),
        )

    @property
    def pinnochio_robot(self) -> RobotWrapper:
        """Get Pinnochio robot model."""
        if self._pinnochio_robot is None:
            self._pinnochio_robot = load_robot_description(self.robot_type.value[1])
        return self._pinnochio_robot

    @property
    def joint_names(self) -> list[str]:
        """Get list of joint names."""
        if self._joint_names is None:
            self._joint_names = list(self.pinnochio_robot.model.names[1:])
        return self._joint_names

    @property
    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """Get dictionary of joint limits."""
        if self._joint_limits is None:
            pm = self.pinnochio_robot.model
            self._joint_limits = {
                jname: (pm.lowerPositionLimit[i], pm.upperPositionLimit[i])
                for i, jname in enumerate(self.joint_names)
            }
        return self._joint_limits

    def gripper_open_width_to_open_amount(
        self, gripper_open_width: float, gripper_joint_names: list[str]
    ) -> float:
        """Convert gripper joint positions to open amount.

        Args:
            gripper_joint_names: List of gripper joint names
            gripper_joint_positions: Dictionary mapping gripper
                joint names to position values

        Returns:
            Amount to open gripper (0-1)
        """
        joint_name = gripper_joint_names[0]  # Assume joints are symmetric
        min_pos, max_pos = self.joint_limits[joint_name]
        joint_pos = np.clip(gripper_open_width / 2.0, min_pos, max_pos)
        assert min_pos <= joint_pos <= max_pos, "Gripper joint position out of range"
        open_amount = (joint_pos - min_pos) / (max_pos - min_pos)
        assert 0.0 <= open_amount <= 1.0, "Gripper open amount out of range"
        return open_amount

    def process_gripper_state(
        self,
        gripper_joint_names: list[str],
        gripper_open_amount: float | None = None,
        gripper_open_width: float | None = None,
    ) -> dict[str, float]:
        """Process gripper state to get joint positions.

        Args:
            gripper_joint_names: List of gripper joint names
            gripper_open_amount: Amount to open gripper (0-1)
            gripper_open_width: Width to open gripper (meters)

        Returns:
            Dictionary mapping gripper joint names to position values
        """
        if gripper_open_amount is not None and gripper_open_width is not None:
            raise ValueError(
                "Only one of gripper_open_amount or "
                "gripper_open_width should be provided"
            )
        if gripper_open_amount is None and gripper_open_width is None:
            raise ValueError(
                "One of gripper_open_amount or gripper_open_width should be provided"
            )
        joint_positions = {}
        for joint_name in gripper_joint_names:
            min_pos, max_pos = self.joint_limits[joint_name]

            # Calculate joint position based on input value
            if gripper_open_amount is not None:
                assert (
                    0 <= gripper_open_amount <= 1
                ), "Gripper open amount should be in [0, 1]"
                # Treat as percentage open
                joint_pos = min_pos + (max_pos - min_pos) * gripper_open_amount
            else:
                assert isinstance(
                    gripper_open_width, float
                ), "Expected gripper_open_width to be float instance"
                # Treat as absolute width (usually in meters)
                # For typical grippers, divide by 2 (half width on each finger)
                # But ensure result is within limits
                joint_pos = np.clip(gripper_open_width / 2.0, min_pos, max_pos)
            joint_positions[joint_name] = float(joint_pos)
        return joint_positions

    def tcp_to_joint_positions(self, tcp_pose: list, ee_frame: str) -> dict[str, float]:
        """Convert TCP pose to joint positions using IK.

        Args:
            tcp_pose: List containing [x, y, z, qw, qx, qy, qz]
            ee_frame: End-effector frame name

        Returns:
            List of joint positions
        """
        # TODO: we need a way to get smooth trajectories out

        robot = self.pinnochio_robot
        xyz = tcp_pose[:3]
        quat = np.array(tcp_pose[3:7])
        quat = quat / np.linalg.norm(quat)

        # Convert to SE3 transform
        rotation_matrix = R.from_quat(quat).as_matrix()
        target_pose = pinocchio.SE3(rotation_matrix, np.array(xyz))

        # Create frame task for end-effector
        ee_task = FrameTask(ee_frame, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        ee_task.set_target(target_pose)

        # Create configuration limit
        config_limit = ConfigurationLimit(robot.model, config_limit_gain=0.5)

        # Initialize configuration
        q0 = (robot.model.lowerPositionLimit + robot.model.upperPositionLimit) / 2
        configuration = pink.Configuration(robot.model, robot.data, q0)

        # IK parameters
        dt = 1e-2
        stop_thres = 1e-6
        max_steps = 1000

        # Run IK
        error_norm = np.linalg.norm(ee_task.compute_error(configuration))
        nb_steps = 0

        while error_norm > stop_thres and nb_steps < max_steps:
            dv = pink.solve_ik(
                configuration,
                tasks=[ee_task],
                limits=[config_limit],
                dt=dt,
                damping=1e-6,
                solver="quadprog",
            )
            q_out = pinocchio.integrate(robot.model, configuration.q, dv * dt)
            configuration = pink.Configuration(robot.model, robot.data, q_out)
            pinocchio.updateFramePlacements(robot.model, robot.data)
            error_norm = np.linalg.norm(ee_task.compute_error(configuration))
            nb_steps += 1

        if error_norm > stop_thres:
            raise ValueError("IK did not converge")
        return {j: q.item() for j, q in zip(robot.model.names[1:], configuration.q[:])}
