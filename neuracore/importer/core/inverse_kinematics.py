"""Utility functions for Inverse Kinematics (IK) calculations."""

import numpy as np
import pink
import pinocchio
from pink.limits import ConfigurationLimit
from pink.tasks import FrameTask
from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation as R


class InverseKinematics:
    """Utility class for Inverse Kinematics (IK) calculations."""

    def __init__(self, urdf_path: str, packages_dir: str):
        """Initialize robot with specified URDF path.

        Args:
            urdf_path: Path to the URDF file.
            packages_dir: Path to the packages directory.
        """
        self.urdf_path = urdf_path
        self.packages_dir = packages_dir
        self.robot = self.build_pinnochio_robot()

    def build_pinnochio_robot(self) -> RobotWrapper:
        """Build Pinnochio robot model from URDF."""
        robot = RobotWrapper.BuildFromURDF(self.urdf_path, self.packages_dir)
        return robot

    def _run_ik(
        self,
        target_pose: pinocchio.SE3,
        ee_frame: str,
        q0: np.ndarray,
    ) -> dict[str, float]:
        """Run IK to find joint positions for a given target pose.

        Args:
            target_pose: Target SE3 pose
            ee_frame: End-effector frame name
            q0: Initial joint configuration

        Returns:
            Dictionary mapping joint names to joint positions

        Raises:
            ValueError: If IK does not converge
        """
        ee_task = FrameTask(ee_frame, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        ee_task.set_target(target_pose)

        config_limit = ConfigurationLimit(self.robot.model, config_limit_gain=0.5)
        configuration = pink.Configuration(self.robot.model, self.robot.data, q0)

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
            q_out = pinocchio.integrate(self.robot.model, configuration.q, dv * dt)
            configuration = pink.Configuration(self.robot.model, self.robot.data, q_out)
            pinocchio.updateFramePlacements(self.robot.model, self.robot.data)
            error_norm = np.linalg.norm(ee_task.compute_error(configuration))
            nb_steps += 1

        if error_norm > stop_thres:
            raise ValueError("IK did not converge")
        return {
            j: q.item() for j, q in zip(self.robot.model.names[1:], configuration.q[:])
        }

    def end_effector_to_joint_positions(
        self,
        end_effector_pose: list,
        ee_frame: str,
        prev_ik_solution: list[float] | None = None,
    ) -> dict[str, float]:
        """Convert end effector pose to joint positions using IK.

        Args:
            end_effector_pose: List containing [x, y, z, qw, qx, qy, qz]
            ee_frame: End-effector frame name
            prev_ik_solution: Previous IK solution to use as initial guess

        Returns:
            Dictionary mapping joint names to joint positions

        Raises:
            ValueError: If IK does not converge
        """
        xyz = end_effector_pose[:3]
        quat = np.array(end_effector_pose[3:7])
        quat = quat / np.linalg.norm(quat)

        # Convert to SE3 transform
        rotation_matrix = R.from_quat(quat).as_matrix()
        target_pose = pinocchio.SE3(rotation_matrix, np.array(xyz))

        # Initialise with previous IK solution if available
        if prev_ik_solution is not None:
            try:
                q0 = np.array(prev_ik_solution)
                return self._run_ik(target_pose, ee_frame, q0)
            except ValueError:
                # Failed to find IK solution by initialising with previous solution
                # Try to initialise with midpoint of joint limits
                pass

        # Initialise with midpoint of joint limits
        q0 = (
            self.robot.model.lowerPositionLimit + self.robot.model.upperPositionLimit
        ) / 2
        try:
            return self._run_ik(target_pose, ee_frame, q0)
        except ValueError:
            raise
