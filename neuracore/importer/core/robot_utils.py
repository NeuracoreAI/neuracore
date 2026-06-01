"""Utility functions for kinematics calculations."""

import numpy as np
import pinocchio
from scipy.spatial.transform import Rotation as R


class RobotUtils:
    """Utility class for kinematics calculations."""

    def __init__(self, urdf_path: str, packages_dir: str):
        """Initialize robot with specified URDF path.

        Args:
            urdf_path: Path to the URDF file.
            packages_dir: Path to the packages directory.
        """
        self.model = pinocchio.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

    def _run_ik(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        ee_frame: str,
        q0: np.ndarray,
    ) -> dict[str, float]:
        """Run DLS IK to find joint positions for a given target pose.

        Uses damped least squares: J^T (J J^T + λI)^{-1} error.

        Args:
            target_pos: Target position [x, y, z]
            target_rot: Target rotation matrix (3x3)
            ee_frame: End-effector frame name
            q0: Initial joint configuration

        Returns:
            Dictionary mapping joint names to joint positions

        Raises:
            ValueError: If IK does not converge
        """
        frame_id = self.model.getFrameId(ee_frame)
        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        M_target = pinocchio.SE3(target_rot, target_pos)

        stop_thres = 1e-6
        max_steps = 200
        damping = 1e-4
        n_restarts = 5

        rng = np.random.default_rng(0)
        candidates = [q0] + [rng.uniform(lower, upper) for _ in range(n_restarts - 1)]

        error_norm = float("inf")
        for q_init in candidates:
            q = np.clip(q_init, lower, upper)
            for _ in range(max_steps):
                pinocchio.computeJointJacobians(self.model, self.data, q)
                pinocchio.updateFramePlacements(self.model, self.data)

                err = pinocchio.log6(
                    self.data.oMf[frame_id].inverse() * M_target
                ).vector
                error_norm = float(np.linalg.norm(err))

                if error_norm < stop_thres:
                    break

                J = pinocchio.getFrameJacobian(
                    self.model, self.data, frame_id, pinocchio.ReferenceFrame.LOCAL
                )
                J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(6))
                q = np.clip(q + J_pinv @ err, lower, upper)

            if error_norm < stop_thres:
                break

        if error_norm >= stop_thres:
            raise ValueError("IK did not converge")

        return {name: float(q[i]) for i, name in enumerate(self.model.names[1:])}

    def joint_positions_to_end_effector_pose(
        self,
        joint_positions: dict[str, float],
        ee_frame: str,
    ) -> np.ndarray:
        """Compute end-effector pose from joint positions using forward kinematics.

        Args:
            joint_positions: Dictionary mapping joint names to joint positions.
            ee_frame: End-effector frame name.

        Returns:
            numpy array containing [x, y, z, qx, qy, qz, qw] (position and quaternion
            in xyzw order).
        """
        q_default = pinocchio.neutral(self.model)
        q = np.array([
            joint_positions.get(name, q_default[i])
            for i, name in enumerate(self.model.names[1:])
        ])
        pinocchio.forwardKinematics(self.model, self.data, q)
        pinocchio.updateFramePlacements(self.model, self.data)

        frame_id = self.model.getFrameId(ee_frame)
        if frame_id >= len(self.data.oMf):
            raise ValueError(f"Unknown frame: {ee_frame}")

        placement = self.data.oMf[frame_id]
        quat_xyzw = R.from_matrix(placement.rotation).as_quat()
        return np.concatenate([placement.translation, quat_xyzw])

    def end_effector_to_joint_positions(
        self,
        end_effector_pose: list,
        ee_frame: str,
        prev_ik_solution: list[float] | None = None,
    ) -> dict[str, float]:
        """Convert end effector pose to joint positions using IK.

        Args:
            end_effector_pose: List containing [x, y, z, qx, qy, qz, qw]
            ee_frame: End-effector frame name
            prev_ik_solution: Previous IK solution to use as initial guess

        Returns:
            Dictionary mapping joint names to joint positions

        Raises:
            ValueError: If IK does not converge
        """
        xyz = np.array(end_effector_pose[:3])
        quat = np.array(end_effector_pose[3:7])
        quat /= np.linalg.norm(quat)
        rot = R.from_quat(quat).as_matrix()

        if prev_ik_solution is not None:
            try:
                return self._run_ik(xyz, rot, ee_frame, np.array(prev_ik_solution))
            except ValueError:
                pass

        q0 = (self.model.lowerPositionLimit + self.model.upperPositionLimit) / 2
        return self._run_ik(xyz, rot, ee_frame, q0)
