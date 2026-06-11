"""Utility functions for kinematics calculations."""

import numpy as np
import pink
import pinocchio
from pink.limits import ConfigurationLimit
from pink.tasks import FrameTask
from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation as R


class RobotUtils:
    """Utility class for kinematics calculations."""

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

    def _chain_joint_names(self, ee_frame: str) -> set[str]:
        """Return the set of joint names on the kinematic chain to ee_frame.

        Joints downstream of (or sibling to) ee_frame -- e.g. gripper fingers
        when ee_frame is the gripper mount link -- are excluded. IK solves only
        for the chain; including downstream joints in the result would surface
        their init values as if they were a solved state, masking the real
        motion that VISUAL_JOINT_POSITIONS / PARALLEL_GRIPPER_OPEN_AMOUNTS
        provide.
        """
        model = self.robot.model
        frame_id = model.getFrameId(ee_frame)
        parent_joint = model.frames[frame_id].parent
        chain: set[str] = set()
        while parent_joint > 0:
            chain.add(model.names[parent_joint])
            parent_joint = model.parents[parent_joint]
        return chain

    def _q_to_joint_dict(self, q: np.ndarray) -> dict[str, float]:
        """Map a pinocchio configuration vector to {joint_name: angle}.

        Handles multi-DOF joints (e.g. continuous joints stored as (cos, sin))
        by using each joint's idx_q/nq, then converting (cos, sin) -> angle via
        atan2 so the returned value is a single scalar per joint name.
        """
        model = self.robot.model
        result: dict[str, float] = {}
        for jid in range(1, model.njoints):
            joint = model.joints[jid]
            name = model.names[jid]
            idx = joint.idx_q
            if joint.nq == 1:
                result[name] = float(q[idx])
            elif joint.nq == 2:
                # SO(2) continuous joint stored as (cos, sin)
                result[name] = float(np.arctan2(q[idx + 1], q[idx]))
            # Floating/planar joints (nq>=3) are not returned as a scalar.
        return result

    def _safe_midpoint_q(self) -> np.ndarray:
        """Limit-midpoint q with continuous joints reset to identity (1, 0)."""
        model = self.robot.model
        q = (model.lowerPositionLimit + model.upperPositionLimit) / 2.0
        q = np.asarray(q, dtype=float).copy()
        for jid in range(1, model.njoints):
            joint = model.joints[jid]
            if joint.nq == 2:
                q[joint.idx_q] = 1.0
                q[joint.idx_q + 1] = 0.0
        return q

    def _joint_dict_to_q(self, joint_positions: dict[str, float]) -> np.ndarray:
        """Build a pinocchio q from a {joint_name: angle} dict.

        Unspecified joints default to the safe midpoint configuration. Continuous
        joints (nq=2) are encoded as (cos(angle), sin(angle)).
        """
        model = self.robot.model
        q = self._safe_midpoint_q()
        for jid in range(1, model.njoints):
            name = model.names[jid]
            if name not in joint_positions:
                continue
            joint = model.joints[jid]
            idx = joint.idx_q
            val = joint_positions[name]
            if joint.nq == 1:
                q[idx] = val
            elif joint.nq == 2:
                q[idx] = float(np.cos(val))
                q[idx + 1] = float(np.sin(val))
        return q

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
        full = self._q_to_joint_dict(np.asarray(configuration.q))
        chain = self._chain_joint_names(ee_frame)
        return {name: angle for name, angle in full.items() if name in chain}

    def joint_positions_to_end_effector_pose(
        self,
        joint_positions: dict[str, float],
        ee_frame: str,
    ) -> list[float]:
        """Compute end-effector pose from joint positions using forward kinematics.

        Args:
            joint_positions: Dictionary mapping joint names to joint positions.
            ee_frame: End-effector frame name.

        Returns:
            numpy array containing [x, y, z, qx, qy, qz, qw] (position and quaternion
            in xyzw order).
        """
        q = self._joint_dict_to_q(joint_positions)
        pinocchio.forwardKinematics(self.robot.model, self.robot.data, q)
        pinocchio.updateFramePlacements(self.robot.model, self.robot.data)

        frame_id = self.robot.model.getFrameId(ee_frame)
        if frame_id >= len(self.robot.data.oMf):
            raise ValueError(f"Unknown frame: {ee_frame}")

        placement = self.robot.data.oMf[frame_id]
        xyz = placement.translation
        quat_xyzw = R.from_matrix(placement.rotation).as_quat()

        return np.concatenate([xyz, quat_xyzw])

    def end_effector_to_joint_positions(
        self,
        end_effector_pose: list,
        ee_frame: str,
        prev_ik_solution: dict[str, float] | list[float] | None = None,
    ) -> dict[str, float]:
        """Convert end effector pose to joint positions using IK.

        Args:
            end_effector_pose: List containing [x, y, z, qx, qy, qz, qw]
            ee_frame: End-effector frame name
            prev_ik_solution: Previous IK solution to use as initial guess.
                Accepts either a {joint_name: angle} dict (preferred; safe
                for multi-DOF joints) or a raw q vector matching model.nq.

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

        # Pick the initial guess. Accept either a {joint_name: angle} dict
        # (preferred; multi-DOF safe) or a raw q vector of length model.nq.
        # Anything else (None, wrong shape, etc.) falls back to safe midpoint.
        if isinstance(prev_ik_solution, dict):
            q0 = self._joint_dict_to_q(prev_ik_solution)
        elif prev_ik_solution is not None and (
            (arr := np.asarray(prev_ik_solution, dtype=float)).shape
            == (self.robot.model.nq,)
        ):
            q0 = arr
        else:
            q0 = self._safe_midpoint_q()

        return self._run_ik(target_pose, ee_frame, q0)
