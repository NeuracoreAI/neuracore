"""Transfer cube tasks with joint and end-effector control modes."""

from pathlib import Path
from typing import List, Tuple

import mujoco
import numpy as np

from .base_env import BimanualViperXTask, Observation

# Global variable for box pose coordination
BOX_POSE: List[np.ndarray] = [None]

THIS_DIR = Path(__file__).parent.resolve()
ROBOT_ASSETS = THIS_DIR / "assets" / "robots"
VX300S_DIR = ROBOT_ASSETS / "vx300s"
BIMANUAL_VIPERX_MJCF_PATH = VX300S_DIR / "bimanual_viperx_transfer_cube.xml"
BIMANUAL_EE_VIPERX_MJCF_PATH = VX300S_DIR / "bimanual_viperx_ee_transfer_cube.xml"
BIMANUAL_VIPERX_URDF_PATH = VX300S_DIR / "bimanual_viperx_transfer_cube.urdf"


class TransferCubeTask(BimanualViperXTask):
    """Transfer cube task with joint position control."""

    def __init__(self) -> None:
        """Initialize joint control transfer cube task."""
        super().__init__(str(BIMANUAL_VIPERX_MJCF_PATH))

    def before_step(self, action: np.ndarray) -> None:
        """Process joint control action before stepping.

        Args:
            action: 14-element action array
                [left_arm(6), left_grip(1), right_arm(6), right_grip(1)].
        """
        left_arm_action = action[:6]
        right_arm_action = action[7:13]
        normalized_left_gripper = action[6]
        normalized_right_gripper = action[13]

        left_gripper = self.unnormalize_gripper_position(normalized_left_gripper)
        right_gripper = self.unnormalize_gripper_position(normalized_right_gripper)

        full_left_gripper = [left_gripper, -left_gripper]
        full_right_gripper = [right_gripper, -right_gripper]

        env_action = np.concatenate([
            left_arm_action,
            full_left_gripper,
            right_arm_action,
            full_right_gripper,
        ])

        self.data.ctrl[:] = env_action

    def initialize_episode(self) -> None:
        """Initialize episode with robot and box placement."""
        # Set initial joint positions
        self.data.qpos[:16] = self.START_ARM_POSE
        self.data.ctrl[:] = self.START_ARM_POSE

        # Set box pose if specified globally
        if BOX_POSE[0] is not None:
            self.data.qpos[-7:] = BOX_POSE[0]

        # Let environment settle
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

    def get_env_state(self) -> np.ndarray:
        """Get box pose as environment state.

        Returns:
            Box pose array (position + quaternion).
        """
        return self.data.qpos.copy()[16:]

    def get_reward(self) -> float:
        """Calculate reward based on contact conditions.

        Returns:
            Reward value (0-4 based on task progress).
        """
        contact_pairs = self._get_all_contact_pairs()

        touch_left_gripper = (
            "red_box",
            "vx300s_left/10_left_gripper_finger",
        ) in contact_pairs
        touch_right_gripper = (
            "red_box",
            "vx300s_right/10_right_gripper_finger",
        ) in contact_pairs
        touch_table = ("red_box", "table") in contact_pairs

        if touch_left_gripper and not touch_table:  # Successful transfer
            return 4
        elif touch_left_gripper:  # Attempted transfer
            return 3
        elif touch_right_gripper and not touch_table:  # Lifted
            return 2
        elif touch_right_gripper:  # Grasped
            return 1
        else:
            return 0

    def _get_all_contact_pairs(self) -> List[Tuple[str, str]]:
        """Get all current contact pairs in simulation.

        Returns:
            List of (geom1_name, geom2_name) contact pairs.
        """
        contact_pairs = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
            )
            geom2_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
            )
            if geom1_name and geom2_name:
                contact_pairs.append((geom1_name, geom2_name))
        return contact_pairs


class TransferCubeEETask(BimanualViperXTask):
    """Transfer cube task with end-effector mocap control."""

    def __init__(self) -> None:
        """Initialize end-effector control transfer cube task."""
        super().__init__(str(BIMANUAL_EE_VIPERX_MJCF_PATH))

    def before_step(self, action: np.ndarray) -> None:
        """Process mocap control action before stepping.

        Args:
            action: 16-element action array
                [left_pose(7), left_grip(1), right_pose(7), right_grip(1)].
        """
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        # Set mocap positions and orientations
        mocap_left_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "mocap_left"
        )
        if mocap_left_id != -1:
            self.data.mocap_pos[0] = action_left[:3]
            self.data.mocap_quat[0] = action_left[3:7]

        mocap_right_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "mocap_right"
        )
        if mocap_right_id != -1:
            self.data.mocap_pos[1] = action_right[:3]
            self.data.mocap_quat[1] = action_right[3:7]

        # Set gripper controls
        g_left_ctrl = self.unnormalize_gripper_position(action_left[7])
        g_right_ctrl = self.unnormalize_gripper_position(action_right[7])
        self.data.ctrl[:] = np.array(
            [g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]
        )

    def initialize_robots(self) -> None:
        """Initialize robot positions and mocap targets."""
        # Reset joint positions
        self.data.qpos[:16] = self.START_ARM_POSE

        # Reset mocap to align with end effector
        self.data.mocap_pos[0] = [-0.31718881, 0.5, 0.29525084]
        self.data.mocap_quat[0] = [1, 0, 0, 0]
        self.data.mocap_pos[1] = [0.31718881, 0.49999888, 0.29525084]
        self.data.mocap_quat[1] = [1, 0, 0, 0]

        # Reset gripper control to closed position
        close_gripper_control = np.array([
            self.GRIPPER_POSITION_CLOSE,
            -self.GRIPPER_POSITION_CLOSE,
            self.GRIPPER_POSITION_CLOSE,
            -self.GRIPPER_POSITION_CLOSE,
        ])
        self.data.ctrl[:] = close_gripper_control

        # Let environment settle
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

    def initialize_episode(self) -> None:
        """Initialize episode with randomized box placement."""
        self.initialize_robots()

        # Randomize box position
        cube_pose = self.sample_box_pose()
        box_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "red_box_joint"
        )
        if box_joint_id != -1:
            box_start_idx = self.model.jnt_qposadr[box_joint_id]
            self.data.qpos[box_start_idx : box_start_idx + 7] = cube_pose

    def get_env_state(self) -> np.ndarray:
        """Get box pose as environment state.

        Returns:
            Box pose array (position + quaternion).
        """
        return self.data.qpos.copy()[16:]

    def get_observation(self) -> Observation:
        """Get observation including mocap poses and gripper controls.

        Returns:
            Enhanced observation with mocap and gripper data.
        """
        obs = super().get_observation()

        # Add mocap poses
        obs.mocap_pose_left = np.concatenate(
            [self.data.mocap_pos[0].copy(), self.data.mocap_quat[0].copy()]
        )
        obs.mocap_pose_right = np.concatenate(
            [self.data.mocap_pos[1].copy(), self.data.mocap_quat[1].copy()]
        )
        obs.gripper_ctrl = self.data.ctrl.copy()

        return obs

    def get_reward(self) -> float:
        """Calculate reward based on contact conditions.

        Returns:
            Reward value (0-4 based on task progress).
        """
        contact_pairs = self._get_all_contact_pairs()

        touch_left_gripper = (
            "red_box",
            "vx300s_left/10_left_gripper_finger",
        ) in contact_pairs
        touch_right_gripper = (
            "red_box",
            "vx300s_right/10_right_gripper_finger",
        ) in contact_pairs
        touch_table = ("red_box", "table") in contact_pairs

        if touch_left_gripper and not touch_table:  # Successful transfer
            return 4
        elif touch_left_gripper:  # Attempted transfer
            return 3
        elif touch_right_gripper and not touch_table:  # Lifted
            return 2
        elif touch_right_gripper:  # Grasped
            return 1
        else:
            return 0

    def _get_all_contact_pairs(self) -> List[Tuple[str, str]]:
        """Get all current contact pairs in simulation.

        Returns:
            List of (geom1_name, geom2_name) contact pairs.
        """
        contact_pairs = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
            )
            geom2_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
            )
            if geom1_name and geom2_name:
                contact_pairs.append((geom1_name, geom2_name))
        return contact_pairs


def make_sim_env() -> TransferCubeTask:
    """Create joint control transfer cube environment.

    Returns:
        Configured TransferCubeTask instance.
    """
    return TransferCubeTask()


def make_ee_sim_env() -> TransferCubeEETask:
    """Create end-effector control transfer cube environment.

    Returns:
        Configured TransferCubeEETask instance.
    """
    return TransferCubeEETask()
