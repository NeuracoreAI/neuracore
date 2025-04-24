import collections
import os

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from .constants import (
    DT,
    PUPPET_GRIPPER_POSITION_CLOSE,
    PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN,
    START_ARM_POSE,
    VX300S_DIR,
)


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def make_ee_sim_env():
    """
    Environment for simulated robot bi-manual manipulation, with end-effector control.
    Action space:
    [left_arm_pose (7),             # position and quaternion for end effector
    left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
    right_arm_pose (7),            # position and quaternion for end effector
    right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)
    """
    xml_path = os.path.join(VX300S_DIR, "bimanual_viperx_ee_transfer_cube.xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Model file not found: {xml_path}")
    physics = mujoco.Physics.from_xml_path(xml_path)
    task = TransferCubeEETask(random=False)
    env = control.Environment(
        physics,
        task,
        time_limit=20,
        control_timestep=DT,
        n_sub_steps=None,
        flat_observation=False,
    )
    return env


class BimanualViperXEETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        # set mocap position and quat
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # set gripper
        g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
        g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
        np.copyto(
            physics.data.ctrl,
            np.array([g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]),
        )

    def initialize_robots(self, physics):
        # reset joint position
        physics.named.data.qpos[:16] = START_ARM_POSE

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side
        np.copyto(physics.data.mocap_pos[0], [-0.31718881, 0.5, 0.29525084])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        # right
        np.copyto(
            physics.data.mocap_pos[1], np.array([0.31718881, 0.49999888, 0.29525084])
        )
        np.copyto(physics.data.mocap_quat[1], [1, 0, 0, 0])

        # reset gripper control
        close_gripper_control = np.array([
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
        ])
        np.copyto(physics.data.ctrl, close_gripper_control)
        # Let env settle; e.g. grippersa go to mocap position
        for _ in range(100):
            physics.step()

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        joint_dict = {}
        for i in range(16):
            joint_name = physics.model.id2name(i, "joint")
            joint_dict[joint_name] = qpos_raw[i]
        return joint_dict

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        joint_dict = {}
        for i in range(16):
            joint_name = physics.model.id2name(i, "joint")
            joint_dict[joint_name] = qvel_raw[i]
        return joint_dict

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(physics)
        obs["qvel"] = self.get_qvel(physics)
        obs["env_state"] = self.get_env_state(physics)
        obs["images"] = dict()
        obs["images"]["top"] = physics.render(height=480, width=640, camera_id="top")
        obs["images"]["angle"] = physics.render(
            height=480, width=640, camera_id="angle"
        )
        obs["images"]["vis"] = physics.render(
            height=480, width=640, camera_id="front_close"
        )
        # used in scripted policy to obtain starting pose
        obs["mocap_pose_left"] = np.concatenate(
            [physics.data.mocap_pos[0], physics.data.mocap_quat[0]]
        ).copy()
        obs["mocap_pose_right"] = np.concatenate(
            [physics.data.mocap_pos[1], physics.data.mocap_quat[1]]
        ).copy()

        # used when replaying joint trajectory
        obs["gripper_ctrl"] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError


class TransferCubeEETask(BimanualViperXEETask):

    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = sample_box_pose()
        box_start_idx = physics.model.name2id("red_box_joint", "joint")
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # Return reward based on contact conditions
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = (
            "red_box",
            "vx300s_left/10_left_gripper_finger",
        ) in all_contact_pairs
        touch_right_gripper = (
            "red_box",
            "vx300s_right/10_right_gripper_finger",
        ) in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table:  # lifted
            reward = 2
        if touch_left_gripper:  # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table:  # successful transfer
            reward = 4
        return reward
