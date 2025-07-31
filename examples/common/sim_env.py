import collections
import os

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from .constants import (
    DT,
    PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN,
    START_ARM_POSE,
    VX300S_DIR,
)

BOX_POSE = [None]  # to be changed from outside


def make_sim_env():
    """
    Environment for simulated robot bi-manual manipulation, with joint position control
    """
    xml_path = os.path.join(VX300S_DIR, "bimanual_viperx_transfer_cube.xml")
    physics = mujoco.Physics.from_xml_path(xml_path)
    task = TransferCubeTask(random=False)
    env = control.Environment(
        physics,
        task,
        time_limit=20,
        control_timestep=DT,
        n_sub_steps=None,
        flat_observation=False,
    )
    return env


class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7 : 7 + 6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7 + 6]

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
            normalized_left_gripper_action
        )
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
            normalized_right_gripper_action
        )

        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        env_action = np.concatenate([
            left_arm_action,
            full_left_gripper_action,
            right_arm_action,
            full_right_gripper_action,
        ])
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)
        # Let env settle
        for _ in range(100):
            physics.step()

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
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(physics)
        obs["qvel"] = self.get_qvel(physics)
        obs["env_state"] = self.get_env_state(physics)

        # Define cameras and resolution in one place
        camera_names = ["top", "angle", "front_close"]
        resolution = (480, 640)
        height, width = resolution

        # Capture RGB and Depth in loops
        obs["images"] = {}
        obs["depth"] = {}
        for cam in camera_names:
            obs["images"][cam] = physics.render(
                height=height, width=width, camera_id=cam
            )
            depth = physics.render(
                height=height, width=width, camera_id=cam, depth=True
            )
            obs["depth"][cam] = depth.reshape(height, width).astype(np.float16)

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            if BOX_POSE[0] is not None:
                physics.named.data.qpos[-7:] = BOX_POSE[0]
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
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
