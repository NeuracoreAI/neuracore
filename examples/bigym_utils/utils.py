import numpy as np
from bigym.action_modes import JointPositionActionMode
from bigym.envs.reach_target import ReachTarget
from bigym.utils.observation_config import CameraConfig, ObservationConfig

FREQUENCY = 20

JOINT_NAMES: list[str] = [
    "left_shoulder_pitch",  # [ 0]
    "left_shoulder_roll",  # [ 1]
    "left_shoulder_yaw",  # [ 2]
    "left_elbow",  # [ 3]
    "left_gripper_right_driver",  # [ 4]  robotiq_2f85_left/right_driver_joint
    "left_gripper_right_coupler",  # [ 5]  robotiq_2f85_left/right_coupler_joint
    "left_gripper_right_spring",  # [ 6]  robotiq_2f85_left/right_spring_link_joint
    "left_gripper_right_follower",  # [ 7]  robotiq_2f85_left/right_follower_joint
    "left_gripper_left_driver",  # [ 8]  robotiq_2f85_left/left_driver_joint
    "left_gripper_left_coupler",  # [ 9]  robotiq_2f85_left/left_coupler_joint
    "left_gripper_left_spring",  # [10]  robotiq_2f85_left/left_spring_link_joint
    "left_gripper_left_follower",  # [11]  robotiq_2f85_left/left_follower_joint
    "left_wrist",  # [12]
    "right_shoulder_pitch",  # [13]
    "right_shoulder_roll",  # [14]
    "right_shoulder_yaw",  # [15]
    "right_elbow",  # [16]
    "right_gripper_right_driver",  # [17]  robotiq_2f85_right/right_driver_joint
    "right_gripper_right_coupler",  # [18]  robotiq_2f85_right/right_coupler_joint
    "right_gripper_right_spring",  # [19]  robotiq_2f85_right/right_spring_link_joint
    "right_gripper_right_follower",  # [20]  robotiq_2f85_right/right_follower_joint
    "right_gripper_left_driver",  # [21]  robotiq_2f85_right/left_driver_joint
    "right_gripper_left_coupler",  # [22]  robotiq_2f85_right/left_coupler_joint
    "right_gripper_left_spring",  # [23]  robotiq_2f85_right/left_spring_link_joint
    "right_gripper_left_follower",  # [24]  robotiq_2f85_right/left_follower_joint
    "right_wrist",  # [25]
    "pelvis_x",  # [26]
    "pelvis_y",  # [27]
    "pelvis_rz",  # [28]
    # "h1_floating_base",           # [29]  beyond obs len=29, never reached by zip()
]

JOINT_ACTUATORS: list[str] = [
    "floating_base_x",
    "floating_base_y",
    "floating_base_z",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist",
    "gripper_left",
    "gripper_right",
]


def make_env() -> ReachTarget:
    """Create a ReachTarget environment with a fixed configuration."""
    return ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig("head", resolution=(84, 84)),
                CameraConfig("left_wrist", resolution=(84, 84)),
                CameraConfig("right_wrist", resolution=(84, 84)),
            ]
        ),
        control_frequency=FREQUENCY,
        render_mode="human",
    )


def obs_to_joint_dict(
    obs: dict[str, np.ndarray],
    joint_names: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Convert observation proprioception to joint position/velocity dicts.

    Assumes:
        obs["proprioception"] = [qpos..., qvel...] concatenated.
    """
    obs_proprioception = obs["proprioception"].astype(float)

    mid = len(obs_proprioception) // 2
    robot_qpos = obs_proprioception[:mid]
    robot_qvel = obs_proprioception[mid:]

    qpos: dict[str, float] = dict(zip(joint_names, robot_qpos))
    qvel: dict[str, float] = dict(zip(joint_names, robot_qvel))
    return qpos, qvel


def obs_to_imgs(
    obs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Extract camera images from observation (CHW -> HWC)."""
    return {
        "head": obs["rgb_head"].transpose(1, 2, 0),
        "left_wrist": obs["rgb_left_wrist"].transpose(1, 2, 0),
        "right_wrist": obs["rgb_right_wrist"].transpose(1, 2, 0),
    }


def action_to_joint_action_dict(
    action,
    joint_names: list[str],
) -> dict[str, float]:
    """Convert action array to joint position dict."""
    joint_action: dict[str, float] = dict(zip(joint_names, action))
    return joint_action
