import numpy as np
from bigym.action_modes import JointPositionActionMode
from bigym.envs.reach_target import ReachTarget
from bigym.utils.observation_config import CameraConfig, ObservationConfig

FREQUENCY = 20

JOINT_NAMES: list[str] = [
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "right_driver_joint",
    "right_coupler_joint",
    "right_spring_link_joint",
    "right_follower_joint",
    "left_driver_joint",
    "left_coupler_joint",
    "left_spring_link_joint",
    "left_follower_joint",
    "left_wrist",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_driver_joint",
    "right_coupler_joint",
    "right_spring_link_joint",
    "right_follower_joint",
    "left_driver_joint",
    "left_coupler_joint",
    "left_spring_link_joint",
    "left_follower_joint",
    "right_wrist",
    "pelvis_x",
    "pelvis_y",
    "pelvis_rz",
    "h1_floating_base",
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
