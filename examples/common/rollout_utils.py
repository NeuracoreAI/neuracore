import numpy as np

from .constants import EPISODE_LENGTH, PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from .ee_sim_env import make_ee_sim_env
from .scripted_policy import PickAndTransferPolicy


def rollout_policy(
    inject_noise: bool = False,
    onscreen_render: bool = False,
    render_cam_name: str = "angle",
) -> tuple[list[dict], dict, float]:
    """
    Rolls out the pick and transfer policy and returns the action trajectory.

    Args:
        inject_noise: Whether to inject noise into the actions
        onscreen_render: Whether to render onscreen
        render_cam_name: Name of camera to render from

    Returns:
        action_traj: List of actions to replay
        subtask_info: Environment state info
        episode_max_reward: Maximum reward achieved
    """
    # Setup environment and policy
    env = make_ee_sim_env()
    ts = env.reset()
    episode = [ts]
    policy = PickAndTransferPolicy(inject_noise)

    # Setup visualization if needed
    if onscreen_render:
        import matplotlib.pyplot as plt

        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation["images"][render_cam_name])
        plt.ion()

    # Execute policy
    for step in range(EPISODE_LENGTH):
        action = policy(ts)
        ts = env.step(action)
        episode.append(ts)

        if onscreen_render:
            plt_img.set_data(ts.observation["images"][render_cam_name])
            plt.pause(0.002)

    if onscreen_render:
        plt.close()

    # Calculate rewards
    episode_max_reward = np.max([ts.reward for ts in episode[1:]])

    # Convert joint trajectory to action trajectory
    joint_trajectory = [ts.observation["qpos"] for ts in episode]
    gripper_ctrl_traj = [ts.observation["gripper_ctrl"] for ts in episode]

    action_traj = []
    for joint_dict, ctrl in zip(joint_trajectory, gripper_ctrl_traj):
        joint_action = {}
        # Split into left and right joint actions
        left_joint_actions = {k: v for k, v in list(joint_dict.items())[:6]}
        right_joint_actions = {k: v for k, v in list(joint_dict.items())[6 + 2 : -2]}

        # Combine actions with normalized gripper positions
        joint_action.update(left_joint_actions)
        joint_action["vx300s_left/gripper_open"] = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(
            ctrl[0]
        )
        joint_action.update(right_joint_actions)
        joint_action["vx300s_right/gripper_open"] = (
            PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
        )

        action_traj.append(joint_action)

    # Get initial environment state
    subtask_info = episode[0].observation["env_state"].copy()

    return action_traj, subtask_info, episode_max_reward
