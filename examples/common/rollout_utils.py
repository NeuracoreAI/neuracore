"""Utilities for rolling out policies and collecting trajectories."""

from .base_env import BimanualViperXTask
from .scripted_policy import PickAndTransferPolicy
from .transfer_cube import BOX_POSE, make_ee_sim_env


def rollout_policy(
    inject_noise: bool = False,
    onscreen_render: bool = False,
    render_cam_name: str = "angle",
) -> list[dict[str, float]]:
    """Roll out the pick and transfer policy and return action trajectory.

    Args:
        inject_noise: Whether to inject noise into policy actions.
        onscreen_render: Whether to render episode onscreen.
        render_cam_name: Camera name for rendering.

    Returns:
        List of action dictionaries for each timestep.
    """
    # Setup environment and policy
    env = make_ee_sim_env()
    obs = env.reset()
    episode = [obs]
    policy = PickAndTransferPolicy(inject_noise)

    # Setup visualization if requested
    plt_img = None
    if onscreen_render:
        import matplotlib.pyplot as plt

        ax = plt.subplot()
        plt_img = ax.imshow(obs.cameras[render_cam_name].rgb)
        plt.ion()

    # Execute policy for full episode
    for step in range(BimanualViperXTask.EPISODE_LENGTH):
        # Create timestep-like object for policy
        ts = type("TimeStep", (), {"observation": obs.model_dump()})()
        action = policy(ts)

        obs, reward, done = env.step(action)
        obs.reward = reward
        episode.append(obs)

        # Update visualization
        if onscreen_render and plt_img is not None:
            plt_img.set_data(obs.cameras[render_cam_name].rgb)
            plt.pause(0.002)

        if done:
            break

    # Clean up visualization
    if onscreen_render:
        plt.close()

    # Extract action trajectory
    action_traj = []
    for obs in episode:
        if hasattr(obs, "qpos") and hasattr(obs, "gripper_ctrl"):
            joint_action = _extract_joint_action(obs)
            action_traj.append(joint_action)

    # Store initial environment state globally
    if episode:
        BOX_POSE[0] = episode[0].env_state.copy()

    return action_traj


def _extract_joint_action(obs) -> dict[str, float]:
    """Extract joint action dictionary from observation.

    Args:
        obs: Observation containing joint positions and gripper control.

    Returns:
        Dictionary mapping joint names to action values.
    """
    joint_dict = obs.qpos.copy()
    ctrl = obs.gripper_ctrl

    # Split into left and right joint actions
    joint_items = list(joint_dict.items())
    left_joint_actions = {k: v for k, v in joint_items[:6]}
    right_joint_actions = {k: v for k, v in joint_items[8:14]}  # Skip gripper joints

    # Create complete action dictionary
    joint_action = {}
    joint_action.update(left_joint_actions)
    joint_action["vx300s_left/gripper_open"] = (
        BimanualViperXTask.normalize_gripper_position(ctrl[0])
    )
    joint_action.update(right_joint_actions)
    joint_action["vx300s_right/gripper_open"] = (
        BimanualViperXTask.normalize_gripper_position(ctrl[2])
    )

    return joint_action
