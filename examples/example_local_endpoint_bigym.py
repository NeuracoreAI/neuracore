import argparse
import time
from typing import Any, List, Tuple, Dict

import numpy as np
import neuracore as nc

from bi_gym_utils.utils import (
    JOINT_NAMES,
    JOINT_ACTUATORS,
    make_env,
    obs_to_joint_dict,
    obs_to_imgs,
)


TRAINING_JOB_NAME = "MyTrainingJob"

def run_rollout(
    env: Any,
    policy: Any,
    num_steps: int = 100,
    sleep_per_step: float = 0.05,
) -> bool:
    """
    Run a single rollout in the environment using the given policy.

    Parameters
    ----------
    env : gym-like environment
        Environment with reset() and step() methods.
    policy : neuracore policy
        Policy object with predict() method.
    num_steps : int
        Maximum number of steps for the rollout.
    sleep_per_step : float
        Sleep duration between steps (to slow down visualization).

    Returns
    -------
    bool
        True if the episode succeeded (reward == 1.0 at any step), else False.
    """
    obs, info = env.reset()

    horizon = 1
    actions: List[np.ndarray] = []

    for step_idx in range(num_steps):
        # Log joint states
        qpos, qvel = obs_to_joint_dict(obs, JOINT_NAMES)
        nc.log_joint_positions(qpos)

        # Log image(s)
        images = obs_to_imgs(obs)
        nc.log_rgb("rgb_head", images["rgb_head"])
        # nc.log_rgb("rgb_left_wrist", images["rgb_left_wrist"])
        # nc.log_rgb("rgb_right_wrist", images["rgb_right_wrist"])

        idx_in_horizon = step_idx % horizon

        # Re-plan at the start of each horizon
        if idx_in_horizon == 0:
            print(f"Step {step_idx} / {num_steps}")
            predicted_sync_points = policy.predict(timeout=5)

            joint_target_positions = [
                sp.joint_target_positions for sp in predicted_sync_points
            ]

            actions = [
                jtp.numpy(order=JOINT_ACTUATORS)
                for jtp in joint_target_positions
                if jtp is not None
            ]

            if not actions:
                raise RuntimeError("Policy returned no valid actions.")

            horizon = len(actions)

        a = actions[idx_in_horizon]
        action = np.clip(a, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(sleep_per_step)

        if reward == 1.0:
            print("Episode succeeded!")
            return True

        if terminated or truncated:
            break

    return False


def main(
    num_rollouts: int,
) -> None:
    nc.login()

    nc.connect_robot(
        robot_name="Mujoco UnitreeH1 Example",
        overwrite=False,
    )

    # If you have a train run name, you can use it to connect to a local. E.g.:
    policy = nc.policy(
        train_run_name=TRAINING_JOB_NAME,
    )
    # If you know the path to the local model.nc.zip file, you can use it directly as:
    # policy = nc.policy(model_file=PATH/TO/MODEL.nc.zip)

    # Alternatively, you can connect to a local endpoint that has been started
    # policy = nc.policy_local_server(train_run_name=TRAINING_JOB_NAME)

    # Optional. Set the checkpoint to the last epoch.
    # Note by default, model is loaded from the last epoch.
    # policy.set_checkpoint(epoch=-1)

    # Optional: policy.set_checkpoint(epoch=-1)

    success_count = 0
    env = make_env()

    try:
        for episode_idx in range(num_rollouts):
            print(f"\n=== Episode {episode_idx} ===")
            succeeded = run_rollout(
                env=env,
                policy=policy,
                num_steps=100,
            )

            if succeeded:
                success_count += 1

            success_rate = success_count / (episode_idx + 1)
            print(
                f"Episode {episode_idx} done | "
                f"successes: {success_count} | "
                f"success rate: {success_rate:.2f}"
            )

            time.sleep(3)
    finally:
        env.close()
        policy.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Mujoco UnitreeH1 policy rollouts via neuracore."
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=50,
        help="Number of rollouts (episodes) to run.",
    )

    args = parser.parse_args()
    main(
        num_rollouts=args.num_rollouts,
    )
