import argparse
import time
from typing import Any, cast

import numpy as np
import torch
from bigym_utils.utils import (
    JOINT_ACTUATORS,
    JOINT_NAMES,
    make_env,
    obs_to_imgs,
    obs_to_joint_dict,
)
from neuracore_types import (
    BatchedJointData,
    BatchedNCData,
    DataSpec,
    DataType,
    JointData,
    RGBCameraData,
    SynchronizedPoint,
)

import neuracore as nc

TRAINING_JOB_NAME = "MyTrainingJob"
ROBOT_NAME = "Mujoco UnitreeH1 Example"
CAMERA_NAMES = ["head"]

# Specification of the order that will be fed into the model
MODEL_INPUT_ORDER: DataSpec = {
    DataType.JOINT_POSITIONS: JOINT_NAMES[:-1],
    DataType.RGB_IMAGES: CAMERA_NAMES,
}

MODEL_OUTPUT_ORDER: DataSpec = {
    DataType.JOINT_TARGET_POSITIONS: JOINT_ACTUATORS,
}


def run_rollout(
    env: Any,
    policy: Any,
    num_steps: int = 100,
    sleep_per_step: float = 0.05,
) -> bool:
    """
    Run a single rollout using the provided policy in the given environment.
    Creates sync points manually without logging data to the robot.
    Returns True if the episode succeeded, False otherwise.

    Args:
        env: The Bigym environment to run the rollout in.
        policy: The neuracore policy to use for action selection.
        num_steps: Maximum number of steps to run in the episode.
        sleep_per_step: Time to sleep between steps to control speed.

    Returns:
        bool: True if the episode succeeded, False otherwise.
    """
    obs, info = env.reset()

    horizon = 1

    for step_idx in range(num_steps):
        # Get joint states and images
        qpos, qvel = obs_to_joint_dict(obs, JOINT_NAMES)
        images = obs_to_imgs(obs)

        # Create a sync point manually without logging data to the robot
        sync_point = SynchronizedPoint(
            data={
                DataType.JOINT_POSITIONS: {
                    k: JointData(value=v) for k, v in qpos.items()
                },
                DataType.RGB_IMAGES: {
                    "head": RGBCameraData(frame=images["head"]),
                },
            }
        )

        idx_in_horizon = step_idx % horizon

        # Re-plan at the start of each horizon
        if idx_in_horizon == 0:
            print(f"Step {step_idx} / {num_steps}")
            predictions: dict[DataType, dict[str, BatchedNCData]] = policy.predict(
                sync_point=sync_point, timeout=5
            )

            joint_target_positions = cast(
                dict[str, BatchedJointData],
                predictions[DataType.JOINT_TARGET_POSITIONS],
            )

            # Concatenate joint targets in the order specified by JOINT_ACTUATORS
            batched_action = (
                torch.cat(
                    [joint_target_positions[name].value for name in JOINT_ACTUATORS],
                    dim=1,
                )
                .cpu()
                .numpy()
            )

            actions = batched_action[0]  # Get the first (and only) in the batch
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
        mjcf_path="bigym/bigym/envs/xmls/h1/h1.xml",  # Update path as needed
        overwrite=True,
    )
    # If you know the path to the local model.nc.zip file
    # you can use it directly without connecting to a robot
    policy = nc.policy(
        model_file="PATH/TO/MODEL.nc.zip",
        model_input_order=MODEL_INPUT_ORDER,
        model_output_order=MODEL_OUTPUT_ORDER,
    )

    # Optional. Set the checkpoint to the last epoch.
    # Note by default, model is loaded from the last epoch.
    # policy.set_checkpoint(epoch=-1)

    success_count = 0
    env = make_env()

    try:
        for episode_idx in range(num_rollouts):
            print(f"\n=== Episode {episode_idx} ===")
            succeeded = run_rollout(
                env=env,
                policy=policy,
                num_steps=100,
                sleep_per_step=0.05,
            )

            if succeeded:
                success_count += 1

            success_rate = success_count / (episode_idx + 1)
            print(
                f"Episode {episode_idx} done | "
                f"successes: {success_count} | "
                f"success rate: {success_rate:.2f}"
            )
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
