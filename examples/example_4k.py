"""This example demonstrates sending two heavy 4K RGB traces and one non-video trace to Neuracore."""

import argparse
import time

import numpy as np
from common.rollout_utils import rollout_policy
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH, make_sim_env

import neuracore as nc


def make_4k_rgb_frame():
    """Generate a heavy 4K RGB frame."""
    return np.random.randint(
        0,
        256,
        size=(2160, 3840, 3),
        dtype=np.uint8,
    )


def main(args):
    """Main function for running the robot demo and logging with neuracore."""

    nc.login()
    nc.connect_robot(
        robot_name="Mujoco VX300s",
        urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
        overwrite=False,
    )

    record = args["record"]
    num_episodes = args["num_episodes"]

    if record:
        nc.create_dataset(
            name="My Heavy 4K RGB Dataset",
            description="Dataset with two heavy 4K RGB traces and one non-video trace.",
        )
        print("Created Dataset...")

    try:
        for episode_idx in range(num_episodes):
            print(f"Starting episode {episode_idx}")

            action_traj = rollout_policy()

            env = make_sim_env()
            obs = env.reset()

            if record:
                nc.start_recording()

            t = time.time()

            # Keep only one non-video trace
            nc.log_joint_positions(positions=obs.qpos, timestamp=t)

            # Two heavy 4K RGB traces
            nc.log_rgb("heavy_4k_rgb_1", make_4k_rgb_frame(), timestamp=t)
            nc.log_rgb("heavy_4k_rgb_2", make_4k_rgb_frame(), timestamp=t)

            for action in action_traj:
                obs, reward, done = env.step(np.array(list(action.values())))
                t += 0.02

                # Keep only one non-video trace
                nc.log_joint_positions(positions=obs.qpos, timestamp=t)

                # Two heavy 4K RGB traces
                nc.log_rgb("heavy_4k_rgb_1", make_4k_rgb_frame(), timestamp=t)
                nc.log_rgb("heavy_4k_rgb_2", make_4k_rgb_frame(), timestamp=t)

            if record:
                print("Finishing recording...")
                nc.stop_recording(wait=False)
                print("Finished recording!")

            print(f"Episode {episode_idx} done")

    except KeyboardInterrupt:
        if record:
            nc.cancel_recording()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_episodes",
        type=int,
        help="Number of episodes to run",
        default=50,
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Whether to record with neuracore",
        default=False,
    )

    main(vars(parser.parse_args()))