import argparse
import os

import numpy as np
from common.constants import BIMANUAL_VIPERX_URDF_PATH
from common.rollout_utils import rollout_policy
from common.sim_env import BOX_POSE, make_sim_env

import neuracore as nc

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def main(args):
    """Main function for running the robot demo and logging with neuracore."""

    # Initialize neuracore
    nc.login()
    nc.connect_robot(
        robot_name="Mujoco VX300s",
        urdf_path=BIMANUAL_VIPERX_URDF_PATH,
        overwrite=False,
    )

    # Setup parameters
    record = args["record"]
    num_episodes = args["num_episodes"]
    camera_names = ["angle"]

    if record:
        nc.create_dataset(
            name="MP4 Dataset V2", description="This is an example dataset"
        )
        print("Created Dataset...")

    success = []
    for episode_idx in range(num_episodes):
        print(f"Starting episode {episode_idx}")

        # Get action trajectory from policy rollout
        action_traj, subtask_info, max_reward = rollout_policy()

        # Setup environment for replay with neuracore logging
        BOX_POSE[0] = subtask_info
        env = make_sim_env()
        ts = env.reset()

        # Start recording if enabled
        if record:
            nc.start_recording()

        # Log initial state
        nc.log_joints(ts.observation["qpos"])
        for cam_name in camera_names:
            nc.log_rgb(cam_name, ts.observation["images"][cam_name])

        # Execute action trajectory while logging
        episode_replay = [ts]
        for action in action_traj:
            ts = env.step(list(action.values()))

            nc.log_joints(ts.observation["qpos"])
            nc.log_action(action)
            for cam_name in camera_names:
                nc.log_rgb(cam_name, ts.observation["images"][cam_name])

            episode_replay.append(ts)

        # Stop recording if enabled
        if record:
            print("Finishing recording...")
            nc.stop_recording()
            print("Finished recording!")

        # Calculate episode results
        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])

        if episode_max_reward == env.task.max_reward:
            success.append(1)
            print(f"Episode {episode_idx} Successful, return: {episode_return}")
        else:
            success.append(0)
            print(f"Episode {episode_idx} Failed")

    print(
        f"Success rate: {np.mean(success)*100:.1f}% ({np.sum(success)}/{len(success)})"
    )


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
