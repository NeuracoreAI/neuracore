import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from common.rollout_utils import rollout_policy
from common.transfer_cube import make_sim_env

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def main(args):
    """Main function for running the robot demo and logging with neuracore."""

    # Setup parameters
    args["record"]
    num_episodes = args["num_episodes"]
    onscreen_render = True
    render_cam_name = "angle"

    for episode_idx in range(num_episodes):
        print(f"Starting episode {episode_idx}")

        # Get action trajectory from policy rollout
        action_traj = rollout_policy()

        # Setup environment for replay with neuracore logging
        env = make_sim_env()
        obs = env.reset()

        # Setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(obs.cameras[render_cam_name].rgb)
            plt.ion()

        # Log initial state
        t = time.time()

        # Execute action trajectory while logging
        for action in action_traj:
            obs, reward, done = env.step(np.array(list(action.values())))
            if onscreen_render:
                plt_img.set_data(obs.cameras[render_cam_name].rgb)
                plt.pause(0.002)
            t += 0.02

        print(f"Episode {episode_idx} done")


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
