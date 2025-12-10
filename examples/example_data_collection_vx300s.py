import argparse
import os
import time

import numpy as np
from common.rollout_utils import rollout_policy
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH, make_sim_env

import neuracore as nc

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def main(args):
    """Main function for running the robot demo and logging with neuracore."""

    # Initialize neuracore
    nc.login()
    nc.connect_robot(
        robot_name="Mujoco VX300s",
        urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
        overwrite=False,
    )

    # Setup parameters
    record = args["record"]
    num_episodes = args["num_episodes"]

    if record:
        nc.create_dataset(
            name="My Example Dataset",
            description="This is an example dataset",
        )
        print("Created Dataset...")

    try:
        for episode_idx in range(num_episodes):
            print(f"Starting episode {episode_idx}")

            # Get action trajectory from policy rollout
            action_traj = rollout_policy()

            # Setup environment for replay with neuracore logging
            env = make_sim_env()
            obs = env.reset()

            # Start recording if enabled
            if record:
                nc.start_recording()

            # Log initial state
            t = time.time()
            CUSTOM_DATA = np.array([1, 2, 3, 4, 5])
            CAM_NAME = "angle"
            nc.log_custom_1d("my_custom_data", CUSTOM_DATA, timestamp=t)
            nc.log_joint_positions(positions=obs.qpos, timestamp=t)
            nc.log_joint_velocities(velocities=obs.qvel, timestamp=t)
            nc.log_language(
                name="instruction",
                language="Pick up the cube and pass it to the other robot",
                timestamp=t,
            )
            nc.log_rgb(CAM_NAME, obs.cameras[CAM_NAME].rgb, timestamp=t)

            # Execute action trajectory while logging
            for action in action_traj:
                obs, reward, done = env.step(np.array(list(action.values())))

                t += 0.02
                nc.log_custom_1d("my_custom_data", CUSTOM_DATA, timestamp=t)
                nc.log_joint_positions(positions=obs.qpos, timestamp=t)
                nc.log_joint_velocities(velocities=obs.qvel, timestamp=t)
                nc.log_language(
                    name="instruction",
                    language="Pick up the cube and pass it to the other robot",
                    timestamp=t,
                )
                nc.log_joint_target_positions(target_positions=action, timestamp=t)
                nc.log_rgb(name=CAM_NAME, rgb=obs.cameras[CAM_NAME].rgb, timestamp=t)

            # Stop recording if enabled
            if record:
                print("Finishing recording...")
                nc.stop_recording()
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
