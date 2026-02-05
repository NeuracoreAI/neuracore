import argparse
import logging
import os
import time

import numpy as np
from common.rollout_utils import rollout_policy
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH, make_sim_env

import neuracore as nc
from neuracore.data_daemon.lifecycle.daemon_lifecycle import ensure_daemon_running

logger = logging.getLogger(__name__)


def main(args: dict) -> None:
    """Run the robot demo and log via nc.log_* APIs (data daemon under the hood)."""

    record = args["record"]
    # num_episodes = args["num_episodes"]
    num_episodes = 5

    if args["api_url"]:
        os.environ["NEURACORE_API_URL"] = args["api_url"]
    if args["api_key"]:
        os.environ["NEURACORE_API_KEY"] = args["api_key"]

    # Initialize neuracore
    nc.login()
    nc.connect_robot(
        robot_name="Mujoco VX300s",
        urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
        overwrite=False,
    )

    if record:
        nc.create_dataset(
            name="My data daemon example set",
            description="This is an example dataset (data daemon logging)",
        )
        print("Created Dataset...")

        # start daemon process
        ensure_daemon_running()
    try:
        for episode_idx in range(num_episodes):
            print(f"Starting episode {episode_idx}")

            action_traj = rollout_policy()

            env = make_sim_env()
            obs = env.reset()

            if record:
                nc.start_recording()

            # Until here it is fine and working.

            t = time.time()
            custom_data = np.array([1, 2, 3, 4, 5])
            nc.log_custom_1d("my_custom_data", custom_data, timestamp=t)
            nc.log_joint_positions(positions=obs.qpos, timestamp=t)
            # nc.log_joint_velocities(velocities=obs.qvel, timestamp=t)
            # nc.log_language(
            #     name="instruction",
            #     language="Pick up the cube and pass it to the other robot",
            #     timestamp=t,
            # )
            # nc.log_rgb(cam_name, obs.cameras[cam_name].rgb, timestamp=t)

            for action in action_traj:
                obs, reward, done = env.step(np.array(list(action.values())))
                t += 0.02
                nc.log_custom_1d("my_custom_data", custom_data, timestamp=t)
                nc.log_joint_positions(positions=obs.qpos, timestamp=t)
                # nc.log_joint_velocities(velocities=obs.qvel, timestamp=t)
                # nc.log_language(
                #     name="instruction",
                #     language="Pick up the cube and pass it to the other robot",
                #     timestamp=t,
                # )
                # nc.log_joint_target_positions(
                #     target_positions=action,
                #     timestamp=t,
                # )
                # nc.log_rgb(name=cam_name, rgb=obs.cameras[cam_name].rgb, timestamp=t)

                if done:
                    break

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
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="Optional Neuracore API base URL (overrides NEURACORE_API_URL).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional Neuracore API key (overrides NEURACORE_API_KEY).",
    )

    main(vars(parser.parse_args()))
