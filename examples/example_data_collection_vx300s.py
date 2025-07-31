import argparse
import os
import time

import numpy as np
from common.constants import BIMANUAL_VIPERX_URDF_PATH
from common.rollout_utils import rollout_policy
from common.sim_env import BOX_POSE, make_sim_env
from common.utils import (
    generate_random_point_cloud,
    generate_random_rgb,
    get_camera_matrices,
)

import neuracore as nc

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def main(args):
    """Main function for running the robot demo and logging with neuracore."""
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
    num_points = 2500

    if record:
        nc.create_dataset(
            name="My Example Dataset",
            description="My Example Dataset",
        )

    success = []
    for episode_idx in range(num_episodes):

        if record:
            nc.start_recording()

        # Get action trajectory
        action_traj, subtask_info, max_reward = rollout_policy()
        BOX_POSE[0] = subtask_info
        env = make_sim_env()
        ts = env.reset()

        # Initial logging
        t = time.time()
        CUSTOM_DATA = [1, 2, 3, 4, 5]
        nc.log_custom_data("my_custom_data", CUSTOM_DATA, timestamp=t)
        nc.log_joint_positions(ts.observation["qpos"], timestamp=t)
        nc.log_joint_velocities(ts.observation["qvel"], timestamp=t)
        nc.log_language("Pick up the cube and pass it to the other robot", timestamp=t)

        # Log initial camera point cloud
        for cam_name in camera_names:
            points = generate_random_point_cloud(num_points).astype(np.float16)
            rgb_points = generate_random_rgb(num_points)
            intrinsics, extrinsics = get_camera_matrices()

            # --- Log ---
            nc.log_point_cloud(
                camera_id=cam_name,
                points=points.astype(np.float32),
                rgb_points=rgb_points.astype(np.uint8),
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                timestamp=t,
            )

        # --- Run trajectory ---
        episode_replay = [ts]
        for step_idx, action in enumerate(action_traj):
            # Measure sim step
            ts = env.step(list(action.values()))

            t += 0.02
            nc.log_custom_data("my_custom_data", CUSTOM_DATA, timestamp=t)
            nc.log_joint_positions(ts.observation["qpos"], timestamp=t)
            nc.log_joint_velocities(ts.observation["qvel"], timestamp=t)
            nc.log_joint_target_positions(action, timestamp=t)

            for cam_name in camera_names:
                # Measure point cloud generation
                points = generate_random_point_cloud(num_points).astype(np.float16)
                rgb_points = generate_random_rgb(num_points)
                intrinsics, extrinsics = get_camera_matrices()

                # --- Log ---
                nc.log_point_cloud(
                    camera_id=cam_name,
                    points=points.astype(np.float32),
                    rgb_points=rgb_points.astype(np.uint8),
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                    timestamp=t,
                )

            episode_replay.append(ts)

        if record:
            nc.stop_recording()

        # Evaluate episode success
        np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        success.append(1 if episode_max_reward == env.task.max_reward else 0)

        print(
            f"Success rate: {np.mean(success) * 100:.1f}% "
            f"({np.sum(success)}/{len(success)})"
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
