import argparse
import os
import time
import numpy as np
import mujoco
import neuracore as nc

from examples.common.utils import depth_to_point_cloud, get_camera_matrices
from common.constants import BIMANUAL_VIPERX_URDF_PATH
from common.rollout_utils import rollout_policy
from common.sim_env import BOX_POSE, make_sim_env

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
            name="My Example Dataset with custom data 2",
            description="This is an example dataset 2",
        )
        print("Created Dataset...")
        nc.start_recording()
        print(f"[INFO] Recording started")

    success = []
    for episode_idx in range(num_episodes):
        print(f"Starting episode {episode_idx}")

        # Get action trajectory from policy rollout
        action_traj, subtask_info, max_reward = rollout_policy()

        # Setup environment for replay with neuracore logging
        BOX_POSE[0] = subtask_info
        env = make_sim_env()
        ts = env.reset()

        # Log initial state
        t = time.time()
        CUSTOM_DATA = [1, 2, 3, 4, 5]
        nc.log_custom_data("my_custom_data", CUSTOM_DATA, timestamp=t)
        nc.log_joint_positions(ts.observation["qpos"], timestamp=t)
        nc.log_joint_velocities(ts.observation["qvel"], timestamp=t)
        nc.log_language("Pick up the cube and pass it to the other robot", timestamp=t)

        for cam_name in camera_names:
            rgb_img = ts.observation["images"][cam_name]
            depth_img = ts.observation["depth"][cam_name]
            height, width = depth_img.shape
            cam_id = mujoco.mj_name2id(env.physics.model.ptr, mujoco.mjtObj.mjOBJ_CAMERA, str(cam_name))
            intrinsics, extrinsics = get_camera_matrices(env.physics.model, env.physics.data, cam_id, height, width)
            points = depth_to_point_cloud(depth_img, intrinsics, extrinsics)
            
            nc.log_point_cloud(
                camera_id=cam_name,
                points=points,
                rgb_points=rgb_img.reshape(-1, 3).astype(np.uint8),
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                timestamp=t
            )
            print(f"[DEBUG] Logged point cloud for {cam_name} at {t}, shape: {points.shape}")

        # Execute action trajectory while logging
        episode_replay = [ts]
        for step_idx, action in enumerate(action_traj):
            ts = env.step(list(action.values()))
            t += 0.02
            nc.log_custom_data("my_custom_data", CUSTOM_DATA, timestamp=t)
            nc.log_joint_positions(ts.observation["qpos"], timestamp=t)
            nc.log_joint_velocities(ts.observation["qvel"], timestamp=t)
            nc.log_language("Pick up the cube and pass it to the other robot", timestamp=t)
            nc.log_joint_target_positions(action, timestamp=t)

            # Log point cloud every frame 
            for cam_name in camera_names:
                rgb_img = ts.observation["images"][cam_name]
                depth_img = ts.observation["depth"][cam_name]
                height, width = depth_img.shape
                cam_id = mujoco.mj_name2id(env.physics.model.ptr, mujoco.mjtObj.mjOBJ_CAMERA, str(cam_name))
                intrinsics, extrinsics = get_camera_matrices(env.physics.model, env.physics.data, cam_id, height, width)
                points = depth_to_point_cloud(depth_img, intrinsics, extrinsics)
                
                nc.log_point_cloud(
                    camera_id=cam_name,
                    points=points,
                    rgb_points=rgb_img.reshape(-1, 3).astype(np.uint8),
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                    timestamp=t
                )
                print(f"[DEBUG] Logged point cloud for {cam_name} at {t}, shape: {points.shape}")

            episode_replay.append(ts)

        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        if episode_max_reward == env.task.max_reward:
            success.append(1)
            print(f"Episode {episode_idx} Successful, return: {episode_return}")
        else:
            success.append(0)
            print(f"Episode {episode_idx} Failed")

    if record:
        nc.stop_recording()
        print(f"[INFO] Recording stopped")

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
