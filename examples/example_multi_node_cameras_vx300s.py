"""Follower node of a two machine recording test on the VX300s environment.

This node logs the camera into whatever recording the leader node opens.
Start it first, then run example_multi_node_joints_vx300s.py on a second
machine with the same org and dataset name.
"""

import argparse
import time

import numpy as np
from common.rollout_utils import rollout_policy
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH, make_sim_env

import neuracore as nc

ROBOT_NAME = "Mujoco VX300s"
CAM_NAME = "angle"
STEP_PERIOD_S = 0.02
POLL_PERIOD_S = 0.01


def wait_for_recording(want: bool) -> None:
    """Block until the robot recording state equals want."""
    while nc.is_recording() != want:
        time.sleep(POLL_PERIOD_S)


def main(args):
    """Log camera frames into recordings the joint node opens."""
    nc.login()
    nc.connect_robot(
        robot_name=ROBOT_NAME,
        urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
        overwrite=False,
    )
    nc.create_dataset(
        name=args["dataset_name"],
        description="Two machine recording test",
    )

    for episode_idx in range(args["num_episodes"]):
        action_traj = rollout_policy()
        env = make_sim_env()
        obs = env.reset()

        print(f"Episode {episode_idx}: waiting for the joint node to start")
        wait_for_recording(True)
        print(f"Episode {episode_idx}: recording {nc.get_cloud_recording_id()}")

        frames = 0
        next_tick = time.time()
        for action in action_traj:
            if not nc.is_recording():
                break
            nc.log_rgb(CAM_NAME, obs.cameras[CAM_NAME].rgb, timestamp=time.time())
            frames += 1
            obs, _, _ = env.step(np.array(list(action.values())))

            next_tick += STEP_PERIOD_S
            time.sleep(max(0.0, next_tick - time.time()))

        wait_for_recording(False)
        print(f"Episode {episode_idx}: done, logged {frames} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--dataset_name", type=str, default="Multi Node VX300s Test")
    main(vars(parser.parse_args()))
