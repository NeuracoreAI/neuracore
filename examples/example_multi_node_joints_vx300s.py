"""Leader node of a two machine recording test on the VX300s environment.

This node owns the recording: it starts and stops it and logs the joint
streams. Run example_multi_node_cameras_vx300s.py on a second machine to log
the camera into the same recording.
"""

import argparse
import time

import numpy as np
from common.rollout_utils import rollout_policy
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH, make_sim_env

import neuracore as nc

ROBOT_NAME = "Mujoco VX300s"
STEP_PERIOD_S = 0.02


def main(args):
    """Record joint data into recordings shared with the camera node."""
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

    try:
        for episode_idx in range(args["num_episodes"]):
            action_traj = rollout_policy()
            env = make_sim_env()
            obs = env.reset()

            nc.start_recording()
            print(f"Episode {episode_idx}: recording {nc.get_cloud_recording_id()}")

            next_tick = time.time()
            for action in action_traj:
                t = time.time()
                nc.log_joint_positions(positions=obs.qpos, timestamp=t)
                nc.log_joint_velocities(velocities=obs.qvel, timestamp=t)
                nc.log_joint_target_positions(target_positions=action, timestamp=t)
                obs, _, _ = env.step(np.array(list(action.values())))

                next_tick += STEP_PERIOD_S
                time.sleep(max(0.0, next_tick - time.time()))

            nc.stop_recording(wait=True)
            print(f"Episode {episode_idx}: done")
            time.sleep(args["gap_s"])
    except KeyboardInterrupt:
        if nc.is_recording():
            nc.cancel_recording()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--dataset_name", type=str, default="Multi Node VX300s Test")
    parser.add_argument(
        "--gap_s",
        type=float,
        default=2.0,
        help="Pause between episodes so the camera node sees the stop",
    )
    main(vars(parser.parse_args()))
