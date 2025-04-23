from pathlib import Path

import matplotlib.pyplot as plt
from common.constants import BIMANUAL_VIPERX_URDF_PATH
from common.rollout_utils import rollout_policy
from common.sim_env import BOX_POSE, make_sim_env

import neuracore as nc
from neuracore.core.nc_types import DataType

THIS_DIR = Path(__file__).parent
TRAINING_JOB_NAME = "MyWorldModelTrainingJob"


def main():
    camera_names = ["angle"]
    render_cam_name = "angle"

    nc.login()
    nc.connect_robot(
        robot_name="Mujoco VX300s",
        urdf_path=BIMANUAL_VIPERX_URDF_PATH,
        overwrite=False,
    )
    endpoint = nc.connect_local_endpoint(train_run_name=TRAINING_JOB_NAME)

    # Get action trajectory from policy rollout
    # We will use this to generate actions for out world model
    action_traj, subtask_info, max_reward = rollout_policy()

    # Setup environment for replay with neuracore logging
    BOX_POSE[0] = subtask_info
    env = make_sim_env()
    ts = env.reset()

    # Log initial state
    nc.log_joint_positions(ts.observation["qpos"])
    for cam_name in camera_names:
        nc.log_rgb(cam_name, ts.observation["images"][cam_name])

    # For showing the future frames
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation["images"][render_cam_name])
    plt.ion()

    for action in action_traj:
        ts = env.step(list(action.values()))

        nc.log_joint_positions(ts.observation["qpos"])
        for cam_name in camera_names:
            nc.log_rgb(cam_name, ts.observation["images"][cam_name])

        prediction = endpoint.predict()
        # Produces (T, CAMERAS, ...)
        future_frames = prediction.outputs[DataType.RGB_IMAGE]
        future_frames[0, 0]

        # Show future
        plt_img.set_data(ts.observation["images"][render_cam_name])
        plt.pause(0.002)
        plt.close()

    plt.close()
    endpoint.disconnect()


if __name__ == "__main__":
    main()
