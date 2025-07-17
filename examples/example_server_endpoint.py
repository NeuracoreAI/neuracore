import sys

import matplotlib.pyplot as plt
from common.constants import BIMANUAL_VIPERX_URDF_PATH, EPISODE_LENGTH
from common.ee_sim_env import sample_box_pose
from common.sim_env import BOX_POSE, make_sim_env

import neuracore as nc
from neuracore import EndpointError

ENDPOINT_NAME = "MyExampleEndpoint"


def main():

    nc.login()
    nc.connect_robot(
        robot_name="Mujoco VX300s",
        urdf_path=BIMANUAL_VIPERX_URDF_PATH,
        overwrite=False,
    )

    try:
        policy = nc.policy_remote_server(ENDPOINT_NAME)
    except EndpointError:
        print(f"Please ensure that the endpoint '{ENDPOINT_NAME}' is running.")
        print(
            "Once you have trained a model, endpoints can be satrted at https://neuracore.app/dashboard/endpoints"
        )
        sys.exit(1)

    onscreen_render = True
    render_cam_name = "angle"
    obs_camera_names = ["angle"]

    success = 0
    for episode_idx in range(1):
        print(f"{episode_idx=}")

        # Setup the environment
        env = make_sim_env()
        BOX_POSE[0] = sample_box_pose()
        ts = env.reset()

        # Setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation["images"][render_cam_name])
            plt.ion()

        episode_max = 0
        horizon = 1

        # Run episode
        for i in range(EPISODE_LENGTH):
            nc.log_joint_positions(ts.observation["qpos"])
            for key, value in ts.observation["images"].items():
                if key in obs_camera_names:
                    nc.log_rgb(key, value)
            idx_in_horizon = i % horizon
            if idx_in_horizon == 0:
                predicted_sync_points = policy.predict()
                joint_target_positions = [
                    sp.joint_target_positions for sp in predicted_sync_points
                ]
                actions = [
                    jtp.numpy() for jtp in joint_target_positions if jtp is not None
                ]
                horizon = len(actions)

            a = actions[idx_in_horizon]
            ts = env.step(a)
            episode_max = max(episode_max, ts.reward)

            if onscreen_render:
                plt_img.set_data(ts.observation["images"][render_cam_name])
                plt.pause(0.002)
        plt.close()

        # Log results
        if episode_max == env.task.max_reward:
            success += 1
            print(f"{episode_idx=} Successful")
        else:
            print(f"{episode_idx=} Failed")

    policy.disconnect()


if __name__ == "__main__":
    main()
