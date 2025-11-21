import sys
from typing import cast

import matplotlib.pyplot as plt
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH, make_sim_env
from neuracore_types import DataType, JointData

import neuracore as nc
from neuracore import EndpointError

ENDPOINT_NAME = "MyExampleEndpoint"
JOINT_GROUP_NAME = "arm"


def main():

    nc.login()
    nc.connect_robot(
        robot_name="Mujoco VX300s",
        urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
        overwrite=False,
    )

    try:
        policy = nc.policy_remote_server(ENDPOINT_NAME)
    except EndpointError:
        print(f"Please ensure that the endpoint '{ENDPOINT_NAME}' is running.")
        print(
            "Once you have trained a model, endpoints can be started at https://neuracore.app/dashboard/endpoints"
        )
        sys.exit(1)

    onscreen_render = True
    render_cam_name = "angle"
    obs_camera_names = ["angle"]

    for episode_idx in range(1):
        print(f"{episode_idx=}")

        # Setup the environment
        env = make_sim_env()
        obs = env.reset()

        # Setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(obs.cameras[render_cam_name].rgb)
            plt.ion()

        horizon = 1

        # Run episode
        for i in range(400):
            nc.log_joint_positions(name=JOINT_GROUP_NAME, positions=obs.qpos)
            for key, value in obs.cameras.items():
                if key in obs_camera_names:
                    nc.log_rgb(name=key, rgb=value.rgb)
            idx_in_horizon = i % horizon
            if idx_in_horizon == 0:
                predicted_sync_points = policy.predict(timeout=5)
                joint_target_positions = [
                    sp.data[DataType.JOINT_TARGET_POSITIONS]
                    for sp in predicted_sync_points
                ]
                actions = []
                for jtp in joint_target_positions:
                    joint_data = cast(JointData, jtp[JOINT_GROUP_NAME])
                    actions.append(
                        [joint_data.values[jname] for jname in env.ACTION_KEYS]
                    )
                horizon = len(actions)

            a = actions[idx_in_horizon]
            obs, reward, done = env.step(a)

            if onscreen_render:
                plt_img.set_data(obs.cameras[render_cam_name].rgb)
                plt.pause(0.002)
        plt.close()

    policy.disconnect()


if __name__ == "__main__":
    main()
