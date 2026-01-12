import sys
from typing import cast

import matplotlib.pyplot as plt
import torch
from common.base_env import BimanualViperXTask
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH, make_sim_env
from neuracore_types import BatchedJointData, BatchedNCData, DataType

import neuracore as nc
from neuracore import EndpointError

ENDPOINT_NAME = "MyExampleEndpoint"
CAMERA_NAMES = ["angle"]


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
            "Once you have trained a model, endpoints can be started at https://neuracore.com/dashboard/endpoints"
        )
        sys.exit(1)

    onscreen_render = True
    render_cam_name = "angle"

    for episode_idx in range(10):
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

            nc.log_joint_positions(positions=obs.qpos)

            for key, value in obs.cameras.items():
                if key in CAMERA_NAMES:
                    nc.log_rgb(key, value.rgb)

            idx_in_horizon = i % horizon
            if idx_in_horizon == 0:
                predictions: dict[DataType, dict[str, BatchedNCData]] = policy.predict(
                    timeout=5
                )
                joint_target_positions = cast(
                    dict[str, BatchedJointData],
                    predictions[DataType.JOINT_TARGET_POSITIONS],
                )
                left_arm = torch.cat(
                    [
                        joint_target_positions[name].value
                        for name in BimanualViperXTask.LEFT_ARM_JOINT_NAMES
                    ],
                    dim=2,
                )
                right_arm = torch.cat(
                    [
                        joint_target_positions[name].value
                        for name in BimanualViperXTask.RIGHT_ARM_JOINT_NAMES
                    ],
                    dim=2,
                )
                left_open_amount = joint_target_positions[
                    BimanualViperXTask.LEFT_GRIPPER_OPEN
                ].value
                right_open_amount = joint_target_positions[
                    BimanualViperXTask.RIGHT_GRIPPER_OPEN
                ].value
                batched_action = (
                    torch.cat(
                        [left_arm, left_open_amount, right_arm, right_open_amount],
                        dim=2,
                    )
                    .cpu()
                    .numpy()
                )
                # Get first batch: (horizon, num_joints)
                mj_action = batched_action[0]
                horizon = len(mj_action)

            obs, reward, done = env.step(mj_action[idx_in_horizon])

            if onscreen_render:
                plt_img.set_data(obs.cameras[render_cam_name].rgb)
                plt.pause(0.002)

            if done:
                print(f"Episode {episode_idx} done")
                break
        if reward == 4:
            print(f"Episode {episode_idx} successful.")
        else:
            print(f"Episode {episode_idx} failed.")

        plt.close()

    policy.disconnect()


if __name__ == "__main__":
    main()
