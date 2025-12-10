from typing import cast

import matplotlib.pyplot as plt
import torch
from common.base_env import BimanualViperXTask
from common.transfer_cube import BOX_POSE, make_sim_env
from neuracore_types import (
    BatchedJointData,
    BatchedNCData,
    BatchedParallelGripperOpenAmountData,
    DataSpec,
    DataType,
    JointData,
    RGBCameraData,
    SynchronizedPoint,
)

import neuracore as nc

TRAINING_JOB_NAME = "MyTrainingJob"
ROBOT_NAME = "Mujoco VX300s"
CAMERA_NAMES = ["angle"]

# Specification of the order that will be fed into the model
MODEL_INPUT_ORDER: DataSpec = {
    DataType.JOINT_POSITIONS: BimanualViperXTask.LEFT_ARM_JOINT_NAMES
    + BimanualViperXTask.RIGHT_ARM_JOINT_NAMES,
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: ["left_arm", "right_arm"],
    DataType.RGB_IMAGES: CAMERA_NAMES,
}

MODEL_OUTPUT_ORDER: DataSpec = {
    DataType.JOINT_TARGET_POSITIONS: (
        BimanualViperXTask.LEFT_ARM_JOINT_NAMES
        + BimanualViperXTask.RIGHT_ARM_JOINT_NAMES
    ),
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: ["left_arm", "right_arm"],
}


def main():
    # If you know the path to the local model.nc.zip file
    # you can use it directly without connecting to a robot
    policy = nc.policy(
        model_file="PATH/TO/MODEL.nc.zip",
        model_input_order=MODEL_INPUT_ORDER,
        model_output_order=MODEL_OUTPUT_ORDER,
    )

    # Optional. Set the checkpoint to the last epoch.
    # Note by default, model is loaded from the last epoch.
    # policy.set_checkpoint(epoch=-1)

    onscreen_render = True
    render_cam_name = CAMERA_NAMES[0]
    num_rollouts = 10

    for episode_idx in range(num_rollouts):
        print(f"{episode_idx=}")

        # Setup the environment
        env = make_sim_env()
        # resample the initial cube pose
        BOX_POSE[0] = env.sample_box_pose()
        obs = env.reset()

        # Setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(obs.cameras[render_cam_name].rgb)
            plt.ion()

        horizon = 1
        # Run episode
        for i in range(400):
            # Create a sync point manually without logging data to the robot
            SynchronizedPoint(
                data={
                    DataType.JOINT_POSITIONS: {
                        k: JointData(value=v) for k, v in obs.qpos.items()
                    },
                    DataType.RGB_IMAGES: {
                        render_cam_name: RGBCameraData(
                            frame=obs.cameras[render_cam_name].rgb
                        ),
                    },
                }
            )

            idx_in_horizon = i % horizon
            if idx_in_horizon == 0:
                predictions: dict[DataType, dict[str, BatchedNCData]] = policy.predict(
                    timeout=5
                )
                joint_target_positions = cast(
                    dict[str, BatchedJointData],
                    predictions[DataType.JOINT_TARGET_POSITIONS],
                )
                open_amounts = cast(
                    dict[str, BatchedParallelGripperOpenAmountData],
                    predictions[DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS],
                )
                left_arm = torch.cat(
                    [
                        joint_target_positions[name].value
                        for name in BimanualViperXTask.LEFT_ARM_JOINT_NAMES
                    ],
                    dim=1,
                )
                right_arm = torch.cat(
                    [
                        joint_target_positions[name].value
                        for name in BimanualViperXTask.RIGHT_ARM_JOINT_NAMES
                    ],
                    dim=1,
                )
                left_open_amount = open_amounts[
                    BimanualViperXTask.LEFT_GRIPPER_OPEN
                ].open_amount
                right_open_amount = open_amounts[
                    BimanualViperXTask.RIGHT_GRIPPER_OPEN
                ].open_amount
                batched_action = torch.cat(
                    [left_arm, left_open_amount, right_arm, right_open_amount], dim=1
                ).numpy()
                mj_action = batched_action[0]  # Get the first (and only) in the batch
                horizon = len(mj_action)

            a = mj_action[idx_in_horizon]
            obs, reward, done = env.step(a)

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
