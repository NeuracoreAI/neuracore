"""This example demonstrates how you can run a local policy rollout
in a VX300s environment using Neuracore."""

from pathlib import Path
from typing import cast

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import torch
from common.base_env import BimanualViperXTask
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH, BOX_POSE, make_sim_env
from neuracore_types import BatchedJointData, BatchedNCData, DataSpec, DataType

import neuracore as nc

TRAINING_JOB_NAME = "MyTrainingJob"

CAMERA_NAMES = ["angle"]

TRAIN_INPUT_ORDER: list[str] = [
    "vx300s_left/wrist_angle",
    "vx300s_left/waist",
    "vx300s_left/right_finger",
    "vx300s_right/left_finger",
    "vx300s_right/wrist_angle",
    "vx300s_left/shoulder",
    "vx300s_left/wrist_rotate",
    "vx300s_right/shoulder",
    "vx300s_left/elbow",
    "vx300s_right/wrist_rotate",
    "vx300s_right/waist",
    "vx300s_right/elbow",
    "vx300s_left/left_finger",
    "vx300s_left/forearm_roll",
    "vx300s_right/forearm_roll",
    "vx300s_right/right_finger",
]

TRAIN_OUTPUT_ORDER: list[str] = ["vx300s_left/wrist_angle",
    "vx300s_left/waist",
    "vx300s_left/gripper_open",
    "vx300s_right/wrist_angle",
    "vx300s_left/shoulder",
    "vx300s_left/wrist_rotate",
    "vx300s_right/shoulder",
    "vx300s_left/elbow",
    "vx300s_right/wrist_rotate",
    "vx300s_right/waist",
    "vx300s_right/elbow",
    "vx300s_left/forearm_roll",
    "vx300s_right/forearm_roll",
    "vx300s_right/gripper_open",
]

# Specification of the order that will be fed into the model
MODEL_INPUT_ORDER: DataSpec = {
    # Map the simulated joint order to the training-time joint order
    # so that Pi05 sees proprio in the same order it was trained on.
    DataType.JOINT_POSITIONS: TRAIN_INPUT_ORDER,
    DataType.RGB_IMAGES: CAMERA_NAMES,
}

MODEL_OUTPUT_ORDER: DataSpec = {
    DataType.JOINT_TARGET_POSITIONS: TRAIN_OUTPUT_ORDER,
}

FREQUENCY = 49

def _save_episode_video(frames: list, video_path: Path, fps: int) -> None:
    if not frames:
        return
    video_path.parent.mkdir(parents=True, exist_ok=True)
    # Explicitly specify codec so PyAV does not get None
    imageio.mimwrite(str(video_path), frames, fps=fps, codec="libx264")
    print(f"Saved episode video to {video_path}")


def main():
    nc.login()
    nc.connect_robot(
        robot_name="Mujoco VX300s",
        urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
        overwrite=False,
    )
    # If you have a train run name, you can use it to connect to a local. E.g.:
    # policy = nc.policy(
    #     train_run_name=TRAINING_JOB_NAME,
    #     model_input_order=MODEL_INPUT_ORDER,
    #     model_output_order=MODEL_OUTPUT_ORDER,
    # )

    # If you know the path to the local model.nc.zip file, you can use it directly as:
    policy = nc.policy(
        model_file="/home/kewang/.neuracore/training/runs/gracious-murdock/artifacts/model.nc.zip",
        model_input_order=MODEL_INPUT_ORDER,
        model_output_order=MODEL_OUTPUT_ORDER,
    )

    # Alternatively, you can connect to a local endpoint that has been started
    # policy = nc.policy_local_server(
    #     train_run_name=TRAINING_JOB_NAME,
    #     model_input_order=MODEL_INPUT_ORDER,
    #     model_output_order=MODEL_OUTPUT_ORDER,
    # )

    # Optional. Set the checkpoint to the last epoch.
    # Note by default, model is loaded from the last epoch.
    # policy.set_checkpoint(epoch=-1)

    onscreen_render = False
    save_video = True
    render_cam_name = CAMERA_NAMES[0]
    num_rollouts = 10

    for episode_idx in range(num_rollouts):
        print(f"{episode_idx=}")

        # Setup the environment
        env = make_sim_env()
        # resample the initial cube pose
        BOX_POSE[0] = env.sample_box_pose()
        obs = env.reset()

        frames: list = []
        if save_video:
            frames.append(obs.cameras[render_cam_name].rgb.copy())

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
                horizon = int(len(mj_action)/2)

            obs, reward, done = env.step(mj_action[idx_in_horizon])

            if save_video:
                frames.append(obs.cameras[render_cam_name].rgb.copy())

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

        if save_video:
            video_path = Path("videos") / f"vx300s_episode_{episode_idx}.mp4"
            _save_episode_video(frames, video_path, fps=FREQUENCY // 2)

        plt.close()

    policy.disconnect()


if __name__ == "__main__":
    main()
