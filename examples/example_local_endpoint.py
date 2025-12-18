from pathlib import Path
from typing import cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from common.base_env import BimanualViperXTask
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH, BOX_POSE, make_sim_env
from neuracore_types import BatchedJointData, BatchedNCData, DataSpec, DataType

import neuracore as nc

TRAINING_JOB_NAME = "MyTrainingJob"
VIDEO_OUTPUT_DIR = Path(__file__).parent / "videos"

CAMERA_NAMES = ["angle"]

# Specification of the order that will be fed into the model
MODEL_INPUT_ORDER: DataSpec = {
    DataType.JOINT_POSITIONS: (
        BimanualViperXTask.LEFT_ARM_JOINT_NAMES
        + BimanualViperXTask.LEFT_GRIPPER_JOINT_NAMES
        + BimanualViperXTask.RIGHT_ARM_JOINT_NAMES
        + BimanualViperXTask.RIGHT_GRIPPER_JOINT_NAMES
    ),
    DataType.RGB_IMAGES: CAMERA_NAMES,
}

MODEL_OUTPUT_ORDER: DataSpec = {
    DataType.JOINT_TARGET_POSITIONS: (
        BimanualViperXTask.LEFT_ARM_JOINT_NAMES
        + BimanualViperXTask.RIGHT_ARM_JOINT_NAMES
    ),
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: ["left_arm", "right_arm"],
}


def save_video(
    frames: list[np.ndarray],
    output_path: Path,
    fps: int = 30,
) -> None:
    """Save a list of RGB frames as an MP4 video.

    Args:
        frames: List of RGB numpy arrays (H, W, 3).
        output_path: Path to save the video file.
        fps: Frames per second for the output video.
    """
    if not frames:
        print("No frames to save.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]
    codec = cv2.VideoWriter_fourcc(*"mp4v")  # cspell:ignore fourcc
    writer = cv2.VideoWriter(str(output_path), codec, fps, (width, height))

    for frame in frames:
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)

    writer.release()
    print(f"Video saved to {output_path}")


def save_video(
    frames: list[np.ndarray],
    output_path: Path,
    fps: int = 30,
) -> None:
    """Save a list of RGB frames as an MP4 video.

    Args:
        frames: List of RGB numpy arrays (H, W, 3).
        output_path: Path to save the video file.
        fps: Frames per second for the output video.
    """
    if not frames:
        print("No frames to save.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]
    codec = cv2.VideoWriter_fourcc(*"mp4v")  # cspell:ignore fourcc
    writer = cv2.VideoWriter(str(output_path), codec, fps, (width, height))

    for frame in frames:
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)

    writer.release()
    print(f"Video saved to {output_path}")


def main():
    nc.login()
    nc.connect_robot(
        robot_name="Mujoco VX300s",
        urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
        overwrite=False,
    )
    # If you have a train run name, you can use it to connect to a local. E.g.:
    policy = nc.policy(
        train_run_name=TRAINING_JOB_NAME,
        model_input_order=MODEL_INPUT_ORDER,
        model_output_order=MODEL_OUTPUT_ORDER,
    )

    # Alternatively, you can connect to a local endpoint that has been started
    # policy = nc.policy_local_server(
    #     train_run_name=TRAINING_JOB_NAME,
    #     model_input_order=MODEL_INPUT_ORDER,
    #     model_output_order=MODEL_OUTPUT_ORDER,
    # )
    # policy = nc.policy_local_server(
    #     train_run_name=TRAINING_JOB_NAME,
    #     model_input_order=MODEL_INPUT_ORDER,
    #     model_output_order=MODEL_OUTPUT_ORDER,
    # )

    # Optional. Set the checkpoint to the last epoch.
    # Note by default, model is loaded from the last epoch.
    # policy.set_checkpoint(epoch=-1)

    onscreen_render = False
    save_video_enabled = True
    render_cam_name = CAMERA_NAMES[0]
    num_rollouts = 10

    # Create video output directory
    if save_video_enabled:
        VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Videos will be saved to: {VIDEO_OUTPUT_DIR}")

    for episode_idx in range(num_rollouts):
        print(f"{episode_idx=}")

        # Setup the environment
        env = make_sim_env()
        # resample the initial cube pose
        BOX_POSE[0] = env.sample_box_pose()
        obs = env.reset()
        (
            arm_joint_positions,
            arm_joint_velocities,
            left_arm_gripper_open,
            right_arm_gripper_open,
        ) = env.extract_state()

        # Setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(obs.cameras[render_cam_name].rgb)
            plt.ion()

        # Collect frames for video saving
        episode_frames: list[np.ndarray] = []

        horizon = 1
        # Run episode
        for i in range(400):

            (
                arm_joint_positions,
                arm_joint_velocities,
                left_arm_gripper_open,
                right_arm_gripper_open,
            ) = env.extract_state()

            nc.log_joint_positions(positions=arm_joint_positions)

            nc.log_parallel_gripper_open_amounts(
                {"left_arm": left_arm_gripper_open, "right_arm": right_arm_gripper_open}
            )

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
                left_open_amount = open_amounts["left_arm"].open_amount
                right_open_amount = open_amounts["right_arm"].open_amount
                batched_action = (
                    torch.cat(
                        [left_arm, left_open_amount, right_arm, right_open_amount],
                        dim=2,
                    )
                    .cpu()
                    .numpy()
                )
                mj_action = batched_action[0]  # Get the first (and only) in the batch
                horizon = len(mj_action)

            obs, reward, done = env.step(mj_action[idx_in_horizon])

            # Collect frame for video
            if save_video_enabled:
                frame = obs.cameras[render_cam_name].rgb
                episode_frames.append(frame.copy())

            if onscreen_render:
                plt_img.set_data(obs.cameras[render_cam_name].rgb)
                plt.pause(0.002)

            if done:
                print(f"Episode {episode_idx} done")
                break

        # Determine success/failure
        success = reward == 4
        if success:
            print(f"Episode {episode_idx} successful.")
        else:
            print(f"Episode {episode_idx} failed.")

        # Save video for this episode
        if save_video_enabled and episode_frames:
            status = "success" if success else "fail"
            video_path = VIDEO_OUTPUT_DIR / f"episode_{episode_idx:03d}_{status}.mp4"
            save_video(episode_frames, video_path, fps=30)

        plt.close()

    policy.disconnect()


if __name__ == "__main__":
    main()
