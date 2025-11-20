import logging
from pathlib import Path

import av
import matplotlib.pyplot as plt
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH, BOX_POSE, make_sim_env

import neuracore as nc

TRAINING_JOB_NAME = "MyTrainingJob"

logger = logging.getLogger(__name__)


def save_frames_to_video(
    frames, output_path: Path, width: int, height: int, fps: int = 30
):
    """Save a list of RGB frames to an MP4 video file.

    Args:
        frames: List of numpy arrays of shape (H, W, 3) with dtype uint8
        output_path: Path to save the video file
        width: Video width in pixels
        height: Video height in pixels
        fps: Frames per second for the output video
    """
    with av.open(str(output_path), mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height

        for frame_array in frames:
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush encoder
        for packet in stream.encode(None):
            container.mux(packet)


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    nc.login()
    nc.connect_robot(
        robot_name="Mujoco VX300s",
        urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
        overwrite=False,
    )
    # If you have a train run name, you can use it to connect to a local. E.g.:
    policy = nc.policy(
        train_run_name=TRAINING_JOB_NAME,
    )

    # If you know the path to the local model.nc.zip file, you can use it directly as:
    # policy = nc.policy(model_file=PATH_TO_MODEL.NC.ZIP)

    # Alternatively, you can connect to a local endpoint that has been started
    # policy = nc.policy_local_server(train_run_name=TRAINING_JOB_NAME)

    # Optional. Set the checkpoint to the last epoch.
    # Note by default, model is loaded from the last epoch.
    # policy.set_checkpoint(epoch=-1)

    onscreen_render = True
    save_video = False  # Set to True to save video files
    video_output_dir = Path("videos")  # Directory to save videos
    render_cam_name = "top"
    obs_camera_names = ["top"]
    num_rollouts = 30
    successful_rollouts = 0

    # Create video output directory if saving videos
    if save_video:
        video_output_dir.mkdir(exist_ok=True)

    for episode_idx in range(num_rollouts):
        logger.info(f"{episode_idx=}")

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

        # Collect frames for video saving
        frames = [] if save_video else None
        if save_video:
            # Get frame dimensions from first observation
            first_frame = obs.cameras[render_cam_name].rgb
            frame_height, frame_width = first_frame.shape[:2]
            # Add initial frame
            frames.append(first_frame.copy())

        horizon = 1
        # Run episode
        for i in range(400):
            nc.log_joint_positions(obs.qpos)
            for key, value in obs.cameras.items():
                if key in obs_camera_names:
                    nc.log_rgb(key, value.rgb)
            idx_in_horizon = i % horizon
            if idx_in_horizon == 0:
                predicted_sync_points = policy.predict(timeout=5)
                joint_target_positions = [
                    sp.joint_target_positions for sp in predicted_sync_points
                ]
                actions = [
                    jtp.numpy(order=env.ACTION_KEYS)
                    for jtp in joint_target_positions
                    if jtp is not None
                ]
                horizon = int(len(actions) * 0.5)
            a = actions[idx_in_horizon]
            obs, reward, done = env.step(a)

            # Collect frame for video
            if save_video:
                frames.append(obs.cameras[render_cam_name].rgb.copy())

            if onscreen_render:
                plt_img.set_data(obs.cameras[render_cam_name].rgb)
                plt.pause(0.002)

            if done:
                logger.info(f"Episode {episode_idx} done")
                break
        if reward == 4:
            successful_rollouts += 1
            print(f"Episode {episode_idx} successful.")
        else:
            print(f"Episode {episode_idx} failed with reward {reward}.")
        print(f"Successful rollouts: {successful_rollouts}/{num_rollouts}")
        # save the video
        if save_video:
            save_frames_to_video(
                frames,
                video_output_dir / f"episode_{episode_idx}.mp4",
                frame_width,
                frame_height,
            )

        plt.close()

    policy.disconnect()


if __name__ == "__main__":
    main()
