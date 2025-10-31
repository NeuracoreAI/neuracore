import matplotlib.pyplot as plt
import numpy as np
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH, BOX_POSE, make_sim_env

import neuracore as nc

TRAINING_JOB_NAME = "MyTrainingJob"


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
    # )
    # If you know the path to the local model.nc.zip file, you can use it directly as:
    policy = nc.policy(model_file="/home/kewang/Downloads/model.nc.zip")

    # Alternatively, you can connect to a local endpoint that has been started
    # policy = nc.policy_local_server(train_run_name=TRAINING_JOB_NAME)

    # Optional. Set the checkpoint to the last epoch.
    # Note by default, model is loaded from the last epoch.
    # policy.set_checkpoint(epoch=-1)

    onscreen_render = True
    render_cam_name = "angle"
    obs_camera_names = ["angle"]
    num_rollouts = 10

    for episode_idx in range(num_rollouts):
        print(f"{episode_idx=}")

        # Setup the environment
        env = make_sim_env()
        # resample the initial cube pose
        BOX_POSE[0] = env.sample_box_pose()
        obs = env.reset()
        for i in range(3):
            env.initialize_episode()

        # Setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(obs.cameras[render_cam_name].rgb)
            plt.ion()

        horizon = 1
        # Run episode
        for i in range(200):
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
                # INSERT_YOUR_CODE
                # Apply a moving average filter to smooth actions along the
                # second dimension

                def moving_average_actions(actions, window_size=3):
                    """Smooth actions along second dimension using moving average."""
                    if len(actions) == 0:
                        return actions
                    arr = np.stack(actions)  # shape: [N, D]
                    # Apply moving average along action dimension for each feature
                    pad_width = window_size // 2
                    np.pad(arr, ((0, 0), (pad_width, pad_width)), mode="edge")
                    smoothed = np.zeros_like(arr)
                    for i in range(arr.shape[1]):
                        smoothed[:, i] = np.convolve(
                            arr[:, i], np.ones(window_size) / window_size, mode="same"
                        )
                    return [smoothed[i] for i in range(arr.shape[0])]

                # Smooth the actions if there are more than 1
                if len(actions) > 1:
                    actions = moving_average_actions(actions, window_size=10)
                # INSERT_YOUR_CODE
                # Plot the action values along the second dimension
                # (across actions for each dimension)
                # plt.figure("Action values")
                # plt.clf()
                # if len(actions) > 0:
                #     actions_arr = np.stack(actions)
                #     for dim in range(actions_arr.shape[1]):
                #         plt.plot(actions_arr[:, dim], label=f'dim {dim}')
                #     plt.title("Predicted Action Values Across Actions")
                #     plt.xlabel("Action Index")
                #     plt.ylabel("Value")
                #     plt.legend()
                # plt.pause(0.001)
                horizon = int(len(actions) / 2)
            a = actions[idx_in_horizon]
            obs, reward, done = env.step(a)

            if onscreen_render:
                plt_img.set_data(obs.cameras[render_cam_name].rgb)
                plt.pause(0.002)

            if done:
                print(f"Episode {episode_idx} done")
                break
        plt.close()

    policy.disconnect()


if __name__ == "__main__":
    main()
