import matplotlib.pyplot as plt
from common.transfer_cube import BOX_POSE, make_sim_env
from neuracore_types import CameraData, JointData, SyncPoint

import neuracore as nc

TRAINING_JOB_NAME = "MyTrainingJob"
ROBOT_NAME = "Mujoco VX300s"


def main():
    # If you know the path to the local model.nc.zip file
    # you can use it directly without connecting to a robot
    policy = nc.policy(model_file="PATH/TO/MODEL.nc.zip")

    # Optional. Set the checkpoint to the last epoch.
    # Note by default, model is loaded from the last epoch.
    # policy.set_checkpoint(epoch=-1)

    onscreen_render = True
    render_cam_name = "angle"
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
            sp = SyncPoint(
                joint_positions=JointData(values=obs.qpos),
                rgb_images={
                    render_cam_name: CameraData(frame=obs.cameras[render_cam_name].rgb),
                },
            )
            idx_in_horizon = i % horizon
            if idx_in_horizon == 0:
                # No active robot, so we need to pass in the robot name
                predicted_sync_points = policy.predict(
                    sync_point=sp, robot_name=ROBOT_NAME, timeout=5
                )
                joint_target_positions = [
                    sp.joint_target_positions for sp in predicted_sync_points
                ]
                actions = [
                    jtp.numpy(order=env.ACTION_KEYS)
                    for jtp in joint_target_positions
                    if jtp is not None
                ]
                horizon = len(actions)
            a = actions[idx_in_horizon]
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
