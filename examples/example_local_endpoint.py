from pathlib import Path

import matplotlib.pyplot as plt
from common.constants import EPISODE_LENGTH
from common.ee_sim_env import sample_box_pose
from common.sim_env import BOX_POSE, make_sim_env

import neuracore as nc

THIS_DIR = Path(__file__).parent


def main():
    nc.login()
    policy = nc.connect_local_endpoint(THIS_DIR / "common" / "assets" / "model.mar")
    # If you have a train run name, you can use it to connect to a local. E.g.:
    # policy = nc.connect_local_endpoint(train_run_name="MyTrainRun")
    onscreen_render = True
    render_cam_name = "angle"

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
                nc.log_rgb(key, value)
            idx_in_horizon = i % horizon
            if idx_in_horizon == 0:
                action = policy.predict()
                horizon = action.shape[0]

            a = action[idx_in_horizon]
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
