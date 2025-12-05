import argparse
import time

from bigym_utils.utils import (
    FREQUENCY,
    JOINT_ACTUATORS,
    JOINT_NAMES,
    action_to_joint_action_dict,
    make_env,
    obs_to_imgs,
    obs_to_joint_dict,
)
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata

import neuracore as nc

DT = 1.0 / FREQUENCY


def run_episode(
    episode_idx: int,
    record: bool,
    demo_store: DemoStore,
) -> bool:
    """Run one demonstration episode and optionally record it."""
    print(f"\n=== Starting Episode {episode_idx} ===")

    env = make_env()
    metadata = Metadata.from_env(env)

    # Get a single demo trajectory at the correct frequency
    demo = demo_store.get_demos(metadata, amount=1, frequency=FREQUENCY)[0]

    obs, info = env.reset(seed=demo.seed)
    success = False
    t = time.time()

    try:
        # Start neuracore recording if required
        if record:
            nc.start_recording()

        # Log example custom metadata
        nc.log_custom_data("my_custom_data", [1, 2, 3, 4, 5], timestamp=t)

        # Initial joint + camera logging
        qpos, qvel = obs_to_joint_dict(obs, JOINT_NAMES)
        nc.log_joint_positions(qpos, timestamp=t)
        nc.log_joint_velocities(qvel, timestamp=t)

        images = obs_to_imgs(obs)
        nc.log_rgb("head", images["head"], timestamp=t)

        nc.log_language(
            "Move two plates simultaneously from one draining rack to the other.",
            timestamp=t,
        )

        for step in demo._steps:
            # Apply demo action
            obs, reward, terminated, truncated, info = env.step(
                step.info["demo_action"]
            )
            print(
                f"Reward={reward}, terminated={terminated}, "
                f"truncated={truncated}, info={info}"
            )

            # Increment timestamp
            t += DT

            # Logging at each step
            nc.log_custom_data("my_custom_data", [1, 2, 3, 4, 5], timestamp=t)

            qpos, qvel = obs_to_joint_dict(obs, JOINT_NAMES)
            nc.log_joint_positions(qpos, timestamp=t)
            nc.log_joint_velocities(qvel, timestamp=t)

            images = obs_to_imgs(obs)
            nc.log_rgb("head", images["head"], timestamp=t)

            # Log joint targets
            joint_action = action_to_joint_action_dict(
                step.info["demo_action"], JOINT_ACTUATORS
            )
            nc.log_joint_target_positions(joint_action)

            # Check outcome
            if terminated and not truncated:
                success = True
                print("Episode terminated successfully.")
                break
            if truncated:
                print("Episode truncated (likely time limit).")
                break

    finally:
        # Clean up recording
        if record:
            if success:
                print("Episode successful → finalizing recording...")
                nc.stop_recording(wait=True)
            else:
                print("Episode failed → cancelling recording...")
                nc.cancel_recording()

        env.close()

    print(f"=== Episode {episode_idx} done | success={success} ===")
    return success


def main(num_episodes: int, record: bool, recording_name: str) -> None:
    nc.login()

    # Connect to virtual robot
    robot = nc.connect_robot(
        robot_name="Mujoco UnitreeH1 Example",
        mjcf_path="bigym/bigym/envs/xmls/h1/h1.xml",  # Update path as needed
        overwrite=True,
    )

    print(f"Connected to robot: {robot.id}")
    print(f"Organisation ID: {nc.get_current_org()}")

    if record:
        nc.create_dataset(
            name=recording_name,
            description="Example ReachTarget data collection",
        )
        print("Created dataset.")

    demo_store = DemoStore()

    success_count = 0

    try:
        for episode_idx in range(num_episodes):
            success = run_episode(
                episode_idx=episode_idx,
                record=record,
                demo_store=demo_store,
            )
            if success:
                success_count += 1
                print(f"Successful demos: {success_count}/{episode_idx + 1}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        if record:
            nc.cancel_recording()
    finally:
        print(
            f"\nFinished running {num_episodes} episodes → "
            f"{success_count} succeeded."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ReachTarget demo logging into neuracore."
    )
    parser.add_argument(
        "--num_episodes", type=int, default=50, help="Number of episodes to run."
    )
    parser.add_argument(
        "--record",
        action="store_true",
        default=False,
        help="Enable neuracore recording.",
    )
    parser.add_argument(
        "--recording_name",
        type=str,
        default="Example ReachTarget Data Collection",
        help="Dataset name when recording.",
    )

    args = parser.parse_args()
    main(
        num_episodes=args.num_episodes,
        record=args.record,
        recording_name=args.recording_name,
    )
