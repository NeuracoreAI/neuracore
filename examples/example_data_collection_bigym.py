"""This example demonstrates how you can collect data from a Bigym environment
and record it to Neuracore."""

import argparse
import time
from pathlib import Path
from typing import Any

try:
    import bigym
    from bigym_utils.utils import (
        FREQUENCY,
        JOINT_ACTUATORS,
        JOINT_NAMES,
        action_to_joint_action_dict,
        make_env,
        obs_to_imgs,
        obs_to_joint_dict,
    )
    from demonstrations.demo import Demo
    from demonstrations.demo_store import DemoStore
    from demonstrations.utils import Metadata
except ImportError:
    raise ImportError(
        "bigym is not installed and required to run this example. "
        "Please follow the instructions in the README.md file to install it "
        "from the Github repository."
    )

import neuracore as nc

DT = 1.0 / FREQUENCY


def run_episode(
    episode_idx: int,
    record: bool,
    demo: Demo,
    env: Any,
    render: bool,
    dataset: nc.Dataset | None = None,
) -> bool:
    """Run one demonstration episode and optionally record it."""
    print(f"\n=== Starting Episode {episode_idx} ===")

    obs, info = env.reset(seed=demo.seed)
    success = False
    t = time.time()

    try:
        # Start neuracore recording if required
        if record:
            assert dataset is not None
            dataset.start_recording()

        for step in demo._steps:
            # Log current joint positions and velocities
            qpos, qvel = obs_to_joint_dict(obs, JOINT_NAMES)
            nc.log_joint_positions(qpos, timestamp=t)
            nc.log_joint_velocities(qvel, timestamp=t)

            # Log current camera observation
            images = obs_to_imgs(obs)
            nc.log_rgb("head", images["head"], timestamp=t)

            # Log joint targets
            joint_action = action_to_joint_action_dict(
                step.info["demo_action"], JOINT_ACTUATORS
            )
            nc.log_joint_target_positions(joint_action, timestamp=t)

            # Apply demo action
            obs, reward, terminated, truncated, info = env.step(
                step.info["demo_action"]
            )

            print(
                f"Reward={reward}, terminated={terminated}, "
                f"truncated={truncated}, info={info}"
            )

            if render:
                env.render()

            # Increment timestamp
            t += DT

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
            assert dataset is not None
            if success:
                print("Episode successful → finalizing recording...")
                dataset.stop_recording(wait=True)
            else:
                print("Episode failed → cancelling recording...")
                dataset.cancel_recording()

    print(f"=== Episode {episode_idx} done | success={success} ===")
    return success


def main(num_episodes: int, record: bool, recording_name: str, render: bool) -> None:
    nc.login()

    # Connect to virtual robot (Get MJCF path from installed bigym package)
    # Update path as needed
    mjcf_path = Path(bigym.__file__).parent / "envs" / "xmls" / "h1" / "h1.xml"

    robot = nc.connect_robot(
        robot_name="Mujoco UnitreeH1 Example",
        mjcf_path=str(mjcf_path),
        overwrite=True,
    )

    print(f"Connected to robot: {robot.id}")
    print(f"Organisation ID: {nc.get_current_org()}")

    if record:
        # Create a dataset to start recording episodes into
        created_dataset = nc.create_dataset(
            name=recording_name,
            description="Example Bigym data collection on the ReachTarget environment",
        )
        dataset = nc.connect_dataset(created_dataset)
        print("Created dataset.")
    else:
        dataset = None

    # Create ReachTarget Bigym environment
    env = make_env()
    metadata = Metadata.from_env(env)

    # Retrieve `num_episodes` expert demonstrations from Bigym
    demo_store = DemoStore()
    demos = demo_store.get_demos(metadata, amount=num_episodes, frequency=FREQUENCY)

    success_count = 0

    try:
        for episode_idx in range(num_episodes):
            success = run_episode(
                episode_idx=episode_idx,
                record=record,
                demo=demos[episode_idx],
                env=env,
                render=render,
                dataset=dataset,
            )

            if success:
                success_count += 1
                print(f"Successful demos: {success_count}/{episode_idx + 1}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        if record:
            assert dataset is not None
            dataset.cancel_recording()
    finally:
        env.close()
        print(
            f"\nFinished running {num_episodes} episodes → "
            f"{success_count} succeeded."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Bigym demo logging into neuracore."
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render the environment.",
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
        default="Example Bigym Data Collection",
        help="Dataset name when recording.",
    )

    args = parser.parse_args()
    main(
        num_episodes=args.num_episodes,
        record=args.record,
        recording_name=args.recording_name,
        render=args.render,
    )
