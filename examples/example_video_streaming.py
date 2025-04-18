import random
import time
from functools import cache

import numpy as np
from common.constants import (
    BIMANUAL_VIPERX_URDF_PATH,
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
)

import neuracore as nc

FRAME_LOOP = 30 * 2


@cache
def generate_wave_pattern(width, height, phase_key: int, frequency=0.02):
    """Generate a dynamic wave pattern for grayscale simulation."""
    phase_offset = phase_key * (2 * np.pi / FRAME_LOOP)
    # Create coordinate grids
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Compute wave pattern using vectorized operations
    wave_pattern = 127.5 * (1 + np.sin(frequency * (x + y) + phase_offset))
    image = wave_pattern.astype(np.uint8)

    return image




def simulate_camera_frames(num_frames=1_000_000, width=50, height=50, camera_id=0):
    """Generate test frames with variable timing for each camera."""
    t = 0.0

    for i in range(num_frames):

        gen = np.random.default_rng(camera_id * 31 + 15)
        # Create a test RGB frame using wave patterns for each channel
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for channel in range(3):
            offset = int(gen.random() * FRAME_LOOP)
            phase_key = (i + offset) % FRAME_LOOP
            frame[:, :, channel] = generate_wave_pattern(width, height, phase_key)

        # Generate animated depth frame
        raw_depth_frame = generate_wave_pattern(
            width, height, (i + int(gen.random() * FRAME_LOOP)) % FRAME_LOOP
        )
        float_depth_frame = (raw_depth_frame / 255.0).astype(np.float16)

        # Simulate irregular frame timing
        dt = 0.02 + 0.03 * np.random.random()  # Between 20-50ms
        time.sleep(dt)
        t += dt

        yield frame, float_depth_frame, t

    print(f"Camera {camera_id} Total time: ", t)


def camera_task(camera_id):
    for frame, depth_frame, timestamp in simulate_camera_frames(camera_id=camera_id):
        nc.log_depth(f"Camera {camera_id} Depth", depth_frame, timestamp=timestamp)
        nc.log_rgb(f"Camera {camera_id} RGB", frame, timestamp=timestamp)


def joint_task(num_frames=250):
    joint_names = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES

    # Partition joints into 5 groups
    random.shuffle(joint_names)
    num_groups = 5
    joint_groups = [joint_names[i::num_groups] for i in range(num_groups)]

    # Base angles for all joints
    base_angles = {name: 0.0 for name in joint_names}

    # Time variable for sinusoidal movement
    t = 0.0

    for _ in range(num_frames):
        # Pick a random group of joints to update
        joints_to_update = random.choice(joint_groups)

        # Create a new joint position dictionary
        joint_positions = base_angles.copy()

        for joint in joints_to_update:
            gen = np.random.default_rng(
                np.frombuffer(joint.encode("utf-8"), dtype=np.uint8)
            )
            frequency = 0.1 + 0.05 * gen.random()
            amplitude = 1.5 + 1.5 * gen.random()

            joint_positions[joint] = (
                amplitude * np.sin(frequency * t + gen.random() * np.pi * 2)
                + 0.05 * gen.random()
            )

        # Log the joint positions
        nc.log_joint_positions(
            {joint: joint_positions[joint] for joint in joints_to_update}
        )

        # Sleep for a random duration
        dt = 0.02 + 0.03 * np.random.random()  # Between 20-50ms
        time.sleep(dt)

        # Update time
        t += dt



def main():
    """Main function for running the robot demo and logging with neuracore."""
    # Initialize neuracore
    nc.login()
    nc.connect_robot(
        robot_name="Test Video Robot",
        urdf_path=BIMANUAL_VIPERX_URDF_PATH,
        overwrite=True,
        instance=0,
    )

    nc.create_dataset(name="Test Video Dataset", description="This is a test dataset", tags=["test"])
    print("Created Dataset...")

    nc.start_recording()

    # Run four camera streams concurrently
    from threading import Thread

    threads = [Thread(target=camera_task, args=(i,)) for i in range(0)]

    threads.append(Thread(target=joint_task))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    print("Finishing recording...")
    nc.stop_recording()
    print("Finished recording!")
    nc.stop_live_data()


if __name__ == "__main__":
    main()
