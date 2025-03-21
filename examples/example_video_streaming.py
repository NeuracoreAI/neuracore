import asyncio
import numpy as np
import neuracore as nc
from common.constants import BIMANUAL_VIPERX_URDF_PATH


async def simulate_camera_frames(num_frames=1_000_000, width=640, height=480):
    """Generate test frames with variable timing"""
    t = 0.0
    for i in range(num_frames):
        # Create a test frame (gradient that changes over time)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = int(255 * (i / num_frames))  # Red channel
        frame[:, :, 1] = 255 - int(255 * (i / num_frames))  # Green channel

        # first_half = i < num_frames // 2
        # if first_half:
        #     dt = 0.001
        # else:
        #     dt = 0.05

        # Simulate irregular frame timing
        dt = 0.02 + 0.03 * np.random.random()  # Between 20-50ms
        t += dt
        await asyncio.sleep(dt)

        yield frame, t

    print("Total time: ", t)


async def main():
    """Main function for running the robot demo and logging with neuracore."""

    # Initialize neuracore
    nc.login()
    nc.connect_robot(
        robot_name="Test Video Robot",
        urdf_path=BIMANUAL_VIPERX_URDF_PATH,
        overwrite=True,
    )

    nc.create_dataset(name="Test Video Dataset", description="This is an test dataset")
    print("Created Dataset...")

    nc.start_recording()

    async for frame, time in simulate_camera_frames():
        nc.log_rgb("test", frame, timestamp=time)

    print("Finishing recording...")
    nc.stop_recording()
    print("Finished recording!")
    nc.stop_live_data()


if __name__ == "__main__":
    asyncio.run(main())
