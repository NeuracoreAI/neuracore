import asyncio
import numpy as np
import neuracore as nc
from common.constants import BIMANUAL_VIPERX_URDF_PATH

def generate_wave_pattern(width, height, frame_index, camera_id, channel, frequency=0.02, speed=0.1):
    """Generate a dynamic wave pattern for RGB simulation."""
    wave_pattern = np.zeros((height, width), dtype=np.uint8)
    phase_offset = camera_id * np.pi / 2 + channel * np.pi / 3
    for y in range(height):
        for x in range(width):
            wave_pattern[y, x] = int(
                127.5 * (1 + np.sin(frequency * (x + y) + speed * frame_index + phase_offset))
            )
    return wave_pattern


async def simulate_camera_frames(num_frames=1_000_000, width=640, height=480, camera_id=0):
    """Generate test frames with variable timing for each camera."""
    t = 0.0
    for i in range(num_frames):
        # Create a test RGB frame using wave patterns for each channel
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for channel in range(3):
            frame[:, :, channel] = generate_wave_pattern(width, height, i, camera_id, channel)

        # Generate animated depth frame
        depth_frame = generate_wave_pattern(width, height, i, camera_id, channel=0)

        # Simulate irregular frame timing
        dt = 0.02 + 0.03 * np.random.random()  # Between 20-50ms
        t += dt
        await asyncio.sleep(dt)

        yield frame, depth_frame, t

    print(f"Camera {camera_id} Total time: ", t)


async def camera_task(camera_id):
    async for frame, depth_frame, time in simulate_camera_frames(camera_id=camera_id):
        nc.log_rgb(f"Camera {camera_id} RGB", frame, timestamp=time)
        nc.log_depth(f"Camera {camera_id} Depth", depth_frame, timestamp=time)


async def main():
    """Main function for running the robot demo and logging with neuracore."""

    # Initialize neuracore
    nc.login()
    nc.connect_robot(
        robot_name="Test Video Robot",
        urdf_path=BIMANUAL_VIPERX_URDF_PATH,
        overwrite=True,
    )

    nc.create_dataset(name="Test Video Dataset", description="This is a test dataset")
    print("Created Dataset...")

    nc.start_recording()

    # Run four camera streams concurrently
    await asyncio.gather(*(camera_task(i) for i in range(4)))

    print("Finishing recording...")
    nc.stop_recording()
    print("Finished recording!")
    nc.stop_live_data()


if __name__ == "__main__":
    asyncio.run(main())
