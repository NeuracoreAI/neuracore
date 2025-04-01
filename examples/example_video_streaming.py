import time
import numpy as np
import neuracore as nc
from common.constants import BIMANUAL_VIPERX_URDF_PATH

def generate_wave_pattern(
    width, height, frame_index, camera_id, channel, frequency=0.02, speed=0.2
):
    """Generate a dynamic wave pattern for grayscale simulation using NumPy vectorization."""
    phase_offset = camera_id * np.pi / 2 + channel * np.pi / 3

    # Create coordinate grids
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Compute wave pattern using vectorized operations
    wave_pattern = 127.5 * (1 + np.sin(frequency * (x + y) + speed * frame_index + phase_offset))
    image = wave_pattern.astype(np.uint8)

    return image

def simulate_camera_frames(num_frames=1_000_000, width=640, height=480, camera_id=0):
    """Generate test frames with variable timing for each camera."""
    t = 0.0
    for i in range(num_frames):
        # Create a test RGB frame using wave patterns for each channel
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for channel in range(3):
            frame[:, :, channel] = generate_wave_pattern(
                width, height, i, camera_id, channel
            )

        # Generate animated depth frame
        raw_depth_frame = generate_wave_pattern(width, height, i, camera_id, channel=0)
        float_depth_frame = (raw_depth_frame / 255.0).astype(np.float16)

        # Simulate irregular frame timing
        dt = 0.02 + 0.03 * np.random.random()  # Between 20-50ms
        t += dt
        time.sleep(dt)

        yield frame, float_depth_frame, t
    
    print(f"Camera {camera_id} Total time: ", t)

def camera_task(camera_id):
    for frame, depth_frame, timestamp in simulate_camera_frames(camera_id=camera_id):
        # print("logging depth")
        nc.log_depth(f"Camera {camera_id} Depth", depth_frame, timestamp=timestamp)
        nc.log_rgb(f"Camera {camera_id} RGB", frame, timestamp=timestamp)

def main():
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
    from threading import Thread
    threads = [Thread(target=camera_task, args=(i,)) for i in range(3)]
    
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
