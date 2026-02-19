from typing import cast

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from neuracore_types import (
    DataType,
    JointData,
    RGBCameraData,
    RobotDataSpec,
    SynchronizedPoint,
)

import neuracore as nc


def visualize_episode(
    joint_positions: list[dict[str, JointData]],
    camera_data: list[dict[str, RGBCameraData]],
    timestamps: list[float],
    start_time: float = 0.0,
    end_time: float = 0.0,
):
    """Visualize an episode with joint positions and camera images side by side."""
    # Extract joint values from the first joint in the dict at each timestep
    # Assumes all dicts have the same keys and we want to visualize all joints
    joint_names = list(joint_positions[0].keys())
    jps = np.array([
        [joint_positions[t][name].value for name in joint_names]
        for t in range(len(joint_positions))
    ])

    # Extract frames from the first camera in the dict at each timestep
    # Assumes we want to visualize the first camera
    first_camera_name = list(camera_data[0].keys())[0]
    images = np.array(
        [camera_data[t][first_camera_name].frame for t in range(len(camera_data))]
    )

    # Calculate relative times from timestamps
    relative_times = np.array([t - start_time for t in timestamps])

    # Add a "fake" point at the end using end_time
    jps = np.vstack([jps, jps[-1:]])
    images = np.vstack([images, images[-1:]])
    relative_times = np.append(relative_times, end_time - start_time)

    # Create a more compact figure
    fig = plt.figure(figsize=(12, 4))

    # Plot joint positions
    ax1 = plt.subplot(1, 2, 1)
    for joint_idx, joint_name in enumerate(joint_names):
        ax1.plot(relative_times, jps[:, joint_idx], label=joint_name)
    ax1.set_title("Joint Positions")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Camera feed
    ax2 = plt.subplot(1, 2, 2)
    img_display = ax2.imshow(images[0])
    ax2.set_title("Camera Feed")
    ax2.axis("off")

    # Time indicator and timestamp
    time_line = ax1.axvline(x=relative_times[0], color="r")
    timestamp_text = ax2.text(
        0.02,
        0.95,
        f"Time: {relative_times[0]:.2f}s / {relative_times[-1]:.2f}s",
        transform=ax2.transAxes,
        color="white",
        bbox=dict(facecolor="black", alpha=0.7),
    )

    plt.tight_layout()

    def update(frame):
        img_display.set_array(images[frame])
        time_line.set_xdata([relative_times[frame], relative_times[frame]])
        timestamp_text.set_text(
            f"Time: {relative_times[frame]:.2f}s / {relative_times[-1]:.2f}s"
        )
        return [img_display, time_line, timestamp_text]

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(images), interval=50, blit=True, repeat=True
    )

    # Add play/pause button
    button_ax = plt.axes([0.45, 0.01, 0.1, 0.04])
    button = plt.Button(button_ax, "Play/Pause")

    def toggle_pause(event):
        if ani.running:
            ani.event_source.stop()
        else:
            ani.event_source.start()
        ani.running ^= True

    button.on_clicked(toggle_pause)
    ani.running = True

    plt.show()


def main():
    nc.login()

    # ASU Table Top is one of the many public/shared datasets you have access to
    dataset = nc.get_dataset("ASU Table Top")
    data_types_to_synchronize = [DataType.JOINT_POSITIONS, DataType.RGB_IMAGES]

    robot_data_spec: RobotDataSpec = {}
    robot_ids_dataset = dataset.robot_ids
    for robot_id in robot_ids_dataset:
        data_type_to_names = dataset.get_full_data_spec(robot_id)
        robot_data_spec[robot_id] = {
            data_type: data_type_to_names[data_type]
            for data_type in data_types_to_synchronize
        }

    synced_dataset = dataset.synchronize(
        frequency=1,
        robot_data_spec=robot_data_spec,
    )
    print(f"Number of episodes: {len(dataset)}")
    joint_positions = []
    camera_data = []
    timestamps = []

    print("Streaming first episode from dataset")
    for episode in synced_dataset[:1]:
        for step in episode:
            step = cast(SynchronizedPoint, step)
            joint_positions.append(step[DataType.JOINT_POSITIONS])
            camera_data.append(step[DataType.RGB_IMAGES])
            timestamps.append(step.timestamp)

    print(f"Episode length t: {episode.end_time - episode.start_time} seconds")
    visualize_episode(
        joint_positions, camera_data, timestamps, episode.start_time, episode.end_time
    )


if __name__ == "__main__":
    main()
