import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import neuracore as nc


def visualize_episode(
    joint_positions: list[dict[float]], first_camera_images: list[np.ndarray]
):
    """Visualize an episode with joint positions and camera images side by side."""
    jps = np.array([list(jps.values()) for jps in joint_positions])
    print(jps)
    images = np.array(first_camera_images)

    # Create a more compact figure
    fig = plt.figure(figsize=(12, 4))

    # Plot joint positions
    ax1 = plt.subplot(1, 2, 1)
    for joint_idx in range(jps.shape[1]):
        ax1.plot(jps[:, joint_idx], label=f"Joint {joint_idx}")
    ax1.set_title("Joint Positions")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Position")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Camera feed
    ax2 = plt.subplot(1, 2, 2)
    img_display = ax2.imshow(images[0])
    ax2.set_title("Camera Feed")
    ax2.axis("off")

    # Time indicator and timestamp
    time_line = ax1.axvline(x=0, color="r")
    timestamp_text = ax2.text(
        0.02,
        0.95,
        f"Timestep: 0/{len(images)}",
        transform=ax2.transAxes,
        color="white",
        bbox=dict(facecolor="black", alpha=0.7),
    )

    plt.tight_layout()

    def update(frame):
        img_display.set_array(images[frame])
        time_line.set_xdata([frame, frame])
        timestamp_text.set_text(f"Timestep: {frame}/{len(images)}")
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


nc.login()
# CMU Play Fusion is one of the many public/shared datasets you have access to
dataset = nc.get_dataset("CMU Play Fusion")
print(f"Number of episodes: {len(dataset)}")
joint_positions = []
first_camera_images = []
for episode in dataset[:1]:
    print(f"Episode length is {len(episode)}")
    for step in episode:
        joint_positions.append(step["joint_positions"])
        if "images" in step:
            for cam_id, img in step["images"].items():
                first_camera_images.append(img)
                break

visualize_episode(joint_positions, first_camera_images)
