"""Load episode data (e.g. RGB video/npy) for streaming tests."""

from pathlib import Path

import numpy as np


def load_rgb_frames(episode_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load RGB timestamps and frames from an episode directory.

    - If rgb_timestamps.npy does not exist, returns None (no RGB data).
    - If rgb_timestamps.npy exists but neither rgb_video.mkv nor rgb_video.npy
      exists, raises FileNotFoundError.
    - Otherwise loads video or npy into memory as uint8 RGB frames and returns
      (timestamps, frames) so the caller can build events with pre-loaded images.

    Returns:
        (timestamps, frames) as numpy arrays, or None if no RGB timestamps file.
    """
    ts_rgb_path = episode_dir / "rgb_timestamps.npy"
    video_path = episode_dir / "rgb_video.mkv"

    if not ts_rgb_path.exists():
        raise FileNotFoundError(f"RGB timestamps file not found at {ts_rgb_path}")

    if not video_path.exists():
        raise FileNotFoundError(
            f"RGB timestamps found at {ts_rgb_path} but "
            f"{video_path.name} not found in {episode_dir}"
        )

    ts_rgb = np.load(ts_rgb_path)

    import imageio.v3 as iio

    frames = iio.imread(str(video_path))
    # Video was saved as BGR0; convert to RGB
    if frames.ndim == 4 and frames.shape[-1] == 4:
        bgr = frames[:, :, :, :3]
        frames = bgr[:, :, :, ::-1]
    frames = np.asarray(frames, dtype=np.uint8)

    if len(frames) == 0 or len(ts_rgb) != len(frames):
        raise ValueError(
            f"RGB timestamps length ({len(ts_rgb)}) does not match "
            f"frames length ({len(frames)}) in {episode_dir}"
        )

    return ts_rgb, frames
