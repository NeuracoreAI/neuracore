"""On-disk video frame cache: its lock protocol and its decoding step.

Frames live at ``<cache_dir>/<recording_id>/<data_type>/<sensor_id>/<idx>.png``.
RGB frames cached at a fixed size live instead in one uint8 array of shape
(frames, height, width, 3) at <sensor_id>.<height>x<width>/frames.npy, next to
the full resolution directory.
A directory is published with a single ``os.replace`` once decoding finishes, so
one that exists is always complete; while it is being produced, a sibling
``<sensor_id>.recording.lock`` marks it as owned.
"""

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from neuracore_types import DataType
from PIL import Image

logger = logging.getLogger(__name__)

STALE_LOCK_TIMEOUT_S = 300
"""Age at which a lock is assumed abandoned by a dead worker."""

PNG_COMPRESSION_LEVEL = 3
"""zlib level used when writing cached frames."""

FRAME_ARRAY_FILENAME = "frames.npy"
"""File holding every frame of a camera cached at a fixed size."""

FRAME_ARRAY_DECODE_CHUNK = 64
"""Frames decoded before each resize and write to the frame array."""

_RGB_VIDEO_FILENAME_PREFERENCE = ("lossless.mp4", "lossy.mp4")
_DEPTH_VIDEO_FILENAME_PREFERENCE = ("lossless.mp4",)

_FFMPEG_AVAILABLE: bool | None = None

# `-fps_mode` is accepted from ffmpeg 5.1 onwards and is the only spelling
# accepted from 8.0, where the legacy `-vsync` name was removed. Resolved
# once per process and cached, like _FFMPEG_AVAILABLE above.
_FPS_MODE_ARG = "-fps_mode"
_VSYNC_ARG = "-vsync"
_FFMPEG_FRAME_SYNC_ARG: str | None = None


def video_filename_preference(camera_type: DataType) -> tuple[str, ...]:
    """Return the video filenames to try for a camera type, most preferred first.

    Args:
        camera_type: Type of camera (e.g. rgb_images, depth_images).

    Returns:
        Candidate filenames in preference order.
    """
    if camera_type == DataType.DEPTH_IMAGES:
        return _DEPTH_VIDEO_FILENAME_PREFERENCE
    return _RGB_VIDEO_FILENAME_PREFERENCE


def frames_dir_for(
    cache_dir: Path,
    recording_id: str,
    data_type: DataType,
    sensor_id: str,
    rgb_frame_size: tuple[int, int] | None = None,
) -> Path:
    """Return the directory a camera's cached frames are published to.

    Args:
        cache_dir: Root of the recording cache.
        recording_id: Recording the camera belongs to.
        data_type: Type of camera.
        sensor_id: Camera identifier.
        rgb_frame_size: Height and width RGB frames are fitted within, or None
            for full resolution frames.

    Returns:
        The frames directory, named after the frame size when RGB frames are
        cached at a fixed size.
    """
    camera_type_dir = cache_dir / recording_id / data_type.value
    if rgb_frame_size is None or data_type != DataType.RGB_IMAGES:
        return camera_type_dir / sensor_id
    height, width = rgb_frame_size
    return camera_type_dir / f"{sensor_id}.{height}x{width}"


def fit_within(height: int, width: int, size: tuple[int, int]) -> tuple[int, int]:
    """Return the largest frame shape within size that keeps the aspect ratio.

    Uses the same rounding as the ResizePad preprocessing method, so ResizePad
    leaves a fitted frame at its shape and only pads it.

    Args:
        height: Source frame height.
        width: Source frame width.
        size: Target height and width.

    Returns:
        Fitted height and width.
    """
    scale = min(size[0] / height, size[1] / width)
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def resize_frames(frames: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a stack of uint8 frames with bilinear interpolation.

    Args:
        frames: Frames of shape (frames, height, width, 3).
        shape: Target height and width.

    Returns:
        Resized uint8 frames of shape (frames, height, width, 3).
    """
    if frames.shape[1:3] == shape:
        return frames

    import torch

    channels_first = torch.from_numpy(frames).permute(0, 3, 1, 2)
    resized = torch.nn.functional.interpolate(
        channels_first, size=shape, mode="bilinear"
    )
    return resized.permute(0, 2, 3, 1).contiguous().numpy()


def decode_video_to_array(
    video_location: Path, output_file: Path, size: tuple[int, int]
) -> None:
    """Decode every frame of a video, fit it within size and write one npy array.

    The frames stream to disk in chunks, so memory stays bounded for any video
    length.

    Args:
        video_location: Path to the video file.
        output_file: Path of the npy file to write.
        size: Height and width to fit each frame within.
    """
    import av

    with av.open(str(video_location)) as container, open(output_file, "wb") as out:
        stream = container.streams.video[0]
        stream.thread_count = 1
        height, width = fit_within(
            stream.codec_context.height, stream.codec_context.width, size
        )

        def write_header(num_frames: int) -> int:
            np.lib.format.write_array_header_1_0(
                out,
                {
                    "descr": np.lib.format.dtype_to_descr(np.dtype(np.uint8)),
                    "fortran_order": False,
                    "shape": (num_frames, height, width, 3),
                },
            )
            return out.tell()

        header_length = write_header(0)
        num_frames = 0
        chunk: list[np.ndarray] = []

        def flush() -> None:
            nonlocal num_frames
            resize_frames(np.stack(chunk), (height, width)).tofile(out)
            num_frames += len(chunk)
            chunk.clear()

        for frame in container.decode(stream):
            chunk.append(frame.to_ndarray(format="rgb24"))
            if len(chunk) == FRAME_ARRAY_DECODE_CHUNK:
                flush()
        if chunk:
            flush()

        # The npy header pads the first axis for growth, so the final frame
        # count fits in the space the placeholder header took.
        out.seek(0)
        if write_header(num_frames) != header_length:
            raise RuntimeError(f"Frame array header for {output_file} changed size")


def lock_file_for(frames_dir: Path) -> Path:
    """Return the lock guarding a frames directory.

    The lock is a sibling of the directory, not inside it, so the directory can
    be published atomically.

    Args:
        frames_dir: Directory the frames are published to.

    Returns:
        Path of the guarding lock file.
    """
    return frames_dir.parent / f"{frames_dir.name}.recording.lock"


def point_cloud_lock_file_for(sensor_root: Path) -> Path:
    """Return the lock guarding a point cloud sensor's cached frames.

    Unlike video frames, point cloud frames are written individually into
    ``sensor_root`` rather than published by renaming it, so the lock lives
    inside that directory.

    Args:
        sensor_root: Directory the sensor's frames are written to.

    Returns:
        Path of the guarding lock file.
    """
    return sensor_root / ".recording.lock"


def create_decoding_lock(lock_file: Path) -> bool:
    """Try to take a decoding lock.

    Args:
        lock_file: Path of the lock to create.

    Returns:
        True if the lock was taken, False if another worker already holds it.
    """
    try:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file.touch(exist_ok=False)
    except FileExistsError:
        return False
    return True


def acquire_decoding_lock(lock_file: Path, sensor_id: str) -> None:
    """Take a decoding lock, refusing to proceed if another worker holds it.

    Args:
        lock_file: Path of the lock to create.
        sensor_id: Sensor the lock guards, used in the error message.

    Raises:
        RuntimeError: If another process already holds the lock.
    """
    if not create_decoding_lock(lock_file):
        raise RuntimeError(
            f"Another process is already decoding video for camera {sensor_id}"
        )


def delete_decoding_lock(lock_file: Path) -> None:
    """Remove a decoding lock if present.

    Args:
        lock_file: Path of the lock to remove.
    """
    lock_file.unlink(missing_ok=True)


def check_stale_lock_file(lock_file: Path, timeout: int = STALE_LOCK_TIMEOUT_S) -> bool:
    """Whether a lock is old enough to be treated as abandoned.

    Args:
        lock_file: Path to the lock file.
        timeout: Age in seconds after which the lock is considered stale.

    Returns:
        True if the lock exists and is stale, False otherwise.
    """
    if not lock_file.exists():
        return False
    return (time.time() - lock_file.stat().st_mtime) > timeout


def clear_stale_lock(lock_file: Path, guarded_dir: Path) -> None:
    """Drop an abandoned lock and whatever partial output it guarded.

    Args:
        lock_file: Path to the stale lock file.
        guarded_dir: Directory to clear along with the lock.
    """
    logger.warning(f"Stale lock file detected at {lock_file}. Removing lock.")
    delete_decoding_lock(lock_file)
    shutil.rmtree(guarded_dir, ignore_errors=True)
    logger.info(f"Removed stale lock and cleared cache at {guarded_dir}.")


def wait_for_lock_release(
    lock_file: Path, parent_folder_path: Path, check_interval: int = 1
) -> None:
    """Block until a decoding lock is released, or clear it if it is stale.

    Args:
        lock_file: Path to the lock file.
        parent_folder_path: Directory the lock guards, cleared if it is stale.
        check_interval: Seconds between checks.
    """
    while lock_file.exists():
        if check_stale_lock_file(lock_file):
            clear_stale_lock(lock_file, parent_folder_path)
            break
        time.sleep(check_interval)


def _resolve_frame_sync_arg() -> str:
    """Return the flag the local ffmpeg accepts for passthrough frame timing.

    Probes with one synthetic frame piped to the null muxer: `-fps_mode` fails
    with "Unrecognized option" on ffmpeg < 5.1, so any other outcome (success,
    or a failure unrelated to the flag) is treated as accepted.

    Returns:
        "-fps_mode" if the local ffmpeg accepts it, otherwise "-vsync".
    """
    global _FFMPEG_FRAME_SYNC_ARG
    if _FFMPEG_FRAME_SYNC_ARG is not None:
        return _FFMPEG_FRAME_SYNC_ARG

    frame = bytes([128]) * (16 * 16 * 3 // 2)  # one 16x16 yuv420p frame
    try:
        probe = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "-video_size",
                "16x16",
                "-i",
                "-",
                _FPS_MODE_ARG,
                "passthrough",
                "-f",
                "null",
                "-",
            ],
            input=frame,
            capture_output=True,
        )
        rejected = b"Unrecognized option" in probe.stderr
    except FileNotFoundError:
        rejected = True

    _FFMPEG_FRAME_SYNC_ARG = _VSYNC_ARG if rejected else _FPS_MODE_ARG
    return _FFMPEG_FRAME_SYNC_ARG


def decode_video(video_location: Path, video_frame_cache_path: Path) -> None:
    """Extract every frame from a video and write them as PNGs.

    Args:
        video_location: Path to the video file.
        video_frame_cache_path: Directory to write the frames into.
    """
    global _FFMPEG_AVAILABLE

    # Lazily determine ffmpeg availability once
    if _FFMPEG_AVAILABLE is None:
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            _FFMPEG_AVAILABLE = True
        except (FileNotFoundError, subprocess.CalledProcessError):
            _FFMPEG_AVAILABLE = False
            logger.warning(
                "ffmpeg not found. Falling back to PyAV for video decoding. "
                "Install ffmpeg for significantly faster decoding."
            )

    if _FFMPEG_AVAILABLE:
        output_pattern = str(video_frame_cache_path / "%d.png")
        frame_sync_arg = _resolve_frame_sync_arg()
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    str(video_location),
                    frame_sync_arg,
                    "passthrough",
                    "-pix_fmt",
                    "rgb24",
                    "-compression_level",
                    str(PNG_COMPRESSION_LEVEL),
                    "-start_number",
                    "0",
                    output_pattern,
                    "-y",
                    "-loglevel",
                    "error",
                ],
                check=True,
                capture_output=True,
            )
            return
        except subprocess.CalledProcessError:
            logger.error("ffmpeg failed during decoding, falling back to PyAV")
            _FFMPEG_AVAILABLE = False  # Permanently disable ffmpeg for this run

    # PyAV fallback (executed only once ffmpeg is known unavailable)
    import av

    with av.open(str(video_location)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            frame_image = Image.fromarray(frame.to_rgb().to_ndarray())
            frame_file = video_frame_cache_path / f"{i}.png"
            frame_image.save(frame_file, compress_level=PNG_COMPRESSION_LEVEL)


def publish_decoded_frames(
    video_path: Path,
    staging_dir: Path,
    frames_dir: Path,
    frame_size: tuple[int, int] | None = None,
) -> None:
    """Decode a video into a staging directory and publish it atomically.

    Does nothing if the frames were published in the meantime, so a reader only
    ever sees a complete directory.

    Args:
        video_path: Video to decode.
        staging_dir: Empty directory on the same filesystem to decode into.
        frames_dir: Final location to publish the frames to.
        frame_size: Height and width to fit frames within, written as one frame
            array. None writes full resolution PNG frames.
    """
    if frames_dir.exists():
        return
    if frame_size is None:
        decode_video(video_path, staging_dir)
    else:
        decode_video_to_array(
            video_path, staging_dir / FRAME_ARRAY_FILENAME, frame_size
        )
    os.replace(staging_dir, frames_dir)
