"""On-disk video frame cache: its lock protocol and its decoding step.

Frames live at ``<cache_dir>/<recording_id>/<data_type>/<sensor_id>/<idx>.png``.
A depth camera with JPEG-XL frames keeps them at <sensor_id>/lossless.bin,
where the trace entry of each frame holds its byte range.
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
from neuracore_types.utils.depth_utils import metres_to_depth_u16
from PIL import Image

from neuracore.core.utils.depth_utils import rgb_to_depth_storage

logger = logging.getLogger(__name__)

STALE_LOCK_TIMEOUT_S = 300
"""Age at which a lock is assumed abandoned by a dead worker."""

PNG_COMPRESSION_LEVEL = 3
"""zlib level used when writing cached frames."""

DEPTH_FRAMES_FILENAME = "lossless.bin"
"""Name of the JPEG-XL depth frames file inside its camera directory."""

_JXL_SIGNATURES = (b"\xff\x0a", b"\x00\x00\x00\x0cJXL \r\n\x87\n")

_RGB_VIDEO_FILENAME_PREFERENCE = ("lossless.mp4", "lossy.mp4")
_DEPTH_VIDEO_FILENAME_PREFERENCE = (DEPTH_FRAMES_FILENAME, "lossless.mp4")

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
    video_path: Path, staging_dir: Path, frames_dir: Path
) -> None:
    """Decode a video into a staging directory and publish it atomically.

    Does nothing if the frames were published in the meantime, so a reader only
    ever sees a complete directory.

    Args:
        video_path: Video to decode.
        staging_dir: Empty directory on the same filesystem to decode into.
        frames_dir: Final location to publish the frames to.
    """
    if frames_dir.exists():
        return
    decode_video(video_path, staging_dir)
    os.replace(staging_dir, frames_dir)


def decode_depth_frame(payload: bytes) -> np.ndarray:
    """Decode one JPEG-XL depth frame.

    Args:
        payload: JPEG-XL codestream of the frame.

    Returns:
        The uint16 frame of sensor units.

    Raises:
        ValueError: If the frame decodes to another dtype.
    """
    import imagecodecs

    frame = np.asarray(imagecodecs.jpegxl_decode(payload))
    if frame.dtype != np.uint16:
        raise ValueError(f"Depth frame decodes as {frame.dtype}, expected uint16")
    return frame


def is_depth_frames_file(media_path: Path) -> bool:
    """Return whether a downloaded depth file holds JPEG-XL frames.

    Args:
        media_path: Downloaded depth file.

    Returns:
        True for JPEG-XL frames, False for a video.
    """
    with open(media_path, "rb") as handle:
        header = handle.read(max(len(signature) for signature in _JXL_SIGNATURES))
    return header.startswith(_JXL_SIGNATURES)


def publish_depth_frames(media_path: Path, staging_dir: Path, frames_dir: Path) -> None:
    """Publish a downloaded depth frames file or packed depth video atomically.

    A frames file moves into the staging directory unchanged. A packed depth
    video decodes to PNG frames.
    Does nothing if the frames were published in the meantime, so a reader only
    ever sees a complete directory.

    Args:
        media_path: Downloaded depth frames file or packed depth video, on the
            same filesystem as the staging directory.
        staging_dir: Empty directory on the same filesystem to stage into.
        frames_dir: Final location to publish the frames to.
    """
    if frames_dir.exists():
        return
    if not is_depth_frames_file(media_path):
        publish_decoded_frames(media_path, staging_dir, frames_dir)
        return
    os.replace(media_path, staging_dir / DEPTH_FRAMES_FILENAME)
    os.replace(staging_dir, frames_dir)


def publish_camera_frames(
    camera_type: DataType, media_path: Path, staging_dir: Path, frames_dir: Path
) -> None:
    """Publish a downloaded camera video or depth frames file atomically.

    Args:
        camera_type: Data type of the camera.
        media_path: Downloaded video or depth frames file, on the same
            filesystem as the staging directory.
        staging_dir: Empty directory on the same filesystem to stage into.
        frames_dir: Final location to publish the frames to.
    """
    publish = (
        publish_depth_frames
        if camera_type == DataType.DEPTH_IMAGES
        else publish_decoded_frames
    )
    publish(media_path, staging_dir, frames_dir)


def read_depth_frame(frames_dir: Path, offset: int, length: int) -> np.ndarray:
    """Read one JPEG-XL depth frame from a published directory.

    Args:
        frames_dir: Published depth camera directory holding lossless.bin.
        offset: Byte offset of the frame in lossless.bin.
        length: Length in bytes of the frame.

    Returns:
        The uint16 frame of shape (height, width) in sensor units.

    Raises:
        ValueError: If the file ends inside the frame.
    """
    frames_path = frames_dir / DEPTH_FRAMES_FILENAME
    with open(frames_path, "rb") as frames:
        frames.seek(offset)
        payload = frames.read(length)
    if len(payload) != length:
        raise ValueError(
            f"Depth frames file {frames_path} ends inside the frame at byte {offset}"
        )
    return decode_depth_frame(payload)


def read_packed_depth_frame(frames_dir: Path, frame_idx: int) -> np.ndarray:
    """Read one frame of a packed depth video from a published directory.

    Args:
        frames_dir: Published depth camera directory of packed PNG frames.
        frame_idx: Index of the frame to read.

    Returns:
        The uint16 frame of shape (height, width) in units of DEPTH_GRID_SCALE_M.
    """
    packed = np.asarray(Image.open(frames_dir / f"{frame_idx}.png"))
    return metres_to_depth_u16(rgb_to_depth_storage(packed))
