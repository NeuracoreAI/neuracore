"""Synchronized recording iterator."""

import copy
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import requests
import wget
from neuracore_types import CameraData, DataType, RobotDataSpec, SynchronizationDetails
from neuracore_types import SynchronizedEpisode as SynchronizedEpisodeModel
from neuracore_types import SynchronizedPoint, SynchronizeRecordingRequest
from PIL import Image

from neuracore.core.data.cache_manager import CacheManager

from ..auth import get_auth
from ..const import API_URL

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from neuracore.core.data.dataset import Dataset

MAX_DECODING_ATTEMPTS = 3
_FFMPEG_AVAILABLE: bool | None = None


class SynchronizedRecording:
    """Synchronized recording iterator."""

    def __init__(
        self,
        dataset: "Dataset",
        recording_id: str,
        robot_id: str,
        instance: int,
        start_time: float,
        end_time: float,
        frequency: int = 0,
        robot_data_spec: RobotDataSpec | None = None,
        prefetch_videos: bool = False,
    ):
        """Initialize episode iterator for a specific recording.

        Args:
            dataset: Parent Dataset instance.
            recording_id: Recording ID string.
            robot_id: The robot that created this recording.
            instance: The instance of the robot that created this recording.
            start_time: Start time of the recording.
            end_time: End time of the recording.
            frequency: Frequency at which to synchronize the recording.
            robot_data_spec: Robot data specification for synchronization.
            prefetch_videos: Whether to prefetch video data to cache on initialization.
        """
        self.dataset = dataset
        self.id = recording_id
        self.frequency = frequency
        self.robot_data_spec = robot_data_spec
        self.cache_dir: Path = dataset.cache_dir
        self.robot_id = robot_id
        self.instance = instance
        self.start_time = start_time
        self.end_time = end_time

        self._episode_synced = self._get_synced_data()
        self._episode_length = len(self._episode_synced.observations)
        self.cache_manager = CacheManager(
            self.cache_dir,
        )
        self._iter_idx = 0
        self._suppress_wget_progress = True

        if prefetch_videos:
            cache = self.dataset.cache_dir / self.id
            # Check if cache directory exists and contains any files
            self._wait_for_lock_release(cache / ".recording.lock", cache)
            # NOTE: this is to start video prefetching frames into cache
            self._get_sync_point(0)

    def _get_synced_data(self) -> SynchronizedEpisodeModel:
        """Retrieve synchronized metadata for the recording.

        Returns:
            SynchronizedEpisode object containing synchronized frames and metadata.

        Raises:
            requests.HTTPError: If the API request fails.
        """
        auth = get_auth()
        response = requests.post(
            f"{API_URL}/org/{self.dataset.org_id}/synchronize/synchronize-recording",
            json=SynchronizeRecordingRequest(
                recording_id=self.id,
                synchronization_details=SynchronizationDetails(
                    frequency=self.frequency,
                    robot_data_spec=self.robot_data_spec,
                    max_delay_s=sys.float_info.max,
                    allow_duplicates=True,
                ),
            ).model_dump(mode="json"),
            headers=auth.get_headers(),
        )
        response.raise_for_status()
        return SynchronizedEpisodeModel.model_validate(response.json())

    def _get_video_url(self, camera_type: DataType, camera_id: str) -> str:
        """Get streaming URL for a specific camera's video data.

        Args:
            camera_type: Type of camera (e.g., "rgbs", "depths").
            camera_id: Unique identifier for the camera.

        Returns:
            URL string for downloading the video file.

        Raises:
            requests.HTTPError: If the API request fails.
        """
        auth = get_auth()
        response = requests.get(
            f"{API_URL}/org/{self.dataset.org_id}/recording/{self.id}/download_url",
            params={"filepath": f"{camera_type.value}/{camera_id}/lossless.mp4"},
            headers=auth.get_headers(),
        )
        response.raise_for_status()
        return response.json()["url"]

    def _decode_video(self, video_location: Path, video_frame_cache_path: Path) -> None:
        """Extract frames from video and cache them to disk.

        Args:
            video_location: Path to the video file.
            video_frame_cache_path: Path to the directory where video frames are cached.
        """
        """Extract frames from video and cache them to disk."""
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
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        str(video_location),
                        "-vsync",
                        "0",
                        "-q:v",
                        "1",
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
                frame_image.save(frame_file)

    def _download_video_and_cache_frames_to_disk(
        self, camera_type: DataType, camera_id: str, video_frame_cache_path: Path
    ) -> None:
        """Download video and cache individual frames as images.

        Args:
            camera_type: Type of camera (e.g., "rgbs", "depths").
            camera_id: Unique identifier for the camera.
            video_frame_cache_path: Path to the directory where video frames are cached.
        """
        lock_file = video_frame_cache_path / ".recording.lock"
        lock_acquired = self._create_decoding_lock(lock_file, camera_id)

        try:
            # Create a temporary video file path
            self.cache_manager.ensure_space_available()

            with tempfile.TemporaryDirectory() as temp_dir:
                # Download video to temporary directory
                video_location = Path(temp_dir) / f"{camera_id}{camera_type.value}.mp4"
                wget.download(
                    self._get_video_url(camera_type, camera_id),
                    str(video_location),
                    bar=None if self._suppress_wget_progress else wget.bar_thermometer,
                )
                # Decode video to frames and cache them to disk
                self._decode_video(video_location, video_frame_cache_path)
        finally:
            if lock_acquired:
                self._delete_decoding_lock(lock_file)

    def _create_decoding_lock(self, lock_file: Path, camera_id: str) -> bool:
        """Create an exclusive lock file for decoding."""
        try:
            # Create the lock file exclusively
            lock_file.parent.mkdir(parents=True, exist_ok=True)
            lock_file.touch(exist_ok=False)
        except FileExistsError as exc:
            raise RuntimeError(
                f"Another process is already decoding video for camera {camera_id}"
            ) from exc
        return True

    def _delete_decoding_lock(self, lock_file: Path) -> None:
        """Remove the decoding lock file if present."""
        lock_file.unlink(missing_ok=True)

    def _check_stale_lock_file(self, lock_file: Path, timeout: int = 300) -> bool:
        """Check if a lock file is stale based on a timeout.

        Args:
            lock_file: Path to the lock file.
            timeout: Time in seconds after which the lock is considered stale.
                    (default: 300s/5min)

        Returns:
            True if the lock file is stale, False otherwise.
        """
        if not lock_file.exists():
            return False
        lock_mtime = lock_file.stat().st_mtime
        if (time.time() - lock_mtime) > timeout:
            return True
        return False

    def _wait_for_lock_release(
        self, lock_file: Path, parent_folder_path: Path, check_interval: int = 1
    ) -> None:
        """Wait for a lock file to be released.

        Args:
            lock_file: Path to the lock file.
            parent_folder_path: Path to the parent folder containing the lock file.
            check_interval: Time in seconds between checks.
        """
        # Check if the lock is stale
        while lock_file.exists():
            if self._check_stale_lock_file(lock_file):
                logger.warning(
                    f"Stale lock file detected at {lock_file}. Removing lock."
                )
                self._delete_decoding_lock(lock_file)
                shutil.rmtree(parent_folder_path, ignore_errors=True)
                logger.info(
                    f"Removed stale lock and cleared cache at {parent_folder_path}."
                )
                break
            time.sleep(check_interval)

    def _get_frame_from_disk_cache(
        self,
        camera_type: DataType,
        camera_data: dict[str, CameraData],
        transform_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> dict[str, CameraData]:
        """Get video frame from disk cache for camera data.

        Args:
            camera_type: DataType indicating the type of camera data.
            camera_data: Dictionary of camera data with camera IDs as keys.
            frame_idx: Index of the frame to retrieve.
            transform_fn: Optional function to transform frames (e.g., rgb_to_depth).

        Returns:
            Dictionary of CameraData with populated frames.
        """
        for cam_id, cam_data in camera_data.items():
            cam_id_rgb_root = self.cache_dir / f"{self.id}" / camera_type.value / cam_id
            lock_file = cam_id_rgb_root / ".recording.lock"
            self._wait_for_lock_release(lock_file, cam_id_rgb_root)

            if not cam_id_rgb_root.exists():
                # Not in cache, download video and cache frames to disk
                cam_id_rgb_root.mkdir(parents=True, exist_ok=True)
                self._download_video_and_cache_frames_to_disk(
                    camera_type, cam_id, cam_id_rgb_root
                )

            frame_file = cam_id_rgb_root / f"{cam_data.frame_idx}.png"
            frame = Image.open(frame_file)

            if transform_fn:
                frame = Image.fromarray(transform_fn(np.array(frame)))
            camera_data[cam_id].frame = frame
        return camera_data

    def _insert_camera_data_intro_sync_point(
        self, sync_point: SynchronizedPoint
    ) -> SynchronizedPoint:
        """Populate video frames for a given sync point.

        Args:
            sync_point: SynchronizedPoint object containing camera data.

        Returns:
            SynchronizedPoint object with populated video frames.
        """
        if DataType.RGB_IMAGES in sync_point.data:
            sync_point.data[DataType.RGB_IMAGES] = self._get_frame_from_disk_cache(
                DataType.RGB_IMAGES, sync_point.data[DataType.RGB_IMAGES]
            )
        if DataType.DEPTH_IMAGES in sync_point.data:
            sync_point.data[DataType.DEPTH_IMAGES] = self._get_frame_from_disk_cache(
                DataType.DEPTH_IMAGES, sync_point.data[DataType.DEPTH_IMAGES]
            )
        return sync_point

    def _get_sync_point(self, idx: int) -> SynchronizedPoint:
        """Get synchronized data point at a specific index.

        Args:
            idx: Index of the sync point to retrieve.

        Returns:
            SynchronizedPoint object containing synchronized data
                for the specified index.
        """
        # Copy for two reasons:
        # 1. we dont't want self._episode_synced.observations to hold the real image
        #    data in the ram. Because it will become large over time
        # 2. If the user modifies the returned sync point, it won't affect the loader.
        sync_point = copy.deepcopy(self._episode_synced.observations[idx])
        sync_point = self._insert_camera_data_intro_sync_point(sync_point)
        return sync_point

    def __iter__(self) -> "SynchronizedRecording":
        """Initialize iteration over the episode.

        Returns:
            SynchronizedRecording instance for iteration.
        """
        self._iter_idx = 0
        return self

    def __len__(self) -> int:
        """Get the number of timesteps in the episode.

        Returns:
            int: Number of timesteps in the episode.
        """
        return self._episode_length

    def __getitem__(
        self, idx: int | slice
    ) -> SynchronizedPoint | list[SynchronizedPoint]:
        """Support for indexing episode data.

        Args:
            idx: Integer index or slice object for accessing sync points.

        Returns:
            SynchronizedPoint object for single index or list of
                SynchronizedPoint objects for slice.

        Raises:
            IndexError: If the index is out of range.
            TypeError: If the index is not an integer or slice.
        """
        if isinstance(idx, slice):
            # Handle slice objects
            start, stop, step = idx.indices(len(self))
            return [cast(SynchronizedPoint, self[i]) for i in range(start, stop, step)]

        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        return self._get_sync_point(idx)

    def __next__(self) -> SynchronizedPoint:
        """Get the next synchronized data point in the episode.

        Returns:
            SynchronizedPoint object containing synchronized data for the next timestep.

        Raises:
            StopIteration: When all timesteps have been processed.
        """
        if self._iter_idx >= len(self._episode_synced.observations):
            raise StopIteration
        sync_point = self._get_sync_point(self._iter_idx)
        self._iter_idx += 1
        return sync_point
