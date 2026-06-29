"""Synchronized recording iterator."""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import wget
from neuracore_types import (
    CameraData,
    CrossEmbodimentUnion,
    DataType,
    PointCloudData,
    SynchronizationDetails,
)
from neuracore_types import SynchronizedEpisode as SynchronizedEpisodeModel
from neuracore_types import SynchronizedPoint, SynchronizeRecordingRequest
from neuracore_types.nc_data.point_cloud_data import (
    POINT_CLOUD_TRACE_BIN_FILENAME,
    decode_point_cloud_frame,
)
from PIL import Image

from neuracore.core.data.cache_manager import CacheManager
from neuracore.core.utils.depth_utils import rgb_to_depth_storage
from neuracore.core.utils.http_session import thread_local_session

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
        recording_name: str | None,
        robot_id: str,
        instance: int,
        frequency: int = 0,
        cross_embodiment_union: CrossEmbodimentUnion | None = None,
        prefetch_videos: bool = False,
    ):
        """Initialize episode iterator for a specific recording.

        Args:
            dataset: Parent Dataset instance.
            recording_id: Recording ID string.
            recording_name: Recording Name string.
            robot_id: The robot that created this recording.
            instance: The instance of the robot that created this recording.
            frequency: Frequency at which to synchronize the recording.
            cross_embodiment_union: Union of embodiment descriptions for
                synchronization.
            prefetch_videos: Whether to prefetch video data to cache on initialization.
        """
        self.dataset = dataset
        self.id = recording_id
        self.name = recording_name
        self.frequency = frequency
        self.cross_embodiment_union = cross_embodiment_union
        self.cache_dir: Path = dataset.cache_dir
        self.robot_id = robot_id
        self.instance = instance

        self._episode_synced = self._get_synced_data()
        self._episode_length = len(self._episode_synced.observations)

        # Use start_time and end_time from the synchronized episode,
        # as they reflect trim_start_end settings from synchronization
        self.start_time = self._episode_synced.start_time
        self.end_time = self._episode_synced.end_time
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
        session = thread_local_session()
        response = session.post(
            f"{API_URL}/org/{self.dataset.org_id}/synchronize/synchronize-recording",
            json=SynchronizeRecordingRequest(
                recording_id=self.id,
                synchronization_details=SynchronizationDetails(
                    frequency=self.frequency,
                    cross_embodiment_union=self.cross_embodiment_union,
                    max_delay_s=sys.float_info.max,
                    allow_duplicates=True,
                ),
            ).model_dump(mode="json"),
            headers=auth.get_headers(),
        )
        response.raise_for_status()
        return SynchronizedEpisodeModel.model_validate(response.json())

    def _get_recording_file_url(self, filepath: str) -> str:
        """Get a signed download URL for a file in this recording.

        Args:
            filepath: Recording-root-relative path
                (e.g. ``rgbs/cam1/lossless.mp4``).

        Returns:
            URL string for downloading the file.
        """
        auth = get_auth()
        session = thread_local_session()
        response = session.get(
            f"{API_URL}/org/{self.dataset.org_id}/recording/{self.id}/download_url",
            params={"filepath": filepath},
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
        # The lock lives beside the frames directory (not inside it) so that the
        # frames directory can be published atomically with os.replace.
        video_frame_cache_path.parent.mkdir(parents=True, exist_ok=True)
        # The lock is a sibling of the frames directory (not inside it) so the
        # frames directory can be published atomically once decoding completes.
        lock_file = (
            video_frame_cache_path.parent
            / f"{video_frame_cache_path.name}.recording.lock"
        )
        lock_acquired = self._create_decoding_lock(lock_file, camera_id)

        try:
            # Another process may have published this cache while we waited for
            # the lock; nothing left to do.
            if video_frame_cache_path.exists():
                return

            self.cache_manager.ensure_space_available()

            # Stage the download+decode in a temp dir on the same filesystem, then
            # publish atomically. A reader sees either a complete frames directory
            # or none at all -- never a partially decoded one.
            with tempfile.TemporaryDirectory(
                dir=video_frame_cache_path.parent
            ) as temp_dir:
                staging_dir = Path(temp_dir) / "frames"
                staging_dir.mkdir()
                video_location = Path(temp_dir) / f"{camera_id}{camera_type.value}.mp4"
                wget.download(
                    self._get_recording_file_url(
                        f"{camera_type.value}/{camera_id}/lossless.mp4"
                    ),
                    str(video_location),
                    bar=None if self._suppress_wget_progress else wget.bar_thermometer,
                )
                # Decode into staging, then atomically move into place.
                self._decode_video(video_location, staging_dir)
                os.replace(staging_dir, video_frame_cache_path)
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
        # Create new dict with new CameraData instances to avoid mutating originals
        result = {}
        for cam_id, cam_data in camera_data.items():
            cam_id_rgb_root = self.cache_dir / f"{self.id}" / camera_type.value / cam_id
            # The lock is a sibling of the frames directory (not inside it) so
            # the frames directory can be published atomically once decoding
            # completes.
            lock_file = (
                cam_id_rgb_root.parent / f"{cam_id_rgb_root.name}.recording.lock"
            )
            self._wait_for_lock_release(lock_file, cam_id_rgb_root)

            if not cam_id_rgb_root.exists():
                # Not in cache: download and decode. The frames directory is
                # published atomically, so its existence means it is complete.
                self._download_video_and_cache_frames_to_disk(
                    camera_type, cam_id, cam_id_rgb_root
                )

            frame_file = cam_id_rgb_root / f"{cam_data.frame_idx}.png"
            frame = Image.open(frame_file)

            if transform_fn:
                frame = Image.fromarray(transform_fn(np.array(frame)))

            result[cam_id] = cam_data.model_copy(update={"frame": frame})

        return result

    def _download_bytes(self, url: str) -> bytes:
        """Download a remote file and return its contents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "download.bin"
            wget.download(
                url,
                str(destination),
                bar=None if self._suppress_wget_progress else wget.bar_thermometer,
            )
            return destination.read_bytes()

    def _cache_point_cloud_frames_to_disk(
        self, sensor_id: str, sensor_root: Path
    ) -> None:
        """Download trace files and cache decoded point cloud frames to disk."""
        trace_prefix = f"{DataType.POINT_CLOUDS.value}/{sensor_id}"
        trace_json = json.loads(
            self._download_bytes(
                self._get_recording_file_url(f"{trace_prefix}/trace.json")
            ).decode("utf-8")
        )
        if not isinstance(trace_json, list):
            raise RuntimeError("Point cloud trace.json must be a JSON array")

        trace_bin_path = sensor_root / POINT_CLOUD_TRACE_BIN_FILENAME
        if trace_bin_path.exists():
            trace_bin = trace_bin_path.read_bytes()
        else:
            trace_bin = self._download_bytes(
                self._get_recording_file_url(
                    f"{trace_prefix}/{POINT_CLOUD_TRACE_BIN_FILENAME}"
                )
            )

        for entry_idx, entry in enumerate(trace_json):
            if not isinstance(entry, dict):
                raise RuntimeError("Invalid point cloud trace frame metadata")

            frame_idx = entry.get("frame_idx", entry_idx)
            frame_file = sensor_root / f"{frame_idx}.npz"
            if frame_file.exists():
                continue

            offset = entry.get("offset")
            length = entry.get("length")
            if not isinstance(offset, int) or not isinstance(length, int):
                raise RuntimeError(
                    f"Invalid point cloud frame offset/length for frame_idx={frame_idx}"
                )
            decoded = decode_point_cloud_frame(trace_bin[offset : offset + length])

            save_kwargs: dict[str, Any] = {"points": decoded.points}
            if decoded.rgb_points is not None:
                save_kwargs["rgb_points"] = decoded.rgb_points
            np.savez_compressed(frame_file, **save_kwargs)

    def _get_point_cloud_from_disk_cache(
        self, point_cloud_data: dict[str, PointCloudData]
    ) -> dict[str, PointCloudData]:
        """Load point cloud arrays from disk cache."""
        result: dict[str, PointCloudData] = {}
        for sensor_id, pc_data in point_cloud_data.items():
            sensor_root = (
                self.cache_dir / f"{self.id}" / DataType.POINT_CLOUDS.value / sensor_id
            )
            lock_file = sensor_root / ".recording.lock"
            self._wait_for_lock_release(lock_file, sensor_root)

            frame_file = sensor_root / f"{pc_data.frame_idx}.npz"
            if not sensor_root.exists() or not frame_file.exists():
                sensor_root.mkdir(parents=True, exist_ok=True)
                self._download_point_cloud_and_cache_frames_to_disk(
                    sensor_id, sensor_root
                )

            frame_file = sensor_root / f"{pc_data.frame_idx}.npz"
            with np.load(frame_file) as cached:
                points = cached["points"]
                rgb_points = cached["rgb_points"] if "rgb_points" in cached else None

            result[sensor_id] = pc_data.model_copy(
                update={"points": points, "rgb_points": rgb_points}
            )
        return result

    def _download_point_cloud_and_cache_frames_to_disk(
        self, sensor_id: str, point_cloud_cache_path: Path
    ) -> None:
        """Download point cloud trace files and cache frames to disk."""
        lock_file = point_cloud_cache_path / ".recording.lock"
        lock_acquired = self._create_decoding_lock(lock_file, sensor_id)

        try:
            self.cache_manager.ensure_space_available()
            self._cache_point_cloud_frames_to_disk(sensor_id, point_cloud_cache_path)
        finally:
            if lock_acquired:
                self._delete_decoding_lock(lock_file)

    def _load_sync_point_payloads(
        self, sync_point: SynchronizedPoint
    ) -> SynchronizedPoint:
        """Load lazy sensor payloads from disk cache for a sync point.

        Args:
            sync_point: Sync point with metadata-only camera and point cloud entries.

        Returns:
            Sync point with camera frames and point cloud arrays populated.
        """
        # Build new data dict with loaded frames
        new_data = {}
        for data_type, data_dict in sync_point.data.items():
            if data_type == DataType.RGB_IMAGES:
                new_data[data_type] = self._get_frame_from_disk_cache(
                    DataType.RGB_IMAGES, data_dict
                )
            elif data_type == DataType.DEPTH_IMAGES:
                new_data[data_type] = self._get_frame_from_disk_cache(
                    DataType.DEPTH_IMAGES, data_dict, rgb_to_depth_storage
                )
            elif data_type == DataType.POINT_CLOUDS:
                new_data[data_type] = self._get_point_cloud_from_disk_cache(data_dict)
            else:
                # create NEW instances to avoid shared references
                new_data[data_type] = {
                    name: nc_data.model_copy() for name, nc_data in data_dict.items()
                }

        return SynchronizedPoint(
            timestamp=sync_point.timestamp,
            robot_id=sync_point.robot_id,
            data=new_data,
        )

    def _get_sync_point(self, idx: int) -> SynchronizedPoint:
        """Get synchronized data point at a specific index.

        Args:
            idx: Index of the sync point to retrieve.

        Returns:
            SynchronizedPoint object containing synchronized data
                for the specified index.
        """
        sync_point = self._episode_synced.observations[idx]
        return self._load_sync_point_payloads(sync_point)

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
