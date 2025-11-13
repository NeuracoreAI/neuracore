"""Synchronized recording iterator."""

import copy
import logging
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Union, cast

import av
import numpy as np
import requests
import wget
from neuracore_types import CameraData, DataType, SyncedData, SyncPoint
from PIL import Image

from neuracore.core.data.cache_manager import CacheManager

from ..auth import get_auth
from ..const import API_URL
from ..utils.depth_utils import rgb_to_depth

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from neuracore.core.data.dataset import Dataset

MAX_DECODING_ATTEMPTS = 3


class SynchronizedRecording:
    """Synchronized recording iterator."""

    def __init__(
        self,
        dataset: "Dataset",
        recording_id: str,
        robot_id: str,
        instance: int,
        frequency: int = 0,
        data_types: Optional[list[DataType]] = None,
        prefetch_videos: bool = False,
    ):
        """Initialize episode iterator for a specific recording.

        Args:
            dataset: Parent Dataset instance.
            recording_id: Recording ID string.
            robot_id: The robot that created this recording.
            instance: The instance of the robot that created this recording.
            frequency: Frequency at which to synchronize the recording.
            data_types: List of DataType to include in synchronization.
            prefetch_videos: Whether to prefetch video data to cache on initialization.
        """
        self.dataset = dataset
        self.id = recording_id
        self.frequency = frequency
        self.data_types = data_types or []
        self.cache_dir: Path = dataset.cache_dir
        self.robot_id = robot_id
        self.instance = instance

        self._recording_synced = self._get_synced_data()
        self._episode_length = len(self._recording_synced.frames)
        self.cache_manager = CacheManager(
            self.cache_dir,
        )
        self._iter_idx = 0
        self._suppress_wget_progress = True

        if prefetch_videos:
            cache = self.dataset.cache_dir / f"{self.id}" / f"{self.frequency}Hz"
            if not cache.exists():
                self._get_sync_point(0)

    def _get_synced_data(self) -> SyncedData:
        """Retrieve synchronized metadata for the recording.

        Returns:
            SyncedData object containing synchronized frames and metadata.

        Raises:
            requests.HTTPError: If the API request fails.
        """
        auth = get_auth()
        response = requests.post(
            f"{API_URL}/org/{self.dataset.org_id}/synchronize/synchronize-recording",
            json={
                "recording_id": self.id,
                "frequency": self.frequency,
                "data_types": self.data_types,
            },
            headers=auth.get_headers(),
        )
        response.raise_for_status()
        return SyncedData.model_validate(response.json())

    def _get_video_url(self, camera_type: str, camera_id: str) -> str:
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
            params={"filepath": f"{camera_type}/{camera_id}/video.mp4"},
            headers=auth.get_headers(),
        )
        response.raise_for_status()
        return response.json()["url"]

    def _download_video_and_cache_frames_to_disk(
        self, camera_type: str, camera_id: str, video_frame_cache_path: Path
    ) -> None:
        """Download video and cache individual frames as images.

        Args:
            camera_type: Type of camera (e.g., "rgbs", "depths").
            camera_id: Unique identifier for the camera.
            video_frame_cache_path: Path to the directory where video frames are cached.
        """
        # Create a temporary video file path
        self.cache_manager.ensure_space_available()
        with tempfile.TemporaryDirectory() as temp_dir:
            video_location = Path(temp_dir) / f"{camera_id}{camera_type}.mp4"
            wget.download(
                self._get_video_url(camera_type, camera_id),
                str(video_location),
                bar=None if self._suppress_wget_progress else wget.bar_adaptive,
            )
            container = av.open(str(video_location))
            try:
                for i, frame in enumerate(container.decode(video=0)):
                    frame_image = Image.fromarray(frame.to_rgb().to_ndarray())
                    frame_file = video_frame_cache_path / f"{i}.png"
                    frame_image.save(frame_file)
            finally:
                container.close()

    def _get_frame_from_disk_cache(
        self,
        camera_type: str,
        camera_data: dict[str, CameraData],
        transform_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> dict[str, CameraData]:
        """Get video frame from disk cache for camera data.

        Args:
            camera_type: Type of camera (e.g., "rgbs", "depths").
            camera_data: Dictionary of camera data with camera IDs as keys.
            frame_idx: Index of the frame to retrieve.
            transform_fn: Optional function to transform frames (e.g., rgb_to_depth).

        Returns:
            Dictionary of CameraData with populated frames.
        """
        for cam_id, cam_data in camera_data.items():
            cam_id_rgb_root = (
                self.cache_dir
                / f"{self.id}"
                / f"{self.frequency}Hz"
                / camera_type
                / cam_id
            )
            frame_file = cam_id_rgb_root / f"{cam_data.frame_idx}.png"
            if not cam_id_rgb_root.exists():
                # Not in cache, download video and cache frames to disk
                cam_id_rgb_root.mkdir(parents=True, exist_ok=True)
                self._download_video_and_cache_frames_to_disk(
                    camera_type, cam_id, cam_id_rgb_root
                )

            # Check if frame is cached
            last_num_frames = -1
            attempts_left = MAX_DECODING_ATTEMPTS
            while True:
                try:
                    # Make sure the frame is successfully cached and decoded
                    frame = Image.open(frame_file)
                    break
                except Exception:
                    # Check if decoding is progressing
                    current_num_frames = len(
                        [1 for i in cam_id_rgb_root.iterdir() if i.suffix == ".png"]
                    )
                    if current_num_frames == last_num_frames:
                        attempts_left -= 1
                        if attempts_left <= 0:
                            raise RuntimeError(
                                f"Decoding timed out for recording {self.id}"
                            )
                    else:
                        last_num_frames = current_num_frames
                        attempts_left = MAX_DECODING_ATTEMPTS

                    # Wait for decoding to progress and try again
                    time.sleep(5)

            if transform_fn:
                frame = Image.fromarray(transform_fn(np.array(frame)))
            camera_data[cam_id].frame = frame
        return camera_data

    def _insert_camera_data_intro_sync_point(self, sync_point: SyncPoint) -> SyncPoint:
        """Populate video frames for a given sync point.

        Args:
            sync_point: SyncPoint object containing camera data.

        Returns:
            SyncPoint object with populated video frames.
        """
        if sync_point.rgb_images is not None:
            sync_point.rgb_images = self._get_frame_from_disk_cache(
                "rgbs", sync_point.rgb_images
            )
        if sync_point.depth_images is not None:
            sync_point.depth_images = self._get_frame_from_disk_cache(
                "depths", sync_point.depth_images, transform_fn=rgb_to_depth
            )
        return sync_point

    def _get_sync_point(self, idx: int) -> SyncPoint:
        """Get synchronized data point at a specific index.

        Args:
            idx: Index of the sync point to retrieve.

        Returns:
            SyncPoint object containing synchronized data for the specified index.
        """
        # Copy for two reasons:
        # 1. we dont't want self._recording_synced.frames to hold the real image
        #    data in the ram. Because it will become large over time
        # 2. If the user modifies the returned sync point, it won't affect the loader.
        sync_point = copy.deepcopy(self._recording_synced.frames[idx])
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

    def __getitem__(self, idx: Union[int, slice]) -> Union[SyncPoint, list[SyncPoint]]:
        """Support for indexing episode data.

        Args:
            idx: Integer index or slice object for accessing sync points.

        Returns:
            SyncPoint object for single index or list of SyncPoint objects for slice.

        Raises:
            IndexError: If the index is out of range.
            TypeError: If the index is not an integer or slice.
        """
        if isinstance(idx, slice):
            # Handle slice objects
            start, stop, step = idx.indices(len(self))
            return [cast(SyncPoint, self[i]) for i in range(start, stop, step)]

        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        return self._get_sync_point(idx)

    def __next__(self) -> SyncPoint:
        """Get the next synchronized data point in the episode.

        Returns:
            SyncPoint object containing synchronized data for the next timestep.

        Raises:
            StopIteration: When all timesteps have been processed.
        """
        if self._iter_idx >= len(self._recording_synced.frames):
            raise StopIteration
        sync_point = self._get_sync_point(self._iter_idx)
        self._iter_idx += 1
        return sync_point
