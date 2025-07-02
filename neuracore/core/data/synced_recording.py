"""Synchronized recording iterator."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Union, cast

import av
import numpy as np
import requests
import wget
from PIL import Image

from neuracore.core.data.cache_manager import CacheManager

from ..auth import get_auth
from ..const import API_URL
from ..nc_types import CameraData, DataType, SyncedData, SyncPoint
from ..utils.depth_utils import rgb_to_depth

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from neuracore.core.data.dataset import Dataset


class SynchronizedRecording:
    """Synchronized recording iterator."""

    def __init__(
        self,
        dataset: "Dataset",
        recording_id: str,
        frequency: int = 0,
        data_types: Optional[list[DataType]] = None,
    ):
        """Initialize episode iterator for a specific recording.

        Args:
            dataset: Parent Dataset instance.
            recording_id: Recording ID string.
            frequency: Frequency at which to synchronize the recording.
            data_types: List of DataType to include in synchronization.
        """
        self.dataset = dataset
        self.id = recording_id
        self.frequency = frequency
        self.data_types = data_types or []
        self.cache_dir: Optional[Path] = dataset.cache_dir

        self._recording_synced = self._get_synced_data()
        _rgb = self._recording_synced.frames[0].rgb_images
        _depth = self._recording_synced.frames[0].depth_images
        self._camera_ids = {
            "rgbs": list(_rgb.keys()) if _rgb else [],
            "depths": list(_depth.keys()) if _depth else [],
        }
        self._episode_length = len(self._recording_synced.frames)

        self.cache_manager = CacheManager(
            self.cache_dir,
        )

        self._iter_idx = 0
        self._suppress_wget_progress = True

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

    def _download_video(
        self, camera_type: str, camera_id: str, video_cache_path: Path
    ) -> None:
        """Download a single video using wget with progress bar.

        Args:
            camera_type: Type of camera (e.g., "rgbs", "depths").
            camera_id: Unique identifier for the camera.
            video_cache_path: Path to the directory where videos are cached.
        """
        video_file = video_cache_path / f"{camera_id}.mp4"

        if video_file.exists():
            logger.info(f"Video already exists: {video_file}")
            return

        # Ensure cache space is available before downloading
        self.cache_manager.ensure_space_available()
        url = self._get_video_url(camera_type, camera_id)
        wget.download(
            url,
            str(video_file),
            bar=None if self._suppress_wget_progress else wget.bar_adaptive,
        )

    def _ensure_videos_downloaded(self, camera_type: str) -> None:
        """Download videos for specific camera type if not already downloaded.

        Args:
            camera_type: Type of camera (e.g., "rgbs", "depths").
        """
        if not self.cache_dir:
            return

        camera_types_to_download = (
            [camera_type] if camera_type else list(self._camera_ids.keys())
        )

        for cam_type in camera_types_to_download:
            video_cache_path = self.cache_dir / f"{self.id}_videos" / cam_type

            # Check if videos already exist
            if video_cache_path.exists() and any(video_cache_path.glob("*.mp4")):
                continue

            video_cache_path.mkdir(parents=True, exist_ok=True)

            camera_ids = self._camera_ids[cam_type]
            if not camera_ids:
                continue

            num_parallel_downloads = int(
                os.environ.get("NEURACORE_NUM_PARALLEL_VIDEO_DOWNLOADS", "4")
            )
            if num_parallel_downloads <= 1:
                for camera_id in camera_ids:
                    self._download_video(cam_type, camera_id, video_cache_path)
            else:
                with ThreadPoolExecutor(max_workers=num_parallel_downloads) as executor:
                    future_to_task = {
                        executor.submit(
                            self._download_video, cam_type, camera_id, video_cache_path
                        ): camera_id
                        for camera_id in camera_ids
                    }

                    for future in as_completed(future_to_task):
                        camera_id = future_to_task[future]
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(
                                f"Failed to download {cam_type}/{camera_id}: {e}"
                            )
                            raise

    def _get_video_frames(
        self,
        camera_type: str,
        cam_metadata: dict[str, CameraData],
        t0_cam_metadata: dict[str, CameraData],
    ) -> list[Image.Image]:
        """Get video frames for multiple cameras with timing synchronization.

        Args:
            camera_type: Type of camera (e.g., "rgbs", "depths").
            cam_metadata: Dictionary of camera metadata with camera IDs as keys.
            t0_cam_metadata: Metadata for the first frame of each camera type.

        Returns:
            List of synchronized PIL Image frames for each camera.

        Raises:
            ValueError: If no frames are found for a camera at the specified timestamp.
        """
        camera_ids = list(cam_metadata.keys())

        if self.cache_dir:
            video_cache_path = self.cache_dir / f"{self.id}_videos" / camera_type
            video_cache_path.mkdir(parents=True, exist_ok=True)

            # Ensure cache space and download videos if needed
            for cam_id in camera_ids:
                video_file = video_cache_path / f"{cam_id}.mp4"
                if not video_file.exists():
                    # Ensure space before downloading
                    self.cache_manager.ensure_space_available()
                    self._download_video(camera_type, cam_id, video_cache_path)

        image_frame_for_each_camera = []
        for cam_id, cam_data in cam_metadata.items():
            video_file = video_cache_path / f"{cam_id}.mp4"
            container = av.open(str(video_file))
            video_stream = container.streams.video[0]

            # Calculate target frame based on timestamp difference
            start_time = t0_cam_metadata[cam_id].timestamp
            ts = cam_data.timestamp - start_time
            target_pts = int(ts / float(video_stream.time_base))

            # Seek to approximate position
            container.seek(target_pts, stream=video_stream)

            # Find the closest frame to our target
            cam_frame: Optional[np.ndarray] = None
            for frame in container.decode(video=0):
                frame_pts = frame.pts
                diff = frame_pts - target_pts

                if diff >= 0:
                    cam_frame = Image.fromarray(frame.to_rgb().to_ndarray())
                    break

            if cam_frame is None:
                raise ValueError(
                    f"No frame found for {camera_type}/{cam_id} "
                    f"at timestamp {cam_data.timestamp}"
                )

            image_frame_for_each_camera.append(cam_frame)
            container.close()

        return image_frame_for_each_camera

    def _populate_video_frames(
        self,
        camera_data: dict[str, CameraData],
        transform_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        """Populate video frames for camera data.

        Args:
            camera_data: Dictionary of camera data with camera IDs as keys.
            transform_fn: Optional function to transform frames (e.g., rgb_to_depth).
        """
        camera_type = "rgbs" if transform_fn is None else "depths"

        # Ensure videos for this camera type are downloaded
        self._ensure_videos_downloaded(camera_type)

        # Get t0 metadata for timing calculations
        t0_cam_metadata = getattr(
            self._recording_synced.frames[0], f"{camera_type.rstrip('s')}_images", {}
        )

        # Get frames using the original timing-based method
        frame_lists = self._get_video_frames(camera_type, camera_data, t0_cam_metadata)

        # Apply transforms and assign frames to camera data
        for i, (camera_id, cam_data) in enumerate(camera_data.items()):
            if i < len(frame_lists) and frame_lists[i] is not None:
                frame = frame_lists[i]
                if transform_fn:
                    frame = Image.fromarray(transform_fn(np.array(frame)))
                cam_data.frame = frame
            else:
                logger.error(f"No frame available for {camera_type}/{camera_id}")
                cam_data.frame = None

    def __next__(self) -> SyncPoint:
        """Get the next synchronized data point in the episode.

        Returns:
            SyncPoint object containing synchronized data for the next timestep.

        Raises:
            StopIteration: When all timesteps have been processed.
        """
        if self._iter_idx >= len(self._recording_synced.frames):
            raise StopIteration

        # Get sync point data
        sync_point = self._recording_synced.frames[self._iter_idx]

        if sync_point.rgb_images is not None:
            self._populate_video_frames(sync_point.rgb_images)
        if sync_point.depth_images is not None:
            self._populate_video_frames(
                sync_point.depth_images, transform_fn=rgb_to_depth
            )

        self._iter_idx += 1
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

        # Get sync point data
        sync_point: SyncPoint = self._recording_synced.frames[idx]

        # Temporarily set iter_idx for _populate_video_frames
        original_iter_idx = self._iter_idx
        self._iter_idx = idx

        try:
            if sync_point.rgb_images is not None:
                self._populate_video_frames(sync_point.rgb_images)

            if sync_point.depth_images is not None:
                self._populate_video_frames(
                    sync_point.depth_images, transform_fn=rgb_to_depth
                )
        finally:
            # Restore original iter_idx
            self._iter_idx = original_iter_idx

        return sync_point
