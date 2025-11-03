"""Synchronized recording iterator (optimized)."""

import copy
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
        robot_id: str,
        instance: int,
        frequency: int = 0,
        data_types: Optional[list[DataType]] = None,
    ):
        """Initialize episode iterator for a specific recording.

        Args:
            dataset: Parent Dataset instance.
            recording_id: Recording ID string.
            robot_id: The robot that created this recording.
            instance: The instance of the robot that created this recording.
            frequency: Frequency at which to synchronize the recording.
            data_types: List of DataType to include in synchronization.
        """
        self.dataset = dataset
        self.id = recording_id
        self.frequency = frequency
        self.data_types = data_types or []
        self.cache_dir: Optional[Path] = dataset.cache_dir
        self.robot_id = robot_id
        self.instance = instance

        self._recording_synced = self._get_synced_data()
        _rgb = self._recording_synced.frames[0].rgb_images
        _depth = self._recording_synced.frames[0].depth_images
        self._camera_ids = {
            "rgbs": list(_rgb.keys()) if _rgb else [],
            "depths": list(_depth.keys()) if _depth else [],
        }
        self._episode_length = len(self._recording_synced.frames)

        self.cache_manager = CacheManager(self.cache_dir)

        self._iter_idx = 0
        self._suppress_wget_progress = True

        # Caches for open video containers/streams
        self._video_containers: dict[str, dict[str, av.container.Container]] = {}
        self._video_streams: dict[str, dict[str, av.video.stream.VideoStream]] = {}

        # NEW: per-camera decode state (to avoid seek each step)
        # camera_type -> cam_id -> dict(last_pts:int, last_rel_ts:float)
        self._decode_state: dict[str, dict[str, dict[str, float | int]]] = {
            "rgbs": {},
            "depths": {},
        }

        # NEW: precompute t0 metadata once (absolute timestamps per camera)
        self._t0_by_type: dict[str, dict[str, float]] = {
            "rgbs": {},
            "depths": {},
        }
        if _rgb:
            for cid, cdata in _rgb.items():
                self._t0_by_type["rgbs"][cid] = cdata.timestamp
        if _depth:
            for cid, cdata in _depth.items():
                self._t0_by_type["depths"][cid] = cdata.timestamp

        # Optional: small thread pool for per-step multi-camera decode
        self._per_step_workers = int(
            os.environ.get("NEURACORE_PER_STEP_DECODE_THREADS", "0")
        )
        self._executor: Optional[ThreadPoolExecutor] = None
        if self._per_step_workers and self._per_step_workers > 0:
            self._executor = ThreadPoolExecutor(max_workers=self._per_step_workers)

    # ---------------- API calls ----------------

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

            num_parallel = int(
                os.environ.get("NEURACORE_NUM_PARALLEL_VIDEO_DOWNLOADS", "4")
            )
            if num_parallel <= 1:
                for camera_id in camera_ids:
                    self._download_video(cam_type, camera_id, video_cache_path)
            else:
                with ThreadPoolExecutor(max_workers=num_parallel) as executor:
                    futures = [
                        executor.submit(
                            self._download_video, cam_type, cid, video_cache_path
                        )
                        for cid in camera_ids
                    ]
                    for f in as_completed(futures):
                        f.result()

    # ---------------- optimized video frame access ----------------

    def _ensure_open(self, camera_type: str, cam_id: str, video_file: Path) -> None:
        """Open a video file and initialize decode state.

        Args:
            camera_type: Type of camera (e.g., "rgbs", "depths").
            cam_id: Unique identifier for the camera.
            video_file: Path to the video file.
        """
        if camera_type not in self._video_containers:
            self._video_containers[camera_type] = {}
            self._video_streams[camera_type] = {}
        if cam_id not in self._video_containers[camera_type]:
            container = av.open(str(video_file))
            stream = container.streams.video[0]
            self._video_containers[camera_type][cam_id] = container
            self._video_streams[camera_type][cam_id] = stream
            # initialize decode state
            self._decode_state[camera_type][cam_id] = {
                "last_pts": -1,  # last decoded pts
                "last_rel_ts": -1.0,  # last requested relative seconds
            }

    def _decode_to_target(
        self,
        camera_type: str,
        cam_id: str,
        target_rel_ts: float,
        *,
        allow_reseek_back_gap_s: float = 0.05,
    ) -> np.ndarray:
        """Decode forward to the frame at/after target_rel_ts (seconds from t0).

        We only reseek if the target is sufficiently behind the last decoded
        position.
        """
        container = self._video_containers[camera_type][cam_id]
        stream = self._video_streams[camera_type][cam_id]
        state = self._decode_state[camera_type][cam_id]

        fps = float(stream.average_rate) if stream.average_rate else 0.0
        tbase = float(stream.time_base) if stream.time_base else (1.0 / max(fps, 1.0))
        target_pts = int(target_rel_ts / tbase)

        last_pts = int(state["last_pts"])
        last_rel_ts = float(state["last_rel_ts"])

        # If we are going forward (normal __next__), decode forward without seek.
        # Only reseek if target is quite earlier than last decoded time (random access).
        going_back = (last_pts >= 0) and (
            target_rel_ts + allow_reseek_back_gap_s < last_rel_ts
        )

        if last_pts < 0 or going_back:
            # Initial access or big backward jump -> seek once near the target
            container.seek(target_pts, any_frame=False, stream=stream)
            state["last_pts"] = -1  # reset cursor

        # Decode forward until we hit or pass target_pts
        for packet in container.demux(stream):
            for frame in packet.decode():
                fpts = frame.pts if frame.pts is not None else None
                if fpts is None:
                    # estimate via last_pts + 1 frame
                    fpts = last_pts + int(1.0 / tbase) if last_pts >= 0 else target_pts
                state["last_pts"] = fpts
                cur_rel_ts = fpts * tbase
                state["last_rel_ts"] = cur_rel_ts
                if fpts >= target_pts:
                    # Return as NumPy RGB to avoid extra conversions
                    return frame.to_rgb().to_ndarray()

        raise ValueError(
            f"No frame found for {camera_type}/{cam_id} at rel_ts={target_rel_ts:.6f}s"
        )

    def _get_video_frames(
        self,
        camera_type: str,
        cam_metadata: dict[str, CameraData],
    ) -> dict[str, np.ndarray]:
        """Get video frames for multiple cameras with timing synchronization.

        Args:
            camera_type: Type of camera (e.g., "rgbs", "depths").
            cam_metadata: Dictionary of camera metadata with camera IDs as keys.

        Returns:
            Dictionary mapping camera IDs to NumPy RGB arrays.

        Raises:
            ValueError: If no frames are found for a camera at the specified timestamp.
        """
        camera_ids = list(cam_metadata.keys())
        out: dict[str, np.ndarray] = {}

        if not camera_ids:
            return out

        # Paths & ensure downloaded
        if self.cache_dir:
            video_cache_path = self.cache_dir / f"{self.id}_videos" / camera_type
            video_cache_path.mkdir(parents=True, exist_ok=True)
            for cam_id in camera_ids:
                video_file = video_cache_path / f"{cam_id}.mp4"
                if not video_file.exists():
                    self.cache_manager.ensure_space_available()
                    self._download_video(camera_type, cam_id, video_cache_path)

        # Open if needed and decode each camera (optionally in parallel)
        def _job(cid: str) -> tuple[str, np.ndarray]:
            video_cache_path = (
                self.cache_dir / f"{self.id}_videos" / camera_type
                if self.cache_dir
                else None
            )
            video_file = (
                (video_cache_path / f"{cid}.mp4")
                if video_cache_path
                else Path(f"{cid}.mp4")
            )
            self._ensure_open(camera_type, cid, video_file)
            # relative timestamp from precomputed t0
            t0 = self._t0_by_type[camera_type].get(cid, 0.0)
            rel_ts = cam_metadata[cid].timestamp - t0
            arr = self._decode_to_target(camera_type, cid, rel_ts)
            return cid, arr

        if self._executor:
            futures = [self._executor.submit(_job, cid) for cid in camera_ids]
            for f in as_completed(futures):
                cid, arr = f.result()
                out[cid] = arr
        else:
            for cid in camera_ids:
                c, arr = _job(cid)
                out[c] = arr

        return out

    # ---------------- public helpers ----------------

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

        # Ensure videos for this camera type are downloaded once
        self._ensure_videos_downloaded(camera_type)

        # Decode all cameras for this step
        rgb_map = self._get_video_frames(camera_type, camera_data)

        for cam_id, cam_data in camera_data.items():
            arr = rgb_map.get(cam_id, None)
            if arr is None:
                logger.error(f"No frame available for {camera_type}/{cam_id}")
                cam_data.frame = None
                continue
            # Optional transform (e.g., depth)
            if transform_fn is not None:
                # rgb_to_depth expects ndarray; returns ndarray
                arr = transform_fn(arr)
            # Assign as PIL Image at the end to keep API unchanged
            if arr.ndim == 2:
                img = Image.fromarray(arr)
            else:
                img = Image.fromarray(arr.astype(np.uint8))
            cam_data.frame = img

    # ---------------- lifecycle ----------------

    def _close_video_containers(self) -> None:
        """Close all open video containers and clear caches."""
        for camera_type in self._video_containers:
            for _, container in self._video_containers[camera_type].items():
                try:
                    container.close()
                except Exception:
                    pass
        self._video_containers.clear()
        self._video_streams.clear()
        self._decode_state = {"rgbs": {}, "depths": {}}

    def close(self) -> None:
        """Explicitly close all resources.

        This method should be called when done accessing the recording to free
        file handles and shutdown thread pools. It is safe to call multiple times.
        """
        self._close_video_containers()
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __del__(self) -> None:
        """Cleanup when object is garbage collected (best effort)."""
        try:
            if hasattr(self, "_video_containers"):
                self.close()
        except Exception:
            pass

    # ---------------- iteration / indexing ----------------

    def __next__(self) -> SyncPoint:
        """Get the next synchronized data point in the episode.

        Returns:
            SyncPoint object containing synchronized data for the next timestep.

        Raises:
            StopIteration: When all timesteps have been processed.
        """
        if self._iter_idx >= len(self._recording_synced.frames):
            raise StopIteration

        # we dont't want self._recording_synced.frames to hold the real video
        # data in the ram. Because it will be very large with the increasing
        # number of frames and cause out of memory error. Instead we create
        # a local copy of the sync point to avoid this issue.
        sync_point = copy.deepcopy(self._recording_synced.frames[self._iter_idx])

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
        # Reset decode state/containers between full passes
        self._close_video_containers()
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

        idx = idx if idx >= 0 else idx + len(self)
        if not 0 <= idx < len(self):
            raise IndexError("Index out of range")

        # we dont't want self._recording_synced.frames to hold the real video
        # data in the ram. Because it will be very large with the increasing
        # number of frames and cause out of memory error. Instead we create
        # a local copy of the sync point to avoid this issue.
        sync_point: SyncPoint = copy.deepcopy(self._recording_synced.frames[idx])

        # Temporarily set iter_idx and keep decode cursors viable
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
