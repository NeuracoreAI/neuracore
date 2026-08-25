"""Synchronized recording iterator."""

import json
import logging
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import requests
from neuracore_types import (
    CameraData,
    CrossEmbodimentUnion,
    DataType,
    PointCloudData,
    SynchronizationDetails,
)
from neuracore_types import SynchronizedEpisode as SynchronizedEpisodeModel
from neuracore_types import (
    SynchronizedPoint,
    SynchronizeRecordingProgress,
    SynchronizeRecordingRequest,
    SynchronizeRecordingStartResponse,
    SynchronizeRecordingStatus,
)
from neuracore_types.nc_data.point_cloud_data import decode_point_cloud_frame
from PIL import Image
from pydantic import ValidationError as PydanticValidationError

from neuracore.core.data.cache_manager import CacheManager
from neuracore.core.data.frame_cache import (
    acquire_decoding_lock,
    delete_decoding_lock,
    lock_file_for,
    point_cloud_lock_file_for,
    publish_decoded_frames,
    video_filename_preference,
    wait_for_lock_release,
)
from neuracore.core.exceptions import SynchronizationError
from neuracore.core.utils.depth_utils import rgb_to_depth_storage
from neuracore.core.utils.download import download_bytes, stream_to_file
from neuracore.core.utils.http_session import thread_local_session

from ..auth import get_auth
from ..const import API_URL

logger = logging.getLogger(__name__)

POINT_CLOUD_TRACE_BIN_FILE = "trace.bin"
POINT_CLOUD_TRACE_INDEX_FILE = "trace.json"

if TYPE_CHECKING:
    from neuracore.core.data.dataset import Dataset

SYNCED_RECORDING_POLL_INTERVAL_S = 2.0
"""Seconds between polls of an in-progress recording synchronization."""

SYNCED_RECORDING_TIMEOUT_S = 1800.0
"""Seconds to wait for a recording to synchronize before giving up."""

SYNCED_EPISODE_DOWNLOAD_TIMEOUT_S: tuple[float, float] = (15.0, 120.0)
"""``(connect, read)`` seconds for the synchronized-episode object download.

The read budget bounds the wait for each chunk of the body rather than the whole
transfer. Object storage streaming a multi-megabyte episode can stall for far
longer than an API metadata call ever should, so this download does not inherit
``http_session.DEFAULT_TIMEOUT``. Matches the budget the equally large dataset
statistics result is fetched with.
"""


def _describe_download_failure(error: requests.RequestException) -> str:
    """Summarize a download failure without disclosing the signed URL.

    ``requests`` puts the full URL, signing credentials included, in its own
    error messages, so none of them can be surfaced to the caller.

    Args:
        error: The failure raised by ``requests``.

    Returns:
        The HTTP status code where the server answered, otherwise the name of
        the error class.
    """
    if error.response is not None:
        return f"HTTP {error.response.status_code}"
    return type(error).__name__


class SynchronizedRecording:
    """Synchronized recording iterator."""

    def __init__(
        self,
        dataset: "Dataset",
        recording_id: str,
        recording_name: str | None,
        robot_id: str,
        instance: int,
        synchronization_details: SynchronizationDetails,
        prefetch_videos: bool = False,
        episode_synced: SynchronizedEpisodeModel | None = None,
    ):
        """Initialize episode iterator for a specific recording.

        Args:
            dataset: Parent Dataset instance.
            recording_id: Recording ID string.
            recording_name: Recording Name string.
            robot_id: The robot that created this recording.
            instance: The instance of the robot that created this recording.
            synchronization_details: The full synchronization parameters. The
                server keys stored synchronized data on all of them, so these
                must match the parameters the data was synchronized with or the
                recording is synchronized again under a different key.
            prefetch_videos: Whether to prefetch video data to cache on initialization.
            episode_synced: Already-fetched synchronized metadata for this
                recording. When omitted, it is requested here.
        """
        self.dataset = dataset
        self.id = recording_id
        self.name = recording_name
        self.synchronization_details = synchronization_details
        self.cache_dir: Path = dataset.cache_dir
        self.robot_id = robot_id
        self.instance = instance

        self._episode_synced = (
            episode_synced if episode_synced is not None else self._get_synced_data()
        )
        self._episode_length = len(self._episode_synced.observations)

        # Use start_time and end_time from the synchronized episode,
        # as they reflect trim_start_end settings from synchronization
        self.start_time = self._episode_synced.start_time
        self.end_time = self._episode_synced.end_time
        self.cache_manager = CacheManager(
            self.cache_dir,
        )
        self._iter_idx = 0

        if prefetch_videos:
            cache = self.dataset.cache_dir / self.id
            # Check if cache directory exists and contains any files
            wait_for_lock_release(cache / ".recording.lock", cache)
            # NOTE: this is to start video prefetching frames into cache
            self._get_sync_point(0)

    @property
    def frequency(self) -> int:
        """Frequency in Hz this recording was synchronized at."""
        return self.synchronization_details.frequency

    @property
    def cross_embodiment_union(self) -> CrossEmbodimentUnion | None:
        """Cross-embodiment union this recording was synchronized with."""
        return self.synchronization_details.cross_embodiment_union

    def _get_synced_data(self) -> SynchronizedEpisodeModel:
        """Retrieve synchronized metadata for the recording.

        Synchronization is asynchronous: the API starts it and then reports its
        progress, and the finished episode is downloaded straight from object
        storage rather than served back through the API.

        Returns:
            SynchronizedEpisode object containing synchronized frames and metadata.

        Raises:
            requests.HTTPError: If a synchronization API request fails.
            SynchronizationError: If the synchronization fails or exceeds its
                deadline, if the API reports a state the caller cannot act on,
                if the download fails, or if the downloaded episode is not a
                valid synchronized episode.
        """
        job = self._start_synchronization()
        return self._download_synced_data(self._await_synced_data_url(job))

    def _start_synchronization(self) -> SynchronizeRecordingStartResponse:
        """Ask the API to synchronize this recording.

        Returns:
            The synchronization's initial state, identifying the artifact to
            poll for.

        Raises:
            requests.HTTPError: If the API request fails.
            SynchronizationError: If the response is not tracking metadata.
        """
        auth = get_auth()
        session = thread_local_session(retry_transient=True)
        response = session.post(
            f"{API_URL}/org/{self.dataset.org_id}/synchronize/trigger-synchronize-recording",
            json=SynchronizeRecordingRequest(
                recording_id=self.id,
                synchronization_details=self.synchronization_details,
            ).model_dump(mode="json"),
            headers=auth.get_headers(),
        )
        response.raise_for_status()
        try:
            return SynchronizeRecordingStartResponse.model_validate_json(
                response.content
            )
        except PydanticValidationError as exc:
            raise SynchronizationError(
                f"Synchronization start response for recording {self.id} is not"
                f" tracking metadata ({exc.error_count()} validation errors)"
            ) from exc

    def _poll_synchronization(
        self,
        job: SynchronizeRecordingStartResponse | SynchronizeRecordingProgress,
    ) -> SynchronizeRecordingProgress:
        """Read the current state of a started synchronization.

        Args:
            job: The state the synchronization was started with.

        Returns:
            The synchronization's latest state.

        Raises:
            requests.HTTPError: If the API request fails.
            SynchronizationError: If the response is not tracking metadata.
        """
        auth = get_auth()
        session = thread_local_session(retry_transient=True)
        synchronize_recording_id = (
            job.synchronized_recording_id
            if isinstance(job, SynchronizeRecordingStartResponse)
            else job.synchronize_recording_id
        )
        response = session.get(
            f"{API_URL}/org/{self.dataset.org_id}/synchronize"
            f"/synchronize-recording-progress/{synchronize_recording_id}",
            params={"recording_id": job.recording_id},
            headers=auth.get_headers(),
        )
        response.raise_for_status()
        try:
            return SynchronizeRecordingProgress.model_validate_json(response.content)
        except PydanticValidationError as exc:
            raise SynchronizationError(
                f"Synchronization progress response for recording {self.id} is not"
                f" tracking metadata ({exc.error_count()} validation errors)"
            ) from exc

    def _await_synced_data_url(self, job: SynchronizeRecordingStartResponse) -> str:
        """Poll a synchronization until its episode can be downloaded.

        An artifact the server has already cached reports READY on the first
        poll, so nothing sleeps in the common case.

        Args:
            job: The state the synchronization was started with.

        Returns:
            Signed object-storage URL for the synchronized episode. Its query
            string carries temporary credentials, so it must never be logged.

        Raises:
            requests.HTTPError: If a poll fails.
            SynchronizationError: If the synchronization fails, exceeds its
                deadline, or reports READY without a download URL.
        """
        deadline = time.monotonic() + SYNCED_RECORDING_TIMEOUT_S
        while True:
            job = self._poll_synchronization(job)

            if job.status is SynchronizeRecordingStatus.READY:
                if not job.download_url:
                    raise SynchronizationError(
                        f"Synchronizing recording {self.id} reported READY without"
                        " a download URL"
                    )
                return job.download_url

            if job.status is SynchronizeRecordingStatus.FAILED:
                raise SynchronizationError(
                    f"Synchronizing recording {self.id} failed:"
                    f" {job.error or 'no reason given'}"
                )

            if time.monotonic() >= deadline:
                raise SynchronizationError(
                    f"Timed out after {SYNCED_RECORDING_TIMEOUT_S:.0f}s waiting for"
                    f" recording {self.id} to synchronize (status"
                    f" {job.status.value}). The synchronization is still running;"
                    " reading the recording again resumes waiting rather than"
                    " starting over."
                )

            time.sleep(SYNCED_RECORDING_POLL_INTERVAL_S)

    def _download_synced_data(self, download_url: str) -> SynchronizedEpisodeModel:
        """Download and validate the synchronized episode behind a signed URL.

        Args:
            download_url: Signed object-storage URL for the episode JSON.

        Returns:
            The validated synchronized episode.

        Raises:
            SynchronizationError: If the download fails, or the downloaded body
                is not a valid synchronized episode.
        """
        session = thread_local_session(retry_transient=True, retry_read_timeout=True)
        try:
            response = session.get(
                download_url, timeout=SYNCED_EPISODE_DOWNLOAD_TIMEOUT_S
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise SynchronizationError(
                f"Failed to download synchronized episode for recording {self.id}"
                f" ({_describe_download_failure(exc)})"
            ) from None

        try:
            return SynchronizedEpisodeModel.model_validate_json(response.content)
        except PydanticValidationError as exc:
            raise SynchronizationError(
                f"Downloaded synchronized episode for recording {self.id} is not a"
                f" valid synchronized episode ({exc.error_count()} validation errors)"
            ) from exc

    def _get_recording_file_url(self, filepath: str) -> str:
        """Get a signed download URL for a file in this recording.

        Args:
            filepath: Recording-root-relative path
                (e.g. ``rgbs/cam1/lossless.mp4``).

        Returns:
            URL string for downloading the file.
        """
        auth = get_auth()
        session = thread_local_session(retry_transient=True)
        response = session.get(
            f"{API_URL}/org/{self.dataset.org_id}/recording/{self.id}/download_url",
            params={"filepath": filepath},
            headers=auth.get_headers(),
        )
        response.raise_for_status()
        return response.json()["url"]

    def _get_video_url(self, camera_type: DataType, camera_id: str) -> str:
        """Get streaming URL for a specific camera's video data.

        Args:
            camera_type: Type of camera (e.g., "rgbs", "depths").
            camera_id: Unique identifier for the camera.

        Returns:
            URL string for downloading the video file.

        Raises:
            requests.HTTPError: If every candidate is absent (404), or for any
                non-404 HTTP error.
        """
        filename_preference = video_filename_preference(camera_type)

        for video_filename in filename_preference:
            try:
                return self._get_recording_file_url(
                    f"{camera_type.value}/{camera_id}/{video_filename}"
                )
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    continue
                raise

        raise requests.HTTPError(
            f"No candidate filename found for recording {self.id} "
            f"(camera {camera_type.value}/{camera_id}); tried: {filename_preference}"
        )

    def _get_point_cloud_url(self, sensor_id: str, filename: str) -> str:
        """Get a signed download URL for a point cloud trace file.

        Args:
            sensor_id: Unique identifier for the point cloud sensor.
            filename: Trace file name (e.g. trace.json, trace.bin).

        Returns:
            URL string for downloading the trace file.
        """
        return self._get_recording_file_url(
            f"{DataType.POINT_CLOUDS.value}/{sensor_id}/{filename}"
        )

    def _download_video_and_cache_frames_to_disk(
        self, camera_type: DataType, camera_id: str, video_frame_cache_path: Path
    ) -> None:
        """Download video and cache individual frames as images.

        Args:
            camera_type: Type of camera (e.g., "rgbs", "depths").
            camera_id: Unique identifier for the camera.
            video_frame_cache_path: Path to the directory where video frames are cached.
        """
        video_frame_cache_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = lock_file_for(video_frame_cache_path)
        acquire_decoding_lock(lock_file, camera_id)

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
                stream_to_file(
                    self._get_video_url(camera_type, camera_id), video_location
                )
                publish_decoded_frames(
                    video_location, staging_dir, video_frame_cache_path
                )
        finally:
            delete_decoding_lock(lock_file)

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
            lock_file = lock_file_for(cam_id_rgb_root)
            wait_for_lock_release(lock_file, cam_id_rgb_root)

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

    def _cache_point_cloud_frames_to_disk(
        self, sensor_id: str, sensor_root: Path
    ) -> None:
        """Download trace files and cache decoded point cloud frames to disk."""
        trace_json = json.loads(
            download_bytes(
                self._get_point_cloud_url(sensor_id, POINT_CLOUD_TRACE_INDEX_FILE)
            ).decode("utf-8")
        )
        if not isinstance(trace_json, list):
            raise RuntimeError("Point cloud trace.json must be a JSON array")

        trace_bin_path = sensor_root / POINT_CLOUD_TRACE_BIN_FILE
        if trace_bin_path.exists():
            trace_bin = trace_bin_path.read_bytes()
        else:
            trace_bin = download_bytes(
                self._get_point_cloud_url(sensor_id, POINT_CLOUD_TRACE_BIN_FILE)
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
            lock_file = point_cloud_lock_file_for(sensor_root)
            wait_for_lock_release(lock_file, sensor_root)

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
        lock_file = point_cloud_lock_file_for(point_cloud_cache_path)
        acquire_decoding_lock(lock_file, sensor_id)

        try:
            self.cache_manager.ensure_space_available()
            self._cache_point_cloud_frames_to_disk(sensor_id, point_cloud_cache_path)
        finally:
            delete_decoding_lock(lock_file)

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
