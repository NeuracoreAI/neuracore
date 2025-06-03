"""Dataset management and streaming for Neuracore robot recordings.

This module provides classes for managing datasets, streaming episodes,
and iterating over synchronized robot data including video frames and
sensor information. It supports both organizational and shared datasets
with efficient streaming capabilities.
"""

import concurrent
import logging
import queue
import threading
from typing import Callable, Optional

import numpy as np
import requests

from .auth import Auth, get_auth
from .const import API_URL
from .exceptions import DatasetError
from .nc_types import CameraData, SyncedData, SyncPoint
from .utils.depth_utils import rgb_to_depth
from .utils.video_url_streamer import VideoStreamer

logger = logging.getLogger(__name__)

CHUNK_SIZE = 256 * 1024  # Multiples of 256KB


class Dataset:
    """Represents a dataset containing robot demonstration recordings.

    This class provides access to collections of robot recordings that can be
    streamed for analysis or used for training machine learning models. It
    supports both organizational and shared datasets with efficient iteration
    over episodes and synchronized data access.
    """

    def __init__(self, dataset_dict: dict, recordings: list[dict] = None):
        """Initialize a dataset from server response data.

        Args:
            dataset_dict: Dictionary containing dataset metadata from the server.
            recordings: Optional list of recording dictionaries. If not provided,
                recordings will be fetched from the server.
        """
        self._dataset_dict = dataset_dict
        self.id = dataset_dict["id"]
        self.name = dataset_dict["name"]
        self.size_bytes = dataset_dict["size_bytes"]
        self.tags = dataset_dict["tags"]
        self.is_shared = dataset_dict["is_shared"]
        self._recording_idx = 0
        self._previous_iterator = None
        if recordings is None:
            self.num_episodes = dataset_dict["num_demonstrations"]
            auth = get_auth()
            response = requests.get(
                f"{API_URL}/datasets/{self.id}/recordings", headers=auth.get_headers()
            )
            response.raise_for_status()
            data = response.json()
            self._recordings = data["recordings"]
        else:
            self.num_episodes = len(recordings)
            self._recordings = recordings

    @staticmethod
    def get(name: str, non_exist_ok: bool = False) -> "Dataset":
        """Retrieve an existing dataset by name.

        Searches through both organizational and shared datasets to find
        a dataset with the specified name.

        Args:
            name: Name of the dataset to retrieve.
            non_exist_ok: If True, returns None when dataset is not found
                instead of raising an exception.

        Returns:
            The Dataset instance if found, or None if non_exist_ok is True
            and the dataset doesn't exist.

        Raises:
            DatasetError: If the dataset is not found and non_exist_ok is False.
        """
        dataset_jsons = Dataset._get_datasets()
        for dataset in dataset_jsons:
            if dataset["name"] == name:
                return Dataset(dataset)
        if non_exist_ok:
            return None
        raise DatasetError(f"Dataset '{name}' not found.")

    @staticmethod
    def create(
        name: str,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        shared: bool = False,
    ) -> "Dataset":
        """Create a new dataset or return existing one with the same name.

        Creates a new dataset with the specified parameters. If a dataset
        with the same name already exists, returns the existing dataset
        instead of creating a duplicate.

        Args:
            name: Unique name for the dataset.
            description: Optional description of the dataset contents and purpose.
            tags: Optional list of tags for organizing and searching datasets.
            shared: Whether the dataset should be shared/open-source.
                Note that setting shared=True is only available to specific
                members allocated by the Neuracore team.

        Returns:
            The newly created Dataset instance, or existing dataset if
            name already exists.
        """
        ds = Dataset.get(name, non_exist_ok=True)
        if ds is None:
            ds = Dataset._create_dataset(name, description, tags, shared=shared)
        else:
            logger.info(f"Dataset '{name}' already exist.")
        return ds

    @staticmethod
    def _create_dataset(
        name: str,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        shared: bool = False,
    ) -> "Dataset":
        """Create a new dataset via API call.

        Args:
            name: Unique name for the dataset.
            description: Optional description of the dataset.
            tags: Optional list of tags for the dataset.
            shared: Whether the dataset should be shared.
                Note that setting shared=True is only available to specific
                members allocated by the Neuracore team.

        Returns:
            The newly created Dataset instance.

        Raises:
            requests.HTTPError: If the API request fails.
        """
        auth: Auth = get_auth()
        response = requests.post(
            f"{API_URL}/datasets",
            headers=auth.get_headers(),
            json={
                "name": name,
                "description": description,
                "tags": tags,
                "is_shared": shared,
            },
        )
        response.raise_for_status()
        dataset_json = response.json()
        return Dataset(dataset_json)

    @staticmethod
    def _get_datasets() -> list[dict]:
        """Fetch all available datasets from the server.

        Retrieves both organizational and shared datasets using concurrent
        requests for improved performance.

        Returns:
            List of dataset dictionaries containing metadata.

        Raises:
            requests.HTTPError: If either API request fails.
        """
        auth: Auth = get_auth()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            org_data_req = executor.submit(
                requests.get, f"{API_URL}/datasets", headers=auth.get_headers()
            )
            shared_data_req = executor.submit(
                requests.get, f"{API_URL}/datasets/shared", headers=auth.get_headers()
            )
            org_data, shared_data = org_data_req.result(), shared_data_req.result()
        org_data.raise_for_status()
        shared_data.raise_for_status()
        return org_data.json() + shared_data.json()

    def as_pytorch_dataset(self, **kwargs):
        """Convert to PyTorch dataset format.

        Returns:
            PyTorch dataset compatible object.

        Raises:
            NotImplementedError: This feature is not yet implemented.
        """
        raise NotImplementedError("PyTorch dataset conversion not yet implemented")

    def __iter__(self) -> "Dataset":
        """Initialize iterator over episodes in the dataset.

        Returns:
            Self for iteration over episodes.
        """
        return self

    def __len__(self) -> int:
        """Get the number of episodes in the dataset.

        Returns:
            Number of demonstration episodes in the dataset.
        """
        return self.num_episodes

    def __getitem__(self, idx):
        """Support for indexing and slicing dataset episodes.

        Args:
            idx: Integer index or slice object for accessing episodes.

        Returns:
            For integer indices: EpisodeIterator for the specified episode.
            For slices: New Dataset containing the selected episodes.

        Raises:
            IndexError: If the index is out of range.
            TypeError: If the index is not an integer or slice.
        """
        if isinstance(idx, slice):
            # Handle slice
            recordings = self._recordings[idx.start : idx.stop : idx.step]
            ds = Dataset(self._dataset_dict, recordings)
            return ds
        else:
            # Handle single index
            if isinstance(idx, int):
                if idx < 0:  # Handle negative indices
                    idx += len(self._recordings)
                if not 0 <= idx < len(self._recordings):
                    raise IndexError("Dataset index out of range")
                return EpisodeIterator(self, self._recordings[idx])
            raise TypeError(
                f"Dataset indices must be integers or slices, not {type(idx)}"
            )

    def __next__(self):
        """Get the next episode in the dataset iteration.

        Returns:
            EpisodeIterator for the next episode.

        Raises:
            StopIteration: When all episodes have been processed.
        """
        if self._recording_idx >= len(self._recordings):
            raise StopIteration

        recording = self._recordings[self._recording_idx]
        self._recording_idx += 1  # Increment counter
        if self._previous_iterator is not None:
            self._previous_iterator.close()
            del self._previous_iterator
        self._previous_iterator = EpisodeIterator(self, recording)
        return self._previous_iterator

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context manager.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        if self._previous_iterator is not None:
            self._previous_iterator.close()


class EpisodeIterator:
    """Iterator for streaming synchronized data from a single recording episode.

    This class provides efficient streaming access to robot demonstration data
    including video frames from multiple cameras, depth data, and sensor
    information. It manages concurrent video streams and synchronizes data
    according to the episode's timestamp information.
    """

    def __init__(self, dataset, recording):
        """Initialize episode iterator for a specific recording.

        Args:
            dataset: Parent Dataset instance.
            recording: Recording dictionary containing episode metadata.
        """
        self.dataset = dataset
        self.recording = recording
        self.id = recording["id"]
        self.size_bytes = recording["total_bytes"]
        self._running = False
        self._recording_synced = self._get_synced_data()
        _rgb = self._recording_synced.frames[0].rgb_images
        _depth = self._recording_synced.frames[0].depth_images
        self._camera_ids = {
            "rgbs": list(_rgb.keys()) if _rgb else [],
            "depths": list(_depth.keys()) if _depth else [],
        }
        self._episode_length = len(self._recording_synced.frames)

    def _get_synced_data(self) -> SyncedData:
        """Retrieve synchronized metadata for the recording.

        Returns:
            SyncedData containing frame timing and camera information.

        Raises:
            requests.HTTPError: If the API request fails.
        """
        auth = get_auth()
        response = requests.post(
            f"{API_URL}/visualization/demonstrations/{self.recording['id']}/sync",
            headers=auth.get_headers(),
        )
        response.raise_for_status()
        return SyncedData.model_validate(response.json())

    def _get_video_url(self, camera_type: str, camera_id: str) -> str:
        """Get streaming URL for a specific camera's video data.

        Args:
            camera_type: Type of camera data ("rgbs" or "depths").
            camera_id: Unique identifier for the camera.

        Returns:
            URL for streaming the video data.

        Raises:
            requests.HTTPError: If the API request fails.
        """
        auth = get_auth()
        response = requests.get(
            f"{API_URL}/recording/{self.recording['id']}/download_url",
            params={"filepath": f"{camera_type}/{camera_id}/video.mp4"},
            headers=auth.get_headers(),
        )
        response.raise_for_status()
        return response.json()["url"]

    def _stream_data_loop(self, camera_type: str, camera_id: str):
        """Stream video frames from a camera in a background thread.

        Downloads and queues video frames for synchronized access. Runs
        in a separate thread to enable concurrent streaming from multiple
        cameras.

        Args:
            camera_type: Type of camera data ("rgbs" or "depths").
            camera_id: Unique identifier for the camera.
        """
        camera_url = self._get_video_url(camera_type, camera_id)
        with VideoStreamer(camera_url) as streamer:
            for i, frame in enumerate(streamer):
                self._msg_queues[camera_id].put((frame, i))
        # Signal end of data stream
        self._msg_queues[camera_id].put((None, None))

    def close(self):
        """Explicitly close the iterator and clean up streaming threads.

        Stops all background streaming threads and releases resources.
        Should be called when done processing the episode to prevent
        resource leaks.
        """
        if self._running:
            self._running = False
            for t in self._threads:
                t.join(timeout=2.0)

    def _populate_video_frames(
        self,
        camera_data: dict[str, CameraData],
        transform_fn: Callable[[np.ndarray], np.ndarray] = None,
    ):
        """Populate camera data with frames from streaming queues.

        Retrieves frames from the appropriate streaming thread queues and
        applies optional transformations. Synchronizes frames based on
        frame indices to ensure temporal alignment.

        Args:
            camera_data: Dictionary mapping camera IDs to CameraData objects.
            transform_fn: Optional function to transform frames before assignment.
        """
        for camera_id, cam_data in camera_data.items():
            while True:
                try:
                    frame, frame_idx = self._msg_queues[camera_id].get(timeout=10.0)
                except queue.Empty:
                    frame = None
                if frame is None:
                    break
                if frame_idx == cam_data.frame_idx:
                    cam_data.frame = transform_fn(frame) if transform_fn else frame
                    break

    def __next__(self) -> SyncPoint:
        """Get the next synchronized data point in the episode.

        Retrieves the next frame of synchronized data including RGB images,
        depth images, and sensor information. Video frames are populated
        from streaming threads with appropriate transformations applied.

        Returns:
            SyncPoint containing all synchronized data for this timestep.

        Raises:
            StopIteration: When all frames in the episode have been processed.
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

    def __iter__(self):
        """Initialize iteration over the episode with background streaming.

        Sets up streaming threads for all cameras and initializes queues
        for frame synchronization.

        Returns:
            Self for iteration over synchronized data points.
        """
        self._iter_idx = 0
        self._msg_queues: dict[str, queue.Queue] = {}
        self._threads: list[threading.Thread] = []
        self._running = True
        for cam_type, camera_ids in self._camera_ids.items():
            for camera_id in camera_ids:
                self._msg_queues[camera_id] = queue.Queue()
                thread = threading.Thread(
                    target=self._stream_data_loop, args=(cam_type, camera_id)
                )
                thread.daemon = True
                thread.start()
                self._threads.append(thread)
        return self

    def __enter__(self):
        """Context manager entry point.

        Returns:
            Self for use in with statements.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point that ensures proper cleanup.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        self.close()

    def __del__(self):
        """Cleanup when object is destroyed to prevent resource leaks."""
        self.close()

    def __len__(self) -> int:
        """Get the number of timesteps in the episode.

        Returns:
            Number of synchronized data points in the episode.
        """
        return self._episode_length

    def __getitem__(self, idx):
        """Support for indexing and slicing episode data.

        Args:
            idx: Index or slice for accessing specific timesteps.

        Raises:
            NotImplementedError: This feature is not yet implemented.
        """
        raise NotImplementedError("Indexing not yet implemented for EpisodeIterator")
