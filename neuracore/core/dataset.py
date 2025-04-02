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
    """Represents a dataset that can be streamed or used for training."""

    def __init__(self, dataset_dict: dict, recordings: list[dict] = None):
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
        """Convert to PyTorch dataset format."""
        raise NotImplementedError("PyTorch dataset conversion not yet implemented")

    def __iter__(self) -> "Dataset":
        """Returns an iterator over episodes in the dataset."""
        return self

    def __len__(self) -> int:
        """Returns the number of episodes in the dataset."""
        return self.num_episodes

    def __getitem__(self, idx):
        """Support for indexing and slicing."""
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
        if self._previous_iterator is not None:
            self._previous_iterator.close()


class EpisodeIterator:

    def __init__(self, dataset, recording):
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
        """Get synchronized data for the recording."""
        auth = get_auth()
        response = requests.post(
            f"{API_URL}/visualization/demonstrations/{self.recording['id']}/sync",
            headers=auth.get_headers(),
        )
        response.raise_for_status()
        return SyncedData.model_validate(response.json())

    def _get_video_url(self, camera_type: str, camera_id: str) -> str:
        """Get video URL for the given camera ID."""
        auth = get_auth()
        response = requests.get(
            f"{API_URL}/recording/{self.recording['id']}/download_url",
            params={"filepath": f"{camera_type}/{camera_id}/video.mp4"},
            headers=auth.get_headers(),
        )
        response.raise_for_status()
        return response.json()["url"]

    def _stream_data_loop(self, camera_type: str, camera_id: str):
        """Stream data from the video URL."""
        camera_url = self._get_video_url(camera_type, camera_id)
        with VideoStreamer(camera_url) as streamer:
            for i, frame in enumerate(streamer):
                self._msg_queues[camera_id].put((frame, i))
        # Signal end of data stream
        self._msg_queues[camera_id].put((None, None))

    def close(self):
        """Explicitly close with proper cleanup."""
        if self._running:
            self._running = False
            for t in self._threads:
                t.join(timeout=2.0)

    def _populate_video_frames(
        self,
        camera_data: dict[str, CameraData],
        transform_fn: Callable[[np.ndarray], np.ndarray] = None,
    ):
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
        """Get next frame with proper thread state handling and auto-cleanup."""
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
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()

    def __len__(self) -> int:
        """Returns the number of steps in the episode."""
        return self._episode_length

    def __getitem__(self, idx):
        """Support for indexing and slicing."""
        raise NotImplementedError("Indexing not yet implemented for EpisodeIterator")
