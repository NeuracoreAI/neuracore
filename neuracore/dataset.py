import asyncio
import base64
import concurrent
import io
import json
import logging
import threading

import numpy as np
import requests
import websockets
from PIL import Image

from .auth import Auth, get_auth
from .const import API_URL

logger = logging.getLogger(__name__)


class DatasetError(Exception):
    """Exception raised for errors in the dataset module."""

    pass


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
        name: str, description: str | None = None, tags: list[str] | None = None
    ) -> "Dataset":
        ds = Dataset.get(name, non_exist_ok=True)
        if ds is None:
            ds = Dataset._create_dataset(name, description, tags)
        else:
            logger.info(f"Dataset '{name}' already exist.")
        return ds

    @staticmethod
    def _create_dataset(
        name: str, description: str | None = None, tags: list[str] | None = None
    ) -> "Dataset":
        auth: Auth = get_auth()
        response = requests.post(
            f"{API_URL}/datasets",
            headers=auth.get_headers(),
            json={"name": name, "description": description, "tags": tags},
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
                if not 0 <= idx < len(self.recordings):
                    raise IndexError("Dataset index out of range")
                return EpisodeIterator(self, self.recordings[idx])
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
        self._frame_count = 0
        recording_preview = self._get_recording_preview()
        self.episode_length = recording_preview["statistics"]["total_frames"]

    def _run_event_loop(self):
        """Run async event loop in separate thread with proper cleanup."""
        try:
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._stream_data())
        finally:
            # Clean up the loop
            self._loop.close()

    async def _stream_data(self):
        """Stream data from WebSocket with proper cleanup."""
        try:
            auth = get_auth()
            base_url = API_URL.replace("http://", "ws://").replace("https://", "wss://")

            # Connect to the new timestep stream endpoint
            url = (
                f"{base_url}/visualization/demonstrations/"
                f"{self.recording['id']}/timestep/stream"
            )

            async with websockets.connect(
                url,
                additional_headers=auth.get_headers(),
                max_size=None,
            ) as websocket:
                # Process frames as they arrive
                while self._running:
                    try:
                        msg = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(msg)

                        if data.get("type") == "frame":
                            frame_data = data.get("data", {})
                            processed_data = self._process_frame(frame_data)
                            await self._msg_queue.put(processed_data)
                            self._frame_count += 1

                            # Break if we've received all frames
                            if self._frame_count >= self.episode_length:
                                logger.info(f"Received all {self._frame_count} frames")
                                break

                        elif data.get("type") == "error":
                            error_msg = data.get("message", "Unknown error")
                            logger.error(f"WebSocket error: {error_msg}")
                            await self._msg_queue.put(
                                DatasetError(f"Stream error: {error_msg}")
                            )
                            break

                    except asyncio.TimeoutError:
                        if not self._running:
                            break
                        continue

                    except websockets.exceptions.ConnectionClosed:
                        logger.info("WebSocket connection closed")
                        break

                    except Exception as e:
                        logger.error(f"Stream processing error: {str(e)}")
                        await self._msg_queue.put(
                            DatasetError(f"Stream error: {str(e)}")
                        )
                        break

                # Signal end of data stream
                await self._msg_queue.put(None)

        except Exception as e:
            logger.error(f"Stream setup error: {str(e)}")
            await self._msg_queue.put(DatasetError(f"Stream setup error: {str(e)}"))
        finally:
            # Always put None on queue to signal end
            try:
                await self._msg_queue.put(None)
            except Exception:
                pass
            self._running = False

    def close(self):
        """Explicitly close with proper cleanup."""
        if self._running:
            self._running = False
            self._thread.join(timeout=2.0)

    def _get_recording_preview(self):
        """Get preview data for the recording."""
        auth = get_auth()
        url = f"{API_URL}/visualization/demonstrations/{self.recording['id']}/preview"
        if self.dataset.is_shared:
            url += "?is_shared=true"
        response = requests.get(url, headers=auth.get_headers())
        response.raise_for_status()
        return response.json()

    def _process_frame(self, frame_data: dict) -> dict:
        """Process raw frame data into numpy arrays."""
        processed = {
            "timestamp": frame_data["timestamp"],
            "joint_positions": {
                k: float(v) for k, v in frame_data["joint_positions"].items()
            },
        }

        # Process images if present
        if "images" in frame_data:
            processed["images"] = {}
            for cam_id, img_data in frame_data["images"].items():
                # Decode base64 image
                img_bytes = base64.b64decode(img_data)
                img = Image.open(io.BytesIO(img_bytes))
                processed["images"][cam_id] = np.array(img)

        return processed

    def __next__(self):
        """Get next frame with proper thread state handling and auto-cleanup."""
        while self._loop.is_running() or self._running:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._msg_queue.get(), self._loop
                )
                data = future.result(timeout=1.0)
            except (asyncio.TimeoutError, concurrent.futures.TimeoutError):
                continue
            if data is None:
                self._running = False
                break
            return data
        raise StopIteration

    def __iter__(self):
        self._msg_queue = asyncio.Queue()
        self._received_data = False

        # Thread control
        self._running = True
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_event_loop)
        self._thread.daemon = True
        self._thread.start()

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
        return self.episode_length

    def __getitem__(self, idx):
        """Support for indexing and slicing."""
        raise NotImplementedError("Indexing not yet implemented for EpisodeIterator")
