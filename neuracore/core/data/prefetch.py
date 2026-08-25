"""Concurrent prefetch of synchronized metadata and recording videos.

Every network request is issued from a single thread on one asyncio event loop,
with ``inflight_requests`` outstanding at a time; decoding runs in a small thread
pool. A bounded queue joins the two stages so videos awaiting decode cannot pile
up on disk.

Each camera's frames are decoded into a staging directory and published with a
single ``os.replace`` under a sibling lock file, so a frames directory that
exists is always complete. Whatever this prefetch skips or fails to fetch is
downloaded lazily by ``SynchronizedRecording._get_frame_from_disk_cache``.
"""

import asyncio
import logging
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp
from neuracore_types import DataType, SynchronizationDetails
from neuracore_types import SynchronizedEpisode as SynchronizedEpisodeModel
from neuracore_types import (
    SynchronizeRecordingProgress,
    SynchronizeRecordingRequest,
    SynchronizeRecordingStartResponse,
    SynchronizeRecordingStatus,
)
from tqdm import tqdm

from neuracore.core.auth import get_auth
from neuracore.core.const import API_URL
from neuracore.core.data.frame_cache import (
    check_stale_lock_file,
    clear_stale_lock,
    create_decoding_lock,
    delete_decoding_lock,
    lock_file_for,
    publish_decoded_frames,
    video_filename_preference,
)
from neuracore.core.data.synced_recording import (
    SYNCED_RECORDING_POLL_INTERVAL_S,
    SYNCED_RECORDING_TIMEOUT_S,
)
from neuracore.core.exceptions import SynchronizationError
from neuracore.core.utils.download import DOWNLOAD_CHUNK_SIZE
from neuracore.core.utils.http_session import retry_connection_failures

if TYPE_CHECKING:
    from neuracore.core.data.dataset import Dataset
    from neuracore.core.data.recording import Recording

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENT_PREFETCH_REQUESTS = 16
"""Requests kept outstanding at once. Throughput flattens out beyond this."""

_SOCKET_CONNECT_TIMEOUT_S = 15.0
_SOCKET_READ_TIMEOUT_S = 120.0

_VIDEO_DATA_TYPES = (DataType.RGB_IMAGES, DataType.DEPTH_IMAGES)


@dataclass
class _PendingDecode:
    """A downloaded video waiting to be decoded and published."""

    recording_id: str
    camera_id: str
    video_path: Path
    staging_dir: Path
    frames_dir: Path
    lock_file: Path
    temp_dir: tempfile.TemporaryDirectory


class VideoPrefetcher:
    """Fetches synchronized metadata and videos for a whole dataset at once.

    Attributes:
        episodes: Synchronized episode metadata by recording index, populated by
            ``run``.
    """

    def __init__(
        self,
        dataset: "Dataset",
        recordings: list["Recording"],
        synchronization_details: SynchronizationDetails,
        inflight_requests: int = DEFAULT_CONCURRENT_PREFETCH_REQUESTS,
        decode_workers: int = 4,
        download_videos: bool = True,
    ):
        """Initialize a prefetcher for one synchronized dataset.

        Args:
            dataset: Parent dataset, used for its org and cache directory.
            recordings: Recordings to prefetch, in dataset index order.
            synchronization_details: Parameters the data was synchronized with.
            inflight_requests: Network requests kept outstanding at once.
            decode_workers: Threads used to run ffmpeg.
            download_videos: Whether to download videos, or only fetch the
                synchronized metadata.
        """
        self.dataset = dataset
        self.recordings = recordings
        self.synchronization_details = synchronization_details
        self.inflight_requests = max(1, inflight_requests)
        self.decode_workers = max(1, decode_workers)
        self.download_videos = download_videos
        self.episodes: dict[int, SynchronizedEpisodeModel] = {}
        self._failures = 0
        self._lock = threading.Lock()
        # Created once the event loop is running.
        self._api_requests: asyncio.Semaphore | None = None
        self._transfers: asyncio.Semaphore | None = None

    def run(self) -> dict[int, SynchronizedEpisodeModel]:
        """Fetch metadata and, if enabled, download and decode every video.

        Failures for individual recordings or cameras are logged and skipped
        rather than raised, leaving them to the lazy download path.

        Returns:
            Synchronized episode metadata by recording index. A recording whose
            metadata could not be fetched is absent.
        """
        asyncio.run(self._run_async())
        if self._failures:
            logger.warning(
                f"{self._failures} video(s) could not be prefetched and will be "
                "downloaded on demand during training"
            )
        return self.episodes

    async def _run_async(self) -> None:
        """Run the metadata stage and then, if enabled, the video stages."""
        # Metadata and signed-URL calls are short; a video transfer holds its
        # connection for seconds. They get separate budgets so a burst of
        # transfers cannot starve the small requests that queue up the next
        # ones, and the connector has to allow for both at once.
        self._api_requests = asyncio.Semaphore(self.inflight_requests)
        self._transfers = asyncio.Semaphore(self.inflight_requests)
        connector = aiohttp.TCPConnector(
            limit=2 * self.inflight_requests, ttl_dns_cache=300
        )
        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=_SOCKET_CONNECT_TIMEOUT_S,
            sock_read=_SOCKET_READ_TIMEOUT_S,
        )
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            middlewares=(retry_connection_failures,),
        ) as session:
            if self.download_videos:
                await self._fetch_and_download(session)
            else:
                await self._fetch_all_metadata(session)

    async def _fetch_all_metadata(self, session: aiohttp.ClientSession) -> None:
        """Request every recording's synchronized metadata concurrently."""
        api_requests = self._api_requests
        assert api_requests is not None
        progress = tqdm(
            total=len(self.recordings),
            desc=f"Fetching synced data ({self.inflight_requests} in flight)",
            unit="Recording",
        )

        async def fetch(index: int, recording: "Recording") -> None:
            async with api_requests:
                try:
                    self.episodes[index] = await self._get_synced_data(
                        session, recording.id
                    )
                except Exception as exc:
                    logger.warning(
                        f"Could not fetch synced data for recording "
                        f"{recording.id}: {exc}"
                    )
                finally:
                    progress.update(1)

        try:
            await asyncio.gather(*[
                fetch(index, recording)
                for index, recording in enumerate(self.recordings)
            ])
        finally:
            progress.close()

    async def _get_synced_data(
        self, session: aiohttp.ClientSession, recording_id: str
    ) -> SynchronizedEpisodeModel:
        """Synchronize one recording and download its episode metadata.

        Args:
            session: Shared client session.
            recording_id: Recording to synchronize.

        Returns:
            The synchronized episode for the recording.
        """
        base_url = f"{API_URL}/org/{self.dataset.org_id}/synchronize"
        async with session.post(
            f"{base_url}/trigger-synchronize-recording",
            json=SynchronizeRecordingRequest(
                recording_id=recording_id,
                synchronization_details=self.synchronization_details,
            ).model_dump(mode="json"),
            headers=get_auth().get_headers(),
        ) as response:
            response.raise_for_status()
            job = SynchronizeRecordingStartResponse.model_validate(
                await response.json()
            )

        deadline = asyncio.get_running_loop().time() + SYNCED_RECORDING_TIMEOUT_S
        while True:
            async with session.get(
                f"{base_url}/synchronize-recording-progress/"
                f"{job.synchronized_recording_id}",
                params={"recording_id": recording_id},
                headers=get_auth().get_headers(),
            ) as response:
                response.raise_for_status()
                progress = SynchronizeRecordingProgress.model_validate(
                    await response.json()
                )

            if progress.status is SynchronizeRecordingStatus.READY:
                if not progress.download_url:
                    raise SynchronizationError(
                        f"Synchronizing recording {recording_id} reported READY "
                        "without a download URL"
                    )
                break
            if progress.status is SynchronizeRecordingStatus.FAILED:
                raise SynchronizationError(
                    f"Synchronizing recording {recording_id} failed: "
                    f"{progress.error or 'no reason given'}"
                )
            if asyncio.get_running_loop().time() >= deadline:
                raise SynchronizationError(
                    f"Timed out after {SYNCED_RECORDING_TIMEOUT_S:.0f}s waiting "
                    f"for recording {recording_id} to synchronize"
                )
            await asyncio.sleep(SYNCED_RECORDING_POLL_INTERVAL_S)

        try:
            async with session.get(progress.download_url) as response:
                response.raise_for_status()
                payload = await response.json()
        except aiohttp.ClientError as exc:
            # aiohttp includes the signed URL (and its credentials) in request
            # errors, so only expose the response status or exception class.
            status = (
                exc.status if isinstance(exc, aiohttp.ClientResponseError) else None
            )
            detail = f"HTTP {status}" if status is not None else type(exc).__name__
            raise SynchronizationError(
                f"Failed to download synchronized episode for recording "
                f"{recording_id} ({detail})"
            ) from None
        return SynchronizedEpisodeModel.model_validate(payload)

    async def _fetch_and_download(self, session: aiohttp.ClientSession) -> None:
        """Fetch metadata and download videos as one overlapped pipeline.

        Each recording's downloads start as soon as its own metadata arrives, so
        bytes begin moving almost immediately instead of waiting for every
        recording's metadata. Downloads run on this thread's event loop, ffmpeg
        in a thread pool, joined by a bounded queue that caps how many videos
        sit staged on disk awaiting decode.
        """
        metadata_progress = tqdm(
            total=len(self.recordings),
            desc=f"Fetching synced data ({self.inflight_requests} in flight)",
            unit="Recording",
        )
        # The video total is not known until each recording's metadata says how
        # many cameras it has, so it grows as the pipeline discovers them.
        video_progress = tqdm(total=0, desc="Downloading videos", unit="Video")
        queue: asyncio.Queue[_PendingDecode | None] = asyncio.Queue(
            maxsize=2 * self.decode_workers
        )
        api_requests = self._api_requests
        assert api_requests is not None
        loop = asyncio.get_running_loop()

        with ThreadPoolExecutor(max_workers=self.decode_workers) as executor:

            async def decode_consumer() -> None:
                """Drain the queue, decoding each video off the event loop."""
                while True:
                    pending = await queue.get()
                    if pending is None:  # shutdown sentinel
                        queue.task_done()
                        return
                    try:
                        await loop.run_in_executor(
                            executor, _decode_and_publish, pending
                        )
                    except Exception as exc:
                        logger.warning(
                            f"Could not decode video for camera "
                            f"{pending.camera_id} of recording "
                            f"{pending.recording_id}: {exc}"
                        )
                        self._record_failure()
                    finally:
                        queue.task_done()

            async def download(target: "_DownloadTarget") -> None:
                """Stage one camera's video, then hand it to the decoders."""
                pending = None
                try:
                    pending = await self._download_video(session, target)
                except Exception as exc:
                    logger.warning(
                        f"Could not download video for camera "
                        f"{target.camera_id} of recording "
                        f"{target.recording_id}: {exc}"
                    )
                    self._record_failure()
                    target.release()
                finally:
                    video_progress.update(1)
                # Queued outside every budget: blocking here while the decoders
                # are saturated must not hold a request slot.
                if pending is not None:
                    await queue.put(pending)

            async def process_recording(index: int, recording: "Recording") -> None:
                """Fetch one recording's metadata, then download its videos."""
                async with api_requests:
                    try:
                        self.episodes[index] = await self._get_synced_data(
                            session, recording.id
                        )
                    except Exception as exc:
                        logger.warning(
                            f"Could not fetch synced data for recording "
                            f"{recording.id}: {exc}"
                        )
                        return
                    finally:
                        metadata_progress.update(1)

                    targets = self._collect_targets_for(recording, self.episodes[index])
                    # Mint each URL while still holding this slot. Semaphores
                    # hand out slots in order, so a mint asking for a slot of
                    # its own would queue behind every recording's metadata
                    # request and not run until that stage had drained -- the
                    # serialisation this pipeline exists to remove.
                    ready = []
                    for target in targets:
                        try:
                            target.url = await self._get_video_url(session, target)
                            ready.append(target)
                        except Exception as exc:
                            logger.warning(
                                f"Could not resolve video for camera "
                                f"{target.camera_id} of recording "
                                f"{target.recording_id}: {exc}"
                            )
                            self._record_failure()
                            target.release()

                if not ready:
                    return
                video_progress.total += len(ready)
                video_progress.refresh()
                await asyncio.gather(*[download(target) for target in ready])

            consumers = [
                asyncio.create_task(decode_consumer())
                for _ in range(self.decode_workers)
            ]
            try:
                await asyncio.gather(*[
                    process_recording(index, recording)
                    for index, recording in enumerate(self.recordings)
                ])
                await queue.join()
            finally:
                for _ in consumers:
                    await queue.put(None)
                await asyncio.gather(*consumers, return_exceptions=True)
                metadata_progress.close()
                video_progress.close()

    def _collect_download_targets(self) -> list["_DownloadTarget"]:
        """Find every camera whose frames are not already cached.

        Returns:
            Targets to download, each holding an acquired decoding lock.
        """
        targets: list[_DownloadTarget] = []
        for index, recording in enumerate(self.recordings):
            episode = self.episodes.get(index)
            if episode is not None:
                targets += self._collect_targets_for(recording, episode)
        return targets

    def _collect_targets_for(
        self, recording: "Recording", episode: SynchronizedEpisodeModel
    ) -> list["_DownloadTarget"]:
        """Find one recording's cameras whose frames are not already cached.

        Args:
            recording: Recording to inspect.
            episode: That recording's synchronized metadata.

        Returns:
            Targets to download, each holding an acquired decoding lock.
        """
        if not episode.observations:
            return []

        targets: list[_DownloadTarget] = []
        observation = episode.observations[0]
        for data_type in _VIDEO_DATA_TYPES:
            for camera_id in observation.data.get(data_type, {}):
                frames_dir = (
                    self.dataset.cache_dir / recording.id / data_type.value / camera_id
                )
                lock_file = lock_file_for(frames_dir)

                # This must not block: it runs on the one thread driving every
                # download, so waiting on a lock would stall the whole prefetch
                # over a single camera. Stale locks are cleared, live ones left
                # to their owner, and the lazy path waits if training later
                # needs those frames.
                if check_stale_lock_file(lock_file):
                    clear_stale_lock(lock_file, frames_dir)
                elif lock_file.exists():
                    continue

                if frames_dir.exists():
                    continue
                frames_dir.parent.mkdir(parents=True, exist_ok=True)
                if not create_decoding_lock(lock_file):
                    # Another process got there first; leave it to them and let
                    # the lazy path pick up anything they miss.
                    continue
                targets.append(
                    _DownloadTarget(
                        recording_id=recording.id,
                        data_type=data_type,
                        camera_id=camera_id,
                        frames_dir=frames_dir,
                        lock_file=lock_file,
                    )
                )
        return targets

    async def _download_video(
        self, session: aiohttp.ClientSession, target: "_DownloadTarget"
    ) -> _PendingDecode:
        """Mint a signed URL for one camera's video and stream it to disk.

        Args:
            session: Shared client session.
            target: Camera to download, with its lock already held.

        Returns:
            The staged video, ready to decode.
        """
        transfers = self._transfers
        assert transfers is not None

        # Stage on the same filesystem as the cache so the frames directory can
        # be published with os.replace once decoding finishes.
        temp_dir = tempfile.TemporaryDirectory(dir=target.frames_dir.parent)
        try:
            staging_dir = Path(temp_dir.name) / "frames"
            staging_dir.mkdir()
            video_path = (
                Path(temp_dir.name) / f"{target.camera_id}{target.data_type.value}.mp4"
            )
            async with transfers:
                url = target.url or await self._get_video_url(session, target)
                await self._stream_video(session, url, video_path)
        except BaseException:
            temp_dir.cleanup()
            raise

        return _PendingDecode(
            recording_id=target.recording_id,
            camera_id=target.camera_id,
            video_path=video_path,
            staging_dir=staging_dir,
            frames_dir=target.frames_dir,
            lock_file=target.lock_file,
            temp_dir=temp_dir,
        )

    async def _stream_video(
        self, session: aiohttp.ClientSession, url: str, destination: Path
    ) -> None:
        """Stream a video's bytes to a file.

        Args:
            session: Shared client session.
            url: Signed URL to read from.
            destination: File to write the body to.
        """
        async with session.get(url) as response:
            response.raise_for_status()
            with open(destination, "wb") as handle:
                async for chunk in response.content.iter_chunked(DOWNLOAD_CHUNK_SIZE):
                    handle.write(chunk)

    async def _get_video_url(
        self, session: aiohttp.ClientSession, target: "_DownloadTarget"
    ) -> str:
        """Get a signed URL for a camera's video, trying each candidate name.

        Args:
            session: Shared client session.
            target: Camera to resolve a URL for.

        Returns:
            Signed download URL.

        Raises:
            FileNotFoundError: If no candidate filename exists.
        """
        preference = video_filename_preference(target.data_type)
        for filename in preference:
            filepath = f"{target.data_type.value}/{target.camera_id}/{filename}"
            url = (
                f"{API_URL}/org/{self.dataset.org_id}"
                f"/recording/{target.recording_id}/download_url"
            )
            async with session.get(
                url,
                params={"filepath": filepath},
                headers=get_auth().get_headers(),
            ) as response:
                if response.status == 404:
                    continue
                response.raise_for_status()
                payload = await response.json()
            return payload["url"]

        raise FileNotFoundError(
            f"No candidate filename found for recording {target.recording_id} "
            f"(camera {target.data_type.value}/{target.camera_id}); "
            f"tried: {preference}"
        )

    def _record_failure(self) -> None:
        """Count one camera left for the lazy download path."""
        with self._lock:
            self._failures += 1


@dataclass
class _DownloadTarget:
    """A camera whose frames are missing from the cache, with its lock held."""

    recording_id: str
    data_type: DataType
    camera_id: str
    frames_dir: Path
    lock_file: Path
    url: str | None = None

    def release(self) -> None:
        """Drop the decoding lock without publishing anything."""
        delete_decoding_lock(self.lock_file)


def _decode_and_publish(pending: _PendingDecode) -> None:
    """Decode a staged video and publish its frames atomically.

    Runs in a worker thread. Always releases the lock and the staging directory,
    so a failure leaves the cache untouched rather than half-populated.

    Args:
        pending: The staged video to decode.
    """
    try:
        publish_decoded_frames(
            pending.video_path, pending.staging_dir, pending.frames_dir
        )
    finally:
        delete_decoding_lock(pending.lock_file)
        pending.temp_dir.cleanup()
