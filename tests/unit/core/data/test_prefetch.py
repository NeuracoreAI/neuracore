"""Tests for the concurrent metadata and video prefetch."""

import asyncio
from unittest.mock import patch

import pytest
from neuracore_types import DataType, SynchronizationDetails

import neuracore as nc
from neuracore.core.data.dataset import Dataset
from neuracore.core.data.prefetch import VideoPrefetcher


@pytest.fixture
def dataset_mock(mock_data_requests, tmp_path, mocked_org_id) -> Dataset:
    """Dataset backed by mocked API responses, caching into tmp_path."""
    nc.login()
    dataset = Dataset.get_by_name("test_dataset")
    dataset.cache_dir = tmp_path / "recording_cache"
    dataset.cache_dir.mkdir(parents=True, exist_ok=True)
    return dataset


@pytest.fixture
def details() -> SynchronizationDetails:
    """Synchronization parameters matching the mocked responses."""
    return SynchronizationDetails(frequency=30, cross_embodiment_union=None)


def _prefetcher(dataset, details, **kwargs) -> VideoPrefetcher:
    recordings = [dataset[idx] for idx in range(len(dataset))]
    return VideoPrefetcher(
        dataset=dataset,
        recordings=recordings,
        synchronization_details=details,
        **kwargs,
    )


class TestMetadataStage:
    """The metadata stage fetches every recording exactly once."""

    def test_fetches_metadata_for_every_recording(self, dataset_mock, details):
        """Every recording index should come back with its episode."""
        prefetcher = _prefetcher(dataset_mock, details, download_videos=False)
        episodes = prefetcher.run()

        assert set(episodes) == set(range(len(dataset_mock)))
        for episode in episodes.values():
            assert episode.observations

    def test_metadata_failure_is_skipped_not_raised(self, dataset_mock, details):
        """A recording whose metadata fails is absent, and does not abort the run."""
        prefetcher = _prefetcher(dataset_mock, details, download_videos=False)
        with patch.object(
            VideoPrefetcher, "_get_synced_data", side_effect=RuntimeError("boom")
        ):
            episodes = prefetcher.run()

        assert episodes == {}

    def test_download_videos_false_skips_transfers(self, dataset_mock, details):
        """Metadata-only runs must not touch the video stage."""
        prefetcher = _prefetcher(dataset_mock, details, download_videos=False)
        with patch.object(VideoPrefetcher, "_download_video") as mock_download:
            prefetcher.run()

        mock_download.assert_not_called()


class TestDownloadTargetCollection:
    """Which cameras the prefetch decides it needs to fetch."""

    def test_uncached_cameras_become_targets(self, dataset_mock, details):
        """With an empty cache, every camera in the sync point is a target."""
        prefetcher = _prefetcher(dataset_mock, details, download_videos=False)
        prefetcher.run()
        prefetcher.download_videos = True

        targets = prefetcher._collect_download_targets()

        assert targets
        # The mocked sync point carries one RGB and one depth camera.
        assert {target.data_type for target in targets} == {
            DataType.RGB_IMAGES,
            DataType.DEPTH_IMAGES,
        }
        # Every target holds its lock, so a second pass finds nothing to do.
        assert all(target.lock_file.exists() for target in targets)

    def test_cached_cameras_are_skipped(self, dataset_mock, details):
        """A published frames directory means there is nothing to download."""
        prefetcher = _prefetcher(dataset_mock, details, download_videos=False)
        prefetcher.run()
        prefetcher.download_videos = True

        # Publish every camera's frames, as a completed decode would.
        for target in prefetcher._collect_download_targets():
            target.release()
            target.frames_dir.mkdir(parents=True, exist_ok=True)

        assert prefetcher._collect_download_targets() == []

    def test_camera_locked_by_another_worker_is_skipped(self, dataset_mock, details):
        """A held lock means another worker owns that camera."""
        prefetcher = _prefetcher(dataset_mock, details, download_videos=False)
        prefetcher.run()
        prefetcher.download_videos = True

        first_pass = prefetcher._collect_download_targets()
        assert first_pass
        # Locks are still held from the first pass, so nothing is claimed twice.
        assert prefetcher._collect_download_targets() == []

    def test_released_lock_can_be_reclaimed(self, dataset_mock, details):
        """Releasing a lock without publishing lets a later attempt retry."""
        prefetcher = _prefetcher(dataset_mock, details, download_videos=False)
        prefetcher.run()
        prefetcher.download_videos = True

        for target in prefetcher._collect_download_targets():
            target.release()

        assert prefetcher._collect_download_targets()


class TestVideoStage:
    """End-to-end behaviour of the download and decode pipeline."""

    def test_prefetch_populates_frame_cache(self, dataset_mock, details):
        """A full run leaves decoded frames published for every camera."""
        prefetcher = _prefetcher(
            dataset_mock, details, download_videos=True, decode_workers=2
        )
        episodes = prefetcher.run()

        assert episodes
        for recording in prefetcher.recordings:
            rgb_root = dataset_mock.cache_dir / recording.id / DataType.RGB_IMAGES.value
            for camera_dir in rgb_root.iterdir():
                assert (camera_dir / "0.png").exists()

    def test_no_locks_are_left_behind(self, dataset_mock, details):
        """Every lock taken must be released, whatever happened to its camera."""
        prefetcher = _prefetcher(
            dataset_mock, details, download_videos=True, decode_workers=2
        )
        prefetcher.run()

        assert list(dataset_mock.cache_dir.rglob("*.recording.lock")) == []

    def test_download_failure_releases_lock_and_is_counted(self, dataset_mock, details):
        """A failed transfer must not leave a lock stranding the camera."""
        prefetcher = _prefetcher(
            dataset_mock, details, download_videos=True, decode_workers=2
        )
        with patch.object(
            VideoPrefetcher, "_download_video", side_effect=RuntimeError("boom")
        ):
            prefetcher.run()

        assert prefetcher._failures > 0
        assert list(dataset_mock.cache_dir.rglob("*.recording.lock")) == []
        # Nothing was published, so a later attempt still sees work to do.
        assert prefetcher._collect_download_targets()

    def test_decode_failure_leaves_no_partial_cache(self, dataset_mock, details):
        """A failed decode must publish nothing rather than a partial directory."""
        prefetcher = _prefetcher(
            dataset_mock, details, download_videos=True, decode_workers=2
        )
        with patch(
            "neuracore.core.data.frame_cache.decode_video",
            side_effect=RuntimeError("boom"),
        ):
            prefetcher.run()

        assert prefetcher._failures > 0
        assert list(dataset_mock.cache_dir.rglob("*.recording.lock")) == []
        assert list(dataset_mock.cache_dir.rglob("*.png")) == []

    def test_frames_published_by_another_worker_are_not_overwritten(
        self, dataset_mock, details
    ):
        """A camera published mid-flight is left exactly as the other worker left it."""
        prefetcher = _prefetcher(
            dataset_mock, details, download_videos=True, decode_workers=1
        )
        prefetcher.download_videos = False
        prefetcher.run()
        prefetcher.download_videos = True

        targets = prefetcher._collect_download_targets()
        target = targets[0]
        for other in targets[1:]:
            other.release()

        # Simulate another process publishing while this download is in flight.
        target.frames_dir.mkdir(parents=True, exist_ok=True)
        sentinel = target.frames_dir / "sentinel.txt"
        sentinel.write_text("published elsewhere")
        target.release()

        # Reclaiming now finds the directory present, so it is left untouched.
        assert target.frames_dir not in [
            t.frames_dir for t in prefetcher._collect_download_targets()
        ]
        assert sentinel.read_text() == "published elsewhere"


class TestMetadataDownloadOverlap:
    """Downloads must not wait for every recording's metadata to arrive."""

    def test_downloads_start_before_all_metadata_is_fetched(
        self, dataset_mock, details
    ):
        """A recording's videos start as soon as its own metadata lands.

        Made deterministic by holding one recording's metadata until a download
        has begun: that can only resolve if downloads are not gated on every
        recording's metadata, so a barrier shows up as the held request timing
        out and its episode going missing.
        """
        prefetcher = _prefetcher(
            dataset_mock, details, download_videos=True, decode_workers=2
        )
        original_metadata = VideoPrefetcher._get_synced_data
        original_download = VideoPrefetcher._download_video
        download_started: list[asyncio.Event] = []
        seen = {"n": 0}

        async def gated_metadata(self, session, recording_id):
            if not download_started:
                download_started.append(asyncio.Event())
            seen["n"] += 1
            if seen["n"] > 1:
                # Every recording after the first waits for a download to begin.
                await asyncio.wait_for(download_started[0].wait(), timeout=5)
            return await original_metadata(self, session, recording_id)

        async def signalling_download(self, session, target):
            if download_started:
                download_started[0].set()
            return await original_download(self, session, target)

        with (
            patch.object(VideoPrefetcher, "_get_synced_data", gated_metadata),
            patch.object(VideoPrefetcher, "_download_video", signalling_download),
        ):
            episodes = prefetcher.run()

        assert len(episodes) == len(prefetcher.recordings)

    def test_one_recording_failing_metadata_does_not_block_the_others(
        self, dataset_mock, details
    ):
        """A metadata failure costs that recording only."""
        prefetcher = _prefetcher(
            dataset_mock, details, download_videos=True, decode_workers=2
        )
        original_metadata = VideoPrefetcher._get_synced_data
        calls = {"n": 0}

        async def fail_first(self, session, recording_id):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return await original_metadata(self, session, recording_id)

        with patch.object(VideoPrefetcher, "_get_synced_data", fail_first):
            episodes = prefetcher.run()

        assert len(episodes) == len(prefetcher.recordings) - 1
        # The surviving recordings still have their frames published.
        published = list(dataset_mock.cache_dir.rglob("0.png"))
        assert published
        assert list(dataset_mock.cache_dir.rglob("*.recording.lock")) == []
