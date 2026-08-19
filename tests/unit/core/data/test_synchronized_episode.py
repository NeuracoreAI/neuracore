"""Tests for SynchronizedRecording class."""

import re
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import requests
from neuracore_types import (
    CameraData,
    DataType,
    JointData,
    SynchronizationDetails,
    SynchronizedEpisode,
    SynchronizedPoint,
    SynchronizeRecordingRequest,
)
from PIL import Image

from neuracore.core.const import API_URL
from neuracore.core.data.synced_recording import (
    SYNCED_EPISODE_DOWNLOAD_TIMEOUT_S,
    SynchronizedRecording,
)
from neuracore.core.exceptions import SynchronizationError
from neuracore.core.utils.http_session import DEFAULT_TIMEOUT, thread_local_session

MODULE = "neuracore.core.data.synced_recording"

SIGNED_URL = "https://storage.example/synced/rec1.json?X-Goog-Signature=secret"


def start_json(
    recording_id: str = "rec1", synchronized_recording_id: str = "synced-rec1"
) -> dict:
    """Build the payload the API answers a synchronization start with.

    Args:
        recording_id: Recording the synchronization was requested for.
        synchronized_recording_id: Artifact the synchronization produces.

    Returns:
        The identifiers and the PENDING status, and nothing else.
    """
    return {
        "recording_id": recording_id,
        "synchronized_recording_id": synchronized_recording_id,
        "status": "PENDING",
    }


def job_json(
    status: str = "PENDING",
    download_url: str | None = None,
    error: str | None = None,
    recording_id: str = "rec1",
    synchronized_recording_id: str = "synced-rec1",
) -> dict:
    """Build a synchronization progress payload.

    Args:
        status: Lifecycle stage the poll reports.
        download_url: Signed URL a READY poll carries.
        error: Failure message a FAILED poll carries.
        recording_id: Recording the synchronization was requested for.
        synchronized_recording_id: Artifact the synchronization produces.

    Returns:
        The progress payload.
    """
    return {
        "recording_id": recording_id,
        "synchronized_recording_id": synchronized_recording_id,
        "status": status,
        "download_url": download_url,
        "error": error,
    }


def build_recording(dataset, recording_id: str = "rec1") -> SynchronizedRecording:
    """Construct a recording, which retrieves its synchronized episode.

    Args:
        dataset: Parent dataset supplying the org and the cache directory.
        recording_id: Recording to synchronize.

    Returns:
        The constructed recording.
    """
    return SynchronizedRecording(
        dataset=dataset,
        recording_id=recording_id,
        recording_name="recording1",
        robot_id="robot1",
        instance=1,
        synchronization_details=SynchronizationDetails(
            frequency=30,
            cross_embodiment_union=None,
        ),
    )


@pytest.mark.usefixtures("mock_login")
class TestSynchronizedRecording:
    """Tests for the SynchronizedRecording class."""

    @pytest.fixture
    def synced_recording(
        self, dataset_mock, mock_data_requests
    ) -> SynchronizedRecording:
        """Create a SynchronizedRecording instance for testing."""
        return build_recording(dataset_mock)

    def test_init(self, synced_recording: SynchronizedRecording, dataset_mock):
        """Test SynchronizedRecording initialization."""
        assert synced_recording.dataset == dataset_mock
        assert synced_recording.id == "rec1"
        assert synced_recording.frequency == 30
        assert synced_recording.robot_id == "robot1"
        assert synced_recording.instance == 1
        assert synced_recording.cross_embodiment_union is None
        assert synced_recording._iter_idx == 0

    def test_init_with_data_types(self, dataset_mock, mock_data_requests):
        """Test initialization with specific data types."""
        from neuracore_types import DataType

        cross_embodiment_union = {
            "robot1": {
                DataType.RGB_IMAGES: [],
                DataType.DEPTH_IMAGES: [],
            }
        }
        synced = SynchronizedRecording(
            dataset=dataset_mock,
            recording_id="rec1",
            recording_name="recording1",
            robot_id="robot1",
            instance=1,
            synchronization_details=SynchronizationDetails(
                frequency=30,
                cross_embodiment_union=cross_embodiment_union,
            ),
        )

        assert synced.cross_embodiment_union == cross_embodiment_union

    def test_get_synced_data(
        self, synced_recording: SynchronizedRecording, synced_data
    ):
        """Test that _get_synced_data correctly retrieves synchronized data."""
        result = synced_recording._episode_synced

        assert result.robot_id == synced_data.robot_id
        assert len(result.observations) == len(synced_data.observations)
        assert result.start_time == synced_data.start_time
        assert result.end_time == synced_data.end_time

    def test_construction_initializes_episode_state(
        self, synced_recording: SynchronizedRecording, synced_data
    ):
        """Construction still populates the episode state from the download."""
        assert synced_recording._episode_synced is not None
        assert synced_recording._episode_length == len(synced_data.observations)
        assert synced_recording.start_time == synced_data.start_time
        assert synced_recording.end_time == synced_data.end_time

    def test_episode_retrieval_makes_three_requests(
        self,
        synced_recording: SynchronizedRecording,
        mock_data_requests,
        mocked_org_id,
        synced_recording_id,
    ):
        """Retrieval starts the synchronization, polls it, then downloads."""
        requested = [
            (request.method, request.url.split("?")[0])
            for request in mock_data_requests.request_history
            if "synchronize" in request.path or "storage.example" in request.hostname
        ]

        base = f"{API_URL}/org/{mocked_org_id}/synchronize"
        assert requested == [
            ("POST", f"{base}/synchronize-recording"),
            (
                "GET",
                f"{base}/synchronized-recording-progress/{synced_recording_id}",
            ),
            ("GET", "https://storage.example/synced-episodes/rec1.json"),
        ]

    def test_failed_download_yields_no_recording(
        self, dataset_mock, mock_data_requests, synced_episode_download_url
    ):
        """A failed artifact download must not produce a partial recording."""
        mock_data_requests.get(
            synced_episode_download_url, status_code=403, text="Forbidden"
        )

        with pytest.raises(SynchronizationError, match="rec1"):
            build_recording(dataset_mock)

    def test_every_request_uses_a_retrying_session(
        self, dataset_mock, mock_data_requests
    ) -> None:
        """Both API calls and the download go through retrying sessions."""
        requested_policies = []

        def record(**kwargs):
            requested_policies.append(kwargs)
            return thread_local_session(**kwargs)

        with patch(f"{MODULE}.thread_local_session", side_effect=record):
            build_recording(dataset_mock)

        assert requested_policies == [
            {"retry_transient": True},  # start
            {"retry_transient": True},  # poll
            # The download is an idempotent GET, so read timeouts are retried too.
            {"retry_transient": True, "retry_read_timeout": True},
        ]

    def test_len(self, synced_recording):
        """Test __len__ returns correct number of frames."""
        assert len(synced_recording) == 2

    def test_iter_reset(self, synced_recording):
        """Test that __iter__ resets the iteration index."""
        synced_recording._iter_idx = 5
        result = iter(synced_recording)

        assert result is synced_recording
        assert synced_recording._iter_idx == 0

    def test_getitem_single_index(
        self, synced_recording: SynchronizedRecording, mock_wget_download
    ):
        """Test accessing a single frame by index."""
        sync_point = synced_recording[0]

        assert isinstance(sync_point, SynchronizedPoint)
        assert DataType.JOINT_POSITIONS in sync_point.data
        assert sync_point.timestamp == 0.0
        joint_data = cast(
            JointData, list(sync_point.data[DataType.JOINT_POSITIONS].values())[0]
        )
        assert joint_data.value == 0.5

    def test_getitem_negative_index(
        self, synced_recording: SynchronizedRecording, mock_wget_download
    ):
        """Test accessing frames with negative indices."""
        sync_point = synced_recording[-1]

        assert isinstance(sync_point, SynchronizedPoint)
        assert sync_point.timestamp == 1.0

    def test_getitem_out_of_range(self, synced_recording):
        """Test that out of range index raises IndexError."""
        with pytest.raises(IndexError, match="Index out of range"):
            _ = synced_recording[10]

    def test_getitem_negative_out_of_range(self, synced_recording):
        """Test that negative out of range index raises IndexError."""
        with pytest.raises(IndexError, match="Index out of range"):
            _ = synced_recording[-10]

    def test_getitem_slice(
        self, synced_recording: SynchronizedRecording, mock_wget_download
    ):
        """Test slicing synchronized recording."""
        frames = synced_recording[0:2]

        assert isinstance(frames, list)
        assert len(frames) == 2
        assert all(isinstance(f, SynchronizedPoint) for f in frames)

    def test_getitem_slice_with_step(
        self,
        dataset_mock,
        mock_data_requests,
        mock_wget_download,
        synced_data_multiple_frames,
    ):
        """Test slicing with step parameter."""
        # Mock the API to serve an artifact with more frames
        download_url = "https://storage.example/synced-episodes/multi-frame.json"
        base = f"{API_URL}/org/{dataset_mock.org_id}/synchronize"
        mock_data_requests.post(
            re.compile(f"{base}/synchronize-recording"),
            json=start_json(synchronized_recording_id="synced-multi-frame"),
            status_code=200,
        )
        mock_data_requests.get(
            f"{base}/synchronized-recording-progress/synced-multi-frame",
            json=job_json(
                "READY",
                download_url=download_url,
                synchronized_recording_id="synced-multi-frame",
            ),
            status_code=200,
        )
        mock_data_requests.get(
            download_url,
            json=synced_data_multiple_frames.model_dump(mode="json"),
            status_code=200,
        )

        synced = build_recording(dataset_mock)

        frames = synced[0:5:2]

        assert len(frames) == 3
        assert frames[0].timestamp == 0.0
        assert frames[1].timestamp == 2.0
        assert frames[2].timestamp == 4.0

    def test_iteration(
        self, synced_recording: SynchronizedRecording, mock_wget_download
    ):
        """Test iterating through synchronized recording."""
        frames = list(synced_recording)

        assert len(frames) == 2
        assert all(isinstance(f, SynchronizedPoint) for f in frames)
        assert frames[0].timestamp == 0.0
        assert frames[1].timestamp == 1.0

    def test_iteration_multiple_times(
        self, synced_recording: SynchronizedRecording, mock_wget_download
    ):
        """Test that the recording can be iterated multiple times."""
        frames1 = list(synced_recording)
        frames2 = list(synced_recording)

        assert len(frames1) == len(frames2)
        assert frames1[0].timestamp == frames2[0].timestamp

    def test_next_stop_iteration(self, synced_recording):
        """Test that __next__ raises StopIteration when exhausted."""
        iter(synced_recording)

        # Exhaust the iterator
        synced_recording._iter_idx = len(synced_recording._episode_synced.observations)

        with pytest.raises(StopIteration):
            next(synced_recording)

    def test_video_caching(
        self, synced_recording: SynchronizedRecording, mock_wget_download, tmp_path
    ):
        """Test that videos are cached correctly."""
        # First access should download and cache
        synced_recording[0]

        # Check that cache directory was created
        cache_path = synced_recording.cache_dir / f"{synced_recording.id}"
        assert cache_path.exists()

    def test_video_cache_reuse(
        self, dataset_mock, mock_data_requests, mock_wget_download, tmp_path
    ):
        """Test that cached videos are reused on subsequent access."""
        # Create cache directory and add a fake cached frame
        cache_path = (
            dataset_mock.cache_dir / "rec1" / DataType.RGB_IMAGES.value / "cam1"
        )
        cache_path.mkdir(parents=True, exist_ok=True)

        # Create a fake cached image
        fake_image = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 128)
        fake_image.save(cache_path / "0.png")

        synced = SynchronizedRecording(
            dataset=dataset_mock,
            recording_id="rec1",
            recording_name="recording1",
            robot_id="robot1",
            instance=1,
            synchronization_details=SynchronizationDetails(
                frequency=30,
                cross_embodiment_union=None,
            ),
        )

        sync_point = cast(SynchronizedPoint, synced[0])

        # Should have loaded from cache
        assert DataType.RGB_IMAGES in sync_point.data
        assert "cam1" in sync_point.data[DataType.RGB_IMAGES]

    def test_prefetch_videos_skip_if_cached(
        self, dataset_mock, mock_data_requests, mock_wget_download
    ):
        """Test that prefetch_videos parameter triggers video download on init."""
        synced = SynchronizedRecording(
            dataset=dataset_mock,
            recording_id="rec1",
            recording_name="recording1",
            robot_id="robot1",
            instance=1,
            synchronization_details=SynchronizationDetails(
                frequency=30,
                cross_embodiment_union=None,
            ),
            prefetch_videos=True,
        )

        # Cache directory should exist after prefetch
        cache_path = synced.cache_dir / f"{synced.id}"
        assert cache_path.exists()

        # Mock wget to track if it's called
        with patch("wget.download") as mock_download:
            SynchronizedRecording(
                dataset=dataset_mock,
                recording_id="rec1",
                recording_name="recording1",
                robot_id="robot1",
                instance=1,
                synchronization_details=SynchronizationDetails(
                    frequency=30,
                    cross_embodiment_union=None,
                ),
                prefetch_videos=True,
            )

            # wget.download should not be called since cache exists
            mock_download.assert_not_called()

    def test_depth_image_processing(
        self, synced_recording: SynchronizedRecording, mock_wget_download
    ):
        """Test that depth images are processed correctly."""
        sync_point = cast(SynchronizedPoint, synced_recording[0])

        for cam_id, cam_data in sync_point.data[DataType.DEPTH_IMAGES].items():
            cam_data = cast(CameraData, cam_data)
            assert cam_data.frame is not None
            assert isinstance(cam_data.frame, Image.Image)

    def test_rgb_to_depth_storage_called_when_retrieving_frame(
        self,
        dataset_mock,
        mock_data_requests,
        mock_wget_download,
        tmp_path,
    ):
        """Test that rgb_to_depth_storage is called when retrieving a frame
        with depth images."""
        rgb_cache = dataset_mock.cache_dir / "rec1" / DataType.RGB_IMAGES.value / "cam1"
        depth_cache = (
            dataset_mock.cache_dir / "rec1" / DataType.DEPTH_IMAGES.value / "cam2"
        )
        rgb_cache.mkdir(parents=True, exist_ok=True)
        depth_cache.mkdir(parents=True, exist_ok=True)
        fake_image = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 128)
        fake_image.save(rgb_cache / "0.png")
        fake_image.save(depth_cache / "0.png")

        synced = SynchronizedRecording(
            dataset=dataset_mock,
            recording_id="rec1",
            recording_name="recording1",
            robot_id="robot1",
            instance=1,
            synchronization_details=SynchronizationDetails(
                frequency=30,
                cross_embodiment_union=None,
            ),
        )

        with patch(
            "neuracore.core.data.synced_recording.rgb_to_depth_storage"
        ) as mock_rgb_to_depth_storage:
            mock_rgb_to_depth_storage.return_value = np.zeros(
                (224, 224), dtype=np.uint8
            )
            _ = synced[0]
            mock_rgb_to_depth_storage.assert_called()

    def test_camera_data_copy_independence(
        self, synced_recording: SynchronizedRecording, mock_wget_download
    ):
        """Test that returned sync points are independent copies."""
        sync_point1 = cast(SynchronizedPoint, synced_recording[0])
        sync_point2 = cast(SynchronizedPoint, synced_recording[0])

        # Should be different objects
        assert sync_point1 is not sync_point2

        # Modifying one shouldn't affect the other
        jp1 = cast(
            JointData, list(sync_point1.data[DataType.JOINT_POSITIONS].values())[0]
        )
        original_value = jp1.value
        jp1.value = 999.0

        jp2 = cast(
            JointData, list(sync_point2.data[DataType.JOINT_POSITIONS].values())[0]
        )
        assert jp2.value == original_value

    def test_cache_manager_initialization(self, synced_recording):
        """Test that cache manager is initialized correctly."""
        assert synced_recording.cache_manager is not None
        assert hasattr(synced_recording.cache_manager, "ensure_space_available")

    def test_suppress_wget_progress(self, synced_recording):
        """Test that wget progress is suppressed by default."""
        assert synced_recording._suppress_wget_progress is True

    def test_different_frequencies_are_stored_on_instances(
        self, dataset_mock, mock_data_requests, mock_wget_download
    ):
        """Test that different instances can retain different frequencies."""
        synced_30 = SynchronizedRecording(
            dataset=dataset_mock,
            recording_id="rec1",
            recording_name="recording1",
            robot_id="robot1",
            instance=1,
            synchronization_details=SynchronizationDetails(
                frequency=30,
                cross_embodiment_union=None,
            ),
        )

        synced_60 = SynchronizedRecording(
            dataset=dataset_mock,
            recording_id="rec1",
            recording_name="recording1",
            robot_id="robot1",
            instance=1,
            synchronization_details=SynchronizationDetails(
                frequency=60,
                cross_embodiment_union=None,
            ),
        )

        assert synced_30.frequency == 30
        assert synced_60.frequency == 60
        assert synced_30.frequency != synced_60.frequency

    def test_create_decoding_lock_creates_file(self, synced_recording, tmp_path):
        """_create_decoding_lock should create lock file when none exists."""
        lock_file = tmp_path / ".decoding.lock"
        synced_recording._create_decoding_lock(lock_file, "cam1")

        assert lock_file.exists()

    def test_create_decoding_lock_raises_when_exists(self, synced_recording, tmp_path):
        """_create_decoding_lock should raise when lock file already exists."""
        lock_file = tmp_path / ".decoding.lock"
        lock_file.touch()

        with pytest.raises(
            RuntimeError,
            match="Another process is already decoding video for camera cam1",
        ):
            synced_recording._create_decoding_lock(lock_file, "cam1")

    def test_delete_decoding_lock_removes_file(self, synced_recording, tmp_path):
        """_delete_decoding_lock should remove lock file if present."""
        lock_file = tmp_path / ".decoding.lock"
        lock_file.touch()

        synced_recording._delete_decoding_lock(lock_file)

        assert not lock_file.exists()

    def test_decode_failure_does_not_publish_partial_cache(
        self, synced_recording, mock_wget_download, dataset_mock
    ):
        """A decode that fails mid-way must not leave a partial frames dir.

        Readers treat an existing frames dir as complete, so a partially written
        dir would later surface as a missing-frame FileNotFoundError. The frames
        dir must appear only once decoding has fully succeeded.
        """
        final_dir = dataset_mock.cache_dir / "rec1" / DataType.RGB_IMAGES.value / "cam1"

        def partial_then_fail(video_location, frames_dir):
            # Simulate the decoder writing some frames, then crashing.
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                frames_dir / "0.png"
            )
            raise RuntimeError("decode crashed")

        with patch.object(
            synced_recording, "_decode_video", side_effect=partial_then_fail
        ):
            with pytest.raises(RuntimeError, match="decode crashed"):
                synced_recording._download_video_and_cache_frames_to_disk(
                    DataType.RGB_IMAGES, "cam1", final_dir
                )

        assert not final_dir.exists()  # nothing published on failure

    def test_successful_decode_publishes_complete_dir_without_leftovers(
        self, synced_recording, mock_wget_download, dataset_mock
    ):
        """A successful decode publishes the complete frames dir and cleans up.

        No decoding lock and no staging temp directory should remain afterwards.
        """
        final_dir = dataset_mock.cache_dir / "rec1" / DataType.RGB_IMAGES.value / "cam1"

        synced_recording._download_video_and_cache_frames_to_disk(
            DataType.RGB_IMAGES, "cam1", final_dir
        )

        assert final_dir.exists()
        assert len(list(final_dir.glob("*.png"))) == 10  # mock video has 10 frames
        # No decoding lock and no staging temp dirs left behind.
        assert not any(final_dir.parent.rglob("*recording.lock"))
        leftovers = [p for p in final_dir.parent.iterdir() if p != final_dir]
        assert leftovers == []

    def test_decode_video_uses_resolved_frame_sync_arg(
        self, synced_recording, tmp_path
    ):
        """The ffmpeg decode command must use the probed flag, not a hard-coded
        `-vsync`: that spelling was removed in ffmpeg 8, so hard-coding it
        breaks decoding on any ffmpeg 8+ install.
        """
        video_location = tmp_path / "video.mp4"
        video_location.touch()
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        with (
            patch(f"{MODULE}._FFMPEG_AVAILABLE", True),
            patch(f"{MODULE}._resolve_frame_sync_arg", return_value="-fps_mode"),
            patch(f"{MODULE}.subprocess.run") as mock_run,
        ):
            synced_recording._decode_video(video_location, frames_dir)

        ffmpeg_args = mock_run.call_args[0][0]
        assert "-fps_mode" in ffmpeg_args
        flag_index = ffmpeg_args.index("-fps_mode")
        assert ffmpeg_args[flag_index + 1] == "passthrough"
        assert "-vsync" not in ffmpeg_args


class TestResolveFrameSyncArg:
    """Tests for _resolve_frame_sync_arg's ffmpeg passthrough flag probe."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        """Clear the module-level memo so each test probes fresh."""
        import neuracore.core.data.synced_recording as synced_recording_module

        synced_recording_module._FFMPEG_FRAME_SYNC_ARG = None
        yield
        synced_recording_module._FFMPEG_FRAME_SYNC_ARG = None

    def test_returns_fps_mode_when_ffmpeg_accepts_it(self):
        """ffmpeg >= 5.1 accepts -fps_mode, so it should be preferred."""
        from neuracore.core.data.synced_recording import _resolve_frame_sync_arg

        with patch(f"{MODULE}.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0, stderr=b"")

            assert _resolve_frame_sync_arg() == "-fps_mode"

    def test_returns_vsync_when_ffmpeg_rejects_fps_mode(self):
        """ffmpeg < 5.1 rejects -fps_mode as unrecognised, so fall back to -vsync."""
        from neuracore.core.data.synced_recording import _resolve_frame_sync_arg

        with patch(f"{MODULE}.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(
                returncode=1, stderr=b"Unrecognized option 'fps_mode'."
            )

            assert _resolve_frame_sync_arg() == "-vsync"

    def test_returns_vsync_when_ffmpeg_binary_is_missing(self):
        """No ffmpeg on PATH resolves to the legacy spelling, matching the
        pre-existing _FFMPEG_AVAILABLE probe's fall-back behaviour.
        """
        from neuracore.core.data.synced_recording import _resolve_frame_sync_arg

        with patch(f"{MODULE}.subprocess.run", side_effect=FileNotFoundError):
            assert _resolve_frame_sync_arg() == "-vsync"

    def test_probes_ffmpeg_only_once(self):
        """The result is memoized: a second call must not spawn ffmpeg again."""
        from neuracore.core.data.synced_recording import _resolve_frame_sync_arg

        with patch(f"{MODULE}.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0, stderr=b"")

            assert _resolve_frame_sync_arg() == "-fps_mode"
            assert _resolve_frame_sync_arg() == "-fps_mode"

        mock_run.assert_called_once()


@pytest.mark.usefixtures("mock_login")
class TestSyncedEpisodeRetrieval:
    """Tests for the start, poll and download flow behind one episode.

    Synchronization is asynchronous: starting it only ever reports tracking
    metadata, the terminal state arrives from the progress endpoint, and the
    episode itself is downloaded straight from object storage.
    """

    @pytest.fixture(autouse=True)
    def no_poll_delay(self, monkeypatch):
        """Remove the poll delay so the loop runs at full speed."""
        monkeypatch.setattr(f"{MODULE}.SYNCED_RECORDING_POLL_INTERVAL_S", 0)

    @pytest.fixture
    def endpoints(self, mocked_org_id):
        """The three requests one episode retrieval makes."""
        base = f"{API_URL}/org/{mocked_org_id}/synchronize"
        return {
            "start": f"{base}/synchronize-recording",
            "progress": f"{base}/synchronized-recording-progress/synced-rec1",
            "download": SIGNED_URL,
        }

    def test_start_response_is_only_tracking_metadata(
        self, mock_data_requests, dataset_mock, endpoints, synced_data
    ):
        """Starting identifies the artifact; the episode arrives from storage."""
        start = mock_data_requests.post(endpoints["start"], json=start_json())
        progress = mock_data_requests.get(
            endpoints["progress"], json=job_json("READY", download_url=SIGNED_URL)
        )
        download = mock_data_requests.get(
            endpoints["download"], json=synced_data.model_dump(mode="json")
        )

        recording = build_recording(dataset_mock)

        # The synchronization request body is unchanged by the async contract.
        assert start.call_count == 1
        assert start.last_request.json() == (
            SynchronizeRecordingRequest(
                recording_id="rec1",
                synchronization_details=recording.synchronization_details,
            ).model_dump(mode="json")
        )
        # Both identifiers the start reported are used to poll the artifact.
        assert progress.call_count == 1
        assert progress.last_request.qs == {"recording_id": ["rec1"]}
        assert download.call_count == 1
        assert isinstance(recording._episode_synced, SynchronizedEpisode)
        assert recording._episode_synced.robot_id == synced_data.robot_id
        assert len(recording) == len(synced_data.observations)

    def test_pending_polls_until_ready(
        self, mock_data_requests, dataset_mock, endpoints, synced_data
    ):
        """Polling continues while the artifact is still being built."""
        mock_data_requests.post(endpoints["start"], json=start_json())
        progress = mock_data_requests.get(
            endpoints["progress"],
            [
                {"json": job_json("PENDING")},
                {"json": job_json("PENDING")},
                {"json": job_json("READY", download_url=SIGNED_URL)},
            ],
        )
        download = mock_data_requests.get(
            endpoints["download"], json=synced_data.model_dump(mode="json")
        )

        recording = build_recording(dataset_mock)

        assert progress.call_count == 3
        assert download.call_count == 1
        assert len(recording) == len(synced_data.observations)

    def test_cached_artifact_is_ready_on_the_first_poll(
        self, mock_data_requests, dataset_mock, endpoints, synced_data
    ):
        """A cached artifact costs one poll and no waiting at all."""
        mock_data_requests.post(endpoints["start"], json=start_json())
        progress = mock_data_requests.get(
            endpoints["progress"], json=job_json("READY", download_url=SIGNED_URL)
        )
        mock_data_requests.get(
            endpoints["download"], json=synced_data.model_dump(mode="json")
        )

        with patch(f"{MODULE}.time.sleep") as sleep:
            build_recording(dataset_mock)

        assert progress.call_count == 1
        sleep.assert_not_called()

    def test_failed_synchronization_raises_with_the_server_error(
        self, mock_data_requests, dataset_mock, endpoints
    ):
        """A FAILED poll surfaces the server's reason and downloads nothing."""
        mock_data_requests.post(endpoints["start"], json=start_json())
        mock_data_requests.get(
            endpoints["progress"],
            json=job_json("FAILED", error="joint stream missing"),
        )
        download = mock_data_requests.get(endpoints["download"], json={})

        with pytest.raises(SynchronizationError, match="joint stream missing") as exc:
            build_recording(dataset_mock)

        assert "rec1" in str(exc.value)
        assert download.call_count == 0

    def test_failed_without_a_reason_still_names_the_recording(
        self, mock_data_requests, dataset_mock, endpoints
    ):
        """A FAILED poll carrying no message still identifies the recording."""
        mock_data_requests.post(endpoints["start"], json=start_json())
        mock_data_requests.get(endpoints["progress"], json=job_json("FAILED"))

        with pytest.raises(SynchronizationError, match="rec1") as exc:
            build_recording(dataset_mock)

        assert "no reason given" in str(exc.value)

    def test_ready_without_a_download_url_is_invalid(
        self, mock_data_requests, dataset_mock, endpoints
    ):
        """READY without a URL is a broken server response, not a retry."""
        mock_data_requests.post(endpoints["start"], json=start_json())
        progress = mock_data_requests.get(
            endpoints["progress"], json=job_json("READY", download_url=None)
        )
        download = mock_data_requests.get(endpoints["download"], json={})

        with pytest.raises(
            SynchronizationError, match="READY without a download URL"
        ) as exc:
            build_recording(dataset_mock)

        assert "rec1" in str(exc.value)
        assert progress.call_count == 1
        assert download.call_count == 0

    def test_polling_gives_up_at_the_deadline(
        self, mock_data_requests, dataset_mock, endpoints, monkeypatch
    ):
        """A synchronization that never finishes fails with a timeout."""
        monkeypatch.setattr(f"{MODULE}.SYNCED_RECORDING_TIMEOUT_S", 0)
        mock_data_requests.post(endpoints["start"], json=start_json())
        progress = mock_data_requests.get(endpoints["progress"], json=job_json())
        download = mock_data_requests.get(endpoints["download"], json={})

        with pytest.raises(SynchronizationError, match="Timed out") as exc:
            build_recording(dataset_mock)

        assert "rec1" in str(exc.value)
        assert "PENDING" in str(exc.value)
        assert progress.call_count == 1
        assert download.call_count == 0

    def test_start_response_is_never_parsed_as_an_episode(
        self, mock_data_requests, dataset_mock, endpoints, synced_data
    ):
        """An inline episode body is rejected instead of being used as one."""
        mock_data_requests.post(
            endpoints["start"], json=synced_data.model_dump(mode="json")
        )
        progress = mock_data_requests.get(
            endpoints["progress"], json=job_json("READY", download_url=SIGNED_URL)
        )
        download = mock_data_requests.get(endpoints["download"], json={})

        with pytest.raises(SynchronizationError, match="start response") as exc:
            build_recording(dataset_mock)

        assert "not tracking metadata" in str(exc.value)
        assert progress.call_count == 0
        assert download.call_count == 0

    def test_malformed_progress_response_is_rejected(
        self, mock_data_requests, dataset_mock, endpoints
    ):
        """A progress body that is not tracking metadata fails clearly."""
        mock_data_requests.post(endpoints["start"], json=start_json())
        mock_data_requests.get(endpoints["progress"], text="not json at all")
        download = mock_data_requests.get(endpoints["download"], json={})

        with pytest.raises(SynchronizationError, match="progress response") as exc:
            build_recording(dataset_mock)

        assert "rec1" in str(exc.value)
        assert download.call_count == 0

    def test_start_failure_is_surfaced_before_any_polling(
        self, mock_data_requests, dataset_mock, endpoints
    ):
        """A failed start propagates and nothing downstream is attempted."""
        mock_data_requests.post(endpoints["start"], status_code=500, text="boom")
        progress = mock_data_requests.get(endpoints["progress"], json=job_json())
        download = mock_data_requests.get(endpoints["download"], json={})

        with pytest.raises(requests.HTTPError):
            build_recording(dataset_mock)

        assert progress.call_count == 0
        assert download.call_count == 0

    def test_auth_headers_are_sent_only_to_the_api(
        self, mock_data_requests, dataset_mock, endpoints, synced_data
    ):
        """Neuracore credentials reach the API but never object storage."""
        start = mock_data_requests.post(endpoints["start"], json=start_json())
        progress = mock_data_requests.get(
            endpoints["progress"], json=job_json("READY", download_url=SIGNED_URL)
        )
        download = mock_data_requests.get(
            endpoints["download"], json=synced_data.model_dump(mode="json")
        )

        build_recording(dataset_mock)

        assert start.last_request.headers["Authorization"]
        assert progress.last_request.headers["Authorization"]
        # The URL is already signed; forwarding Neuracore credentials to object
        # storage would have the request rejected.
        assert "Authorization" not in download.last_request.headers

    def test_download_uses_its_own_read_timeout(
        self, mock_data_requests, dataset_mock, endpoints, synced_data
    ):
        """The object download gets a longer read budget than API calls."""
        mock_data_requests.post(endpoints["start"], json=start_json())
        mock_data_requests.get(
            endpoints["progress"], json=job_json("READY", download_url=SIGNED_URL)
        )
        download = mock_data_requests.get(
            endpoints["download"], json=synced_data.model_dump(mode="json")
        )

        build_recording(dataset_mock)

        assert download.last_request.timeout == SYNCED_EPISODE_DOWNLOAD_TIMEOUT_S
        assert SYNCED_EPISODE_DOWNLOAD_TIMEOUT_S[1] > DEFAULT_TIMEOUT[1]

    def test_download_failure_reports_recording_and_stage(
        self, mock_data_requests, dataset_mock, endpoints
    ):
        """A failed download names the recording without leaking the URL."""
        mock_data_requests.post(endpoints["start"], json=start_json())
        mock_data_requests.get(
            endpoints["progress"], json=job_json("READY", download_url=SIGNED_URL)
        )
        mock_data_requests.get(endpoints["download"], status_code=403, text="Forbidden")

        with pytest.raises(SynchronizationError) as exc:
            build_recording(dataset_mock)

        message = str(exc.value)
        assert "download" in message
        assert "rec1" in message
        assert "HTTP 403" in message
        # The signed URL carries temporary credentials: neither it nor the
        # requests error that embeds it may reach the caller.
        assert "X-Goog-Signature" not in message
        assert exc.value.__cause__ is None

    def test_download_transport_failure_hides_the_signed_url(
        self, mock_data_requests, dataset_mock, endpoints
    ):
        """A transport failure names the recording without exposing the URL."""
        mock_data_requests.post(endpoints["start"], json=start_json())
        mock_data_requests.get(
            endpoints["progress"], json=job_json("READY", download_url=SIGNED_URL)
        )
        mock_data_requests.get(
            endpoints["download"],
            exc=requests.ConnectionError(
                f"Max retries exceeded with url: {SIGNED_URL}"
            ),
        )

        with pytest.raises(SynchronizationError) as exc:
            build_recording(dataset_mock)

        message = str(exc.value)
        assert "rec1" in message
        assert "ConnectionError" in message
        assert "X-Goog-Signature" not in message

    @pytest.mark.parametrize(
        "content",
        [
            b"",
            b"{not json",
            b"\xff\xfe\x00not-utf8",
            b'{"observations": "not-a-list"}',
            b"[1, 2, 3]",
        ],
        ids=["empty", "malformed", "invalid-utf8", "wrong-types", "wrong-shape"],
    )
    def test_invalid_downloaded_artifact_fails_validation(
        self, mock_data_requests, dataset_mock, endpoints, content
    ):
        """A malformed or structurally invalid artifact fails clearly."""
        mock_data_requests.post(endpoints["start"], json=start_json())
        mock_data_requests.get(
            endpoints["progress"], json=job_json("READY", download_url=SIGNED_URL)
        )
        mock_data_requests.get(endpoints["download"], content=content)

        with pytest.raises(SynchronizationError, match="rec1") as exc:
            build_recording(dataset_mock)

        assert "valid synchronized episode" in str(exc.value)
        assert "validation errors" in str(exc.value)
