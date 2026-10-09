"""Tests for the frame cache's lock protocol and decoding step."""

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import imagecodecs
import numpy as np
import pytest
from neuracore_types import DataType

from neuracore.core.data.frame_cache import (
    DEPTH_FRAMES_FILENAME,
    STALE_LOCK_TIMEOUT_S,
    acquire_decoding_lock,
    cache_camera_frames,
    cache_depth_frames,
    check_stale_lock_file,
    clear_stale_lock,
    create_decoding_lock,
    decode_rgb_video,
    delete_decoding_lock,
    get_camera_data_filenames,
    lock_file_for,
    read_depth_frame,
    wait_for_lock_release,
)

MODULE = "neuracore.core.data.frame_cache"


def _depth_frames() -> np.ndarray:
    rng = np.random.default_rng(7)
    frames = rng.integers(1, 65535, size=(3, 6, 8), dtype=np.uint16)
    frames[:, 0, :] = 0
    frames[1, 2:4, 3:5] = 65535
    return frames


class TestCameraDataFilenames:
    """Which candidate filenames each camera type offers."""

    def test_rgb_prefers_lossless_then_lossy(self):
        """RGB falls back to the lossy encode when no lossless one exists."""
        assert get_camera_data_filenames(DataType.RGB_IMAGES) == (
            "lossless.mp4",
            "lossy.mp4",
        )

    def test_depth_is_lossless_only(self):
        """Depth reads only the frames file, never lossy."""
        assert get_camera_data_filenames(DataType.DEPTH_IMAGES) == ("lossless.bin",)

    def test_other_data_types_raise(self):
        """A non-camera data type has no camera data files."""
        with pytest.raises(ValueError, match="No camera data files"):
            get_camera_data_filenames(DataType.JOINT_POSITIONS)


class TestLockFileNaming:
    """The lock must sit beside the frames directory, never inside it."""

    def test_lock_is_a_sibling_of_the_frames_dir(self, tmp_path):
        """Publishing with os.replace requires the lock to be outside the dir."""
        frames_dir = tmp_path / "rgb_images" / "cam1"

        lock_file = lock_file_for(frames_dir)

        assert lock_file.parent == frames_dir.parent
        assert lock_file.name == "cam1.recording.lock"


class TestDecodingLock:
    """Taking, releasing and reclaiming a decoding lock."""

    def test_create_lock_is_exclusive(self, tmp_path):
        """Only the first caller takes the lock."""
        lock_file = tmp_path / "cam1.recording.lock"

        assert create_decoding_lock(lock_file) is True
        assert create_decoding_lock(lock_file) is False

    def test_create_lock_makes_parent(self, tmp_path):
        """The lock can be taken before its directory exists."""
        lock_file = tmp_path / "nested" / "deeper" / "cam1.recording.lock"

        assert create_decoding_lock(lock_file) is True
        assert lock_file.exists()

    def test_acquire_lock_creates_file(self, tmp_path):
        """Acquiring succeeds silently when no lock is held."""
        lock_file = tmp_path / "cam1.recording.lock"

        acquire_decoding_lock(lock_file, DataType.RGB_IMAGES, "cam1")

        assert lock_file.exists()

    @pytest.mark.parametrize(
        ("data_type", "message"),
        [
            (
                DataType.RGB_IMAGES,
                "already decoding and caching RGB frames for camera cam1",
            ),
            (
                DataType.POINT_CLOUDS,
                "already decoding and caching point cloud frames for sensor cam1",
            ),
            (DataType.DEPTH_IMAGES, "already caching depth frames for camera cam1"),
        ],
    )
    def test_acquire_lock_raises_when_held(self, tmp_path, data_type, message):
        """Acquiring a held lock is an error naming the sensor and its work."""
        lock_file = tmp_path / "cam1.recording.lock"
        lock_file.touch()

        with pytest.raises(RuntimeError, match=message):
            acquire_decoding_lock(lock_file, data_type, "cam1")

    def test_acquire_lock_rejects_other_data_types(self, tmp_path):
        """A data type without cached frames has no decoding lock."""
        lock_file = tmp_path / "cam1.recording.lock"

        with pytest.raises(ValueError, match="No decoding lock"):
            acquire_decoding_lock(lock_file, DataType.JOINT_POSITIONS, "cam1")

        assert not lock_file.exists()

    def test_delete_lock_removes_file(self, tmp_path):
        """Releasing removes the lock."""
        lock_file = tmp_path / "cam1.recording.lock"
        lock_file.touch()

        delete_decoding_lock(lock_file)

        assert not lock_file.exists()

    def test_delete_lock_tolerates_missing_file(self, tmp_path):
        """Releasing a lock that is already gone is not an error."""
        delete_decoding_lock(tmp_path / "absent.recording.lock")


class TestStaleLocks:
    """A lock left behind by a dead worker must not block forever."""

    def test_absent_lock_is_not_stale(self, tmp_path):
        """Nothing to reclaim when there is no lock."""
        assert check_stale_lock_file(tmp_path / "absent.recording.lock") is False

    def test_fresh_lock_is_not_stale(self, tmp_path):
        """A lock a live worker just took is still theirs."""
        lock_file = tmp_path / "cam1.recording.lock"
        lock_file.touch()

        assert check_stale_lock_file(lock_file) is False

    def test_old_lock_is_stale(self, tmp_path):
        """A lock older than the timeout is assumed abandoned."""
        lock_file = tmp_path / "cam1.recording.lock"
        lock_file.touch()
        aged = time.time() - STALE_LOCK_TIMEOUT_S - 1
        import os

        os.utime(lock_file, (aged, aged))

        assert check_stale_lock_file(lock_file) is True

    def test_clearing_a_stale_lock_discards_partial_output(self, tmp_path):
        """Whatever the dead worker half-wrote goes with its lock."""
        frames_dir = tmp_path / "cam1"
        frames_dir.mkdir()
        (frames_dir / "0.png").touch()
        lock_file = lock_file_for(frames_dir)
        lock_file.touch()

        clear_stale_lock(lock_file, frames_dir)

        assert not lock_file.exists()
        assert not frames_dir.exists()

    def test_wait_returns_immediately_when_unlocked(self, tmp_path):
        """No lock means no waiting."""
        frames_dir = tmp_path / "cam1"

        wait_for_lock_release(lock_file_for(frames_dir), frames_dir)

    def test_wait_clears_a_stale_lock_rather_than_blocking(self, tmp_path):
        """A stale lock is reclaimed instead of waited on."""
        frames_dir = tmp_path / "cam1"
        frames_dir.mkdir()
        lock_file = lock_file_for(frames_dir)
        lock_file.touch()
        aged = time.time() - STALE_LOCK_TIMEOUT_S - 1
        import os

        os.utime(lock_file, (aged, aged))

        wait_for_lock_release(lock_file, frames_dir)

        assert not lock_file.exists()


class TestDecodeVideo:
    """The ffmpeg invocation used to explode a video into frames."""

    def test_uses_resolved_frame_sync_arg(self, tmp_path):
        """The decode must use the probed flag, not a hard-coded `-vsync`.

        That spelling was removed in ffmpeg 8, so hard-coding it breaks decoding
        on any ffmpeg 8+ install.
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
            decode_rgb_video(video_location, frames_dir)

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
        import neuracore.core.data.frame_cache as frame_cache_module

        frame_cache_module._FFMPEG_FRAME_SYNC_ARG = None
        yield
        frame_cache_module._FFMPEG_FRAME_SYNC_ARG = None

    def test_returns_fps_mode_when_ffmpeg_accepts_it(self):
        """ffmpeg >= 5.1 accepts -fps_mode, so it should be preferred."""
        from neuracore.core.data.frame_cache import _resolve_frame_sync_arg

        with patch(f"{MODULE}.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0, stderr=b"")

            assert _resolve_frame_sync_arg() == "-fps_mode"

    def test_returns_vsync_when_ffmpeg_rejects_fps_mode(self):
        """ffmpeg < 5.1 rejects -fps_mode as unrecognised, so fall back to -vsync."""
        from neuracore.core.data.frame_cache import _resolve_frame_sync_arg

        with patch(f"{MODULE}.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(
                returncode=1, stderr=b"Unrecognized option 'fps_mode'."
            )

            assert _resolve_frame_sync_arg() == "-vsync"

    def test_returns_vsync_when_ffmpeg_binary_is_missing(self):
        """No ffmpeg on PATH resolves to the legacy spelling, matching the
        pre-existing _FFMPEG_AVAILABLE probe's fall-back behaviour.
        """
        from neuracore.core.data.frame_cache import _resolve_frame_sync_arg

        with patch(f"{MODULE}.subprocess.run", side_effect=FileNotFoundError):
            assert _resolve_frame_sync_arg() == "-vsync"

    def test_probes_ffmpeg_only_once(self):
        """The result is memoized: a second call must not spawn ffmpeg again."""
        from neuracore.core.data.frame_cache import _resolve_frame_sync_arg

        with patch(f"{MODULE}.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0, stderr=b"")

            assert _resolve_frame_sync_arg() == "-fps_mode"
            assert _resolve_frame_sync_arg() == "-fps_mode"

        mock_run.assert_called_once()


class TestDepthFrames:
    """Caching JPEG-XL depth frames files and reading frames out of them."""

    def test_published_frames_file_serves_frames_by_byte_range(
        self, tmp_path: Path, depth_frames_fn
    ):
        """A published frames file keeps its bytes and serves any frame exactly."""
        frames = _depth_frames()
        payload, ranges = depth_frames_fn(frames)
        media = tmp_path / "depth.mp4"
        media.write_bytes(payload)
        staging = tmp_path / "staging"
        staging.mkdir()
        frames_dir = tmp_path / "cam"

        cache_depth_frames(media, staging, frames_dir)

        assert (frames_dir / DEPTH_FRAMES_FILENAME).read_bytes() == payload
        for frame_idx in (2, 0, 1):
            frame = read_depth_frame(frames_dir, *ranges[frame_idx])
            assert frame.dtype == np.uint16
            np.testing.assert_array_equal(frame, frames[frame_idx])

    def test_truncated_frames_file_raises(self, tmp_path: Path, depth_frames_fn):
        """A byte range past the end of the file fails."""
        payload, ranges = depth_frames_fn(_depth_frames())
        (tmp_path / DEPTH_FRAMES_FILENAME).write_bytes(payload[:-1])

        with pytest.raises(ValueError, match="ends inside"):
            read_depth_frame(tmp_path, *ranges[-1])

    def test_frame_of_another_dtype_raises(self, tmp_path: Path):
        """A frame that decodes to another dtype fails."""
        payload = imagecodecs.jpegxl_encode(np.ones((2, 2), np.uint8), lossless=True)
        (tmp_path / DEPTH_FRAMES_FILENAME).write_bytes(payload)

        with pytest.raises(ValueError, match="decodes as"):
            read_depth_frame(tmp_path, 0, len(payload))


def test_cache_camera_frames_rejects_other_data_types(tmp_path: Path):
    """Caching frames for a non-camera data type fails."""
    with pytest.raises(ValueError, match="Cannot cache frames"):
        cache_camera_frames(
            DataType.JOINT_POSITIONS, tmp_path / "media", tmp_path, tmp_path / "cam"
        )
