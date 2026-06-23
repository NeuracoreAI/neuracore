"""Tests for CacheManager eviction behavior."""

from neuracore.core.data.cache_manager import CacheManager


def _make_recording(cache_dir, rec_id, num_frames=5):
    """Create a fake cached recording dir with frame files."""
    frames_dir = cache_dir / rec_id / "rgb_images" / "cam1"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_frames):
        (frames_dir / f"{i}.png").write_bytes(b"x" * 10)
    return cache_dir / rec_id


def test_cleanup_evicts_whole_recordings_never_partial(tmp_path):
    """Eviction removes entire recording dirs, leaving survivors fully intact.

    Deleting individual frame files would leave undetectable holes (a present
    directory is assumed complete), so eviction must be all-or-nothing per
    recording.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    rec_a = _make_recording(cache_dir, "recA", num_frames=5)
    rec_b = _make_recording(cache_dir, "recB", num_frames=5)

    cm = CacheManager(cache_dir)
    cm.cleanup_cache(percent_to_remove=50.0)  # 50% of 2 recordings -> 1 removed

    survivors = [d for d in (rec_a, rec_b) if d.exists()]
    removed = [d for d in (rec_a, rec_b) if not d.exists()]
    assert len(survivors) == 1
    assert len(removed) == 1
    # The survivor keeps ALL of its frames -- no partial deletion.
    survivor_frames = list((survivors[0] / "rgb_images" / "cam1").glob("*.png"))
    assert len(survivor_frames) == 5


def test_cleanup_skips_recordings_with_active_lock(tmp_path):
    """A recording with an in-progress decode lock is never evicted."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    locked = _make_recording(cache_dir, "locked")
    unlocked = _make_recording(cache_dir, "unlocked")
    # Active decode lock somewhere inside the locked recording.
    (locked / "rgb_images" / "cam1" / ".recording.lock").write_bytes(b"")

    cm = CacheManager(cache_dir)
    cm.cleanup_cache(percent_to_remove=50.0)

    assert locked.exists()  # skipped despite the lock
    assert not unlocked.exists()  # the unlocked one is evicted instead
