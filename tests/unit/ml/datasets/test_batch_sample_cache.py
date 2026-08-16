"""Tests for the on-disk cache of built training samples."""

import torch
from neuracore_types import BatchedRGBData, DataType

from neuracore.ml.core.ml_types import BatchedTrainingSamples
from neuracore.ml.datasets.batch_sample_cache import BatchSampleCache
from neuracore.ml.preprocessing.base import PreprocessingConfiguration
from neuracore.ml.preprocessing.methods.resize_pad import ResizePad

ROBOT = "robot-1"


def _description(slots: int = 2) -> dict:
    return {ROBOT: {DataType.RGB_IMAGES: {i: f"cam_{i}" for i in range(slots)}}}


def _sample(value: float = 1.0) -> BatchedTrainingSamples:
    return BatchedTrainingSamples(
        inputs={
            DataType.RGB_IMAGES: [
                BatchedRGBData(
                    frame=torch.full((1, 1, 3, 8, 8), value),
                    extrinsics=torch.zeros((1, 1, 4, 4)),
                    intrinsics=torch.zeros((1, 1, 3, 3)),
                )
            ]
        },
        inputs_mask={DataType.RGB_IMAGES: torch.ones(1, 1)},
        outputs={},
        outputs_mask={},
        batch_size=1,
    )


def _cache(tmp_path, horizon: int = 3, resize=(224, 224), slots: int = 2):
    return BatchSampleCache(
        synchronized_dataset_id="sync-1",
        input_cross_embodiment_description=_description(slots),
        output_cross_embodiment_description=_description(slots),
        output_prediction_horizon=horizon,
        input_preprocessing_config=PreprocessingConfiguration(
            {DataType.RGB_IMAGES: [ResizePad(size=resize)]}
        ),
        output_preprocessing_config=PreprocessingConfiguration(),
        root=tmp_path,
    )


def test_round_trip_returns_an_equal_sample(tmp_path):
    cache = _cache(tmp_path)
    original = _sample(value=7.0)

    cache.store("rec-1", 4, original)
    restored = cache.load("rec-1", 4)

    assert restored is not None
    assert torch.equal(
        restored.inputs[DataType.RGB_IMAGES][0].frame,
        original.inputs[DataType.RGB_IMAGES][0].frame,
    )
    assert restored.batch_size == original.batch_size


def test_miss_returns_none(tmp_path):
    assert _cache(tmp_path).load("rec-1", 4) is None


def test_entries_are_keyed_by_recording_and_timestep(tmp_path):
    """Two samples must not collide, including across recordings."""
    cache = _cache(tmp_path)
    cache.store("rec-1", 0, _sample(value=1.0))
    cache.store("rec-1", 1, _sample(value=2.0))
    cache.store("rec-2", 0, _sample(value=3.0))

    def first_pixel(recording_id, timestep):
        sample = cache.load(recording_id, timestep)
        return float(sample.inputs[DataType.RGB_IMAGES][0].frame.flatten()[0])

    assert first_pixel("rec-1", 0) == 1.0
    assert first_pixel("rec-1", 1) == 2.0
    assert first_pixel("rec-2", 0) == 3.0


def test_unreadable_entry_is_discarded_rather_than_raising(tmp_path):
    """A truncated entry costs a rebuild, never a crash."""
    cache = _cache(tmp_path)
    cache.store("rec-1", 4, _sample())
    entry = next(cache.directory.rglob("*.pt"))
    entry.write_bytes(b"not a torch file")

    assert cache.load("rec-1", 4) is None
    assert not entry.exists(), "a bad entry should be removed, not left to fail again"


def test_store_leaves_no_partial_files(tmp_path):
    cache = _cache(tmp_path)
    cache.store("rec-1", 4, _sample())

    assert list(cache.directory.rglob("*.tmp")) == []


def test_key_changes_with_anything_that_shapes_a_sample(tmp_path):
    """Stale entries must not survive a configuration change."""
    baseline = _cache(tmp_path).directory

    assert _cache(tmp_path, horizon=5).directory != baseline
    assert _cache(tmp_path, resize=(128, 128)).directory != baseline
    assert _cache(tmp_path, slots=3).directory != baseline
    assert _cache(tmp_path).directory == baseline, "identical config must be stable"


def test_key_ignores_the_root_directory(tmp_path):
    """Moving the cache elsewhere must not invalidate its entries."""
    other = tmp_path / "elsewhere"
    assert _cache(tmp_path).directory.name == _cache(other).directory.name
