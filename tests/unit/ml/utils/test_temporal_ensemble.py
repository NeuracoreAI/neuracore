"""Unit tests for temporal ensemble merge and ACT ensembler utilities."""

import numpy as np
import pytest

from neuracore.ml.utils.temporal_ensemble import (
    ACTTemporalEnsembler,
    continuity_blend,
    temporal_ensemble_merge,
)


def test_temporal_ensemble_merge_empty_old_returns_new():
    new = np.arange(12, dtype=np.float32).reshape(4, 3)
    merged = temporal_ensemble_merge(np.zeros((0, 3)), new, old_offset=0, m=0.01)
    np.testing.assert_array_equal(merged, new)


def test_temporal_ensemble_merge_favors_newer_at_overlap_start():
    old = np.ones((4, 2), dtype=np.float32)
    new = np.full((6, 2), 3.0, dtype=np.float32)
    merged = temporal_ensemble_merge(old, new, old_offset=2, m=0.0)
    # m=0 → equal weights on overlap → mean of 1 and 3
    np.testing.assert_allclose(merged[:4], 2.0)
    np.testing.assert_allclose(merged[4:], 3.0)


def test_temporal_ensemble_merge_rejects_dim_mismatch():
    with pytest.raises(ValueError, match="action dim"):
        temporal_ensemble_merge(np.ones((2, 2)), np.ones((3, 3)), old_offset=0, m=0.01)


def test_continuity_blend_lerps_head():
    chunk = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]], dtype=np.float32)
    anchor = np.array([0.0, 0.0], dtype=np.float32)
    blended = continuity_blend(chunk, anchor, blend_steps=2)
    np.testing.assert_allclose(blended[0], [5.0, 5.0])
    np.testing.assert_allclose(blended[1], [20.0, 20.0])
    np.testing.assert_allclose(blended[2], [30.0, 30.0])


def test_act_temporal_ensembler_pops_one_action_per_update():
    ens = ACTTemporalEnsembler(m=0.01, chunk_size=4)
    assert not ens.is_warm
    chunk = np.arange(8, dtype=np.float64).reshape(4, 2)
    action = ens.update(chunk)
    assert ens.is_warm
    assert action.shape == (2,)
    np.testing.assert_allclose(action, chunk[0])


def test_act_temporal_ensembler_reset_clears_buffer():
    ens = ACTTemporalEnsembler(m=0.01, chunk_size=3)
    ens.update(np.ones((3, 1)))
    ens.reset()
    assert not ens.is_warm
