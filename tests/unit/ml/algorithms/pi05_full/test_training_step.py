"""End-to-end training step on a synthetic batch."""

import torch

from .conftest import PI05_FULL_TEST_ARGS


def test_training_step_returns_four_finite_losses(
    pi05_full_cls, model_init_description_with_subtask, synthetic_training_batch
):
    model = pi05_full_cls(model_init_description_with_subtask, **PI05_FULL_TEST_ARGS)
    out = model.training_step(synthetic_training_batch)
    expected = {"loss", "flow_mse_loss", "subtask_ce_loss", "fast_ce_loss"}
    assert set(out.losses.keys()) == expected
    assert set(out.metrics.keys()) == expected
    for k in expected:
        assert torch.isfinite(out.losses[k]).all()
    assert out.losses["loss"].requires_grad
