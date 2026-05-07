"""Tests that knowledge insulation severs gradient flow correctly."""

import torch

from neuracore.ml.algorithms.pi05_full.modules import PI05FullPolicy


def _vlm_param(policy: PI05FullPolicy) -> torch.nn.Parameter:
    """Pick a VLM K projection weight that sees gradient via attention."""
    return policy.paligemma_with_expert.paligemma.language_model.layers[
        0
    ].self_attn.k_proj.weight


def _run_training_forward(policy: PI05FullPolicy, knowledge_insulation: bool):
    """Run a tiny training forward and return the loss dict."""
    torch.manual_seed(0)
    bsize = 1
    img = torch.randn(bsize, 3, 224, 224)
    img_mask = torch.ones(bsize, dtype=torch.bool)
    lang_tokens = torch.zeros(bsize, 4, dtype=torch.long)
    lang_masks = torch.ones(bsize, 4, dtype=torch.bool)
    subtask_tokens = torch.zeros(bsize, 4, dtype=torch.long)
    subtask_masks = torch.ones(bsize, 4, dtype=torch.bool)
    fast_tokens = torch.zeros(bsize, 4, dtype=torch.long)
    fast_masks = torch.ones(bsize, 4, dtype=torch.bool)
    actions = torch.zeros(bsize, policy.config.chunk_size, policy.config.max_action_dim)

    policy.train()
    policy.config.knowledge_insulation = knowledge_insulation
    return policy.forward(
        [img],
        [img_mask],
        lang_tokens,
        lang_masks,
        subtask_tokens,
        subtask_masks,
        fast_tokens,
        fast_masks,
        actions,
    )


def test_ki_blocks_flow_gradient_into_vlm(tiny_pi05_full_config):
    """With KI on and only flow loss, VLM K-proj.grad must be zero."""
    policy = PI05FullPolicy(tiny_pi05_full_config)
    losses = _run_training_forward(policy, knowledge_insulation=True)
    losses["flow_mse_loss"].backward()
    assert torch.allclose(
        _vlm_param(policy).grad,
        torch.zeros_like(_vlm_param(policy).grad),
        atol=0.0,
    )


def test_ki_off_allows_flow_gradient_into_vlm(tiny_pi05_full_config):
    """With KI off and only flow loss, VLM K-proj.grad must be non-zero."""
    policy = PI05FullPolicy(tiny_pi05_full_config)
    losses = _run_training_forward(policy, knowledge_insulation=False)
    losses["flow_mse_loss"].backward()
    assert _vlm_param(policy).grad.abs().sum() > 0


def test_ki_does_not_block_subtask_gradient_into_vlm(tiny_pi05_full_config):
    """With KI on and only subtask loss, VLM K-proj.grad must be non-zero."""
    policy = PI05FullPolicy(tiny_pi05_full_config)
    losses = _run_training_forward(policy, knowledge_insulation=True)
    losses["subtask_ce_loss"].backward()
    assert _vlm_param(policy).grad.abs().sum() > 0
