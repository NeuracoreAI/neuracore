"""End-to-end inference: subtask gen -> flow denoise -> actions."""

import torch

from neuracore.ml.algorithms.pi05_full.modules import PI05FullPolicy


def test_sample_actions_with_generated_subtask(tiny_pi05_full_config):
    policy = PI05FullPolicy(tiny_pi05_full_config)
    policy.eval()
    bsize = 1
    img = torch.randn(bsize, 3, 224, 224)
    img_mask = torch.ones(bsize, dtype=torch.bool)
    lang_tokens = torch.zeros(bsize, 4, dtype=torch.long)
    lang_masks = torch.ones(bsize, 4, dtype=torch.bool)

    subtask, subtask_mask = policy.generate_subtask_tokens(
        [img],
        [img_mask],
        lang_tokens,
        lang_masks,
        bos_token_id=2,
    )
    actions = policy.sample_actions(
        [img],
        [img_mask],
        lang_tokens,
        lang_masks,
        subtask_tokens=subtask,
        subtask_masks=subtask_mask,
    )
    assert actions.shape == (
        bsize,
        tiny_pi05_full_config.chunk_size,
        tiny_pi05_full_config.max_action_dim,
    )
    assert torch.isfinite(actions).all()
