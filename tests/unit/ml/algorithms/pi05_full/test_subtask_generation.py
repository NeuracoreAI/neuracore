"""Tests for autoregressive subtask generation at inference."""

import torch

from neuracore.ml.algorithms.pi05_full.modules import PI05FullPolicy


def test_generate_subtask_tokens_returns_correct_shape(tiny_pi05_full_config):
    policy = PI05FullPolicy(tiny_pi05_full_config)
    policy.eval()
    bsize = 2
    img = torch.randn(bsize, 3, 224, 224)
    img_mask = torch.ones(bsize, dtype=torch.bool)
    lang_tokens = torch.zeros(bsize, 4, dtype=torch.long)
    lang_masks = torch.ones(bsize, 4, dtype=torch.bool)
    bos_id = 2  # PaliGemma BOS

    tokens, masks = policy.generate_subtask_tokens(
        [img],
        [img_mask],
        lang_tokens,
        lang_masks,
        bos_token_id=bos_id,
    )
    assert tokens.shape[0] == bsize
    assert tokens.shape[1] <= tiny_pi05_full_config.max_decoding_steps + 1
    assert tokens.dtype == torch.long
    assert masks.shape == tokens.shape
    assert masks.dtype == torch.bool


def test_generate_starts_with_bos(tiny_pi05_full_config):
    policy = PI05FullPolicy(tiny_pi05_full_config)
    policy.eval()
    bsize = 1
    img = torch.randn(bsize, 3, 224, 224)
    img_mask = torch.ones(bsize, dtype=torch.bool)
    lang_tokens = torch.zeros(bsize, 4, dtype=torch.long)
    lang_masks = torch.ones(bsize, 4, dtype=torch.bool)
    bos_id = 2

    tokens, _ = policy.generate_subtask_tokens(
        [img],
        [img_mask],
        lang_tokens,
        lang_masks,
        bos_token_id=bos_id,
    )
    assert tokens[0, 0].item() == bos_id
