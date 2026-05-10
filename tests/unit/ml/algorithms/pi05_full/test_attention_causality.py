"""Within-segment attention must be causal for subtask and FAST blocks.

Bidirectional attention within a segment whose tokens are LM-predicted (subtask
CE, FAST CE) lets the model attend to the next token's embedding directly,
which leaks the target into the prediction even after the causal CE shift.
This file is a regression test for that bug.
"""

# cspell:ignore tril

import torch

from neuracore.ml.algorithms.pi05_full.modules import PI05FullPolicy
from neuracore.ml.algorithms.pi05_full.utils import _make_att_2d_masks


def _build_prefix_2d_mask(policy, *, sub_len: int, fast_len: int):
    bsize = 1
    img = torch.randn(bsize, 3, 224, 224)
    img_mask = torch.ones(bsize, dtype=torch.bool)
    lang_tokens = torch.zeros(bsize, 4, dtype=torch.long)
    lang_masks = torch.ones(bsize, 4, dtype=torch.bool)
    sub_tokens = torch.zeros(bsize, sub_len, dtype=torch.long)
    sub_masks = torch.ones(bsize, sub_len, dtype=torch.bool)
    fast_tokens = torch.zeros(bsize, fast_len, dtype=torch.long)
    fast_masks = torch.ones(bsize, fast_len, dtype=torch.bool)

    _, pad_masks, att_masks, segments = policy._embed_prefix(
        [img],
        [img_mask],
        lang_tokens,
        lang_masks,
        subtask_tokens=sub_tokens,
        subtask_masks=sub_masks,
        fast_tokens=fast_tokens,
        fast_masks=fast_masks,
    )
    return _make_att_2d_masks(pad_masks, att_masks)[0], segments


def test_subtask_block_is_causal_within(tiny_pi05_full_config):
    policy = PI05FullPolicy(tiny_pi05_full_config)
    mask, segments = _build_prefix_2d_mask(policy, sub_len=4, fast_len=4)
    sub = segments["subtask"]
    sub_block = mask[sub, sub]  # (sub_len, sub_len)
    expected = torch.tril(torch.ones_like(sub_block))
    assert torch.equal(sub_block.to(dtype=expected.dtype), expected), (
        "Subtask block must be lower-triangular (causal). Got:\n"
        f"{sub_block.int().tolist()}"
    )


def test_fast_block_is_causal_within(tiny_pi05_full_config):
    policy = PI05FullPolicy(tiny_pi05_full_config)
    mask, segments = _build_prefix_2d_mask(policy, sub_len=4, fast_len=4)
    fast = segments["fast"]
    fast_block = mask[fast, fast]
    expected = torch.tril(torch.ones_like(fast_block))
    assert torch.equal(fast_block.to(dtype=expected.dtype), expected), (
        "FAST block must be lower-triangular (causal). Got:\n"
        f"{fast_block.int().tolist()}"
    )


def test_subtask_attends_to_image_and_language(tiny_pi05_full_config):
    policy = PI05FullPolicy(tiny_pi05_full_config)
    mask, segments = _build_prefix_2d_mask(policy, sub_len=4, fast_len=4)
    sub = segments["subtask"]
    # Every subtask query position should attend to every prefix position
    # before the subtask block (images + language).
    sub_to_prefix = mask[sub, : sub.start]
    assert sub_to_prefix.all(), "Subtask must attend to all images + language"


def test_fast_attends_to_subtask_and_language(tiny_pi05_full_config):
    policy = PI05FullPolicy(tiny_pi05_full_config)
    mask, segments = _build_prefix_2d_mask(policy, sub_len=4, fast_len=4)
    fast = segments["fast"]
    # FAST queries should attend to all images + language + subtask positions.
    fast_to_prefix = mask[fast, : fast.start]
    assert fast_to_prefix.all(), "FAST must attend to all images + language + subtask"
