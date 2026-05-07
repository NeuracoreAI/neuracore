"""Tests for FAST tokenizer loading and action tokenization."""

import numpy as np
import pytest
import torch

from neuracore.ml.algorithms.pi05_full.utils import (
    PI05FullConfig,
    fast_tokenize_actions,
    load_fast_tokenizer,
)


@pytest.fixture(scope="module")
def fast_tokenizer():
    return load_fast_tokenizer("physical-intelligence/fast")


def test_load_fast_tokenizer_returns_tokenizer(fast_tokenizer):
    # The FAST tokenizer is a HF AutoProcessor; we just need encode() to work.
    assert hasattr(fast_tokenizer, "__call__") or hasattr(fast_tokenizer, "encode")


def test_fast_tokenize_returns_padded_ids_and_mask(fast_tokenizer):
    cfg = PI05FullConfig()
    actions = np.random.randn(2, 10, 7).astype(np.float32)  # (B, T, action_dim)
    token_ids, mask = fast_tokenize_actions(
        actions,
        tokenizer=fast_tokenizer,
        max_tokens=cfg.max_fast_tokens,
        skip_tokens=cfg.fast_skip_tokens,
        vocab_size=257152,  # paligemma vocab size
    )
    assert token_ids.shape == (2, cfg.max_fast_tokens)
    assert mask.shape == (2, cfg.max_fast_tokens)
    assert token_ids.dtype == torch.long
    assert mask.dtype == torch.bool


def test_fast_tokens_land_in_paligemma_tail(fast_tokenizer):
    """FAST tokens must map into the last `fast_skip_tokens` slots of the vocab."""
    cfg = PI05FullConfig()
    actions = np.random.randn(1, 10, 7).astype(np.float32)
    token_ids, mask = fast_tokenize_actions(
        actions,
        tokenizer=fast_tokenizer,
        max_tokens=cfg.max_fast_tokens,
        skip_tokens=cfg.fast_skip_tokens,
        vocab_size=257152,
    )
    valid_ids = token_ids[mask]
    if valid_ids.numel() > 0:
        assert (valid_ids >= 257152 - cfg.fast_skip_tokens).all()
        assert (valid_ids < 257152).all()


def test_fast_tokenize_truncation_when_too_long(fast_tokenizer):
    """If FAST emits more tokens than max_fast_tokens, truncate from the right."""
    actions = np.random.randn(1, 200, 16).astype(np.float32)  # large chunk
    token_ids, mask = fast_tokenize_actions(
        actions,
        tokenizer=fast_tokenizer,
        max_tokens=8,  # tiny cap forces truncation
        skip_tokens=128,
        vocab_size=257152,
    )
    assert token_ids.shape == (1, 8)
    # mask should be all True (8 tokens of valid output, none padding)
    assert mask.all()
