"""Tests for PI05FullConfig new fields."""

import pytest

from neuracore.ml.algorithms.pi05_full.utils import PI05FullConfig


def test_default_loss_weights():
    cfg = PI05FullConfig()
    assert cfg.subtask_loss_weight == 10.0
    assert cfg.fast_token_loss_weight == 1.0
    assert cfg.flow_matching_loss_weight == 1.0


def test_knowledge_insulation_default_true():
    cfg = PI05FullConfig()
    assert cfg.knowledge_insulation is True


def test_subtask_token_lengths_have_defaults():
    cfg = PI05FullConfig()
    assert cfg.max_subtask_tokens == 64
    assert cfg.max_fast_tokens == 128


def test_fast_tokenizer_defaults():
    cfg = PI05FullConfig()
    assert cfg.fast_tokenizer_name == "physical-intelligence/fast"
    assert cfg.fast_skip_tokens == 128


def test_subtask_generation_defaults():
    cfg = PI05FullConfig()
    assert cfg.max_decoding_steps == 200
    assert cfg.subtask_temperature == 0.0


def test_negative_loss_weights_rejected():
    cfg = PI05FullConfig(subtask_loss_weight=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        cfg.validate_features()


def test_all_zero_loss_weights_rejected():
    cfg = PI05FullConfig(
        subtask_loss_weight=0.0,
        fast_token_loss_weight=0.0,
        flow_matching_loss_weight=0.0,
    )
    with pytest.raises(ValueError, match="At least one loss weight"):
        cfg.validate_features()
