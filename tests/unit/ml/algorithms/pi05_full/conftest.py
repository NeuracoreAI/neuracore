"""Shared fixtures for pi05_full tests."""

import pytest

from neuracore.ml.algorithms.pi05_full.utils import PI05FullConfig


@pytest.fixture
def tiny_pi05_full_config() -> PI05FullConfig:
    """Smallest possible config for fast tests.

    Uses gemma_300m for both branches and reduces action dimensions and
    chunk sizes so a forward+backward fits comfortably in CPU memory.
    """
    return PI05FullConfig(
        paligemma_variant="gemma_300m",
        action_expert_variant="gemma_300m",
        chunk_size=8,
        max_state_dim=8,
        max_action_dim=8,
        num_inference_steps=2,
        max_subtask_tokens=8,
        max_fast_tokens=8,
        max_decoding_steps=4,
        device="cpu",
        dtype="float32",
        gradient_checkpointing=False,
    )
