"""Real-Time Control (RTC) training utilities for flow-matching algorithms.

Implements training-time RTC from "Robots that Think While They Act":
https://arxiv.org/abs/2501.02955

The core idea: sample a random delay `d` per batch element. The first `d` action
steps are treated as a "clean prefix" - their flow-matching time is set to 0.0
(fully de-noised in PI0's convention), so the model sees the ground-truth actions
there. Loss is computed only on the remaining "postfix" steps.

This trains the model to condition on a clean action prefix when predicting
the remainder of the chunk, enabling smooth chunk-to-chunk rollout at inference.

Delay is configured in **seconds** rather than steps, so the same config works
across datasets with different control frequencies or chunk sizes:

    delay_steps = floor(delay_s * frequency_hz)

The delay is sampled from the closed integer interval [min_delay_steps,
max_delay_steps] (both inclusive). Two distributions are supported:

  - ``"uniform"``: equal probability over the range.
  - ``"exp"``: exponential decay, ``weight ∝ exp(-exp_decay * step)``, so
    smaller delays are sampled more frequently. Useful when the robot spends
    most time with a small carry-over prefix.

PI0 PyTorch convention (note: opposite to the JAX paper):
    t = 0.0  →  clean actions
    t = 1.0  →  pure noise
    x_t = t * noise + (1 - t) * actions

So the prefix gets t = 0.0 (giving x_t = actions, i.e. clean), while the
postfix gets the normally sampled t.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


@dataclass
class RTCTrainingConfig:
    """Configuration for RTC training-time conditioning.

    Delay bounds are expressed in **seconds** and converted to integer steps
    using ``frequency_hz``:

        min_delay_steps = floor(min_delay_s * frequency_hz)
        max_delay_steps = floor(max_delay_s * frequency_hz)

    Delay is then sampled uniformly or exponentially from the **closed**
    interval ``[min_delay_steps, max_delay_steps]``.

    Attributes:
        enabled: Whether to apply RTC during training.
        min_delay_s: Minimum prefix duration in seconds. Sampled delay is
            always at least this many steps. Default 0.0 allows a zero-length
            prefix (equivalent to standard flow-matching for that sample).
        max_delay_s: Maximum prefix duration in seconds (inclusive). Must be
            >= min_delay_s.
        frequency_hz: Control frequency of the robot / dataset in Hz. Used
            to convert seconds to integer step counts. If ``None`` (default),
            the model auto-populates this from ``ModelInitDescription.frequency``
            at init time, so you rarely need to set it manually.
        delay_distribution: How to sample the delay within the range.
            ``"uniform"`` — equal weight over ``[min, max]``.
            ``"exp"`` — exponential decay weight
            ``exp(-exp_decay * step)`` favouring smaller delays.
        exp_decay: Decay rate for the ``"exp"`` distribution. Larger values
            concentrate probability mass near ``min_delay_steps``.
            Only used when ``delay_distribution="exp"``.
    """

    enabled: bool = True
    min_delay_s: float = 0.0
    max_delay_s: float = 0.5
    frequency_hz: float | None = None
    delay_distribution: Literal["uniform", "exp"] = "uniform"
    exp_decay: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration values after dataclass initialization."""
        if self.min_delay_s < 0:
            raise ValueError(f"min_delay_s must be >= 0, got {self.min_delay_s}")
        if self.max_delay_s < self.min_delay_s:
            raise ValueError(
                f"max_delay_s ({self.max_delay_s}) must be >= "
                f"min_delay_s ({self.min_delay_s})"
            )
        if self.frequency_hz is not None and self.frequency_hz <= 0:
            raise ValueError(f"frequency_hz must be > 0, got {self.frequency_hz}")
        if self.exp_decay <= 0:
            raise ValueError(f"exp_decay must be > 0, got {self.exp_decay}")
        if self.delay_distribution not in ("uniform", "exp"):
            raise ValueError(
                f"delay_distribution must be 'uniform' or 'exp', "
                f"got {self.delay_distribution!r}"
            )

    def _resolved_frequency(self) -> float:
        """Return frequency_hz, raising clearly if it was never set."""
        if self.frequency_hz is None:
            raise ValueError(
                "RTCTrainingConfig.frequency_hz is not set. "
                "It is normally auto-populated from the dataset frequency "
                "in the model __init__. If you are constructing the model "
                "outside the standard training pipeline, set frequency_hz "
                "explicitly in RTCTrainingConfig."
            )
        return self.frequency_hz

    @property
    def min_delay_steps(self) -> int:
        """Minimum prefix length in action steps."""
        return math.floor(self.min_delay_s * self._resolved_frequency())

    @property
    def max_delay_steps(self) -> int:
        """Maximum prefix length in action steps (inclusive)."""
        return math.floor(self.max_delay_s * self._resolved_frequency())


def sample_rtc_delay(
    config: RTCTrainingConfig,
    batch_size: int,
    chunk_size: int,
    device: torch.device,
) -> Tensor:
    """Sample a random action-prefix delay for each element in the batch.

    Delay is sampled from the **closed** integer interval
    ``[min_delay_steps, max_delay_steps]``, clamped so the prefix never
    spans the full horizon. When ``min == max``, a constant delay is returned
    with no randomness.

    Distribution options:

    - ``"uniform"``: every value in the range is equally likely.
    - ``"exp"``: weights ``∝ exp(-exp_decay * step)``; smaller delays are
      more probable. This mirrors real-world usage where short carry-over
      prefixes are more common than long ones.

    Args:
        config: RTC training configuration.
        batch_size: Number of samples in the batch.
        chunk_size: Total number of action steps in the prediction horizon.
            Used to clamp the maximum delay.
        device: Target device for the output tensor.

    Returns:
        Integer delay tensor [batch_size] with values in
        ``[min_delay_steps, min(max_delay_steps, chunk_size)]``.
    """
    lo = min(config.min_delay_steps, chunk_size)
    hi = min(config.max_delay_steps, chunk_size)

    if lo == hi:
        return torch.full((batch_size,), lo, dtype=torch.long, device=device)

    if config.delay_distribution == "uniform":
        # randint upper bound is exclusive, so +1 gives closed [lo, hi]
        return torch.randint(lo, hi + 1, (batch_size,), dtype=torch.long, device=device)

    # Exponential: weight ∝ exp(-exp_decay * step), favouring smaller delays
    delay_values = torch.arange(lo, hi + 1, dtype=torch.long, device=device)
    weights = torch.exp(-config.exp_decay * delay_values.to(dtype=torch.float32))
    probs = weights / weights.sum()
    indices = torch.multinomial(probs, batch_size, replacement=True)
    return delay_values[indices]


def apply_rtc_training_time(
    time: Tensor,
    delay: Tensor,
    chunk_size: int,
) -> tuple[Tensor, Tensor]:
    """Create per-step flow-matching time for RTC training.

    Replaces the scalar time with a per-step time tensor where the first
    `delay` steps (the prefix) have time = 0.0 (fully de-noised / clean),
    and the remaining postfix steps keep the original sampled time.

    This is the PI0 PyTorch convention (reversed from the JAX paper):
        prefix time = 0.0  →  x_t = actions (clean ground truth)
        postfix time = t   →  x_t = t * noise + (1-t) * actions (noisy)

    Args:
        time: Scalar flow-matching time per sample [B], values in [0, 1].
        delay: Integer prefix length per sample [B], values in [0, chunk_size].
        chunk_size: Number of action steps in the prediction horizon.

    Returns:
        time_per_step: Per-step time tensor [B, chunk_size]. Prefix steps
            have time = 0.0; postfix steps have the sampled time.
        postfix_mask: Boolean mask [B, chunk_size]. True for postfix steps
            where the loss should be computed.
    """
    delay = torch.clamp(delay, max=chunk_size)
    steps = torch.arange(chunk_size, device=time.device)  # [chunk_size]
    prefix_mask = steps[None, :] < delay[:, None]  # [B, chunk_size]

    time_per_step = time[:, None].expand(-1, chunk_size).clone()
    time_per_step = time_per_step.masked_fill(prefix_mask, 0.0)

    postfix_mask = ~prefix_mask  # [B, chunk_size]
    return time_per_step, postfix_mask


def masked_mean(
    losses: Tensor,
    mask: Tensor | None,
    eps: float = 1e-8,
) -> Tensor:
    """Mean of `losses` over elements selected by `mask`.

    Unlike boolean indexing (``losses[mask].mean()``), this handles the
    degenerate case where the mask is all-False (e.g. every sample had
    delay = chunk_size) by returning 0 instead of NaN.

    When `mask` is None, falls back to ``losses.mean()``.

    Args:
        losses: Per-element MSE loss [B, chunk_size, action_dim].
        mask: Boolean mask [B, chunk_size]. True for elements to include.
            Broadcast to match `losses` along the last dimension.
        eps: Small constant to avoid division by zero.

    Returns:
        Scalar mean loss.
    """
    if mask is None:
        return losses.mean()

    # Expand mask to cover action_dim
    float_mask = mask.to(dtype=losses.dtype)
    while float_mask.dim() < losses.dim():
        float_mask = float_mask.unsqueeze(-1)

    return (losses * float_mask).sum() / float_mask.sum().clamp_min(eps)


def compute_rtc_loss(
    mse_per_element: Tensor,
    postfix_mask: Tensor,
) -> Tensor:
    """Compute mean MSE loss restricted to the postfix action steps.

    Args:
        mse_per_element: Per-element MSE loss [B, chunk_size, action_dim].
        postfix_mask: Boolean mask [B, chunk_size]. True for steps that
            should contribute to the loss.

    Returns:
        Scalar mean loss over all postfix (batch, step, dim) elements.
    """
    return masked_mean(mse_per_element, postfix_mask)
