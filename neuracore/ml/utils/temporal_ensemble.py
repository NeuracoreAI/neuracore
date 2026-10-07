"""Temporal ensemble utilities for overlapping action-chunk policies.

Two forms live here:

* :func:`temporal_ensemble_merge` — prefetch/merge used by the async
  :class:`~neuracore.ml.utils.rtc_controller.ChunkingController` under
  ``mode="temporal_ensemble"``.
* :class:`ACTTemporalEnsembler` — classic ACT Algorithm 2 (predict every
  control step). Standalone; not wired into the async controller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalEnsembleConfig:
    """Configuration for async temporal-ensemble chunking.

    Attributes:
        execution_horizon: ``s``, actions consumed from each chunk before
            the controller triggers a replan.
        m: Exponential decay rate for overlap weights. Larger ``m`` favors
            the newer prediction sooner.
        blend_steps: Optional continuity lerp of the first N actions of a
            newly merged chunk toward the previous remaining head. ``0``
            disables it.
    """

    execution_horizon: int
    m: float = 0.01
    blend_steps: int = 0


def temporal_ensemble_merge(
    old: np.ndarray,
    new: np.ndarray,
    *,
    old_offset: int,
    m: float,
) -> np.ndarray:
    """Blend remaining ``old`` actions with a freshly predicted ``new`` chunk.

    Overlap steps use exponential weights that favor the newer prediction
    (prefetch form, not original ACT): at overlap index ``i``,

    - ``w_new = exp(-m * i)``
    - ``w_old = exp(-m * (old_offset + i))``

    then normalize. After the overlap, the non-overlapping tail of ``new`` is
    appended (leftover ``old`` beyond the overlap is dropped).

    Args:
        old: Remaining actions from the currently executing chunk with shape
            ``(T_old, A)`` (index 0 is the next action to run).
        new: Newly predicted chunk with shape ``(H, A)``.
        old_offset: How many steps of ``old``'s original chunk had already
            been executed when this merge runs (staleness).
        m: Exponential decay rate.

    Returns:
        np.ndarray: Merged chunk with shape ``(H, A)`` matching ``new``.
    """
    old_arr = np.asarray(old, dtype=np.float64)
    new_arr = np.asarray(new, dtype=np.float64)
    if new_arr.ndim != 2:
        raise ValueError(f"new must be 2-D, got shape {new_arr.shape}")
    if old_arr.size == 0:
        return new_arr.astype(np.float32, copy=True)
    if old_arr.ndim != 2:
        raise ValueError(f"old must be 2-D, got shape {old_arr.shape}")
    if old_arr.shape[1] != new_arr.shape[1]:
        raise ValueError(
            f"action dim mismatch: old {old_arr.shape[1]} vs new {new_arr.shape[1]}"
        )

    old_len = old_arr.shape[0]
    new_len = new_arr.shape[0]
    overlap = min(old_len, new_len)
    stale = max(0, int(old_offset))
    decay = max(0.0, float(m))

    merged = np.empty_like(new_arr, dtype=np.float64)
    for i in range(overlap):
        w_new = float(np.exp(-decay * i))
        w_old = float(np.exp(-decay * (stale + i)))
        norm = w_new + w_old
        if norm <= 0.0:
            w_new, w_old, norm = 1.0, 0.0, 1.0
        merged[i] = (w_new * new_arr[i] + w_old * old_arr[i]) / norm

    if new_len > overlap:
        merged[overlap:] = new_arr[overlap:]
    return merged.astype(np.float32)


def continuity_blend(
    chunk: np.ndarray,
    anchor: np.ndarray,
    blend_steps: int,
) -> np.ndarray:
    """Lerp the first ``blend_steps`` rows of ``chunk`` toward ``anchor``.

    Args:
        chunk: Action chunk with shape ``(H, A)``.
        anchor: Previous action with shape ``(A,)``.
        blend_steps: Number of leading steps to blend.

    Returns:
        np.ndarray: Blended chunk with the same shape as ``chunk``.
    """
    if blend_steps <= 0 or chunk.size == 0:
        return np.asarray(chunk, dtype=np.float32).copy()
    out = np.asarray(chunk, dtype=np.float64).copy()
    anchor_arr = np.asarray(anchor, dtype=np.float64).reshape(-1)
    if anchor_arr.shape[0] != out.shape[1]:
        raise ValueError(
            f"anchor dim {anchor_arr.shape[0]} != action dim {out.shape[1]}"
        )
    steps = min(int(blend_steps), out.shape[0])
    for i in range(steps):
        alpha = (i + 1) / steps
        out[i] = (1.0 - alpha) * anchor_arr + alpha * out[i]
    return out.astype(np.float32)


class ACTTemporalEnsembler:
    """Original ACT temporal ensembling (Algorithm 2 / LeRobot online form).

    Weights are ``w_i = exp(-m * i)`` where ``i = 0`` is the **oldest**
    prediction covering a timestep (positive ``m`` favors older actions).
    Call :meth:`update` once per control step with a fresh ``(T, D)`` chunk;
    it returns the ensembled action for the current step and shifts the buffer.

    This is the predict-every-step path and is **not** used by the async
    :class:`~neuracore.ml.utils.rtc_controller.ChunkingController`.
    """

    def __init__(self, m: float, chunk_size: int) -> None:
        """Initialize the online ACT ensembler.

        Args:
            m: Exponential decay rate.
            chunk_size: Expected prediction horizon ``H``.
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.chunk_size = int(chunk_size)
        self.m = float(m)
        self.weights = np.exp(-self.m * np.arange(self.chunk_size, dtype=np.float64))
        self.weights_cumsum = np.cumsum(self.weights)
        self.reset()

    def reset(self) -> None:
        """Clear the online ensemble (call at episode / Play start)."""
        self.ensembled_actions: np.ndarray | None = None
        self.ensembled_actions_count: np.ndarray | None = None

    @property
    def is_warm(self) -> bool:
        """True once at least one chunk has been ingested."""
        return self.ensembled_actions is not None

    def update(self, actions: np.ndarray) -> np.ndarray:
        """Ingest one predicted chunk and pop the next ensembled action.

        Args:
            actions: Shape ``(chunk_size, action_dim)``. Truncated or padded
                with the last row if the model horizon differs from
                ``chunk_size``.

        Returns:
            Shape ``(action_dim,)`` action to execute this step.
        """
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2:
            raise ValueError(f"actions must be 2-D, got shape {actions.shape}")
        if actions.shape[0] != self.chunk_size:
            actions = _resize_chunk(actions, self.chunk_size)

        if self.ensembled_actions is None:
            self.ensembled_actions = actions.copy()
            self.ensembled_actions_count = np.ones((self.chunk_size, 1), dtype=np.int64)
        else:
            assert self.ensembled_actions_count is not None
            count = self.ensembled_actions_count
            overlap = actions[:-1]
            if self.ensembled_actions.shape[0] != overlap.shape[0]:
                raise RuntimeError(
                    "ACT ensembler buffer length mismatch: "
                    f"{self.ensembled_actions.shape[0]} vs {overlap.shape[0]}"
                )
            idx = np.clip(count[:, 0] - 1, 0, self.chunk_size - 1)
            new_idx = np.clip(count[:, 0], 0, self.chunk_size - 1)
            self.ensembled_actions = (
                self.ensembled_actions * self.weights_cumsum[idx, None]
                + overlap * self.weights[new_idx, None]
            ) / self.weights_cumsum[new_idx, None]
            self.ensembled_actions_count = np.clip(count + 1, 1, self.chunk_size)
            self.ensembled_actions = np.concatenate(
                [self.ensembled_actions, actions[-1:]], axis=0
            )
            self.ensembled_actions_count = np.concatenate(
                [
                    self.ensembled_actions_count,
                    np.ones((1, 1), dtype=np.int64),
                ],
                axis=0,
            )

        action = self.ensembled_actions[0].copy()
        self.ensembled_actions = self.ensembled_actions[1:]
        self.ensembled_actions_count = self.ensembled_actions_count[1:]
        return action


def _resize_chunk(actions: np.ndarray, chunk_size: int) -> np.ndarray:
    """Truncate or repeat-pad a chunk to ``chunk_size`` rows."""
    t = actions.shape[0]
    if t == chunk_size:
        return actions
    if t > chunk_size:
        return actions[:chunk_size]
    if t == 0:
        raise ValueError("Cannot resize an empty action chunk")
    pad = np.repeat(actions[-1:], chunk_size - t, axis=0)
    return np.concatenate([actions, pad], axis=0)
