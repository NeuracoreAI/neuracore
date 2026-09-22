"""Dataset helpers for training/validation splits."""

from __future__ import annotations

import logging
from collections import defaultdict
from copy import copy
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Subset

if TYPE_CHECKING:
    from neuracore.ml.datasets.pytorch_synchronized_dataset import (
        PytorchSynchronizedDataset,
    )
    from neuracore.ml.preprocessing.base import PreprocessingConfiguration

logger = logging.getLogger(__name__)

_EMPTY_SPLIT_HINT = (
    "Try adding more recordings to the dataset, or changing the validation split."
)


def _get_episode_sample_ranges(dataset: PytorchSynchronizedDataset) -> list[range]:
    """Return the sample-index range owned by each episode."""
    offsets = list(dataset.episode_start_offsets)
    n_samples = len(dataset)
    ranges: list[range] = []
    for i, start in enumerate(offsets):
        end = offsets[i + 1] if i + 1 < len(offsets) else n_samples
        ranges.append(range(start, end))
    return ranges


def _get_episode_robot_ids(
    dataset: PytorchSynchronizedDataset, n_episodes: int
) -> list[str] | None:
    """Return per-episode robot ids, or None when they cannot be resolved."""
    synchronized = getattr(dataset, "synchronized_dataset", None)
    if synchronized is None:
        return None
    robot_ids: list[str] = []
    for episode_idx in range(n_episodes):
        recording = synchronized[episode_idx]
        robot_id = getattr(recording, "robot_id", None)
        if not robot_id:
            return None
        robot_ids.append(str(robot_id))
    return robot_ids


def _count_val_episodes(n_episodes: int, validation_split: float) -> int:
    """Hold out at least one episode and leave at least one for training."""
    n_val = int(round(n_episodes * validation_split))
    return min(max(n_val, 1), n_episodes - 1)


def _shuffle_indices(n: int, generator: torch.Generator) -> list[int]:
    if n == 0:
        return []
    return torch.randperm(n, generator=generator).tolist()


def _assign_episodes(
    n_episodes: int,
    validation_split: float,
    seed: int,
    robot_ids: list[str] | None,
) -> tuple[list[int], list[int]]:
    """Return ``(train_episode_indices, val_episode_indices)``.

    When robot ids are available, shuffle and split within each robot so a
    multi-recording embodiment is not held out entirely. Single-recording
    robots stay in train. If that would leave either split empty, fall back
    to an unstratified shuffle of every episode.
    """
    generator = torch.Generator().manual_seed(seed)

    def global_split() -> tuple[list[int], list[int]]:
        order = _shuffle_indices(n_episodes, generator)
        n_val = _count_val_episodes(n_episodes, validation_split)
        return order[n_val:], order[:n_val]

    if not robot_ids:
        return global_split()

    groups: dict[str, list[int]] = defaultdict(list)
    for episode_idx, robot_id in enumerate(robot_ids):
        groups[robot_id].append(episode_idx)

    train_episodes: list[int] = []
    val_episodes: list[int] = []
    for robot_id in sorted(groups):
        members = groups[robot_id]
        order = [members[i] for i in _shuffle_indices(len(members), generator)]
        if len(order) < 2:
            train_episodes.extend(order)
            continue
        n_val = _count_val_episodes(len(order), validation_split)
        val_episodes.extend(order[:n_val])
        train_episodes.extend(order[n_val:])

    if not train_episodes or not val_episodes:
        return global_split()
    return train_episodes, val_episodes


def _collect_sample_indices(
    episode_indices: list[int], episode_ranges: list[range]
) -> list[int]:
    """Collect every sample index owned by the given episodes."""
    indices: list[int] = []
    for episode_idx in episode_indices:
        indices.extend(episode_ranges[episode_idx])
    return indices


def split_train_val_datasets(
    dataset: PytorchSynchronizedDataset,
    validation_split: float,
    seed: int,
    inference_input_preprocessing_config: PreprocessingConfiguration,
    inference_output_preprocessing_config: PreprocessingConfiguration,
) -> tuple[Subset, Subset]:
    """Split by whole recordings; val uses inference preprocessing.

    ``validation_split`` is a fraction of episodes, not frames. Every sample
    from a recording goes to exactly one of train or val, so a prediction
    horizon cannot straddle the split.

    ``dataset`` is expected to already carry train preprocessing. The val
    subset is rebased onto a shallow copy configured with inference
    preprocessing.

    Args:
        dataset: Full synchronized dataset with train preprocessing.
        validation_split: Fraction of recordings to hold out for validation.
            Must be strictly between 0 and 1.
        seed: RNG seed for the deterministic episode shuffle.
        inference_input_preprocessing_config: Preprocessing applied to val inputs.
        inference_output_preprocessing_config: Preprocessing applied to val outputs.

    Returns:
        ``(train_subset, val_subset)`` as ``torch.utils.data.Subset`` instances.
    """
    if validation_split <= 0:
        raise ValueError(f"The validation set is empty. {_EMPTY_SPLIT_HINT}")
    if validation_split >= 1:
        raise ValueError(f"The training set is empty. {_EMPTY_SPLIT_HINT}")

    ranges = _get_episode_sample_ranges(dataset)
    n_episodes = len(ranges)
    n_samples = len(dataset)
    if n_samples == 0 and n_episodes == 0:
        raise ValueError(
            f"The training and validation sets are both empty. {_EMPTY_SPLIT_HINT}"
        )
    if n_episodes < 2:
        raise ValueError(
            "Need at least 2 recordings to hold out a validation episode. "
            f"The dataset has {n_episodes} recording(s). {_EMPTY_SPLIT_HINT}"
        )

    train_episodes, val_episodes = _assign_episodes(
        n_episodes,
        validation_split,
        seed,
        _get_episode_robot_ids(dataset, n_episodes),
    )
    train_indices = _collect_sample_indices(train_episodes, ranges)
    val_indices = _collect_sample_indices(val_episodes, ranges)
    if not train_indices:
        raise ValueError(f"The training set is empty. {_EMPTY_SPLIT_HINT}")
    if not val_indices:
        raise ValueError(f"The validation set is empty. {_EMPTY_SPLIT_HINT}")

    logger.info(
        "Split %s recordings (%s samples) into %s train recordings "
        "(%s samples) and %s val recordings (%s samples)",
        n_episodes,
        n_samples,
        len(train_episodes),
        len(train_indices),
        len(val_episodes),
        len(val_indices),
    )

    train_dataset = Subset(dataset, train_indices)

    # Both subsets would otherwise share the train-configured dataset; give val
    # its own copy. Worker-side half only, matching what the dataset
    # constructor keeps. The trainer applies the device-side half of the
    # inference pipeline.
    val_base = copy(dataset)
    val_base.input_preprocessing_config = (
        inference_input_preprocessing_config.split_by_stage()[0]
    )
    val_base.output_preprocessing_config = (
        inference_output_preprocessing_config.split_by_stage()[0]
    )
    # The shallow copy would otherwise share a cache keyed on the train
    # pipeline while storing samples built with the inference one. Re-key it
    # against the preprocessing just assigned above.
    val_base.rebuild_sample_cache()
    val_dataset = Subset(val_base, val_indices)

    return train_dataset, val_dataset
