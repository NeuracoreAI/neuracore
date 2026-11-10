"""Dataset helpers for training/validation splits."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from copy import copy
from typing import TYPE_CHECKING, cast

from torch.utils.data import Subset

from neuracore.core.data.synced_recording import SynchronizedRecording

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
) -> list[str]:
    """Return the robot id of each episode's synchronized recording."""
    synchronized_dataset = dataset.synchronized_dataset
    robot_ids: list[str] = []
    for episode_idx in range(n_episodes):
        synced_recording = cast(
            SynchronizedRecording, synchronized_dataset[episode_idx]
        )
        robot_ids.append(synced_recording.robot_id)
    return robot_ids


def _count_val_episodes(n_episodes: int, validation_split: float) -> int:
    """Hold out at least one episode and leave at least one for training."""
    n_val = int(round(n_episodes * validation_split))
    return min(max(n_val, 1), n_episodes - 1)


def _assign_episodes(
    validation_split: float,
    seed: int,
    robot_ids: list[str],
) -> tuple[list[int], list[int]]:
    """Return ``(train_episode_indices, val_episode_indices)``.

    Hold out a random subset of each robot's episodes so a multi-recording
    embodiment is not held out entirely. Single-recording robots stay in
    train. If no robot has two episodes to split, raise ValueError.
    """
    rng = random.Random(seed)

    episode_groups_by_robot: dict[str, list[int]] = defaultdict(list)
    for episode_idx, robot_id in enumerate(robot_ids):
        episode_groups_by_robot[robot_id].append(episode_idx)

    train_episodes: list[int] = []
    val_episodes: list[int] = []
    for robot_id in sorted(episode_groups_by_robot):
        members = episode_groups_by_robot[robot_id]
        if len(members) < 2:
            train_episodes.extend(members)
            logger.warning(
                "Robot %s has only one episode, so it is not held out for validation.",
                robot_id,
            )
            continue
        n_val = _count_val_episodes(len(members), validation_split)
        held_out = rng.sample(members, n_val)
        train_episodes.extend(episode for episode in members if episode not in held_out)
        val_episodes.extend(held_out)

    if not val_episodes:
        counts = ", ".join(
            f"{robot_id!r} has {len(members)} episode(s)"
            for robot_id, members in sorted(episode_groups_by_robot.items())
        )
        raise ValueError(
            "Need at least 2 episodes for a robot to hold one out for "
            f"validation. {counts}."
        )

    return train_episodes, val_episodes


def _collect_sample_indices(
    episode_indices: list[int], episode_ranges: list[range]
) -> list[int]:
    """Collect every sample index owned by the given episodes."""
    indices: list[int] = []
    for episode_idx in episode_indices:
        indices.extend(episode_ranges[episode_idx])
    return indices


def _collect_rollout_indices(
    val_episodes: list[int],
    episode_ranges: list[range],
    num_rollout_episodes: int,
    seed: int,
    stride: int,
) -> list[int]:
    """Collect the points a robot would see running a few val episodes.

    Picks ``num_rollout_episodes`` val episodes at random. In each, keeps the
    samples at timesteps ``0, stride, 2 * stride, ...``: the inputs a policy
    gets if it starts at timestep 0 and executes ``stride`` actions before
    reading the next input.
    """
    if stride <= 0:
        raise ValueError(f"Rollout stride must be positive, got {stride}.")
    if num_rollout_episodes > len(val_episodes):
        logger.warning(
            "Requested %s validation rollout episode(s), but the validation split "
            "only has %s episode(s). Using all of them.",
            num_rollout_episodes,
            len(val_episodes),
        )
        num_rollout_episodes = len(val_episodes)
    rollout_episodes = random.Random(seed).sample(val_episodes, num_rollout_episodes)
    indices: list[int] = []
    for episode_idx in rollout_episodes:
        indices.extend(episode_ranges[episode_idx][::stride])
    logger.info(
        "Selected %s validation rollout episode(s),"
        " with a total of %s points and a stride of %s.",
        num_rollout_episodes,
        len(indices),
        stride,
    )
    return indices


def split_train_val_datasets(
    dataset: PytorchSynchronizedDataset,
    validation_split: float,
    seed: int,
    inference_input_preprocessing_config: PreprocessingConfiguration,
    inference_output_preprocessing_config: PreprocessingConfiguration,
    num_rollout_episodes: int = 0,
) -> tuple[Subset, Subset, Subset | None]:
    """Split by whole recordings; val uses inference preprocessing.

    ``validation_split`` is a fraction of episodes, not frames. Every sample
    from a recording goes to exactly one of train or val, so a prediction
    horizon cannot straddle the split.

    ``dataset`` is expected to already carry train preprocessing. The val
    subset is rebased onto a shallow copy configured with inference
    preprocessing.

    The rollout subset holds a few val episodes, sampled every
    ``output_prediction_horizon`` timesteps, for validation rollouts. Those
    episodes stay in val as well.

    Args:
        dataset: Full synchronized dataset with train preprocessing.
        validation_split: Fraction of recordings to hold out for validation.
            Must be strictly between 0 and 1.
        seed: RNG seed for the deterministic episode shuffle.
        inference_input_preprocessing_config: Preprocessing applied to val inputs.
        inference_output_preprocessing_config: Preprocessing applied to val outputs.
        num_rollout_episodes: How many val episodes to roll out. 0 turns
            validation rollouts off.

    Returns:
        ``(train_subset, val_subset, rollout_subset)`` as
        ``torch.utils.data.Subset`` instances. ``rollout_subset`` shares val's
        dataset, and is None when ``num_rollout_episodes`` is 0.
    """
    if not 0 < validation_split < 1:
        raise ValueError(
            f"validation_split is {validation_split}; "
            "it must be strictly between 0 and 1."
        )
    if num_rollout_episodes < 0:
        raise ValueError(
            f"num_rollout_episodes must be at least 0, got {num_rollout_episodes}."
        )

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

    rollout_dataset = None
    if num_rollout_episodes > 0:
        rollout_indices = _collect_rollout_indices(
            val_episodes,
            ranges,
            num_rollout_episodes,
            seed,
            stride=dataset.output_prediction_horizon,
        )
        rollout_dataset = Subset(val_base, rollout_indices)

    return train_dataset, val_dataset, rollout_dataset
