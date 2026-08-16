"""Dataset helpers for training/validation splits."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Subset, random_split

if TYPE_CHECKING:
    from neuracore.ml.datasets.pytorch_synchronized_dataset import (
        PytorchSynchronizedDataset,
    )
    from neuracore.ml.preprocessing.base import PreprocessingConfiguration


def split_train_val_datasets(
    dataset: PytorchSynchronizedDataset,
    train_size: int,
    val_size: int,
    seed: int,
    inference_input_preprocessing_config: PreprocessingConfiguration,
    inference_output_preprocessing_config: PreprocessingConfiguration,
) -> tuple[Subset, Subset]:
    """Split into train/val subsets; val uses inference preprocessing.

    ``dataset`` is expected to already carry train preprocessing. Both subsets
    from ``random_split`` share that dataset, so the val subset is rebased onto
    a shallow copy configured with inference preprocessing.

    Args:
        dataset: Full synchronized dataset with train preprocessing.
        train_size: Number of samples in the training subset.
        val_size: Number of samples in the validation subset.
        seed: RNG seed for the deterministic index shuffle.
        inference_input_preprocessing_config: Preprocessing applied to val inputs.
        inference_output_preprocessing_config: Preprocessing applied to val outputs.

    Returns:
        ``(train_subset, val_subset)`` as ``torch.utils.data.Subset`` instances.
    """
    if train_size + val_size != len(dataset):
        raise ValueError(
            f"train_size ({train_size}) + val_size ({val_size}) must equal "
            f"dataset length ({len(dataset)})."
        )
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    # Both subsets share the train-configured dataset; give val its own copy.
    val_base = copy(dataset)
    # Worker-side half only, matching what the dataset constructor keeps. The
    # trainer applies the device-side half of the inference pipeline.
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
    val_dataset = Subset(val_base, val_dataset.indices)

    return train_dataset, val_dataset
