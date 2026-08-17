"""Tests for when weight and gradient histograms are collected."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from neuracore.ml.trainers.distributed_trainer import DistributedTrainer


def _trainer(tmp_path: Path, **kwargs) -> DistributedTrainer:
    model = nn.Linear(2, 2)
    model.configure_optimizers = lambda: [torch.optim.SGD(model.parameters(), lr=0.1)]
    model.configure_schedulers = lambda optimizers, steps: []

    storage_handler = MagicMock()
    storage_handler.log_to_cloud = kwargs.pop("log_to_cloud", False)
    training_logger = MagicMock()
    training_logger.supports_histograms = kwargs.pop("supports_histograms", True)

    loader = MagicMock()
    loader.__len__ = lambda _self: 10
    loader.batch_size = 4

    return DistributedTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        training_logger=training_logger,
        storage_handler=storage_handler,
        output_dir=tmp_path,
        num_epochs=1,
        device=torch.device("cpu"),
        **kwargs,
    )


@pytest.mark.parametrize(
    "kwargs, expected, reason",
    [
        ({}, True, "collected alongside the other step metrics by default"),
        (
            {"supports_histograms": False},
            False,
            "backend discards them, so building them is pure waste",
        ),
        (
            {"rank": 1},
            False,
            "non-zero ranks would race into one TensorBoard directory",
        ),
    ],
)
def test_histograms_are_only_collected_when_they_are_wanted(
    tmp_path, kwargs, expected, reason
):
    assert _trainer(tmp_path, **kwargs)._histograms_enabled is expected, reason


def test_disabled_histograms_do_not_walk_the_model(tmp_path):
    """The cost is iterating every parameter, so the guard must precede it."""
    trainer = _trainer(tmp_path, supports_histograms=False)

    trainer._log_gradients(step=0)
    trainer._log_weights(step=0)

    trainer.training_logger.log_histogram.assert_not_called()


def test_enabled_histograms_reach_the_logger(tmp_path):
    trainer = _trainer(tmp_path)

    trainer._log_weights(step=0)

    assert trainer.training_logger.log_histogram.called


def test_progress_bar_is_off_when_logs_go_to_the_cloud(tmp_path):
    """Nobody reads the bar on a cloud run, and rendering it costs a sync."""
    assert _trainer(tmp_path, log_to_cloud=True)._pbar_enabled is False
    assert _trainer(tmp_path, log_to_cloud=False)._pbar_enabled is True
    assert _trainer(tmp_path, rank=1)._pbar_enabled is False
