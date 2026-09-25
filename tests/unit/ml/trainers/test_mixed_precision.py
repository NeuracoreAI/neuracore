"""Tests for bf16 mixed precision training."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from neuracore.ml import BatchedTrainingOutputs
from neuracore.ml.trainers.distributed_trainer import DistributedTrainer

MODULE = "neuracore.ml.trainers.distributed_trainer"


class _RecordingModel(nn.Module):
    """Model that records whether each forward pass ran under bf16 autocast."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3)
        self.autocast_seen: list[bool] = []

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        return [torch.optim.SGD(self.parameters(), lr=0.1)]

    def configure_schedulers(self, optimizers: list, steps: int) -> list:
        return []

    def training_step(self, batch: SimpleNamespace) -> BatchedTrainingOutputs:
        self.autocast_seen.append(torch.is_autocast_enabled("cpu"))
        loss = self.conv.weight.sum()
        return BatchedTrainingOutputs(losses={"loss": loss}, metrics={})


def _batch() -> SimpleNamespace:
    batch = SimpleNamespace(inputs={}, outputs={})
    batch.to = lambda device: batch
    return batch


def _trainer(tmp_path: Path, model: nn.Module, **kwargs) -> DistributedTrainer:
    training_logger = MagicMock()
    training_logger.supports_histograms = False
    storage_handler = MagicMock()
    storage_handler.log_to_cloud = True
    return DistributedTrainer(
        model=model,
        train_loader=[_batch(), _batch()],
        val_loader=[_batch()],
        training_logger=training_logger,
        storage_handler=storage_handler,
        output_dir=tmp_path,
        num_epochs=1,
        device=torch.device("cpu"),
        **kwargs,
    )


def test_mixed_precision_stays_off_without_native_bf16(tmp_path):
    """A device without native bf16 trains in float32 with its usual layout."""
    model = _RecordingModel()

    trainer = _trainer(tmp_path, model, mixed_precision=True)

    assert trainer.mixed_precision is False
    assert not model.conv.weight.is_contiguous(memory_format=torch.channels_last)


def test_mixed_precision_runs_training_and_validation_under_autocast(tmp_path):
    """Every forward pass runs in bf16 autocast on a channels_last model."""
    model = _RecordingModel()
    with (
        patch(f"{MODULE}._supports_native_bf16", return_value=True),
        patch(f"{MODULE}.MemoryMonitor"),
    ):
        trainer = _trainer(tmp_path, model, mixed_precision=True)
        trainer.train_epoch(epoch=1)
        trainer.validate(epoch=1)

    assert model.conv.weight.is_contiguous(memory_format=torch.channels_last)
    assert model.autocast_seen == [True, True, True]
