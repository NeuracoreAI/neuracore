"""Tests for TF32 matrix multiplies and the channels_last model layout."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from neuracore.ml import BatchedTrainingOutputs
from neuracore.ml.trainers.distributed_trainer import DistributedTrainer

MODULE = "neuracore.ml.trainers.distributed_trainer"


class _ConvModel(nn.Module):
    """Minimal model with one convolution."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3)

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        return [torch.optim.SGD(self.parameters(), lr=0.1)]

    def configure_schedulers(self, optimizers: list, steps: int) -> list:
        return []

    def training_step(self, batch: SimpleNamespace) -> BatchedTrainingOutputs:
        return BatchedTrainingOutputs(
            losses={"loss": self.conv.weight.sum()}, metrics={}
        )


def _trainer(tmp_path: Path, model: nn.Module) -> DistributedTrainer:
    training_logger = MagicMock()
    training_logger.supports_histograms = False
    storage_handler = MagicMock()
    storage_handler.log_to_cloud = True
    return DistributedTrainer(
        model=model,
        train_loader=[SimpleNamespace()],
        val_loader=[SimpleNamespace()],
        training_logger=training_logger,
        storage_handler=storage_handler,
        output_dir=tmp_path,
        num_epochs=1,
        device=torch.device("cpu"),
    )


@pytest.fixture
def restore_tf32_flags():
    """Restore the global TF32 flags a test changes."""
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    yield
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn


def test_device_without_tf32_keeps_the_contiguous_layout(tmp_path):
    """A device without TF32 support trains with its usual layout."""
    model = _ConvModel()

    _trainer(tmp_path, model)

    assert not model.conv.weight.is_contiguous(memory_format=torch.channels_last)


def test_tf32_device_enables_tf32_and_channels_last(tmp_path, restore_tf32_flags):
    """A TF32 capable device gets TF32 matrix multiplies and a channels_last model."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = _ConvModel()

    with patch(f"{MODULE}._supports_tf32", return_value=True):
        _trainer(tmp_path, model)

    assert torch.backends.cuda.matmul.allow_tf32
    assert torch.backends.cudnn.allow_tf32
    assert model.conv.weight.is_contiguous(memory_format=torch.channels_last)
