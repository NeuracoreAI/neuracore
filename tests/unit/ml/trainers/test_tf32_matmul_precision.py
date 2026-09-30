"""Tests for the float32 matrix multiply precision the trainer sets."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from neuracore.ml.trainers.distributed_trainer import DistributedTrainer


@pytest.fixture
def restore_matmul_precision():
    """Restore the global float32 matmul precision a test changes."""
    precision = torch.get_float32_matmul_precision()
    yield
    torch.set_float32_matmul_precision(precision)


def _trainer(tmp_path: Path) -> DistributedTrainer:
    model = nn.Linear(2, 2)
    model.configure_optimizers = lambda: [torch.optim.SGD(model.parameters(), lr=0.1)]
    model.configure_schedulers = lambda optimizers, steps: []
    return DistributedTrainer(
        model=model,
        train_loader=[MagicMock()],
        val_loader=[MagicMock()],
        training_logger=MagicMock(),
        storage_handler=MagicMock(),
        output_dir=tmp_path,
        num_epochs=1,
        device=torch.device("cpu"),
    )


def test_trainer_enables_tf32_matmul_precision(tmp_path, restore_matmul_precision):
    """Set the float32 matmul precision to high so TF32 GPUs use TF32."""
    torch.set_float32_matmul_precision("highest")

    _trainer(tmp_path)

    assert torch.get_float32_matmul_precision() == "high"
