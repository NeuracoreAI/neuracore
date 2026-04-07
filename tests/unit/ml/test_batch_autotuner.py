"""Tests for batch size autotuner."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, Mock, patch

import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from neuracore.ml import BatchedTrainingOutputs, NeuracoreModel
from neuracore.ml.datasets.pytorch_synchronized_dataset import (
    PytorchSynchronizedDataset,
)
from neuracore.ml.trainers.batch_autotuner import (
    BatchSizeAutotuner,
    find_optimal_batch_size,
)


class DummyDataset(Dataset):
    """Simple dataset that returns a tensor sample."""

    def __init__(self, length: int):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # DataLoader will stack these into a batch; len(batch) will reflect batch size.
        return torch.zeros(2)


class DummyModel(torch.nn.Module):
    """Minimal model to exercise autotuner logic."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        return batch

    def training_step(self, batch):
        # Produce a loss that keeps a grad path to the parameter for backward().
        loss = (self.weight * batch.sum()).mean()
        return BatchedTrainingOutputs(
            losses={"loss": loss},
            metrics={},
        )

    def configure_optimizers(self):
        return [torch.optim.SGD(self.parameters(), lr=0.1)]


def test_autotuner_records_gpu_usage():
    train_dataset = DummyDataset(length=16)
    val_dataset = DummyDataset(length=16)
    device = torch.device("cuda:0")

    with patch("torch.cuda.is_available", return_value=True):
        model = cast(NeuracoreModel, DummyModel(device=device))
        autotuner = BatchSizeAutotuner(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_dataloader_kwargs={},
            val_dataloader_kwargs={},
            min_batch_size=2,
            max_batch_size=4,
            num_iterations=2,
        )

    # Patch CUDA helpers and MemoryMonitor behavior
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.synchronize", return_value=None),
        patch("torch.cuda.reset_peak_memory_stats", return_value=None),
        patch("torch.cuda.max_memory_allocated", return_value=1024**3),
        patch("torch.cuda.memory_reserved", return_value=1024**3),  # 1 GB
        patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(total_memory=2 * 1024**3),
        ),
        patch("neuracore.ml.utils.memory_monitor.MemoryMonitor.check_memory"),
        patch("torch.Tensor.to", lambda self, device=None: self),
    ):
        success = autotuner._test_batch_size(batch_size=2)

    assert success is True
    assert autotuner.last_peak_memory_gb == 1.0


def test_find_optimal_runs_probe_batch():
    train_dataset = DummyDataset(length=400)
    val_dataset = DummyDataset(length=100)
    device = torch.device("cuda:0")

    with patch("torch.cuda.is_available", return_value=True):
        model = cast(NeuracoreModel, DummyModel(device=device))
        autotuner = BatchSizeAutotuner(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_dataloader_kwargs={},
            val_dataloader_kwargs={},
            min_batch_size=8,
            max_batch_size=512,
            num_iterations=2,
        )

    with patch.object(
        BatchSizeAutotuner, "_test_batch_size", return_value=True
    ) as mock_test:
        autotuner.find_optimal_batch_size()

    # After reduction, the final validation probes int(safety_factor * optimal).
    # With min=8, max=512, the search stops before mid=512; the last successful
    # binary-search probe is 497 due to granularity=25, therefore int(497 * 0.7).
    assert mock_test.call_args_list[-1][0][0] == int(497 * 0.7)


def test_find_optimal_batch_size_passes_default_min_max_to_batch_size_autotuner():
    """When cfg omits min/max batch size, default range is [2, train_len]."""
    cfg = OmegaConf.create({
        "validation_split": 0.2,
        "seed": 42,
        "num_train_workers": 0,
        "num_val_workers": 0,
    })
    assert "min_batch_size" not in cfg
    assert "max_batch_size" not in cfg

    mock_dataset = Mock(spec=PytorchSynchronizedDataset)
    mock_dataset.__len__ = Mock(return_value=100)
    mock_dataset.collate_fn = lambda x: x

    device = torch.device("cuda:0")
    model = cast(NeuracoreModel, DummyModel(device=device))

    def fake_random_split(dataset, lengths, generator=None):
        assert len(dataset) == 100
        assert lengths == [80, 20]
        return (DummyDataset(80), DummyDataset(20))

    mock_autotuner_instance = MagicMock()
    mock_autotuner_instance.find_optimal_batch_size.return_value = 4

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch.object(model, "to", return_value=model),
        patch(
            "neuracore.ml.trainers.batch_autotuner.random_split",
            side_effect=fake_random_split,
        ),
        patch(
            "neuracore.ml.trainers.batch_autotuner.BatchSizeAutotuner",
            return_value=mock_autotuner_instance,
        ) as mock_autotuner_cls,
    ):
        result = find_optimal_batch_size(cfg, model, mock_dataset, device)

    assert result == 4
    mock_autotuner_cls.assert_called_once()
    kwargs = mock_autotuner_cls.call_args.kwargs
    assert kwargs["min_batch_size"] == 2
    assert kwargs["max_batch_size"] == 80  # clamp to train dataset length
