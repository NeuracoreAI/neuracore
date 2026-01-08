"""Tests for batch size autotuner."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from neuracore.ml import BatchedTrainingOutputs
from neuracore.ml.trainers.batch_autotuner import BatchSizeAutotuner


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
    dataset = DummyDataset(length=16)
    device = torch.device("cuda:0")

    with patch("torch.cuda.is_available", return_value=True):
        model = DummyModel(device=device)
        autotuner = BatchSizeAutotuner(
            dataset=dataset,
            model=model,
            model_kwargs={},
            min_batch_size=2,
            max_batch_size=4,
            num_iterations=1,
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
    dataset = DummyDataset(length=400)
    device = torch.device("cuda:0")

    with patch("torch.cuda.is_available", return_value=True):
        model = DummyModel(device=device)
        autotuner = BatchSizeAutotuner(
            dataset=dataset,
            model=model,
            model_kwargs={},
            min_batch_size=8,
            max_batch_size=512,
            num_iterations=1,
        )

    with patch.object(
        BatchSizeAutotuner, "_test_batch_size", return_value=True
    ) as mock_test:
        autotuner.find_optimal_batch_size()

    # After reduction, the final validation should use the reduced batch size.
    assert mock_test.call_args_list[-1][0][0] == int(512 * 0.7)
