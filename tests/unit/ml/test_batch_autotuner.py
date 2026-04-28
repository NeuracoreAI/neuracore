"""Tests for batch size autotuner."""

import functools
import pickle
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from neuracore.ml import BatchedTrainingOutputs
from neuracore.ml.datasets.pytorch_synchronized_dataset import (
    PytorchSynchronizedDataset,
)
from neuracore.ml.trainers.batch_autotuner import (
    BatchSizeAutotuner,
    BatchSizeValidator,
    _probe_batch_size,
    find_optimal_batch_size,
    is_valid_batch_size,
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
        # Simulate transformer models (e.g. GemmaConfig) that are not picklable.
        self._non_picklable = lambda: None

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


def test_find_optimal_runs_probe_batch():
    train_dataset = DummyDataset(length=400)
    val_dataset = DummyDataset(length=100)
    device = torch.device("cuda:0")

    with patch("torch.cuda.is_available", return_value=True):
        autotuner = BatchSizeAutotuner(
            model_factory=functools.partial(DummyModel, device=device),
            device=device,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_dataloader_kwargs={},
            val_dataloader_kwargs={},
            min_batch_size=8,
            max_batch_size=512,
            num_iterations=2,
        )

    with (
        patch.object(
            BatchSizeValidator, "test_batch_size", return_value=True
        ) as mock_test,
        patch("torch.cuda.reset_peak_memory_stats", return_value=None),
        patch("torch.cuda.max_memory_allocated", return_value=1024**3),
    ):
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
    model_factory = functools.partial(DummyModel, device=device)

    def fake_random_split(dataset, lengths, generator=None):
        assert len(dataset) == 100
        assert lengths == [80, 20]
        return (DummyDataset(80), DummyDataset(20))

    mock_autotuner_instance = MagicMock()
    mock_autotuner_instance.find_optimal_batch_size.return_value = 4

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch(
            "neuracore.ml.trainers.batch_autotuner.random_split",
            side_effect=fake_random_split,
        ),
        patch(
            "neuracore.ml.trainers.batch_autotuner.BatchSizeAutotuner",
            return_value=mock_autotuner_instance,
        ) as mock_autotuner_cls,
    ):
        result = find_optimal_batch_size(cfg, model_factory, mock_dataset, device)

    assert result == 4
    mock_autotuner_cls.assert_called_once()
    kwargs = mock_autotuner_cls.call_args.kwargs
    assert kwargs["min_batch_size"] == 2
    assert kwargs["max_batch_size"] == 80  # clamp to train dataset length


def test_batch_size_validator_test_batch_size_success():
    """BatchSizeValidator returns True when the subprocess reports success."""
    train_dataset = DummyDataset(length=16)
    val_dataset = DummyDataset(length=16)
    device = torch.device("cuda:0")

    with patch("torch.cuda.is_available", return_value=True):
        validator = BatchSizeValidator(
            model_factory=functools.partial(DummyModel, device=device),
            device=device,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_dataloader_kwargs={},
            val_dataloader_kwargs={},
            num_iterations=2,
        )

    with patch.object(
        BatchSizeValidator, "_run_in_subprocess", return_value=True
    ) as mock_run:
        result = validator.test_batch_size(batch_size=4)

    assert result is True
    mock_run.assert_called_once_with(4)


def test_batch_size_validator_test_batch_size_returns_false_on_subprocess_failure():
    """BatchSizeValidator returns False when the subprocess reports failure."""
    train_dataset = DummyDataset(length=16)
    val_dataset = DummyDataset(length=16)
    device = torch.device("cuda:0")

    with patch("torch.cuda.is_available", return_value=True):
        validator = BatchSizeValidator(
            model_factory=functools.partial(DummyModel, device=device),
            device=device,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_dataloader_kwargs={},
            val_dataloader_kwargs={},
            num_iterations=2,
        )

    with patch.object(
        BatchSizeValidator, "_run_in_subprocess", return_value=False
    ) as mock_run:
        result = validator.test_batch_size(batch_size=4)

    assert result is False
    mock_run.assert_called_once_with(4)


def test_batch_size_validator_spawns_subprocess_and_returns_worker_result():
    """BatchSizeValidator spawns a subprocess and uses the queued result."""
    train_dataset = DummyDataset(length=16)
    val_dataset = DummyDataset(length=16)
    device = torch.device("cuda:0")

    with patch("torch.cuda.is_available", return_value=True):
        validator = BatchSizeValidator(
            model_factory=functools.partial(DummyModel, device=device),
            device=device,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_dataloader_kwargs={},
            val_dataloader_kwargs={},
            num_iterations=2,
        )

    fake_queue = MagicMock()
    fake_queue.get_nowait.return_value = ("ok", True)

    fake_proc = MagicMock()
    fake_proc.exitcode = 0
    fake_proc.is_alive.return_value = False

    fake_ctx = MagicMock()
    fake_ctx.Queue.return_value = fake_queue
    fake_ctx.Process.return_value = fake_proc

    with patch(
        "neuracore.ml.trainers.batch_autotuner.multiprocessing.get_context",
        return_value=fake_ctx,
    ) as mock_get_ctx:
        result = validator.test_batch_size(batch_size=4)

    assert result is True
    mock_get_ctx.assert_called_once_with("spawn")
    fake_ctx.Process.assert_called_once()
    fake_proc.start.assert_called_once()
    fake_proc.join.assert_called()


def test_batch_size_validator_treats_nonzero_exit_code_as_failure():
    """A subprocess that exits abnormally is treated as a batch-size failure."""
    train_dataset = DummyDataset(length=16)
    val_dataset = DummyDataset(length=16)
    device = torch.device("cuda:0")

    with patch("torch.cuda.is_available", return_value=True):
        validator = BatchSizeValidator(
            model_factory=functools.partial(DummyModel, device=device),
            device=device,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_dataloader_kwargs={},
            val_dataloader_kwargs={},
            num_iterations=2,
        )

    fake_queue = MagicMock()
    fake_proc = MagicMock()
    fake_proc.exitcode = -9  # e.g. killed by SIGKILL (OOM killer)
    fake_proc.is_alive.return_value = False

    fake_ctx = MagicMock()
    fake_ctx.Queue.return_value = fake_queue
    fake_ctx.Process.return_value = fake_proc

    with patch(
        "neuracore.ml.trainers.batch_autotuner.multiprocessing.get_context",
        return_value=fake_ctx,
    ):
        result = validator.test_batch_size(batch_size=4)

    assert result is False
    fake_queue.get_nowait.assert_not_called()


def test_batch_size_validator_raises_on_worker_failure_result():
    """Unexpected worker failures should propagate as RuntimeError."""
    train_dataset = DummyDataset(length=16)
    val_dataset = DummyDataset(length=16)
    device = torch.device("cuda:0")

    with patch("torch.cuda.is_available", return_value=True):
        validator = BatchSizeValidator(
            model_factory=functools.partial(DummyModel, device=device),
            device=device,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_dataloader_kwargs={},
            val_dataloader_kwargs={},
            num_iterations=2,
        )

    fake_queue = MagicMock()
    fake_queue.get_nowait.return_value = ("fail", "ValueError('shape mismatch')")

    fake_proc = MagicMock()
    fake_proc.exitcode = 0
    fake_proc.is_alive.return_value = False

    fake_ctx = MagicMock()
    fake_ctx.Queue.return_value = fake_queue
    fake_ctx.Process.return_value = fake_proc

    with patch(
        "neuracore.ml.trainers.batch_autotuner.multiprocessing.get_context",
        return_value=fake_ctx,
    ):
        with pytest.raises(
            RuntimeError, match="Unexpected failure while probing batch size 4"
        ):
            validator.test_batch_size(batch_size=4)


def test_batch_size_validator_returns_false_on_worker_oom_failure():
    """Worker-reported OOM remains a normal batch-size failure signal."""
    train_dataset = DummyDataset(length=16)
    val_dataset = DummyDataset(length=16)
    device = torch.device("cuda:0")

    with patch("torch.cuda.is_available", return_value=True):
        validator = BatchSizeValidator(
            model_factory=functools.partial(DummyModel, device=device),
            device=device,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_dataloader_kwargs={},
            val_dataloader_kwargs={},
            num_iterations=2,
        )

    fake_queue = MagicMock()
    fake_queue.get_nowait.return_value = ("ok", False)

    fake_proc = MagicMock()
    fake_proc.exitcode = 0
    fake_proc.is_alive.return_value = False

    fake_ctx = MagicMock()
    fake_ctx.Queue.return_value = fake_queue
    fake_ctx.Process.return_value = fake_proc

    with patch(
        "neuracore.ml.trainers.batch_autotuner.multiprocessing.get_context",
        return_value=fake_ctx,
    ):
        result = validator.test_batch_size(batch_size=4)

    assert result is False


def test_batch_size_validator_requires_cuda_device():
    """BatchSizeValidator rejects non-CUDA devices."""
    train_dataset = DummyDataset(length=16)
    val_dataset = DummyDataset(length=16)
    device = torch.device("cpu")

    with (
        patch("torch.cuda.is_available", return_value=False),
        pytest.raises(ValueError, match="only supported on GPUs"),
    ):
        BatchSizeValidator(
            model_factory=functools.partial(DummyModel, device=device),
            device=device,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_dataloader_kwargs={},
            val_dataloader_kwargs={},
            num_iterations=2,
        )


def test_probe_batch_size_returns_false_on_torch_cuda_oom():
    """CUDA OOM is treated as an expected batch-size failure."""
    model = MagicMock()
    model.configure_optimizers.return_value = []

    with (
        patch("neuracore.ml.trainers.batch_autotuner.MemoryMonitor"),
        patch("neuracore.ml.trainers.batch_autotuner.DataLoader"),
        patch("neuracore.ml.trainers.batch_autotuner._train_probe") as mock_train_probe,
        patch("torch.cuda.reset_peak_memory_stats"),
        patch("torch.cuda.max_memory_allocated", return_value=0),
        patch("torch.cuda.is_available", return_value=False),
    ):
        mock_train_probe.side_effect = torch.cuda.OutOfMemoryError("OOM")
        result = _probe_batch_size(
            model=model,
            train_dataset=DummyDataset(length=8),
            val_dataset=DummyDataset(length=8),
            train_dataloader_kwargs={},
            val_dataloader_kwargs={},
            num_iterations=1,
            batch_size=4,
            device=torch.device("cuda:0"),
        )

    assert result is False


def test_probe_batch_size_raises_on_non_oom_runtime_error():
    """Runtime errors unrelated to OOM should abort probing."""
    model = MagicMock()
    model.configure_optimizers.return_value = []

    with (
        patch("neuracore.ml.trainers.batch_autotuner.MemoryMonitor"),
        patch("neuracore.ml.trainers.batch_autotuner.DataLoader"),
        patch("neuracore.ml.trainers.batch_autotuner._train_probe") as mock_train_probe,
        patch("torch.cuda.reset_peak_memory_stats"),
        patch("torch.cuda.is_available", return_value=False),
    ):
        mock_train_probe.side_effect = RuntimeError("shape mismatch in model head")
        with pytest.raises(RuntimeError, match="shape mismatch in model head"):
            _probe_batch_size(
                model=model,
                train_dataset=DummyDataset(length=8),
                val_dataset=DummyDataset(length=8),
                train_dataloader_kwargs={},
                val_dataloader_kwargs={},
                num_iterations=1,
                batch_size=4,
                device=torch.device("cuda:0"),
            )


def test_probe_batch_size_raises_on_generic_exception():
    """Non-runtime unexpected exceptions should also abort probing."""
    model = MagicMock()
    model.configure_optimizers.return_value = []

    with (
        patch("neuracore.ml.trainers.batch_autotuner.MemoryMonitor"),
        patch("neuracore.ml.trainers.batch_autotuner.DataLoader"),
        patch("neuracore.ml.trainers.batch_autotuner._train_probe") as mock_train_probe,
        patch("torch.cuda.reset_peak_memory_stats"),
        patch("torch.cuda.is_available", return_value=False),
    ):
        mock_train_probe.side_effect = ValueError("bad data shape")
        with pytest.raises(ValueError, match="bad data shape"):
            _probe_batch_size(
                model=model,
                train_dataset=DummyDataset(length=8),
                val_dataset=DummyDataset(length=8),
                train_dataloader_kwargs={},
                val_dataloader_kwargs={},
                num_iterations=1,
                batch_size=4,
                device=torch.device("cuda:0"),
            )


def test_batch_size_validator_handles_non_picklable_model_attributes():
    """test_batch_size succeeds even when model instances have non-picklable attributes.

    Simulates a transformer model where instance attributes (e.g. GemmaConfig)
    are not picklable. The subprocess receives a picklable factory callable rather
    than a model instance, so the pickling error never occurs.
    """
    train_dataset = DummyDataset(length=16)
    val_dataset = DummyDataset(length=16)
    device = torch.device("cuda:0")

    # DummyModel has self._non_picklable = lambda: None — instances are not picklable.
    # However, functools.partial(DummyModel, device=device) is picklable because
    # it holds only the class reference and a plain torch.device, never an instance.
    model_factory = functools.partial(DummyModel, device=device)

    with patch("torch.cuda.is_available", return_value=True):
        validator = BatchSizeValidator(
            model_factory=model_factory,
            device=device,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_dataloader_kwargs={},
            val_dataloader_kwargs={},
            num_iterations=2,
        )

    fake_queue = MagicMock()
    fake_queue.get_nowait.return_value = ("ok", True)
    fake_proc = MagicMock()
    fake_proc.exitcode = 0
    fake_proc.is_alive.return_value = False

    fake_ctx = MagicMock()
    fake_ctx.Queue.return_value = fake_queue
    fake_ctx.Process.return_value = fake_proc

    def start_that_pickles_args():
        # Reproduce what spawn does: pickle-serialize every arg before sending to
        # the subprocess. Skip args[0] (the result_queue IPC object) since it is
        # always an OS primitive, not a plain Python value. Check all remaining
        # args: none of them should be a model instance.
        for arg in fake_ctx.Process.call_args.kwargs["args"][1:]:
            pickle.dumps(arg)

    fake_proc.start.side_effect = start_that_pickles_args

    with patch(
        "neuracore.ml.trainers.batch_autotuner.multiprocessing.get_context",
        return_value=fake_ctx,
    ):
        result = validator.test_batch_size(batch_size=4)

    assert result is True


def test_is_valid_batch_size_clamps_when_exceeding_train_dataset_size():
    """is_valid_batch_size clamps oversized batch_size to train dataset length."""
    cfg = OmegaConf.create({
        "validation_split": 0.2,
        "seed": 42,
        "num_train_workers": 0,
        "num_val_workers": 0,
    })

    mock_dataset = Mock(spec=PytorchSynchronizedDataset)
    mock_dataset.__len__ = Mock(return_value=100)
    mock_dataset.collate_fn = lambda x: x

    device = torch.device("cuda:0")
    model_factory = functools.partial(DummyModel, device=device)

    def fake_random_split(dataset, lengths, generator=None):
        assert lengths == [80, 20]
        return (DummyDataset(80), DummyDataset(20))

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch(
            "neuracore.ml.trainers.batch_autotuner.random_split",
            side_effect=fake_random_split,
        ),
        patch(
            "neuracore.ml.trainers.batch_autotuner.BatchSizeValidator.test_batch_size",
            return_value=True,
        ) as mock_test_batch_size,
    ):
        result = is_valid_batch_size(
            cfg=cfg,
            model_factory=model_factory,
            dataset=mock_dataset,
            batch_size=256,
            device=device,
        )

    assert result is True
    mock_test_batch_size.assert_called_once_with(80)
