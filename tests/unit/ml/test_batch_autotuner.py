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
    BatchProbeResult,
    BatchSizeAutotuner,
    BatchSizeValidator,
    _BatchSizeEstimationError,
    _estimate_max_batch_size,
    _probe_batch_size,
    find_optimal_batch_size,
    is_valid_batch_size,
)
from neuracore.ml.utils.memory_monitor import OutOfMemoryError

GB = 1024**3


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


def test_find_optimal_batch_size_passes_default_min_max_to_batch_size_autotuner():
    """When cfg omits min/max batch size, default range is [2, train_len].

    Minimum defaults to 2, not 1: a single-sample batch breaks BatchNorm layers.
    """
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
    mock_autotuner_instance.estimate_optimal_batch_size.return_value = 4

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


def test_find_optimal_batch_size_default_max_allows():
    """With no cfg max, the default ceiling is length of dataset, so it can exceed 512.

    A fixed constant cap would wrongly limit large-VRAM devices; the dataset size
    is the only true hard ceiling, and the memory-aware estimator finds the real
    VRAM limit below it.
    """
    cfg = OmegaConf.create({
        "validation_split": 0.2,
        "seed": 42,
        "num_train_workers": 0,
        "num_val_workers": 0,
    })
    assert "max_batch_size" not in cfg

    mock_dataset = Mock(spec=PytorchSynchronizedDataset)
    mock_dataset.__len__ = Mock(return_value=10_000)  # train_len 8000 > 512
    mock_dataset.collate_fn = lambda x: x

    device = torch.device("cuda:0")
    model_factory = functools.partial(DummyModel, device=device)

    def fake_random_split(dataset, lengths, generator=None):
        return (DummyDataset(lengths[0]), DummyDataset(lengths[1]))

    mock_autotuner_instance = MagicMock()
    mock_autotuner_instance.estimate_optimal_batch_size.return_value = 4

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
        find_optimal_batch_size(cfg, model_factory, mock_dataset, device)

    kwargs = mock_autotuner_cls.call_args.kwargs
    assert kwargs["max_batch_size"] == 8000  # train_len, well above 512


def test_batch_size_validator_test_batch_size_success():
    """BatchSizeValidator returns True when the subprocess reports success."""
    train_dataset = DummyDataset(length=16)
    device = torch.device("cuda:0")

    with patch("torch.cuda.is_available", return_value=True):
        validator = BatchSizeValidator(
            model_factory=functools.partial(DummyModel, device=device),
            device=device,
            train_dataset=train_dataset,
            train_dataloader_kwargs={},
            num_iterations=2,
        )

    with patch.object(
        BatchSizeValidator,
        "_run_in_subprocess",
        return_value=BatchProbeResult(fitted=True),
    ) as mock_run:
        result = validator.test_batch_size(batch_size=4)

    assert result is True
    mock_run.assert_called_once_with(4)


def test_batch_size_validator_test_batch_size_returns_false_on_subprocess_failure():
    """BatchSizeValidator returns False when the subprocess reports failure."""
    train_dataset = DummyDataset(length=16)
    device = torch.device("cuda:0")

    with patch("torch.cuda.is_available", return_value=True):
        validator = BatchSizeValidator(
            model_factory=functools.partial(DummyModel, device=device),
            device=device,
            train_dataset=train_dataset,
            train_dataloader_kwargs={},
            num_iterations=2,
        )

    with patch.object(
        BatchSizeValidator,
        "_run_in_subprocess",
        return_value=BatchProbeResult(fitted=False),
    ) as mock_run:
        result = validator.test_batch_size(batch_size=4)

    assert result is False
    mock_run.assert_called_once_with(4)


def test_batch_size_validator_spawns_subprocess_and_returns_worker_result():
    """BatchSizeValidator spawns a subprocess and uses the queued result."""
    train_dataset = DummyDataset(length=16)
    device = torch.device("cuda:0")

    with patch("torch.cuda.is_available", return_value=True):
        validator = BatchSizeValidator(
            model_factory=functools.partial(DummyModel, device=device),
            device=device,
            train_dataset=train_dataset,
            train_dataloader_kwargs={},
            num_iterations=2,
        )

    fake_queue = MagicMock()
    fake_queue.get_nowait.return_value = ("ok", BatchProbeResult(fitted=True))

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
    device = torch.device("cuda:0")

    with patch("torch.cuda.is_available", return_value=True):
        validator = BatchSizeValidator(
            model_factory=functools.partial(DummyModel, device=device),
            device=device,
            train_dataset=train_dataset,
            train_dataloader_kwargs={},
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
    device = torch.device("cuda:0")

    with patch("torch.cuda.is_available", return_value=True):
        validator = BatchSizeValidator(
            model_factory=functools.partial(DummyModel, device=device),
            device=device,
            train_dataset=train_dataset,
            train_dataloader_kwargs={},
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
    device = torch.device("cuda:0")

    with patch("torch.cuda.is_available", return_value=True):
        validator = BatchSizeValidator(
            model_factory=functools.partial(DummyModel, device=device),
            device=device,
            train_dataset=train_dataset,
            train_dataloader_kwargs={},
            num_iterations=2,
        )

    fake_queue = MagicMock()
    fake_queue.get_nowait.return_value = ("ok", BatchProbeResult(fitted=False))

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
    device = torch.device("cpu")

    with (
        patch("torch.cuda.is_available", return_value=False),
        pytest.raises(ValueError, match="only supported on GPUs"),
    ):
        BatchSizeValidator(
            model_factory=functools.partial(DummyModel, device=device),
            device=device,
            train_dataset=train_dataset,
            train_dataloader_kwargs={},
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
            train_dataloader_kwargs={},
            num_iterations=1,
            batch_size=4,
            device=torch.device("cuda:0"),
        )

    assert isinstance(result, BatchProbeResult)
    assert result.fitted is False


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
                train_dataloader_kwargs={},
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
                train_dataloader_kwargs={},
                num_iterations=1,
                batch_size=4,
                device=torch.device("cuda:0"),
            )


def test_probe_batch_size_gives_actionable_error_on_batchnorm_single_sample():
    """A BatchNorm 'more than 1 value per channel' error becomes an actionable
    message telling the user to raise min_batch_size."""
    model = MagicMock()
    model.configure_optimizers.return_value = []

    with (
        patch("neuracore.ml.trainers.batch_autotuner.MemoryMonitor"),
        patch("neuracore.ml.trainers.batch_autotuner.DataLoader"),
        patch("neuracore.ml.trainers.batch_autotuner._train_probe") as mock_train_probe,
        patch("torch.cuda.reset_peak_memory_stats"),
        patch("torch.cuda.is_available", return_value=False),
    ):
        mock_train_probe.side_effect = ValueError(
            "Expected more than 1 value per channel when training, got input size "
            "torch.Size([1, 16])"
        )
        with pytest.raises(ValueError, match="min_batch_size >= 2"):
            _probe_batch_size(
                model=model,
                train_dataset=DummyDataset(length=8),
                train_dataloader_kwargs={},
                num_iterations=1,
                batch_size=1,
                device=torch.device("cuda:0"),
            )


def test_batch_size_validator_handles_non_picklable_model_attributes():
    """test_batch_size succeeds even when model instances have non-picklable attributes.

    Simulates a transformer model where instance attributes (e.g. GemmaConfig)
    are not picklable. The subprocess receives a picklable factory callable rather
    than a model instance, so the pickling error never occurs.
    """
    train_dataset = DummyDataset(length=16)
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
            train_dataloader_kwargs={},
            num_iterations=2,
        )

    fake_queue = MagicMock()
    fake_queue.get_nowait.return_value = ("ok", BatchProbeResult(fitted=True))
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


def test_batch_probe_result_defaults():
    """fitted result defaults memory fields to zero and is picklable for spawn."""
    result = BatchProbeResult(fitted=True)
    assert result.peak_reserved_bytes == 0
    assert result.total_bytes == 0
    # Sent across a spawn Queue, so it must round-trip through pickle.
    assert pickle.loads(pickle.dumps(result)) == result


def test_measure_batch_size_returns_probe_result():
    """probe_batch_size surfaces the BatchProbeResult from the subprocess."""
    device = torch.device("cuda:0")
    with patch("torch.cuda.is_available", return_value=True):
        validator = BatchSizeValidator(
            model_factory=functools.partial(DummyModel, device=device),
            device=device,
            train_dataset=DummyDataset(length=64),
            train_dataloader_kwargs={},
        )

    expected = BatchProbeResult(fitted=True, peak_reserved_bytes=5, total_bytes=10)
    with patch.object(
        BatchSizeValidator, "_run_in_subprocess", return_value=expected
    ) as mock_run:
        result = validator.probe_batch_size(16)
        assert validator.test_batch_size(16) is True

    assert result == expected
    assert mock_run.call_args_list[0][0][0] == 16


def test_estimate_max_batch_size_linear_fit():
    """Affine fit solves (budget - intercept) / slope. slope=1GB, intercept=0."""
    out = _estimate_max_batch_size(
        measurements=[(2, 2 * GB), (8, 8 * GB)],
        total_bytes=23 * GB,
        max_gpu_utilization=1.0,
        safety_factor=1.0,
        min_batch_size=1,
        max_batch_size=512,
    )
    assert out == 23


def test_estimate_max_batch_size_applies_budget_and_safety():
    """budget = 24GB*0.9 = 21.6GB; (21.6-0)/1 = 21.6; *0.5 -> 10."""
    out = _estimate_max_batch_size(
        measurements=[(2, 2 * GB), (8, 8 * GB)],
        total_bytes=24 * GB,
        max_gpu_utilization=0.9,
        safety_factor=0.5,
        min_batch_size=1,
        max_batch_size=512,
    )
    assert out == 10


def test_estimate_max_batch_size_clamps_to_max():
    out = _estimate_max_batch_size(
        measurements=[(2, 2 * GB), (8, 8 * GB)],
        total_bytes=10_000 * GB,
        max_gpu_utilization=1.0,
        safety_factor=1.0,
        min_batch_size=1,
        max_batch_size=64,
    )
    assert out == 64


def test_estimate_max_batch_size_clamps_up_to_min():
    # predicted 23 < min 32 -> clamp up to 32.
    out = _estimate_max_batch_size(
        measurements=[(2, 2 * GB), (8, 8 * GB)],
        total_bytes=23 * GB,
        max_gpu_utilization=1.0,
        safety_factor=1.0,
        min_batch_size=32,
        max_batch_size=512,
    )
    assert out == 32


def test_estimate_max_batch_size_raises_on_flat_slope():
    with pytest.raises(_BatchSizeEstimationError):
        _estimate_max_batch_size(
            measurements=[(2, 8 * GB), (8, 8 * GB)],  # slope 0
            total_bytes=24 * GB,
            max_gpu_utilization=1.0,
            safety_factor=1.0,
            min_batch_size=1,
            max_batch_size=512,
        )


def test_estimate_max_batch_size_raises_on_too_few_points():
    with pytest.raises(_BatchSizeEstimationError):
        _estimate_max_batch_size(
            measurements=[(2, 1)],
            total_bytes=24 * GB,
            max_gpu_utilization=1.0,
            safety_factor=1.0,
            min_batch_size=1,
            max_batch_size=512,
        )


@pytest.fixture
def dummy_cfg_and_dataset():
    cfg = OmegaConf.create({
        "validation_split": 0.2,
        "seed": 42,
        "num_train_workers": 0,
        "num_val_workers": 0,
    })
    device = torch.device("cuda:0")
    dataset = Mock(spec=PytorchSynchronizedDataset)
    dataset.__len__ = Mock(return_value=1000)
    dataset.collate_fn = lambda x: x
    model_factory = functools.partial(DummyModel, device=device)

    def fake_random_split(ds, lengths, generator=None):
        return (DummyDataset(lengths[0]), DummyDataset(lengths[1]))

    with patch(
        "neuracore.ml.trainers.batch_autotuner.random_split",
        side_effect=fake_random_split,
    ):
        yield cfg, dataset, model_factory, device


def _make_autotuner(device, max_batch_size=512, min_batch_size=2, safety_factor=0.7):
    with patch("torch.cuda.is_available", return_value=True):
        return BatchSizeAutotuner(
            model_factory=functools.partial(DummyModel, device=device),
            device=device,
            train_dataset=DummyDataset(length=4000),
            train_dataloader_kwargs={},
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            safety_factor=safety_factor,
        )


def test_estimate_optimal_raises_when_min_batch_oom():
    """If even the smallest batch OOMs, nothing fits -> OutOfMemoryError."""
    device = torch.device("cuda:0")
    autotuner = _make_autotuner(device)

    with patch.object(
        BatchSizeValidator,
        "probe_batch_size",
        return_value=BatchProbeResult(fitted=False),
    ):
        with pytest.raises(OutOfMemoryError):
            autotuner.estimate_optimal_batch_size()


def test_estimate_optimal_downscales_when_high_ooms():
    """If the second anchor OOMs (but min fits), downscale from it until it fits."""
    device = torch.device("cuda:0")
    # min=2, high=16; only batch sizes <= 13 fit. high OOMs -> downscale from
    # int(16 * 0.9) = 14 (OOM) -> int(14 * 0.9) = 12 (fits). safety 1.0 keeps it.
    autotuner = _make_autotuner(device, max_batch_size=512, safety_factor=1.0)

    def fits(b):
        return b <= 13

    def fake_probe(b):
        return BatchProbeResult(
            fitted=fits(b),
            peak_reserved_bytes=(b * GB if fits(b) else 0),
            total_bytes=100 * GB,
        )

    with (
        patch.object(BatchSizeValidator, "probe_batch_size", side_effect=fake_probe),
        patch.object(BatchSizeValidator, "test_batch_size", side_effect=fits),
    ):
        result = autotuner.estimate_optimal_batch_size(max_gpu_utilization=1.0)

    assert result == 12


def test_estimate_optimal_returns_high_when_fit_unusable():
    """A non-positive 2-point slope (flat memory) falls back to the largest probe."""
    device = torch.device("cuda:0")
    # min=2, high=16, safety 0.7 -> int(16 * 0.7) = 11.
    autotuner = _make_autotuner(device, max_batch_size=512, safety_factor=0.7)

    # Both probes fit but report identical memory -> slope 0 -> unusable fit.
    with patch.object(
        BatchSizeValidator,
        "probe_batch_size",
        side_effect=lambda b: BatchProbeResult(
            fitted=True, peak_reserved_bytes=5 * GB, total_bytes=100 * GB
        ),
    ):
        result = autotuner.estimate_optimal_batch_size(max_gpu_utilization=1.0)

    assert result == 11  # int(high=16 * safety=0.7)


def test_estimate_optimal_uses_2nd_3rd_slope_and_downscales_on_convex_memory():
    """Third probe fits -> slope(2nd,3rd) predicts -> downscale to a fitting batch.

    Memory is convex, so the prediction still overshoots a little; the geometric
    downscale (no bisection) brings it back to a batch that fits.
    """
    device = torch.device("cuda:0")
    autotuner = _make_autotuner(device, max_batch_size=512, safety_factor=0.7)

    total = 10 * GB
    true_max = 269  # peak(269) ~ total

    def peak_bytes(b):
        # Mildly convex: the third probe (rough // 2) fits.
        return int((1.0 + 0.02 * b + 0.00005 * b * b) * GB)

    def fits(b):
        return peak_bytes(b) <= total

    def fake_probe(b):
        return BatchProbeResult(
            fitted=fits(b),
            peak_reserved_bytes=peak_bytes(b) if fits(b) else 0,
            total_bytes=total,
        )

    with (
        patch.object(
            BatchSizeValidator, "probe_batch_size", side_effect=fake_probe
        ) as mock_probe,
        patch.object(
            BatchSizeValidator, "test_batch_size", side_effect=fits
        ) as mock_confirm,
    ):
        result = autotuner.estimate_optimal_batch_size(max_gpu_utilization=1.0)

    assert 16 < result <= true_max  # used the estimate, above the probes
    assert peak_bytes(result) <= total  # the chosen batch genuinely fits

    # Prove the slope(2nd, 3rd) extrapolation was actually used, not just the
    # third anchor returned: the third anchor is the last affine probe, and the
    # prediction must have extended ABOVE it (a "return the anchor" bug would
    # never probe past it).
    affine_probes = [c.args[0] for c in mock_probe.call_args_list]
    confirm_probes = [c.args[0] for c in mock_confirm.call_args_list]
    third_anchor = affine_probes[-1]  # low < high < third (no retry on this curve)
    assert confirm_probes, "expected a confirmation/downscale probe above the anchor"
    assert max(confirm_probes) > third_anchor
    # ...and the chosen batch reflects a fit found above the anchor.
    assert result > int(third_anchor * 0.7)


def test_estimate_optimal_retries_third_anchor_when_it_ooms():
    """Strongly convex: rough//2 OOMs, so the third anchor is shrunk until it fits.

    The retry yields a real large-batch measurement for the slope(2nd, 3rd) refit
    instead of crawling down one downscale step at a time.
    """
    device = torch.device("cuda:0")
    autotuner = _make_autotuner(device, max_batch_size=512, safety_factor=0.7)

    total = 10 * GB
    true_max = 119  # peak(119) ~ total under the steeper curve

    def peak_bytes(b):
        # Strongly convex: rough // 2 lands above the true max and OOMs.
        return int((1.0 + 0.03 * b + 0.0005 * b * b) * GB)

    def fits(b):
        return peak_bytes(b) <= total

    def fake_probe(b):
        return BatchProbeResult(
            fitted=fits(b),
            peak_reserved_bytes=peak_bytes(b) if fits(b) else 0,
            total_bytes=total,
        )

    with (
        patch.object(
            BatchSizeValidator, "probe_batch_size", side_effect=fake_probe
        ) as mock_probe,
        patch.object(BatchSizeValidator, "test_batch_size", side_effect=fits),
    ):
        result = autotuner.estimate_optimal_batch_size(max_gpu_utilization=1.0)

    assert 16 < result <= true_max
    assert peak_bytes(result) <= total  # the chosen batch genuinely fits
    # low, high, an OOM'd anchor, then a shrunk anchor -> at least 4 probes.
    probed = [c.args[0] for c in mock_probe.call_args_list]
    assert len(probed) >= 4
    assert any(not fits(b) for b in probed)  # at least one anchor OOM'd (retried)


def test_find_optimal_uses_affine_estimate(dummy_cfg_and_dataset):
    """find_optimal_batch_size returns the affine estimate (no separate search)."""
    cfg, dataset, model_factory, device = dummy_cfg_and_dataset
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch.object(
            BatchSizeAutotuner, "estimate_optimal_batch_size", return_value=123
        ) as mock_estimate,
    ):
        result = find_optimal_batch_size(cfg, model_factory, dataset, device)

    assert result == 123
    mock_estimate.assert_called_once()
