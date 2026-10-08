"""Tests for DistributedTrainer epoch-duration reporting."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from neuracore.ml.trainers.distributed_trainer import (
    SECONDS_PER_EPOCH_EMA_ALPHA,
    DistributedTrainer,
)


def _progress_from_call(call):
    """Extract TrainingProgress from a mocked update_training_progress call."""
    if call.args:
        return call.args[0]
    return call.kwargs["progress"]


@pytest.fixture
def trainer(tmp_path: Path) -> DistributedTrainer:
    trainer = DistributedTrainer.__new__(DistributedTrainer)
    trainer.rank = 0
    trainer.num_epochs = 3
    trainer.save_freq = 100
    trainer.save_checkpoints = False
    trainer.global_train_step = 0
    trainer._seconds_per_epoch_ema = None
    trainer._seconds_per_checkpoint_ema = None
    trainer.train_loader = MagicMock()
    trainer.train_loader.sampler = None
    trainer.train_loader.batch_size = 1
    trainer.storage_handler = MagicMock()
    trainer.storage_handler.log_to_cloud = False
    trainer.training_logger = MagicMock()
    trainer.output_dir = tmp_path
    trainer.train_epoch = MagicMock(return_value={})
    trainer.validate = MagicMock(return_value={})
    trainer.save_checkpoint = MagicMock()
    trainer.get_model_without_ddp = MagicMock(return_value=MagicMock())
    return trainer


def test_train_skips_seconds_per_epoch_on_warmup_epoch(trainer: DistributedTrainer):
    trainer.train(start_epoch=1)

    progress_calls = trainer.storage_handler.update_training_progress.call_args_list
    # Initial progress + one call per completed epoch (1, 2, 3).
    assert len(progress_calls) == 4

    warmup = _progress_from_call(progress_calls[1])
    assert warmup.epoch == 1
    assert warmup.seconds_per_epoch is None

    measured_epochs = [_progress_from_call(call) for call in progress_calls[2:]]
    assert [progress.epoch for progress in measured_epochs] == [2, 3]
    for progress in measured_epochs:
        assert progress.seconds_per_epoch is not None
        assert progress.seconds_per_epoch >= 0


def test_train_skips_warmup_for_resumed_start_epoch(trainer: DistributedTrainer):
    trainer.num_epochs = 8
    trainer.train(start_epoch=6)

    progress_calls = trainer.storage_handler.update_training_progress.call_args_list
    by_epoch = {
        _progress_from_call(call).epoch: _progress_from_call(call).seconds_per_epoch
        for call in progress_calls
        if _progress_from_call(call).epoch is not None
        and _progress_from_call(call).epoch >= 6
    }
    assert by_epoch[6] is None
    assert by_epoch[7] is not None
    assert by_epoch[8] is not None


def test_update_ema_seeds_then_smooths():
    assert DistributedTrainer._update_ema(None, 10.0, 0.3) == 10.0
    assert DistributedTrainer._update_ema(10.0, 20.0, 0.3) == pytest.approx(13.0)


def test_record_seconds_per_epoch_applies_ema(trainer: DistributedTrainer):
    first = trainer._record_seconds_per_epoch(10.0)
    second = trainer._record_seconds_per_epoch(20.0)

    assert first == pytest.approx(10.0)
    assert second == pytest.approx(
        SECONDS_PER_EPOCH_EMA_ALPHA * 20.0 + (1.0 - SECONDS_PER_EPOCH_EMA_ALPHA) * 10.0
    )


def test_record_seconds_per_epoch_amortizes_checkpoint(trainer: DistributedTrainer):
    trainer.save_freq = 5
    trainer._seconds_per_checkpoint_ema = 10.0

    # train+val 8s + amortized checkpoint 10/5 = 2s
    assert trainer._record_seconds_per_epoch(8.0) == pytest.approx(10.0)


def test_train_reports_ema_not_raw_epoch_duration(
    trainer: DistributedTrainer, monkeypatch: pytest.MonkeyPatch
):
    """Second measured epoch should report EMA(first, second), not the raw value."""
    trainer.num_epochs = 3
    trainer.save_freq = 100

    # perf_counter call order per epoch without a save:
    # epoch_t0, after train, validate_t0, after validate
    # Warmup epoch 1: train+val = 100s (ignored for SPE)
    # Epoch 2: train+val = 10s -> EMA seeds to 10
    # Epoch 3: train+val = 20s -> EMA = 0.3*20 + 0.7*10 = 13
    clock = iter([
        0.0,
        50.0,
        50.0,
        100.0,  # epoch 1
        100.0,
        105.0,
        105.0,
        110.0,  # epoch 2 -> 10s
        110.0,
        120.0,
        120.0,
        130.0,  # epoch 3 -> 20s
    ])
    monkeypatch.setattr(
        "neuracore.ml.trainers.distributed_trainer.time.perf_counter",
        lambda: next(clock),
    )

    trainer.train(start_epoch=1)

    progress_calls = trainer.storage_handler.update_training_progress.call_args_list
    by_epoch = {
        _progress_from_call(call).epoch: _progress_from_call(call).seconds_per_epoch
        for call in progress_calls
        if _progress_from_call(call).epoch in (2, 3)
    }
    assert by_epoch[2] == pytest.approx(10.0)
    assert by_epoch[3] == pytest.approx(13.0)


def test_train_amortizes_checkpoint_into_seconds_per_epoch(
    trainer: DistributedTrainer, monkeypatch: pytest.MonkeyPatch
):
    """Checkpoint time is amortized across save_freq rather than spiked on save."""
    trainer.num_epochs = 3
    trainer.save_freq = 2
    trainer.save_checkpoints = True

    # Epoch 1 (warmup, no save): train+val=100s, ignored for SPE
    # Epoch 2 (measure + save): train=4, save=10, val=1
    #   checkpoint EMA=10, SPE = 5 + 10/2 = 10
    # Epoch 3 (measure, no save): train=14, val=1
    #   SPE raw = 15 + 5 = 20; EMA = 0.3*20 + 0.7*10 = 13
    clock = iter([
        0.0,
        50.0,
        50.0,
        100.0,  # epoch 1
        100.0,
        104.0,
        104.0,
        114.0,
        114.0,
        115.0,  # epoch 2
        115.0,
        129.0,
        129.0,
        130.0,  # epoch 3
    ])
    monkeypatch.setattr(
        "neuracore.ml.trainers.distributed_trainer.time.perf_counter",
        lambda: next(clock),
    )

    trainer.train(start_epoch=1)

    progress_calls = trainer.storage_handler.update_training_progress.call_args_list
    by_epoch = {
        _progress_from_call(call).epoch: _progress_from_call(call).seconds_per_epoch
        for call in progress_calls
        if _progress_from_call(call).epoch in (1, 2, 3)
    }
    assert by_epoch[1] is None
    assert by_epoch[2] == pytest.approx(10.0)
    assert by_epoch[3] == pytest.approx(13.0)
    trainer.save_checkpoint.assert_called()
    trainer.storage_handler.save_model_artifacts.assert_called()
