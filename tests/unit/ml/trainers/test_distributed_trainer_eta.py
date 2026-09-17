"""Tests for DistributedTrainer epoch-duration reporting."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from neuracore.ml.trainers.distributed_trainer import DistributedTrainer


@pytest.fixture
def trainer(tmp_path: Path) -> DistributedTrainer:
    trainer = DistributedTrainer.__new__(DistributedTrainer)
    trainer.rank = 0
    trainer.num_epochs = 3
    trainer.save_freq = 100
    trainer.save_checkpoints = False
    trainer.global_train_step = 0
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
    return trainer


def test_train_skips_seconds_per_epoch_on_warmup_epoch(trainer: DistributedTrainer):
    trainer.train(start_epoch=1)

    progress_calls = trainer.storage_handler.update_training_progress.call_args_list
    # Initial progress + one call per completed epoch (1, 2, 3).
    assert len(progress_calls) == 4

    warmup_call = progress_calls[1]
    assert warmup_call.kwargs["epoch"] == 1
    assert warmup_call.kwargs.get("seconds_per_epoch") is None

    measured_epochs = [call.kwargs for call in progress_calls[2:]]
    assert [call["epoch"] for call in measured_epochs] == [2, 3]
    for call in measured_epochs:
        assert call["seconds_per_epoch"] is not None
        assert call["seconds_per_epoch"] >= 0


def test_train_skips_warmup_for_resumed_start_epoch(trainer: DistributedTrainer):
    trainer.num_epochs = 8
    trainer.train(start_epoch=6)

    progress_calls = trainer.storage_handler.update_training_progress.call_args_list
    by_epoch = {
        call.kwargs["epoch"]: call.kwargs.get("seconds_per_epoch")
        for call in progress_calls
        if "epoch" in call.kwargs and call.kwargs["epoch"] >= 6
    }
    assert by_epoch[6] is None
    assert by_epoch[7] is not None
    assert by_epoch[8] is not None
