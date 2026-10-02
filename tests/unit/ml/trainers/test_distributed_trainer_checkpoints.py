"""Tests for checkpoint retention in DistributedTrainer."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from neuracore.ml.trainers.distributed_trainer import DistributedTrainer


def _trainer(
    tmp_path: Path,
    keep_last_n_checkpoints: int,
    checkpoint_saving_frequency: int,
) -> DistributedTrainer:
    trainer = DistributedTrainer.__new__(DistributedTrainer)
    trainer.rank = 0
    trainer.save_checkpoints = True
    trainer.keep_last_n_checkpoints = keep_last_n_checkpoints
    trainer.checkpoint_saving_frequency = checkpoint_saving_frequency
    trainer.checkpoint_dir = tmp_path / "checkpoints"
    trainer.global_train_step = 0
    trainer.global_val_step = 0
    trainer.optimizers = []
    trainer.schedulers = []
    trainer.get_model_without_ddp = MagicMock()
    trainer.get_model_without_ddp.return_value.state_dict.return_value = {}
    trainer.storage_handler = MagicMock()
    return trainer


def _deleted_epochs(trainer: DistributedTrainer) -> list[int]:
    epochs = []
    for call in trainer.storage_handler.delete_checkpoint.call_args_list:
        name = call.args[0].name
        epochs.append(int(name.removeprefix("checkpoint_").removesuffix(".pt")))
    return epochs


@pytest.mark.parametrize("checkpoint_saving_frequency", [4, 5])
def test_keeps_recent_window_and_milestone_epochs(
    tmp_path: Path, checkpoint_saving_frequency: int
):
    keep_last_n_checkpoints = 2
    last_epoch = 58
    trainer = _trainer(
        tmp_path,
        keep_last_n_checkpoints=keep_last_n_checkpoints,
        checkpoint_saving_frequency=checkpoint_saving_frequency,
    )

    for epoch in range(1, last_epoch + 1):
        trainer.save_checkpoint(epoch, {})

    saved = set(range(1, last_epoch + 1))
    retained = saved - set(_deleted_epochs(trainer))
    milestones = {epoch for epoch in saved if epoch % checkpoint_saving_frequency == 0}
    recent = set(range(last_epoch - keep_last_n_checkpoints + 1, last_epoch + 1))
    assert retained == milestones | recent


@pytest.mark.parametrize("checkpoint_saving_frequency", [4, 5])
def test_skips_deletion_when_the_expired_epoch_is_a_milestone(
    tmp_path: Path, checkpoint_saving_frequency: int
):
    keep_last_n_checkpoints = 2
    trainer = _trainer(
        tmp_path,
        keep_last_n_checkpoints=keep_last_n_checkpoints,
        checkpoint_saving_frequency=checkpoint_saving_frequency,
    )

    milestone_epoch = checkpoint_saving_frequency
    trainer.save_checkpoint(milestone_epoch + keep_last_n_checkpoints, {})
    assert _deleted_epochs(trainer) == []

    trainer.save_checkpoint(milestone_epoch + keep_last_n_checkpoints + 1, {})
    assert _deleted_epochs(trainer) == [milestone_epoch + 1]
