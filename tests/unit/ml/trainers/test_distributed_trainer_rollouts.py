"""Tests for when validation rollout snapshots run."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from neuracore_types import DataType
from torch.utils.data import Subset

from neuracore.ml.datasets.pytorch_dummy_dataset import PytorchDummyDataset
from neuracore.ml.trainers.distributed_trainer import DistributedTrainer
from neuracore.ml.utils.validation_rollouts import (
    ValidationRolloutConfig,
    write_rollout_manifest,
)


def _dataset() -> PytorchDummyDataset:
    return PytorchDummyDataset(
        input_cross_embodiment_description={
            "robot_0": {DataType.JOINT_POSITIONS: {0: "joint_0"}},
        },
        output_cross_embodiment_description={
            "robot_0": {DataType.JOINT_TARGET_POSITIONS: {0: "joint_0"}},
        },
        num_samples=8,
        num_episodes=2,
        output_prediction_horizon=2,
    )


def _config(frequency: int) -> ValidationRolloutConfig:
    return ValidationRolloutConfig(
        num_points=2,
        frequency=frequency,
        seed=3,
    )


def _trainer(config: ValidationRolloutConfig) -> DistributedTrainer:
    trainer = DistributedTrainer.__new__(DistributedTrainer)
    trainer.rank = 0
    trainer.validation_rollout_config = config
    trainer.num_epochs = 10
    trainer._selected_rollout_validation_indices = [1, 3]
    trainer._input_points_saved = False
    trainer._run_rollout_validation = True
    trainer.val_loader = MagicMock()
    trainer.val_loader.batch_size = 8
    trainer.val_loader.dataset = Subset(_dataset(), [0, 1, 2, 3])
    trainer.device = torch.device("cpu")
    trainer.inference_device_preprocessing = (MagicMock(), MagicMock())
    trainer.output_dir = Path(".")
    trainer.storage_handler = MagicMock()
    trainer.get_model_without_ddp = MagicMock()
    return trainer


def test_input_points_are_saved_on_the_first_rollout_only():
    trainer = _trainer(_config(frequency=5))

    with patch(
        "neuracore.ml.trainers.distributed_trainer.save_validation_rollouts",
        return_value=Path("validation_rollouts"),
    ) as save:
        for epoch in (5, 10):
            trainer._save_validation_rollouts(epoch)

    assert [call.kwargs["save_input_points"] for call in save.call_args_list] == [
        True,
        False,
    ]


def test_manifest_write_and_upload_are_queued_in_order():
    trainer = _trainer(_config(frequency=5))

    trainer._save_rollout_validation_manifest()

    write_job, upload_job = (
        call.args[0]
        for call in trainer.storage_handler.submit_rollout_save.call_args_list
    )
    root_dir = Path(".") / "validation_rollouts"
    expected_points = [
        trainer.val_loader.dataset.dataset.get_sample_identity(i) for i in (1, 3)
    ]
    assert write_job.func is write_rollout_manifest
    assert write_job.args == (root_dir, expected_points)
    assert upload_job.func == trainer.storage_handler.upload_rollout_manifest
    assert upload_job.args == (root_dir,)
    assert trainer._selected_rollout_validation_indices == [1, 3]


@pytest.mark.parametrize(
    ("rank", "config", "num_epochs", "expected"),
    [
        (0, _config(frequency=5), 10, True),
        (0, _config(frequency=10), 10, True),
        (1, _config(frequency=5), 10, False),
        (0, ValidationRolloutConfig(num_points=0, frequency=5, seed=3), 10, False),
        (0, _config(frequency=5), 4, False),
    ],
)
def test_can_run_rollout_validation(rank, config, num_epochs, expected):
    trainer = _trainer(config)
    trainer.rank = rank
    trainer.num_epochs = num_epochs

    assert trainer._can_run_rollout_validation() is expected


def test_rollout_failure_is_logged_and_does_not_stop_training(caplog):
    trainer = _trainer(_config(frequency=1))

    with patch(
        "neuracore.ml.trainers.distributed_trainer.save_validation_rollouts",
        side_effect=RuntimeError("forward failed"),
    ):
        trainer._save_validation_rollouts(1)

    assert "Failed to save validation rollouts for epoch 1" in caplog.text
    assert "forward failed" in caplog.text
