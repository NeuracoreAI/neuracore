"""Tests for when validation rollout snapshots run."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from neuracore_types import DataType
from torch.utils.data import Subset

from neuracore.ml.datasets.pytorch_dummy_dataset import PytorchDummyDataset
from neuracore.ml.trainers.distributed_trainer import DistributedTrainer
from neuracore.ml.utils.validation_rollouts import ValidationRolloutConfig


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


def _trainer(config: ValidationRolloutConfig | None) -> DistributedTrainer:
    trainer = DistributedTrainer.__new__(DistributedTrainer)
    trainer.rank = 0
    trainer.validation_rollout_config = config
    trainer.val_loader = MagicMock()
    trainer.val_loader.batch_size = 8
    trainer.val_loader.dataset = Subset(_dataset(), [0, 1, 2, 3])
    trainer.device = torch.device("cpu")
    trainer.inference_device_preprocessing = (MagicMock(), MagicMock())
    trainer.output_dir = Path(".")
    trainer.storage_handler = MagicMock()
    trainer.get_model_without_ddp = MagicMock()
    return trainer


def test_rollouts_run_only_on_frequency_epochs_when_points_are_configured():
    config = _config(frequency=5)
    trainer = _trainer(config)

    with patch(
        "neuracore.ml.trainers.distributed_trainer.save_validation_rollouts",
        return_value=Path("validation-rollouts"),
    ) as save:
        for epoch in range(1, 11):
            trainer._save_validation_rollouts(epoch)

    assert [call.kwargs["epoch"] for call in save.call_args_list] == [5, 10]
    for call in save.call_args_list:
        assert call.kwargs["config"] is config
        assert call.kwargs["dataset"] is trainer.val_loader.dataset.dataset
        assert call.kwargs["validation_indices"] == [0, 1, 2, 3]
        assert call.kwargs["batch_size"] == 8


def test_rollouts_are_skipped_when_not_configured():
    trainer = _trainer(None)

    with patch(
        "neuracore.ml.trainers.distributed_trainer.save_validation_rollouts"
    ) as save:
        trainer._save_validation_rollouts(5)

    save.assert_not_called()


def test_rollouts_are_skipped_on_non_zero_ranks():
    trainer = _trainer(_config(frequency=5))
    trainer.rank = 1

    with patch(
        "neuracore.ml.trainers.distributed_trainer.save_validation_rollouts"
    ) as save:
        trainer._save_validation_rollouts(5)

    save.assert_not_called()


def test_rollout_failure_is_logged_and_does_not_stop_training(caplog):
    trainer = _trainer(_config(frequency=1))

    with patch(
        "neuracore.ml.trainers.distributed_trainer.save_validation_rollouts",
        side_effect=RuntimeError("forward failed"),
    ):
        trainer._save_validation_rollouts(1)

    assert "Failed to save validation rollouts for epoch 1" in caplog.text
    assert "forward failed" in caplog.text
