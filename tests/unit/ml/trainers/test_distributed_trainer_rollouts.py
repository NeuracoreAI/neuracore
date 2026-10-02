"""Tests for when validation rollout snapshots run."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from neuracore_types import DataType
from torch.utils.data import Subset

from neuracore.ml.datasets.pytorch_dummy_dataset import PytorchDummyDataset
from neuracore.ml.trainers.distributed_trainer import DistributedTrainer


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


def _trainer(
    dataset: PytorchDummyDataset, points: int, frequency: int
) -> DistributedTrainer:
    trainer = DistributedTrainer.__new__(DistributedTrainer)
    trainer.rank = 0
    trainer.validation_rollout_points = points
    trainer.validation_rollout_frequency = frequency
    trainer.validation_rollout_seed = 3
    trainer.device = torch.device("cpu")
    trainer.inference_device_preprocessing = (MagicMock(), MagicMock())
    trainer.output_dir = Path(".")
    trainer.storage_handler = MagicMock()
    trainer.get_model_without_ddp = MagicMock()
    trainer.val_loader = MagicMock()
    trainer.val_loader.dataset = Subset(dataset, list(range(len(dataset))))
    return trainer


def test_rollouts_run_only_on_frequency_epochs_when_points_are_configured():
    trainer = _trainer(_dataset(), points=2, frequency=5)

    with patch(
        "neuracore.ml.trainers.distributed_trainer.save_validation_rollouts",
        return_value=Path("validation-rollouts"),
    ) as save:
        for epoch in range(1, 11):
            trainer._save_validation_rollouts(epoch)

    assert [call.kwargs["epoch"] for call in save.call_args_list] == [5, 10]
    assert all(call.kwargs["num_points"] == 2 for call in save.call_args_list)


def test_rollouts_are_skipped_when_no_points_are_configured():
    trainer = _trainer(_dataset(), points=0, frequency=5)

    with patch(
        "neuracore.ml.trainers.distributed_trainer.save_validation_rollouts"
    ) as save:
        trainer._save_validation_rollouts(5)

    save.assert_not_called()


def test_rollouts_are_skipped_on_non_zero_ranks():
    trainer = _trainer(_dataset(), points=2, frequency=5)
    trainer.rank = 1

    with patch(
        "neuracore.ml.trainers.distributed_trainer.save_validation_rollouts"
    ) as save:
        trainer._save_validation_rollouts(5)

    save.assert_not_called()
