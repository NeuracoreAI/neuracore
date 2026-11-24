"""Tests for checkpoint retention in DistributedTrainer."""

import multiprocessing as mp
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

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


def _loadable_trainer(
    tmp_path: Path, *, rank: int = 0, world_size: int = 1
) -> DistributedTrainer:
    trainer = _trainer(
        tmp_path, keep_last_n_checkpoints=1, checkpoint_saving_frequency=1
    )
    trainer.rank = rank
    trainer.world_size = world_size
    trainer.device = torch.device("cpu")
    trainer.storage_handler.load_checkpoint.return_value = {
        "model_state": {},
        "optimizer_states": [],
        "epoch": 3,
    }
    return trainer


def test_load_checkpoint_downloads_on_rank_zero_then_restores(tmp_path: Path):
    trainer = _loadable_trainer(tmp_path)

    result = trainer.load_checkpoint("checkpoint_3.pt")

    trainer.storage_handler.download_checkpoint.assert_called_once_with(
        "checkpoint_3.pt"
    )
    trainer.storage_handler.load_checkpoint.assert_called_once_with("checkpoint_3.pt")
    trainer.get_model_without_ddp.return_value.load_state_dict.assert_called_once()
    assert result["epoch"] == 3


def test_load_checkpoint_does_not_download_on_nonzero_rank(tmp_path: Path, monkeypatch):
    trainer = _loadable_trainer(tmp_path, rank=1, world_size=2)
    monkeypatch.setattr(
        trainer,
        "_any_rank_failed",
        lambda local_failed: False,
    )

    trainer.load_checkpoint("checkpoint_3.pt")

    trainer.storage_handler.download_checkpoint.assert_not_called()
    trainer.storage_handler.load_checkpoint.assert_called_once_with("checkpoint_3.pt")


def test_load_checkpoint_does_not_read_file_when_rank_zero_download_fails(
    tmp_path: Path,
):
    trainer = _loadable_trainer(tmp_path)
    trainer.storage_handler.download_checkpoint.side_effect = ValueError(
        "Failed to download checkpoint"
    )

    with pytest.raises(ValueError, match="Failed to download checkpoint"):
        trainer.load_checkpoint("checkpoint_3.pt")

    trainer.storage_handler.load_checkpoint.assert_not_called()


def test_load_checkpoint_syncs_rank_zero_download_failure_before_raising(
    tmp_path: Path, monkeypatch
):
    trainer = _loadable_trainer(tmp_path, rank=0, world_size=2)
    trainer.storage_handler.download_checkpoint.side_effect = ValueError(
        "Failed to download checkpoint"
    )
    seen: dict[str, bool] = {}

    def any_rank_failed(local_failed: bool) -> bool:
        seen["local_failed"] = local_failed
        return True

    monkeypatch.setattr(trainer, "_any_rank_failed", any_rank_failed)

    with pytest.raises(ValueError, match="Failed to download checkpoint"):
        trainer.load_checkpoint("checkpoint_3.pt")

    assert seen["local_failed"] is True
    trainer.storage_handler.load_checkpoint.assert_not_called()


def test_load_checkpoint_does_not_read_file_when_another_rank_fails_to_download(
    tmp_path: Path, monkeypatch
):
    trainer = _loadable_trainer(tmp_path, rank=1, world_size=2)
    monkeypatch.setattr(
        trainer,
        "_any_rank_failed",
        lambda local_failed: True,
    )

    with pytest.raises(RuntimeError, match="Another rank failed to download"):
        trainer.load_checkpoint("checkpoint_3.pt")

    trainer.storage_handler.download_checkpoint.assert_not_called()
    trainer.storage_handler.load_checkpoint.assert_not_called()


def _agree_on_resume_load_failure(
    rank: int,
    world_size: int,
    local_failed: bool,
    init_method: str,
    result_path: str,
) -> None:
    """Run the resume-failure collective for one real distributed rank."""
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        trainer = DistributedTrainer.__new__(DistributedTrainer)
        trainer.device = torch.device("cpu")
        agreed = trainer._any_rank_failed(local_failed)
        Path(result_path).write_text("1" if agreed else "0")
    finally:
        dist.destroy_process_group()


def _run_resume_failure_collective(
    local_failures: list[bool], tmp_path: Path
) -> dict[int, bool]:
    """Launch one process per rank and return each rank's agreed result."""
    init_method = f"file://{tmp_path / 'dist_init'}"
    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    result_paths: list[Path] = []
    for rank, local_failed in enumerate(local_failures):
        result_path = tmp_path / f"rank_{rank}.txt"
        result_paths.append(result_path)
        process = context.Process(
            target=_agree_on_resume_load_failure,
            args=(
                rank,
                len(local_failures),
                local_failed,
                init_method,
                str(result_path),
            ),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join(60)
        assert (
            process.exitcode == 0
        ), f"Rank process {process.pid} exited with {process.exitcode}"

    return {
        rank: result_path.read_text() == "1"
        for rank, result_path in enumerate(result_paths)
    }


@pytest.mark.parametrize(
    ("local_failures", "expected"),
    [
        ([False, False], {0: False, 1: False}),
        ([False, True], {0: True, 1: True}),
        ([True, True], {0: True, 1: True}),
    ],
)
def test_resume_failure_collective_agrees_across_ranks(
    tmp_path: Path,
    local_failures: list[bool],
    expected: dict[int, bool],
):
    assert _run_resume_failure_collective(local_failures, tmp_path) == expected
