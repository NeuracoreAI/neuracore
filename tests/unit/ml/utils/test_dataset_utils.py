"""Tests for episode-level train/validation splitting."""

from types import SimpleNamespace

import pytest
from torch.utils.data import Subset

from neuracore.ml.preprocessing.base import PreprocessingConfiguration
from neuracore.ml.utils.dataset_utils import split_train_val_datasets


class _FakeEpisodeDataset:
    """Minimal dataset with the episode mapping split_train_val_datasets needs."""

    def __init__(
        self,
        episode_lengths: list[int],
        robot_ids: list[str] | None = None,
    ) -> None:
        self.episode_start_offsets: list[int] = []
        self.episode_indices: list[int] = []
        start = 0
        recordings: list[SimpleNamespace] = []
        for episode_idx, length in enumerate(episode_lengths):
            self.episode_start_offsets.append(start)
            self.episode_indices.extend([episode_idx] * length)
            start += length
            robot_id = robot_ids[episode_idx] if robot_ids is not None else "robot-a"
            recordings.append(SimpleNamespace(robot_id=robot_id))
        self._n_samples = start
        self.synchronized_dataset = recordings
        self.input_preprocessing_config = PreprocessingConfiguration()
        self.output_preprocessing_config = PreprocessingConfiguration()
        self.cache_rebuilds = 0

    def __len__(self) -> int:
        return self._n_samples

    def rebuild_sample_cache(self) -> None:
        self.cache_rebuilds += 1


_EMPTY_PREPROCESSING = PreprocessingConfiguration()


def _split(
    dataset: _FakeEpisodeDataset,
    validation_split: float = 0.2,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    return split_train_val_datasets(
        dataset,  # type: ignore[arg-type]
        validation_split=validation_split,
        seed=seed,
        inference_input_preprocessing_config=_EMPTY_PREPROCESSING,
        inference_output_preprocessing_config=_EMPTY_PREPROCESSING,
    )


def _episode_ids(subset: Subset) -> set[int]:
    dataset = subset.dataset
    assert isinstance(dataset, _FakeEpisodeDataset)
    return {dataset.episode_indices[idx] for idx in subset.indices}


def test_split_keeps_whole_episodes_on_one_side() -> None:
    dataset = _FakeEpisodeDataset([4, 4, 4, 4, 4])
    train, val = _split(dataset, validation_split=0.2)

    train_episodes = _episode_ids(train)
    val_episodes = _episode_ids(val)
    assert train_episodes.isdisjoint(val_episodes)
    assert train_episodes | val_episodes == set(range(5))
    assert len(val_episodes) == 1
    assert len(train_episodes) == 4
    assert len(train) + len(val) == len(dataset)


def test_split_is_deterministic_for_a_seed() -> None:
    dataset = _FakeEpisodeDataset([3, 5, 2, 8, 4])
    train_a, val_a = _split(dataset, seed=7)
    train_b, val_b = _split(dataset, seed=7)
    assert list(train_a.indices) == list(train_b.indices)
    assert list(val_a.indices) == list(val_b.indices)

    train_c, val_c = _split(dataset, seed=8)
    assert (list(train_a.indices), list(val_a.indices)) != (
        list(train_c.indices),
        list(val_c.indices),
    )


def test_split_holds_out_at_least_one_episode_when_fraction_rounds_down() -> None:
    dataset = _FakeEpisodeDataset([2, 2, 2, 2])
    _, val = _split(dataset, validation_split=0.1)
    assert len(_episode_ids(val)) == 1


def test_split_stratifies_by_robot() -> None:
    dataset = _FakeEpisodeDataset(
        [2, 2, 2, 2, 2, 2, 2, 2],
        robot_ids=["a", "a", "a", "a", "b", "b", "b", "b"],
    )
    train, val = _split(dataset, validation_split=0.25)
    val_episodes = _episode_ids(val)
    val_robots = {dataset.synchronized_dataset[ep].robot_id for ep in val_episodes}
    assert val_robots == {"a", "b"}
    assert _episode_ids(train).isdisjoint(val_episodes)


def test_split_falls_back_when_every_robot_has_one_episode() -> None:
    dataset = _FakeEpisodeDataset(
        [3, 3, 3],
        robot_ids=["a", "b", "c"],
    )
    train, val = _split(dataset, validation_split=0.34)
    assert len(_episode_ids(val)) == 1
    assert len(_episode_ids(train)) == 2


def test_split_rekeys_val_preprocessing_cache() -> None:
    dataset = _FakeEpisodeDataset([2, 2])
    train, val = _split(dataset)
    assert train.dataset is dataset
    assert val.dataset is not dataset
    assert val.dataset.cache_rebuilds == 1
    assert dataset.cache_rebuilds == 0


@pytest.mark.parametrize(
    ("episode_lengths", "validation_split", "match"),
    [
        ([], 0.2, "training and validation sets are both empty"),
        ([5], 0.2, "Need at least 2 recordings"),
        ([2, 2], 0.0, "validation set is empty"),
        ([2, 2], 1.0, "training set is empty"),
    ],
)
def test_split_rejects_empty_sides(
    episode_lengths: list[int],
    validation_split: float,
    match: str,
) -> None:
    dataset = _FakeEpisodeDataset(episode_lengths)
    with pytest.raises(ValueError, match=match):
        _split(dataset, validation_split=validation_split)
