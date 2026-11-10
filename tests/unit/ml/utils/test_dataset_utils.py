"""Tests for episode-level train/validation splitting."""

from itertools import groupby
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
        output_prediction_horizon: int = 1,
    ) -> None:
        self.output_prediction_horizon = output_prediction_horizon
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
    num_rollout_episodes: int = 0,
) -> tuple[Subset, Subset, Subset | None]:
    return split_train_val_datasets(
        dataset,  # type: ignore[arg-type]
        validation_split=validation_split,
        seed=seed,
        inference_input_preprocessing_config=_EMPTY_PREPROCESSING,
        inference_output_preprocessing_config=_EMPTY_PREPROCESSING,
        num_rollout_episodes=num_rollout_episodes,
    )


def _episode_ids(subset: Subset) -> set[int]:
    dataset = subset.dataset
    assert isinstance(dataset, _FakeEpisodeDataset)
    return {dataset.episode_indices[idx] for idx in subset.indices}


def test_split_keeps_whole_episodes_on_one_side() -> None:
    dataset = _FakeEpisodeDataset([4, 4, 4, 4, 4])
    train, val, _ = _split(dataset, validation_split=0.2)

    train_episodes = _episode_ids(train)
    val_episodes = _episode_ids(val)
    assert train_episodes.isdisjoint(val_episodes)
    assert train_episodes | val_episodes == set(range(5))
    assert len(val_episodes) == 1
    assert len(train_episodes) == 4
    assert len(train) + len(val) == len(dataset)


def test_split_is_deterministic_for_a_seed() -> None:
    dataset = _FakeEpisodeDataset([3, 5, 2, 8, 4])
    train_a, val_a, _ = _split(dataset, seed=7)
    train_b, val_b, _ = _split(dataset, seed=7)
    assert list(train_a.indices) == list(train_b.indices)
    assert list(val_a.indices) == list(val_b.indices)

    train_c, val_c, _ = _split(dataset, seed=8)
    assert (list(train_a.indices), list(val_a.indices)) != (
        list(train_c.indices),
        list(val_c.indices),
    )


def test_split_holds_out_at_least_one_episode_when_fraction_rounds_down() -> None:
    dataset = _FakeEpisodeDataset([2, 2, 2, 2])
    _, val, _ = _split(dataset, validation_split=0.1)
    assert len(_episode_ids(val)) == 1


def test_split_stratifies_by_robot() -> None:
    dataset = _FakeEpisodeDataset(
        [2, 2, 2, 2, 2, 2, 2, 2],
        robot_ids=["a", "a", "a", "a", "b", "b", "b", "b"],
    )
    train, val, _ = _split(dataset, validation_split=0.25)
    val_episodes = _episode_ids(val)
    val_robots = {dataset.synchronized_dataset[ep].robot_id for ep in val_episodes}
    assert val_robots == {"a", "b"}
    assert _episode_ids(train).isdisjoint(val_episodes)


def test_split_rejects_when_no_robot_has_two_episodes() -> None:
    dataset = _FakeEpisodeDataset(
        [3, 3, 3],
        robot_ids=["a", "b", "c"],
    )
    with pytest.raises(
        ValueError, match=r"'a' has 1 episode\(s\).*'c' has 1 episode\(s\)"
    ):
        _split(dataset, validation_split=0.34)


def test_split_rekeys_val_preprocessing_cache() -> None:
    dataset = _FakeEpisodeDataset([2, 2])
    train, val, _ = _split(dataset)
    assert train.dataset is dataset
    assert val.dataset is not dataset
    assert val.dataset.cache_rebuilds == 1
    assert dataset.cache_rebuilds == 0


@pytest.mark.parametrize(
    ("episode_lengths", "validation_split", "match"),
    [
        ([], 0.2, "training and validation sets are both empty"),
        ([5], 0.2, "Need at least 2 recordings"),
        ([2, 2], 0.0, r"validation_split is 0\.0.*strictly between 0 and 1"),
        ([2, 2], 1.0, r"validation_split is 1\.0.*strictly between 0 and 1"),
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


def _rollout_points(rollout: Subset) -> list[tuple[int, int]]:
    """Return ``(episode, timestep)`` for each rollout point, in order."""
    dataset = rollout.dataset
    assert isinstance(dataset, _FakeEpisodeDataset)
    points = []
    for idx in rollout.indices:
        episode = dataset.episode_indices[idx]
        points.append((episode, idx - dataset.episode_start_offsets[episode]))
    return points


def test_split_has_no_rollouts_by_default() -> None:
    _, _, rollout = _split(_FakeEpisodeDataset([4, 4, 4, 4, 4]))
    assert rollout is None


def test_split_rolls_out_val_episodes_every_horizon() -> None:
    episode_lengths = [7, 10, 4, 9, 5]
    dataset = _FakeEpisodeDataset(episode_lengths, output_prediction_horizon=3)
    _, val, rollout = _split(dataset, validation_split=0.4, num_rollout_episodes=2)

    assert rollout is not None
    assert rollout.dataset is val.dataset
    assert set(rollout.indices) <= set(val.indices)
    points = _rollout_points(rollout)
    episodes = [episode for episode, _ in points]
    assert set(episodes) == _episode_ids(val)
    # Each episode's points are contiguous and step by the horizon from 0.
    assert len(list(groupby(episodes))) == 2
    for episode in set(episodes):
        expected = list(range(0, episode_lengths[episode], 3))
        assert [t for e, t in points if e == episode] == expected


def test_split_caps_rollout_episodes() -> None:
    dataset = _FakeEpisodeDataset([4] * 10, output_prediction_horizon=2)
    _, val, rollout = _split(dataset, validation_split=0.5, num_rollout_episodes=2)

    assert rollout is not None
    assert len({episode for episode, _ in _rollout_points(rollout)}) == 2
    assert len(_episode_ids(val)) == 5


def test_split_rolls_out_every_val_episode_when_cap_is_larger() -> None:
    dataset = _FakeEpisodeDataset([4] * 5, output_prediction_horizon=2)
    _, val, rollout = _split(dataset, validation_split=0.4, num_rollout_episodes=10)

    assert rollout is not None
    assert {episode for episode, _ in _rollout_points(rollout)} == _episode_ids(val)


def test_split_rollouts_are_deterministic_for_a_seed() -> None:
    dataset = _FakeEpisodeDataset([4] * 10, output_prediction_horizon=2)
    _, _, first = _split(dataset, validation_split=0.5, num_rollout_episodes=2)
    _, _, second = _split(dataset, validation_split=0.5, num_rollout_episodes=2)

    assert first is not None and second is not None
    assert list(first.indices) == list(second.indices)


def test_split_rejects_negative_rollout_episodes() -> None:
    with pytest.raises(ValueError, match="num_rollout_episodes"):
        _split(_FakeEpisodeDataset([4, 4]), num_rollout_episodes=-1)
