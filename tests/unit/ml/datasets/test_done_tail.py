"""Tests for synthesizing the done flag after synchronization."""

import logging
import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from neuracore_types import (
    CrossEmbodimentDescription,
    DataType,
    DoneDataStats,
    EmbodimentDescription,
    JointData,
    JointDataStats,
    SynchronizedDatasetStatistics,
    SynchronizedPoint,
)

from neuracore.core.data.synced_dataset import SynchronizedDataset
from neuracore.core.data.synced_recording import SynchronizedRecording
from neuracore.ml.datasets import pytorch_synchronized_dataset as psd
from neuracore.ml.datasets.done_tail import (
    DONE_SENSOR_NAME,
    DoneTailDataset,
    DoneTailRecording,
    add_done_to_cross_embodiment_description,
    done_tail_length,
    remove_done_from_cross_embodiment_description,
    should_synthesize_done,
)
from neuracore.ml.preprocessing.base import PreprocessingConfiguration
from neuracore.ml.utils.training_config import _validate_synthesize_done

ROBOT_ID = "11111111-1111-1111-1111-111111111111"
OTHER_ROBOT_ID = "22222222-2222-2222-2222-222222222222"
FREQUENCY = 10
POSITION = "joint_0"
VELOCITY = "joint_0_velocity"

FULL_EMBODIMENT: EmbodimentDescription = {
    DataType.JOINT_POSITIONS: {0: POSITION},
    DataType.JOINT_VELOCITIES: {0: VELOCITY},
}


class FakeRecording(SynchronizedRecording):
    """Recording whose joint position at timestep t is t."""

    def __init__(self, length: int, robot_id: str = ROBOT_ID, name: str = "rec"):
        self.length = length
        self.robot_id = robot_id
        self.id = f"{name}-id"
        self.name = name
        self.requested: list[set[DataType]] = []

    @property
    def frequency(self) -> int:
        return FREQUENCY

    def _point(self, timestep: int, data_types: set[DataType]) -> SynchronizedPoint:
        timestamp = timestep / FREQUENCY
        data = {
            DataType.JOINT_POSITIONS: {
                POSITION: JointData(value=float(timestep), timestamp=timestamp)
            },
            DataType.JOINT_VELOCITIES: {
                VELOCITY: JointData(value=1.0, timestamp=timestamp)
            },
        }
        return SynchronizedPoint(
            timestamp=timestamp,
            robot_id=self.robot_id,
            data={k: v for k, v in data.items() if k in data_types},
        )

    def get_sync_point(self, timestep, embodiment_description):
        data_types = set(embodiment_description)
        self.requested.append(data_types)
        if DataType.DONE in data_types:
            raise KeyError("DONE is not logged in this recording")
        return self._point(timestep, data_types)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self._point(idx, set(FULL_EMBODIMENT))


def _joint_stats(count: int) -> JointDataStats:
    stats = JointData(value=0.0).calculate_statistics()
    stats.value.count = np.array([count], dtype=np.int32)
    return stats


class FakeSynchronizedDataset(SynchronizedDataset):
    """Synchronized dataset over fake recordings with server-like statistics."""

    def __init__(self, recordings: list[FakeRecording]):
        self.id = "synced-id"
        self.frequency = FREQUENCY
        self.recordings = recordings
        self._recording_idx = 0
        self.statistics_requests: list[CrossEmbodimentDescription] = []
        self.dataset = MagicMock()
        self.dataset.name = "my dataset"
        self.dataset.data_types = [DataType.JOINT_POSITIONS, DataType.JOINT_VELOCITIES]
        self.dataset.get_full_embodiment_description.side_effect = lambda robot_id: (
            dict(FULL_EMBODIMENT)
        )

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return FakeSynchronizedDataset(self.recordings[idx])
        return self.recordings[idx]

    def __next__(self):
        if self._recording_idx >= len(self.recordings):
            raise StopIteration
        self._recording_idx += 1
        return self.recordings[self._recording_idx - 1]

    def calculate_statistics(
        self,
        input_cross_embodiment_description,
        output_cross_embodiment_description,
    ):
        self.statistics_requests.extend([
            input_cross_embodiment_description,
            output_cross_embodiment_description,
        ])
        for description in self.statistics_requests:
            for embodiment in description.values():
                assert DataType.DONE not in embodiment

        def role_stats(description):
            stats = {}
            for data_type in {dt for e in description.values() for dt in e}:
                count = sum(
                    len(r)
                    for r in self.recordings
                    if data_type in description.get(r.robot_id, {})
                )
                stats[data_type] = [_joint_stats(count)]
            return stats

        return SynchronizedDatasetStatistics(
            synchronized_dataset_id=self.id,
            input_cross_embodiment_description=input_cross_embodiment_description,
            output_cross_embodiment_description=output_cross_embodiment_description,
            dataset_statistics={
                "input": role_stats(input_cross_embodiment_description),
                "output": role_stats(output_cross_embodiment_description),
            },
        )


def _description(*data_types: DataType, robot_id: str = ROBOT_ID):
    return {robot_id: {dt: dict(FULL_EMBODIMENT[dt]) for dt in data_types}}


def _done(sync_point: SynchronizedPoint) -> bool:
    return sync_point.data[DataType.DONE][DONE_SENSOR_NAME].done


def _position(sync_point: SynchronizedPoint) -> float:
    return sync_point.data[DataType.JOINT_POSITIONS][POSITION].value


def _velocity(sync_point: SynchronizedPoint) -> float:
    return sync_point.data[DataType.JOINT_VELOCITIES][VELOCITY].value


class TestDoneTailLength:
    @pytest.mark.parametrize(
        ("tail_s", "frequency", "expected"),
        [(0.5, 10, 5), (0.5, 30, 15), (0.01, 10, 1), (0.26, 10, 3)],
    )
    def test_rounds_to_whole_frames(self, tail_s, frequency, expected):
        assert done_tail_length(tail_s, frequency) == expected

    @pytest.mark.parametrize(("tail_s", "frequency"), [(0, 10), (-1, 10), (0.5, 0)])
    def test_rejects_non_positive_values(self, tail_s, frequency):
        with pytest.raises(ValueError):
            done_tail_length(tail_s, frequency)


class TestShouldSynthesizeDone:
    def _dataset(self, data_types):
        dataset = MagicMock()
        dataset.name = "my dataset"
        dataset.id = "dataset-id"
        dataset.data_types = data_types
        return dataset

    @pytest.mark.parametrize("tail_s", [None, 0])
    def test_off_when_not_requested(self, tail_s):
        dataset = self._dataset([DataType.JOINT_POSITIONS])
        assert not should_synthesize_done(tail_s, dataset)

    def test_on_when_dataset_has_no_done(self):
        dataset = self._dataset([DataType.JOINT_POSITIONS])
        assert should_synthesize_done(0.5, dataset)

    def test_off_with_warning_naming_dataset_when_done_is_logged(self, caplog):
        dataset = self._dataset([DataType.JOINT_POSITIONS, DataType.DONE])
        with caplog.at_level(logging.WARNING):
            assert not should_synthesize_done(0.5, dataset)
        assert "Dataset 'my dataset' already logs DONE" in caplog.text
        assert "dataset-id" not in caplog.text


class TestDescriptions:
    def test_add_done_for_every_robot_without_mutating_input(self):
        description = {
            **_description(DataType.JOINT_POSITIONS),
            **_description(DataType.JOINT_POSITIONS, robot_id=OTHER_ROBOT_ID),
        }
        with_done = add_done_to_cross_embodiment_description(description)
        for robot_id in (ROBOT_ID, OTHER_ROBOT_ID):
            assert with_done[robot_id][DataType.DONE] == {0: DONE_SENSOR_NAME}
            assert DataType.DONE not in description[robot_id]

    def test_remove_undoes_add(self):
        description = _description(DataType.JOINT_POSITIONS)
        with_done = add_done_to_cross_embodiment_description(description)
        assert remove_done_from_cross_embodiment_description(with_done) == description


class TestDoneTailRecording:
    def test_length_includes_tail(self):
        recording = DoneTailRecording(FakeRecording(6), tail_length=3)
        assert len(recording) == 9
        assert recording.num_real_frames == 6

    def test_real_frames_are_not_done_and_tail_repeats_last_frame_as_done(self):
        recording = DoneTailRecording(FakeRecording(6), tail_length=3)
        description = FULL_EMBODIMENT | {DataType.DONE: {0: DONE_SENSOR_NAME}}
        points = [recording.get_sync_point(t, description) for t in range(9)]
        assert [_done(p) for p in points] == [False] * 6 + [True] * 3
        assert [_position(p) for p in points] == [0, 1, 2, 3, 4, 5, 5, 5, 5]

    def test_tail_has_zero_velocity_and_real_frames_keep_theirs(self):
        recording = DoneTailRecording(FakeRecording(4), tail_length=2)
        points = [recording.get_sync_point(t, FULL_EMBODIMENT) for t in range(6)]
        assert [_velocity(p) for p in points] == [1, 1, 1, 1, 0, 0]

    def test_done_is_never_requested_from_the_wrapped_recording(self):
        inner = FakeRecording(4)
        recording = DoneTailRecording(inner, tail_length=2)
        recording.get_sync_point(5, {DataType.DONE: {0: DONE_SENSOR_NAME}})
        assert inner.requested == [set()]

    def test_get_sync_points_clamps_to_extended_length(self):
        recording = DoneTailRecording(FakeRecording(4), tail_length=2)
        points = recording.get_sync_points(3, 100, FULL_EMBODIMENT)
        assert [_done(p) for p in points] == [False, True, True]

    @pytest.mark.parametrize("timestep", [-1, 6])
    def test_get_sync_point_out_of_range(self, timestep):
        recording = DoneTailRecording(FakeRecording(4), tail_length=2)
        with pytest.raises(IndexError):
            recording.get_sync_point(timestep, FULL_EMBODIMENT)

    def test_indexing_and_iteration_cover_the_tail(self):
        recording = DoneTailRecording(FakeRecording(3), tail_length=2)
        assert [_done(p) for p in recording] == [False, False, False, True, True]
        assert _done(recording[-1])
        assert [_position(p) for p in recording[2:]] == [2, 2, 2]

    def test_survives_pickling_for_dataloader_workers(self):
        recording = DoneTailRecording(FakeRecording(3), tail_length=2)
        restored = pickle.loads(pickle.dumps(recording))
        assert len(restored) == 5
        assert _done(restored[4])

    def test_other_attributes_come_from_the_wrapped_recording(self):
        recording = DoneTailRecording(FakeRecording(3, name="abc"), tail_length=2)
        assert recording.id == "abc-id"
        assert recording.name == "abc"
        assert recording.robot_id == ROBOT_ID
        assert recording.frequency == FREQUENCY


class TestDoneTailDataset:
    def test_tail_length_follows_frequency(self):
        dataset = DoneTailDataset(FakeSynchronizedDataset([FakeRecording(4)]), 0.5)
        assert dataset.tail_length == 5

    def test_id_includes_tail_so_caches_do_not_collide(self):
        inner = FakeSynchronizedDataset([FakeRecording(4)])
        assert DoneTailDataset(inner, 0.5).id == "synced-id+done_tail_0.5s"
        assert DoneTailDataset(inner, 0.5).id != DoneTailDataset(inner, 1.0).id

    def test_raw_dataset_view_reports_done(self):
        dataset = DoneTailDataset(FakeSynchronizedDataset([FakeRecording(4)]), 0.5)
        assert DataType.DONE in dataset.dataset.data_types
        embodiment = dataset.dataset.get_full_embodiment_description(ROBOT_ID)
        assert embodiment[DataType.DONE] == {0: DONE_SENSOR_NAME}
        assert embodiment[DataType.JOINT_POSITIONS] == {0: POSITION}
        assert dataset.dataset.name == "my dataset"

    def test_indexing_iteration_and_slicing_return_extended_recordings(self):
        recordings = [FakeRecording(4, name="a"), FakeRecording(6, name="b")]
        dataset = DoneTailDataset(FakeSynchronizedDataset(recordings), 0.2)
        assert len(dataset) == 2
        assert [len(r) for r in dataset] == [6, 8]
        assert dataset[1] is dataset[-1]
        assert isinstance(dataset[0], DoneTailRecording)
        sliced = dataset[1:]
        assert isinstance(sliced, DoneTailDataset)
        assert [r.name for r in sliced] == ["b"]

    def test_statistics_add_done_and_count_the_tail(self):
        recordings = [FakeRecording(4), FakeRecording(6)]
        inner = FakeSynchronizedDataset(recordings)
        dataset = DoneTailDataset(inner, 0.2)  # 2 tail frames per episode
        input_description = _description(DataType.JOINT_POSITIONS)
        output_description = add_done_to_cross_embodiment_description(
            _description(DataType.JOINT_POSITIONS)
        )

        stats = dataset.calculate_statistics(input_description, output_description)

        assert inner.statistics_requests == [
            input_description,
            remove_done_from_cross_embodiment_description(output_description),
        ]
        for role in ("input", "output"):
            joint_stats = stats.dataset_statistics[role][DataType.JOINT_POSITIONS][0]
            assert joint_stats.value.count.tolist() == [10 + 4]
        assert DataType.DONE not in stats.dataset_statistics["input"]
        done_stats = stats.dataset_statistics["output"][DataType.DONE]
        assert len(done_stats) == 1
        assert isinstance(done_stats[0], DoneDataStats)
        assert done_stats[0].done.count.tolist() == [14]
        assert done_stats[0].done.mean.tolist() == pytest.approx([4 / 14])
        assert done_stats[0].done.min.tolist() == [0.0]
        assert done_stats[0].done.max.tolist() == [1.0]

    def test_statistics_only_count_the_tail_for_robots_with_the_slot(self):
        recordings = [FakeRecording(4), FakeRecording(6, robot_id=OTHER_ROBOT_ID)]
        dataset = DoneTailDataset(FakeSynchronizedDataset(recordings), 0.2)
        description = _description(DataType.JOINT_POSITIONS)
        output_description = add_done_to_cross_embodiment_description(description)

        stats = dataset.calculate_statistics(description, output_description)

        joint_stats = stats.dataset_statistics["input"][DataType.JOINT_POSITIONS][0]
        assert joint_stats.value.count.tolist() == [4 + 2]
        done_stats = stats.dataset_statistics["output"][DataType.DONE][0]
        assert done_stats.done.count.tolist() == [6]
        assert done_stats.done.mean.tolist() == pytest.approx([2 / 6])


class TestTrainingDatasetOverDoneTail:
    """The unchanged training dataset trains on the synthesized done flag."""

    @patch("neuracore.login")
    def test_samples_include_the_tail_with_done_labels(
        self, mock_login, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(psd, "DEFAULT_CACHE_DIR", tmp_path)
        recordings = [FakeRecording(6), FakeRecording(4, name="short")]
        dataset = DoneTailDataset(FakeSynchronizedDataset(recordings), 0.3)
        horizon = 2

        training_dataset = psd.PytorchSynchronizedDataset(
            synchronized_dataset=dataset,
            input_cross_embodiment_description=_description(DataType.JOINT_POSITIONS),
            output_cross_embodiment_description=(
                add_done_to_cross_embodiment_description(
                    _description(DataType.JOINT_POSITIONS)
                )
            ),
            output_prediction_horizon=horizon,
            input_preprocessing_config=PreprocessingConfiguration(),
            output_preprocessing_config=PreprocessingConfiguration(),
        )

        # Each episode contributes L + K - 1 samples (the last frame is dropped).
        assert len(training_dataset) == (6 + 3 - 1) + (4 + 3 - 1)

        def done_labels(episode_idx: int, timestep: int) -> list[float]:
            sample = training_dataset.load_sample(episode_idx, timestep)
            done = sample.outputs[DataType.DONE][0].done
            return done.flatten().tolist()

        def input_position(episode_idx: int, timestep: int) -> float:
            sample = training_dataset.load_sample(episode_idx, timestep)
            joint = sample.inputs[DataType.JOINT_POSITIONS][0].value
            return joint.flatten().tolist()[0]

        # DONE predicts the next steps, so the window crosses the end at t=L-2.
        assert done_labels(0, 3) == [0.0, 0.0]
        assert done_labels(0, 4) == [0.0, 1.0]
        assert done_labels(0, 5) == [1.0, 1.0]
        assert done_labels(0, 7) == [1.0, 1.0]
        # Tail inputs repeat the last real frame.
        assert input_position(0, 7) == 5.0
        assert done_labels(1, 2) == [0.0, 1.0]


class TestValidateSynthesizeDone:
    def _dataset(self, data_types):
        dataset = MagicMock()
        dataset.data_types = data_types
        return dataset

    def test_not_set_is_valid(self):
        _validate_synthesize_done(None, self._dataset([]), "algo", set())

    @pytest.mark.parametrize("tail_s", [0, -0.5])
    def test_rejects_non_positive_tail(self, tail_s):
        with pytest.raises(ValueError, match="positive"):
            _validate_synthesize_done(tail_s, self._dataset([]), "algo", set())

    def test_rejects_algorithm_without_done_output(self):
        with pytest.raises(ValueError, match="'algo' does not support DONE"):
            _validate_synthesize_done(
                0.5, self._dataset([]), "algo", {DataType.JOINT_POSITIONS}
            )

    def test_accepts_algorithm_with_done_output(self):
        _validate_synthesize_done(0.5, self._dataset([]), "algo", {DataType.DONE})

    def test_logged_done_skips_the_algorithm_check(self):
        _validate_synthesize_done(
            0.5, self._dataset([DataType.DONE]), "algo", {DataType.JOINT_POSITIONS}
        )
