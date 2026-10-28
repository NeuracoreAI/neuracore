"""Tests for validation rollout selection and file layout."""

import json
from functools import partial
from unittest.mock import MagicMock

import pytest
import torch
from neuracore_types import BatchedEndEffectorPoseData, BatchedJointData, DataType
from PIL import Image
from torch.utils.data import DataLoader, Subset

from neuracore.ml.core.ml_types import BatchedInferenceInputs
from neuracore.ml.datasets.pytorch_dummy_dataset import PytorchDummyDataset
from neuracore.ml.datasets.pytorch_neuracore_dataset import SampleIdentity
from neuracore.ml.preprocessing.base import PreprocessingConfiguration
from neuracore.ml.utils.training_storage_handler import TrainingStorageHandler
from neuracore.ml.utils.validation_rollouts import (
    ValidationRolloutConfig,
    save_validation_rollouts,
    select_validation_rollout_indices,
    write_rollout_manifest,
)


def _storage(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "neuracore.ml.utils.training_storage_handler.get_current_org",
        lambda: "test-org",
    )
    auth = MagicMock()
    auth.get_headers.return_value = {"Authorization": "Bearer test"}
    monkeypatch.setattr(
        "neuracore.ml.utils.training_storage_handler.get_auth",
        lambda: auth,
    )
    return TrainingStorageHandler(local_dir=str(tmp_path))


def _validation_loader(validation_indices):
    base = list(range(max(validation_indices, default=0) + 1))
    return DataLoader(Subset(base, validation_indices))


def _rollout_config(num_points, seed):
    return ValidationRolloutConfig(num_points=num_points, frequency=1, seed=seed)


def test_select_validation_rollout_indices_is_a_deterministic_sample():
    validation_indices = list(range(40))
    loader = _validation_loader(validation_indices)
    config = _rollout_config(num_points=6, seed=7)

    first = select_validation_rollout_indices(loader, config)
    second = select_validation_rollout_indices(loader, config)

    assert first == second
    assert len(first) == 6
    assert len(set(first)) == 6
    assert set(first) <= set(validation_indices)


def test_select_validation_rollout_indices_uses_every_sample_when_split_is_smaller():
    validation_indices = [4, 9, 12]
    loader = _validation_loader(validation_indices)
    config = _rollout_config(num_points=10, seed=3)

    first = select_validation_rollout_indices(loader, config)
    second = select_validation_rollout_indices(loader, config)

    assert first == second
    assert sorted(first) == validation_indices


def test_select_validation_rollout_indices_rejects_empty_split():
    loader = _validation_loader([])

    with pytest.raises(ValueError):
        select_validation_rollout_indices(loader, _rollout_config(4, 0))


@pytest.mark.parametrize(
    ("field", "value"),
    [("num_points", -1), ("frequency", 0), ("frequency", -1)],
)
def test_validation_rollout_config_rejects_invalid_settings(field, value):
    settings = {"num_points": 2, "frequency": 5, field: value}

    with pytest.raises(ValueError, match=field):
        ValidationRolloutConfig(seed=0, **settings)


class _ScriptedModel:
    """Model stand-in that returns one fixed inference output."""

    def __init__(self, outputs: dict) -> None:
        self.training = True
        self.outputs = outputs

    def eval(self) -> None:
        self.training = False

    def train(self, mode: bool = True) -> None:
        self.training = mode

    def forward(self, batch: BatchedInferenceInputs) -> dict:
        return self.outputs


def test_save_validation_rollouts_writes_images_and_trace_json(tmp_path, monkeypatch):
    dataset = PytorchDummyDataset(
        input_cross_embodiment_description={
            "robot_0": {
                DataType.JOINT_POSITIONS: {0: "shoulder", 1: "elbow"},
                DataType.END_EFFECTOR_POSES: {0: "ee"},
                DataType.RGB_IMAGES: {0: "wrist/cam"},
                DataType.DEPTH_IMAGES: {0: "depth"},
            },
        },
        output_cross_embodiment_description={
            "robot_0": {
                DataType.JOINT_TARGET_POSITIONS: {0: "shoulder"},
                DataType.END_EFFECTOR_POSES: {0: "ee"},
            },
        },
        num_samples=1,
        num_episodes=1,
        output_prediction_horizon=2,
    )
    sample = dataset[0]
    sample.inputs[DataType.JOINT_POSITIONS][0].value = torch.tensor([[[0.12]]])
    sample.inputs[DataType.JOINT_POSITIONS][1].value = torch.tensor([[[-0.4]]])
    sample.inputs[DataType.END_EFFECTOR_POSES][0].pose = torch.tensor(
        [[[0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]]]
    )
    sample.outputs[DataType.JOINT_TARGET_POSITIONS][0].value = torch.tensor(
        [[[1.0], [1.1]]]
    )
    sample.outputs[DataType.END_EFFECTOR_POSES][0].pose = torch.tensor([[
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]])
    model = _ScriptedModel({
        DataType.JOINT_TARGET_POSITIONS: [
            BatchedJointData(value=torch.tensor([[[0.1], [0.2], [0.3]]])),
        ],
        DataType.END_EFFECTOR_POSES: [
            BatchedEndEffectorPoseData(
                pose=torch.tensor([[
                    [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]])
            ),
        ],
    })

    storage = _storage(tmp_path, monkeypatch)
    save_validation_rollouts(
        model=model,
        dataset=dataset,
        selected_indices=[0],
        batch_size=1,
        device=torch.device("cpu"),
        inference_device_preprocessing=(
            PreprocessingConfiguration(),
            PreprocessingConfiguration(),
        ),
        output_dir=tmp_path,
        epoch=5,
        storage_handler=storage,
        save_input_points=True,
    )
    storage.wait_for_rollout_saves()

    root = tmp_path / "validation_rollouts"
    epoch_dir = root / "outputs" / "epoch_0005"
    point_dir = root / "input_points" / "dummy-sample-0" / "point_000000"
    rgb_path = point_dir / "inputs" / "images" / "rgb" / "wrist%2Fcam.jpeg"
    depth_path = point_dir / "inputs" / "images" / "depth" / "depth.jpeg"
    assert rgb_path.is_file()
    assert depth_path.is_file()
    Image.open(rgb_path).verify()
    Image.open(depth_path).verify()

    state = json.loads((point_dir / "inputs" / "state_input.json").read_text())
    assert state["JOINT_POSITIONS"]["shoulder"] == pytest.approx(0.12)
    assert state["JOINT_POSITIONS"]["elbow"] == pytest.approx(-0.4)
    assert state["END_EFFECTOR_POSES"]["ee"] == pytest.approx(
        [0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]
    )
    assert "RGB_IMAGES" not in state
    assert "DEPTH_IMAGES" not in state

    prediction = json.loads(
        (epoch_dir / "dummy-sample-0" / "point_000000.json").read_text()
    )
    assert prediction["JOINT_TARGET_POSITIONS"]["shoulder"] == pytest.approx(
        [0.1, 0.2, 0.3]
    )
    assert prediction["END_EFFECTOR_POSES"]["ee"][0] == pytest.approx(
        [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )

    truth = json.loads((point_dir / "ground_truth.json").read_text())
    assert truth["JOINT_TARGET_POSITIONS"]["shoulder"] == pytest.approx([1.0, 1.1])
    pose_horizon = truth["END_EFFECTOR_POSES"]["ee"]
    assert pose_horizon[0] == pytest.approx([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert pose_horizon[1] == pytest.approx([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


class _CountingModel:
    """Model stand-in that echoes each row's joint value as the prediction."""

    def __init__(self) -> None:
        self.training = True
        self.batch_sizes: list[int] = []

    def eval(self) -> None:
        self.training = False

    def train(self, mode: bool = True) -> None:
        self.training = mode

    def forward(self, batch: BatchedInferenceInputs) -> dict:
        self.batch_sizes.append(batch.batch_size)
        value = batch.inputs[DataType.JOINT_POSITIONS][0].value
        horizon = value.expand(-1, 2, -1).contiguous()
        return {
            DataType.JOINT_TARGET_POSITIONS: [BatchedJointData(value=horizon)],
        }


class _DistinctSampleDataset(PytorchDummyDataset):
    """Dummy dataset whose samples carry their index, on alternating robots."""

    def __getitem__(self, idx: int):
        cloned = self.collate_fn([super().__getitem__(idx)])
        cloned.inputs[DataType.JOINT_POSITIONS][0].value = torch.full(
            (1, 1, 1), float(idx + 1)
        )
        cloned.outputs[DataType.JOINT_TARGET_POSITIONS][0].value = torch.full(
            (1, 2, 1), float(idx + 10)
        )
        return cloned

    def get_sample_identity(self, idx: int) -> SampleIdentity:
        robot_id = "robot_0" if idx % 2 == 0 else "robot_1"
        return SampleIdentity(
            recording_id=f"dummy-sample-{idx}",
            timestep=idx,
            robot_id=robot_id,
        )


def _distinct_dataset() -> _DistinctSampleDataset:
    return _DistinctSampleDataset(
        input_cross_embodiment_description={
            "robot_0": {DataType.JOINT_POSITIONS: {0: "shoulder"}},
            "robot_1": {DataType.JOINT_POSITIONS: {0: "elbow"}},
        },
        output_cross_embodiment_description={
            "robot_0": {DataType.JOINT_TARGET_POSITIONS: {0: "shoulder"}},
            "robot_1": {DataType.JOINT_TARGET_POSITIONS: {0: "elbow"}},
        },
        num_samples=6,
        num_episodes=1,
        output_prediction_horizon=2,
    )


def _save(
    dataset,
    model,
    storage,
    tmp_path,
    *,
    epoch: int,
    save_input_points: bool,
    selected_indices=None,
):
    save_validation_rollouts(
        model=model,
        dataset=dataset,
        selected_indices=(
            list(range(6)) if selected_indices is None else selected_indices
        ),
        batch_size=4,
        device=torch.device("cpu"),
        inference_device_preprocessing=(
            PreprocessingConfiguration(),
            PreprocessingConfiguration(),
        ),
        output_dir=tmp_path,
        epoch=epoch,
        storage_handler=storage,
        save_input_points=save_input_points,
    )


def test_save_validation_rollouts_forwards_chunks_and_keeps_each_row(
    tmp_path, monkeypatch
):
    dataset = _distinct_dataset()
    model = _CountingModel()
    storage = _storage(tmp_path, monkeypatch)

    _save(dataset, model, storage, tmp_path, epoch=2, save_input_points=True)
    storage.wait_for_rollout_saves()

    epoch_dir = tmp_path / "validation_rollouts" / "outputs" / "epoch_0002"
    assert model.batch_sizes == [4, 2]
    input_root = tmp_path / "validation_rollouts" / "input_points"
    for index in range(6):
        trace_name = "shoulder" if index % 2 == 0 else "elbow"
        recording_id = f"dummy-sample-{index}"
        point_dir = input_root / recording_id / f"point_{index:06d}"
        state = json.loads((point_dir / "inputs" / "state_input.json").read_text())
        truth = json.loads((point_dir / "ground_truth.json").read_text())
        prediction = json.loads(
            (epoch_dir / recording_id / f"point_{index:06d}.json").read_text()
        )
        assert state["JOINT_POSITIONS"][trace_name] == pytest.approx(float(index + 1))
        assert truth["JOINT_TARGET_POSITIONS"][trace_name] == pytest.approx(
            [float(index + 10), float(index + 10)]
        )
        assert prediction["JOINT_TARGET_POSITIONS"][trace_name] == pytest.approx(
            [float(index + 1), float(index + 1)]
        )


def test_later_epochs_write_only_predictions(tmp_path, monkeypatch):
    dataset = _distinct_dataset()
    model = _CountingModel()
    storage = _storage(tmp_path, monkeypatch)

    _save(
        dataset,
        model,
        storage,
        tmp_path,
        epoch=3,
        save_input_points=False,
        selected_indices=[1, 4],
    )
    storage.wait_for_rollout_saves()

    root = tmp_path / "validation_rollouts"
    epoch_dir = root / "outputs" / "epoch_0003"
    assert not (root / "input_points").exists()
    assert sorted(path.name for path in epoch_dir.rglob("*.json")) == [
        "point_000001.json",
        "point_000004.json",
    ]


def test_manifest_lists_the_selected_points(tmp_path):
    dataset = _distinct_dataset()
    root_dir = tmp_path / "validation_rollouts"

    write_rollout_manifest(root_dir, [dataset.get_sample_identity(i) for i in (3, 0)])

    assert json.loads((root_dir / "manifest.json").read_text()) == {
        "points": [
            {"recording_id": "dummy-sample-3", "timestep": 3, "robot_id": "robot_1"},
            {"recording_id": "dummy-sample-0", "timestep": 0, "robot_id": "robot_0"},
        ],
    }


def test_cloud_upload_sends_manifest_inputs_then_outputs(tmp_path, monkeypatch):
    dataset = _distinct_dataset()
    storage = _storage(tmp_path, monkeypatch)
    storage.log_to_cloud = True
    uploaded: list[str] = []

    def _upload(path, remote_filepath, content_type):
        uploaded.append(remote_filepath)
        return True

    monkeypatch.setattr(storage, "upload_file", _upload)
    root_dir = tmp_path / "validation_rollouts"

    storage.submit_rollout_save(
        partial(write_rollout_manifest, root_dir, [dataset.get_sample_identity(2)])
    )
    storage.submit_rollout_save(partial(storage.upload_rollout_manifest, root_dir))
    _save(
        dataset,
        _CountingModel(),
        storage,
        tmp_path,
        epoch=1,
        save_input_points=True,
        selected_indices=[2],
    )
    storage.wait_for_rollout_saves()

    root = "validation_rollouts"
    point = "dummy-sample-2/point_000002"
    assert uploaded == [
        f"{root}/manifest.json",
        f"{root}/input_points/{point}/ground_truth.json",
        f"{root}/input_points/{point}/inputs/state_input.json",
        f"{root}/outputs/epoch_0001/{point}.json",
    ]


def test_failed_upload_is_left_on_disk_and_does_not_stop_the_rest(
    tmp_path, monkeypatch
):
    dataset = _distinct_dataset()
    storage = _storage(tmp_path, monkeypatch)
    storage.log_to_cloud = True
    uploaded: list[str] = []

    def _upload(path, remote_filepath, content_type):
        if path.name == "ground_truth.json":
            return False
        uploaded.append(remote_filepath)
        return True

    monkeypatch.setattr(storage, "upload_file", _upload)

    _save(
        dataset,
        _CountingModel(),
        storage,
        tmp_path,
        epoch=1,
        save_input_points=True,
        selected_indices=[2],
    )
    storage.wait_for_rollout_saves()

    point = "dummy-sample-2/point_000002"
    root = tmp_path / "validation_rollouts"
    epoch_dir = root / "outputs" / "epoch_0001"
    assert (root / "input_points" / point / "ground_truth.json").exists()
    assert f"validation_rollouts/outputs/epoch_0001/{point}.json" in uploaded
    assert not (epoch_dir / "dummy-sample-2" / "point_000002.json").exists()
