"""Tests for validation rollout selection and file layout."""

import json
from unittest.mock import MagicMock

import pytest
import torch
from neuracore_types import BatchedEndEffectorPoseData, BatchedJointData, DataType
from PIL import Image

from neuracore.ml.core.ml_types import BatchedInferenceInputs
from neuracore.ml.datasets.pytorch_dummy_dataset import PytorchDummyDataset
from neuracore.ml.datasets.pytorch_neuracore_dataset import SampleIdentity
from neuracore.ml.preprocessing.base import PreprocessingConfiguration
from neuracore.ml.utils.training_storage_handler import TrainingStorageHandler
from neuracore.ml.utils.validation_rollouts import (
    ValidationRolloutConfig,
    save_validation_rollouts,
    select_validation_rollout_indices,
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


def test_select_validation_rollout_indices_is_a_deterministic_sample():
    validation_indices = list(range(40))

    first = select_validation_rollout_indices(validation_indices, num_points=6, seed=7)
    second = select_validation_rollout_indices(validation_indices, num_points=6, seed=7)

    assert first == second
    assert len(first) == 6
    assert len(set(first)) == 6
    assert set(first) <= set(validation_indices)


def test_select_validation_rollout_indices_uses_every_sample_when_split_is_smaller():
    validation_indices = [4, 9, 12]

    first = select_validation_rollout_indices(validation_indices, num_points=10, seed=3)
    second = select_validation_rollout_indices(
        validation_indices, num_points=10, seed=3
    )

    assert first == second
    assert sorted(first) == validation_indices


def test_select_validation_rollout_indices_rejects_unusable_requests():
    with pytest.raises(ValueError):
        select_validation_rollout_indices([1, 2], num_points=0, seed=0)
    with pytest.raises(ValueError):
        select_validation_rollout_indices([], num_points=4, seed=0)


@pytest.mark.parametrize("field", ["num_points", "frequency"])
def test_validation_rollout_config_rejects_non_positive_settings(field):
    settings = {"num_points": 2, "frequency": 5, field: 0}

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
    epoch_dir = save_validation_rollouts(
        model=model,
        dataset=dataset,
        validation_indices=[0],
        config=ValidationRolloutConfig(
            num_points=1,
            frequency=1,
            seed=0,
        ),
        batch_size=1,
        device=torch.device("cpu"),
        inference_device_preprocessing=(
            PreprocessingConfiguration(),
            PreprocessingConfiguration(),
        ),
        output_dir=tmp_path,
        epoch=5,
        storage_handler=storage,
    )
    storage.wait_for_rollout_saves()

    point_dir = epoch_dir / "dummy-sample-0" / "point_000000"
    rgb_path = point_dir / "inputs" / "images" / "rgb" / "wrist\\cam.jpeg"
    depth_path = point_dir / "inputs" / "images" / "depths" / "depth.jpeg"
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
        (point_dir / "outputs" / "prediction_horizon.json").read_text()
    )
    assert prediction["JOINT_TARGET_POSITIONS"]["shoulder"] == pytest.approx(
        [0.1, 0.2, 0.3]
    )
    assert prediction["END_EFFECTOR_POSES"]["ee"][0] == pytest.approx(
        [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )

    truth = json.loads((point_dir / "outputs" / "ground_truth.json").read_text())
    assert truth["JOINT_TARGET_POSITIONS"]["shoulder"] == pytest.approx([1.0, 1.1])
    pose_horizon = truth["END_EFFECTOR_POSES"]["ee"]
    assert pose_horizon[0] == pytest.approx([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert pose_horizon[1] == pytest.approx([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    manifest = json.loads((epoch_dir / "manifest.json").read_text())
    assert manifest == {
        "epoch": 5,
        "points": [
            {"recording_id": "dummy-sample-0", "timestep": 0, "robot_id": "robot_0"}
        ],
    }


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


def test_save_validation_rollouts_forwards_chunks_and_keeps_each_row(
    tmp_path, monkeypatch
):
    dataset = _DistinctSampleDataset(
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
    model = _CountingModel()
    storage = _storage(tmp_path, monkeypatch)

    epoch_dir = save_validation_rollouts(
        model=model,
        dataset=dataset,
        validation_indices=list(range(6)),
        config=ValidationRolloutConfig(
            num_points=6,
            frequency=1,
            seed=0,
        ),
        batch_size=4,
        device=torch.device("cpu"),
        inference_device_preprocessing=(
            PreprocessingConfiguration(),
            PreprocessingConfiguration(),
        ),
        output_dir=tmp_path,
        epoch=2,
        storage_handler=storage,
    )
    storage.wait_for_rollout_saves()

    assert model.batch_sizes == [4, 2]
    manifest = json.loads((epoch_dir / "manifest.json").read_text())
    assert len(manifest["points"]) == 6
    for point in manifest["points"]:
        index = point["timestep"]
        robot_id = point["robot_id"]
        trace_name = "shoulder" if robot_id == "robot_0" else "elbow"
        point_dir = epoch_dir / point["recording_id"] / f"point_{index:06d}"
        state = json.loads((point_dir / "inputs" / "state_input.json").read_text())
        truth = json.loads((point_dir / "outputs" / "ground_truth.json").read_text())
        prediction = json.loads(
            (point_dir / "outputs" / "prediction_horizon.json").read_text()
        )
        assert state["JOINT_POSITIONS"][trace_name] == pytest.approx(float(index + 1))
        assert truth["JOINT_TARGET_POSITIONS"][trace_name] == pytest.approx(
            [float(index + 10), float(index + 10)]
        )
        assert prediction["JOINT_TARGET_POSITIONS"][trace_name] == pytest.approx(
            [float(index + 1), float(index + 1)]
        )
