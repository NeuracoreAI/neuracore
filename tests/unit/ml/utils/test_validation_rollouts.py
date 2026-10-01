"""Tests for validation rollout selection and file layout."""

import json
from collections import Counter

import pytest
import torch
from neuracore_types import (
    BatchedDepthData,
    BatchedEndEffectorPoseData,
    BatchedJointData,
    BatchedRGBData,
    DataType,
)
from PIL import Image

from neuracore.ml.datasets.pytorch_dummy_dataset import PytorchDummyDataset
from neuracore.ml.datasets.pytorch_neuracore_dataset import SampleIdentity
from neuracore.ml.utils.validation_rollouts import (
    select_validation_rollout_indices,
    write_rollout_manifest,
    write_rollout_point,
)

_INPUTS = {
    "robot_0": {
        DataType.JOINT_POSITIONS: {0: "shoulder", 1: "elbow"},
        DataType.END_EFFECTOR_POSES: {0: "ee"},
        DataType.RGB_IMAGES: {0: "wrist/cam"},
        DataType.DEPTH_IMAGES: {0: "depth"},
    }
}
_OUTPUTS = {
    "robot_0": {
        DataType.JOINT_TARGET_POSITIONS: {0: "shoulder"},
        DataType.END_EFFECTOR_POSES: {0: "ee"},
    }
}


def _dataset(num_samples: int, num_episodes: int) -> PytorchDummyDataset:
    return PytorchDummyDataset(
        input_cross_embodiment_description=_INPUTS,
        output_cross_embodiment_description=_OUTPUTS,
        num_samples=num_samples,
        num_episodes=num_episodes,
        output_prediction_horizon=3,
    )


def test_get_sample_identity_uses_episode_blocks():
    dataset = _dataset(num_samples=10, num_episodes=2)

    first = dataset.get_sample_identity(0)
    assert first == SampleIdentity("dummy-episode-0", 0, "robot_0")

    next_episode = dataset.get_sample_identity(5)
    assert next_episode.recording_id == "dummy-episode-1"
    assert next_episode.timestep == 0
    assert next_episode.robot_id == "robot_0"


def test_select_validation_rollout_indices_is_deterministic_and_spread():
    dataset = _dataset(num_samples=40, num_episodes=4)
    validation_indices = list(range(40))

    first = select_validation_rollout_indices(
        dataset, validation_indices, num_points=6, seed=7
    )
    second = select_validation_rollout_indices(
        dataset, validation_indices, num_points=6, seed=7
    )

    assert first == second
    assert set(first) <= set(validation_indices)
    counts = Counter(dataset.get_sample_identity(index).recording_id for index in first)
    assert len(counts) == 4
    assert max(counts.values()) - min(counts.values()) <= 1


def test_select_validation_rollout_indices_stays_inside_the_validation_split():
    dataset = _dataset(num_samples=20, num_episodes=2)
    validation_indices = [0, 1, 2]

    selected = select_validation_rollout_indices(
        dataset, validation_indices, num_points=10, seed=1
    )

    assert set(selected) <= set(validation_indices)
    assert len(selected) == 3
    assert (
        select_validation_rollout_indices(
            dataset, validation_indices, num_points=0, seed=1
        )
        == []
    )


def test_write_rollout_point_writes_images_and_trace_json(tmp_path):
    inputs = {
        DataType.JOINT_POSITIONS: {
            "shoulder": BatchedJointData(value=torch.tensor([[[0.12]]])),
            "elbow": BatchedJointData(value=torch.tensor([[[-0.4]]])),
        },
        DataType.END_EFFECTOR_POSES: {
            "ee": BatchedEndEffectorPoseData(
                pose=torch.tensor([[[0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]]])
            ),
        },
        DataType.RGB_IMAGES: {
            "wrist/cam": BatchedRGBData(
                frame=torch.zeros((1, 1, 3, 4, 4)),
                extrinsics=torch.zeros((1, 1, 4, 4)),
                intrinsics=torch.zeros((1, 1, 3, 3)),
            ),
        },
        DataType.DEPTH_IMAGES: {
            "depth": BatchedDepthData(
                frame=torch.ones((1, 1, 1, 4, 4)),
                extrinsics=torch.zeros((1, 1, 4, 4)),
                intrinsics=torch.zeros((1, 1, 3, 3)),
            ),
        },
    }
    predictions = {
        DataType.JOINT_TARGET_POSITIONS: {
            "shoulder": BatchedJointData(value=torch.tensor([[[0.1], [0.2], [0.3]]])),
        },
    }
    ground_truth = {
        DataType.JOINT_TARGET_POSITIONS: {
            "shoulder": BatchedJointData(value=torch.tensor([[[1.0], [1.1], [1.2]]])),
        },
        DataType.END_EFFECTOR_POSES: {
            "ee": BatchedEndEffectorPoseData(
                pose=torch.tensor([[
                    [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]])
            ),
        },
    }

    write_rollout_point(tmp_path, inputs, predictions, ground_truth)
    write_rollout_manifest(
        tmp_path,
        epoch=5,
        points=[SampleIdentity("rec", 42, "robot_0")],
    )

    rgb_path = tmp_path / "inputs" / "images" / "rgb" / "wrist\\cam.png"
    depth_path = tmp_path / "inputs" / "images" / "depths" / "depth.png"
    assert rgb_path.is_file()
    assert depth_path.is_file()
    Image.open(rgb_path).verify()
    Image.open(depth_path).verify()

    state = json.loads((tmp_path / "inputs" / "state_input.json").read_text())
    assert state["JOINT_POSITIONS"]["shoulder"] == pytest.approx(0.12)
    assert state["JOINT_POSITIONS"]["elbow"] == pytest.approx(-0.4)
    assert state["END_EFFECTOR_POSES"]["ee"] == pytest.approx(
        [0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]
    )
    assert "RGB_IMAGES" not in state
    assert "DEPTH_IMAGES" not in state

    prediction = json.loads(
        (tmp_path / "outputs" / "prediction_horizon.json").read_text()
    )
    assert prediction["JOINT_TARGET_POSITIONS"]["shoulder"] == pytest.approx(
        [0.1, 0.2, 0.3]
    )

    truth = json.loads((tmp_path / "outputs" / "ground_truth.json").read_text())
    assert truth["JOINT_TARGET_POSITIONS"]["shoulder"] == pytest.approx([1.0, 1.1, 1.2])
    pose_horizon = truth["END_EFFECTOR_POSES"]["ee"]
    assert pose_horizon[0] == pytest.approx([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert pose_horizon[1] == pytest.approx([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest == {
        "epoch": 5,
        "points": [{"recording_id": "rec", "timestep": 42, "robot_id": "robot_0"}],
    }
