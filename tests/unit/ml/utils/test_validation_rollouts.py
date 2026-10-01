"""Tests for validation rollout selection and file layout."""

import json

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

from neuracore.ml.datasets.pytorch_neuracore_dataset import SampleIdentity
from neuracore.ml.utils.validation_rollouts import (
    select_validation_rollout_indices,
    write_rollout_manifest,
    write_rollout_point,
)


def test_select_validation_rollout_indices_is_a_deterministic_sample():
    validation_indices = list(range(40))

    first = select_validation_rollout_indices(validation_indices, num_points=6, seed=7)
    second = select_validation_rollout_indices(validation_indices, num_points=6, seed=7)

    assert first == second
    assert len(first) == 6
    assert len(set(first)) == 6
    assert set(first) <= set(validation_indices)


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

    rgb_path = tmp_path / "inputs" / "images" / "rgb" / "wrist\\cam.jpeg"
    depth_path = tmp_path / "inputs" / "images" / "depths" / "depth.jpeg"
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
