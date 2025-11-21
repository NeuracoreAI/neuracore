import inspect
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from neuracore_types import (
    DataItemStats,
    DatasetStatistics,
    DataType,
    ModelInitDescription,
    ModelPrediction,
)

from neuracore.ml import (
    BatchedInferenceInputs,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    MaskableData,
)
from neuracore.ml.algorithms.cnnmlp.cnnmlp import CNNMLP
from neuracore.ml.utils.device_utils import get_default_device
from neuracore.ml.utils.validate import run_validation

BS = 2
CAMS = 1
JOINT_POSITION_DIM = 32
OUTPUT_PRED_DIM = JOINT_POSITION_DIM
PRED_HORIZON = 10
DEVICE = get_default_device()


@pytest.fixture
def model_init_description_partial() -> ModelInitDescription:
    dataset_statistics = DatasetStatistics(
        data={
            DataType.JOINT_POSITIONS: {
                "default": DataItemStats(
                    mean=np.zeros(JOINT_POSITION_DIM, dtype=float).tolist(),
                    std=np.ones(JOINT_POSITION_DIM, dtype=float).tolist(),
                )
            },
            DataType.JOINT_TARGET_POSITIONS: {
                "default": DataItemStats(
                    mean=np.zeros(JOINT_POSITION_DIM, dtype=float).tolist(),
                    std=np.ones(JOINT_POSITION_DIM, dtype=float).tolist(),
                )
            },
            DataType.JOINT_VELOCITIES: {
                "default": DataItemStats(
                    mean=np.zeros(JOINT_POSITION_DIM, dtype=float).tolist(),
                    std=np.ones(JOINT_POSITION_DIM, dtype=float).tolist(),
                )
            },
            DataType.JOINT_TORQUES: {
                "default": DataItemStats(
                    mean=np.zeros(JOINT_POSITION_DIM, dtype=float).tolist(),
                    std=np.ones(JOINT_POSITION_DIM, dtype=float).tolist(),
                )
            },
            DataType.RGB_IMAGES: {
                "camera_0": DataItemStats(
                    max_len=CAMS,
                )
            },
        }
    )
    return ModelInitDescription(
        dataset_statistics=dataset_statistics,
        input_data_types=[
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.RGB_IMAGES,
        ],
        output_data_types=[DataType.JOINT_TARGET_POSITIONS],
        output_prediction_horizon=PRED_HORIZON,
    )


@pytest.fixture
def model_init_description_full() -> ModelInitDescription:
    dataset_statistics = DatasetStatistics(
        data={
            DataType.JOINT_POSITIONS: {
                "default": DataItemStats(
                    mean=np.zeros(JOINT_POSITION_DIM, dtype=float).tolist(),
                    std=np.ones(JOINT_POSITION_DIM, dtype=float).tolist(),
                )
            },
            DataType.JOINT_TARGET_POSITIONS: {
                "default": DataItemStats(
                    mean=np.zeros(JOINT_POSITION_DIM, dtype=float).tolist(),
                    std=np.ones(JOINT_POSITION_DIM, dtype=float).tolist(),
                )
            },
            DataType.JOINT_VELOCITIES: {
                "default": DataItemStats(
                    mean=np.zeros(JOINT_POSITION_DIM, dtype=float).tolist(),
                    std=np.ones(JOINT_POSITION_DIM, dtype=float).tolist(),
                )
            },
            DataType.JOINT_TORQUES: {
                "default": DataItemStats(
                    mean=np.zeros(JOINT_POSITION_DIM, dtype=float).tolist(),
                    std=np.ones(JOINT_POSITION_DIM, dtype=float).tolist(),
                )
            },
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: {
                "default": DataItemStats(
                    mean=np.zeros(1, dtype=float).tolist(),
                    std=np.ones(1, dtype=float).tolist(),
                )
            },
            DataType.POSES: {
                "default": DataItemStats(
                    mean=np.zeros(7, dtype=float).tolist(),
                    std=np.ones(7, dtype=float).tolist(),
                )
            },
            DataType.CUSTOM: {
                "sensor_data": DataItemStats(
                    mean=np.zeros(1, dtype=float).tolist(),
                    std=np.ones(1, dtype=float).tolist(),
                )
            },
            DataType.RGB_IMAGES: {
                "camera_0": DataItemStats(
                    max_len=CAMS,
                )
            },
            DataType.DEPTH_IMAGES: {
                "camera_0": DataItemStats(
                    max_len=CAMS,
                )
            },
            DataType.POINT_CLOUDS: {
                "lidar_0": DataItemStats(
                    max_len=1,
                )
            },
            DataType.LANGUAGE: {
                "default": DataItemStats(
                    max_len=512,
                )
            },
        }
    )
    return ModelInitDescription(
        dataset_statistics=dataset_statistics,
        input_data_types=CNNMLP.get_supported_input_data_types(),
        output_data_types=[DataType.JOINT_TARGET_POSITIONS],
        output_prediction_horizon=PRED_HORIZON,
    )


@pytest.fixture
def model_config() -> dict:
    return {}


@pytest.fixture
def sample_batch() -> BatchedTrainingSamples:
    return BatchedTrainingSamples(
        inputs={
            DataType.JOINT_POSITIONS: {
                "default": MaskableData(
                    torch.randn(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                    torch.ones(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                )
            },
            DataType.JOINT_VELOCITIES: {
                "default": MaskableData(
                    torch.randn(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                    torch.ones(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                )
            },
            DataType.JOINT_TORQUES: {
                "default": MaskableData(
                    torch.randn(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                    torch.ones(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                )
            },
            DataType.RGB_IMAGES: {
                "camera_0": MaskableData(
                    torch.randn(BS, CAMS, 3, 224, 224, dtype=torch.float32),
                    torch.ones(BS, CAMS, dtype=torch.float32),
                )
            },
        },
        outputs={
            DataType.JOINT_TARGET_POSITIONS: {
                "default": MaskableData(
                    torch.randn(
                        BS, PRED_HORIZON, JOINT_POSITION_DIM, dtype=torch.float32
                    ),
                    torch.ones(
                        BS, PRED_HORIZON, JOINT_POSITION_DIM, dtype=torch.float32
                    ),
                )
            }
        },
        output_prediction_mask=torch.ones(BS, PRED_HORIZON, dtype=torch.float32),
    )


@pytest.fixture
def sample_batch_full() -> BatchedTrainingSamples:
    return BatchedTrainingSamples(
        inputs={
            DataType.JOINT_POSITIONS: {
                "default": MaskableData(
                    torch.randn(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                    torch.ones(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                )
            },
            DataType.JOINT_VELOCITIES: {
                "default": MaskableData(
                    torch.randn(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                    torch.ones(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                )
            },
            DataType.JOINT_TORQUES: {
                "default": MaskableData(
                    torch.randn(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                    torch.ones(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                )
            },
            DataType.RGB_IMAGES: {
                "camera_0": MaskableData(
                    torch.randn(BS, CAMS, 3, 224, 224, dtype=torch.float32),
                    torch.ones(BS, CAMS, dtype=torch.float32),
                )
            },
            DataType.DEPTH_IMAGES: {
                "camera_0": MaskableData(
                    torch.randn(BS, CAMS, 1, 224, 224, dtype=torch.float32),
                    torch.ones(BS, CAMS, dtype=torch.float32),
                )
            },
            DataType.LANGUAGE: {
                "default": MaskableData(
                    torch.randint(0, 1000, (BS, 512), dtype=torch.int64),
                    torch.ones(BS, 512, dtype=torch.float32),
                )
            },
            DataType.POINT_CLOUDS: {
                "lidar_0": MaskableData(
                    torch.randn(BS, 1, 100, 3, dtype=torch.float32),
                    torch.ones(BS, 1, dtype=torch.float32),
                )
            },
            DataType.POSES: {
                "default": MaskableData(
                    torch.randn(BS, 6, dtype=torch.float32),
                    torch.ones(BS, 1, dtype=torch.float32),
                )
            },
            DataType.CUSTOM: {
                "sensor_data": MaskableData(
                    torch.randn(BS, 1, dtype=torch.float32),
                    torch.ones(BS, 1, dtype=torch.float32),
                )
            },
        },
        outputs={
            DataType.JOINT_TARGET_POSITIONS: {
                "default": MaskableData(
                    torch.randn(
                        BS, PRED_HORIZON, JOINT_POSITION_DIM, dtype=torch.float32
                    ),
                    torch.ones(
                        BS, PRED_HORIZON, JOINT_POSITION_DIM, dtype=torch.float32
                    ),
                )
            }
        },
        output_prediction_mask=torch.ones(BS, PRED_HORIZON, dtype=torch.float32),
    )


@pytest.fixture
def sample_inference_batch() -> BatchedInferenceInputs:
    return BatchedInferenceInputs(
        inputs={
            DataType.JOINT_POSITIONS: {
                "default": MaskableData(
                    torch.randn(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                    torch.ones(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                )
            },
            DataType.JOINT_VELOCITIES: {
                "default": MaskableData(
                    torch.randn(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                    torch.ones(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                )
            },
            DataType.JOINT_TORQUES: {
                "default": MaskableData(
                    torch.randn(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                    torch.ones(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                )
            },
            DataType.RGB_IMAGES: {
                "camera_0": MaskableData(
                    torch.randn(BS, CAMS, 3, 224, 224, dtype=torch.float32),
                    torch.ones(BS, CAMS, dtype=torch.float32),
                )
            },
        }
    )


@pytest.fixture
def mock_dataloader(sample_batch):
    """Create a mock dataloader."""

    def generate_batch():
        return sample_batch

    class MockDataLoader:
        def __iter__(self):
            for _ in range(2):  # 2 batches per epoch
                yield generate_batch()

        def __len__(self):
            return 2

    return MockDataLoader()


def test_model_construction(
    model_init_description_partial: ModelInitDescription, model_config: dict
):
    model = CNNMLP(model_init_description_partial, **model_config)
    model = model.to(DEVICE)
    assert isinstance(model, nn.Module)


def test_model_forward(
    model_init_description_partial: ModelInitDescription,
    model_config: dict,
    sample_inference_batch: BatchedInferenceInputs,
):
    model = CNNMLP(model_init_description_partial, **model_config)
    model = model.to(DEVICE)
    sample_inference_batch = sample_inference_batch.to(DEVICE)
    output = model(sample_inference_batch)
    assert isinstance(output, ModelPrediction)
    assert DataType.JOINT_TARGET_POSITIONS in output.outputs
    assert output.outputs[DataType.JOINT_TARGET_POSITIONS].shape == (
        BS,
        PRED_HORIZON,
        OUTPUT_PRED_DIM,
    )


def test_model_backward(
    model_init_description_partial: ModelInitDescription,
    model_config: dict,
    sample_batch: BatchedTrainingSamples,
):
    model = CNNMLP(model_init_description_partial, **model_config)
    model = model.to(DEVICE)
    sample_batch = sample_batch.to(DEVICE)
    output: BatchedTrainingOutputs = model.training_step(sample_batch)

    # Compute loss
    loss = output.losses["l1_loss"]

    # Perform backward pass
    loss.backward()

    # Check that gradients are computed
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()


def test_model_backward_full_description(
    model_init_description_full: ModelInitDescription,
    model_config: dict,
    sample_batch_full: BatchedTrainingSamples,
):
    model = CNNMLP(model_init_description_full, **model_config)
    model = model.to(DEVICE)
    sample_batch_full = sample_batch_full.to(DEVICE)
    output: BatchedTrainingOutputs = model.training_step(sample_batch_full)

    # Compute loss
    loss = output.losses["l1_loss"]

    # Perform backward pass
    loss.backward()

    # Check that gradients are computed
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert torch.isfinite(param.grad).all()


def test_run_validation(tmp_path: Path, mock_login):
    algorithm_dir = Path(inspect.getfile(CNNMLP)).parent
    _, error_msg = run_validation(
        output_dir=tmp_path,
        algorithm_dir=algorithm_dir,
        port=random.randint(10000, 20000),
        device=DEVICE,
    )
    if len(error_msg) > 0:
        raise RuntimeError(error_msg)
