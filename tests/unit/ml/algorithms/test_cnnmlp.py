import inspect
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from neuracore.core.nc_types import (
    DataItemStats,
    DatasetDescription,
    DataType,
    ModelInitDescription,
    ModelPrediction,
)
from neuracore.ml import (
    BatchedInferenceSamples,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    MaskableData,
)
from neuracore.ml.algorithms.cnnmlp.cnnmlp import CNNMLP
from neuracore.ml.core.ml_types import BatchedData
from neuracore.ml.utils.validate import run_validation

BS = 2
CAMS = 1
JOINT_POSITION_DIM = 32
OUTPUT_PRED_DIM = JOINT_POSITION_DIM
PRED_HORIZON = 10


@pytest.fixture
def model_init_description_partial() -> ModelInitDescription:
    dataset_description = DatasetDescription(
        joint_positions=DataItemStats(
            mean=np.zeros(JOINT_POSITION_DIM, dtype=float),
            std=np.ones(JOINT_POSITION_DIM, dtype=float),
        ),
        joint_target_positions=DataItemStats(
            mean=np.zeros(JOINT_POSITION_DIM, dtype=float),
            std=np.ones(JOINT_POSITION_DIM, dtype=float),
        ),
        joint_velocities=DataItemStats(
            mean=np.zeros(JOINT_POSITION_DIM, dtype=float),
            std=np.ones(JOINT_POSITION_DIM, dtype=float),
        ),
        joint_torques=DataItemStats(
            mean=np.zeros(JOINT_POSITION_DIM, dtype=float),
            std=np.ones(JOINT_POSITION_DIM, dtype=float),
        ),
        rgb_images=DataItemStats(
            max_len=CAMS,
        ),
    )
    return ModelInitDescription(
        dataset_description=dataset_description,
        input_data_types=[
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.RGB_IMAGE,
        ],
        output_data_types=[DataType.JOINT_TARGET_POSITIONS],
        output_prediction_horizon=PRED_HORIZON,
    )


@pytest.fixture
def model_init_description_full() -> ModelInitDescription:
    dataset_description = DatasetDescription(
        joint_positions=DataItemStats(
            mean=np.zeros(JOINT_POSITION_DIM, dtype=float),
            std=np.ones(JOINT_POSITION_DIM, dtype=float),
        ),
        joint_target_positions=DataItemStats(
            mean=np.zeros(JOINT_POSITION_DIM, dtype=float),
            std=np.ones(JOINT_POSITION_DIM, dtype=float),
        ),
        joint_velocities=DataItemStats(
            mean=np.zeros(JOINT_POSITION_DIM, dtype=float),
            std=np.ones(JOINT_POSITION_DIM, dtype=float),
        ),
        joint_torques=DataItemStats(
            mean=np.zeros(JOINT_POSITION_DIM, dtype=float),
            std=np.ones(JOINT_POSITION_DIM, dtype=float),
        ),
        end_effector_states=DataItemStats(
            mean=np.zeros(1, dtype=float),
            std=np.ones(1, dtype=float),
        ),
        poses=DataItemStats(
            mean=np.zeros(7, dtype=float),
            std=np.ones(7, dtype=float),
        ),
        custom_data={
            "sensor_data": DataItemStats(
                mean=np.zeros(1, dtype=float),
                std=np.ones(1, dtype=float),
            )
        },
        rgb_images=DataItemStats(
            max_len=CAMS,
        ),
        depth_images=DataItemStats(
            max_len=CAMS,
        ),
        point_clouds=DataItemStats(
            max_len=1,
        ),
        language=DataItemStats(
            max_len=512,
        ),
    )
    return ModelInitDescription(
        dataset_description=dataset_description,
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
        inputs=BatchedData(
            joint_positions=MaskableData(
                torch.randn(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                torch.ones(BS, JOINT_POSITION_DIM, dtype=torch.float32),
            ),
            joint_velocities=MaskableData(
                torch.randn(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                torch.ones(BS, JOINT_POSITION_DIM, dtype=torch.float32),
            ),
            joint_torques=MaskableData(
                torch.randn(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                torch.ones(BS, JOINT_POSITION_DIM, dtype=torch.float32),
            ),
            rgb_images=MaskableData(
                torch.randn(BS, CAMS, 3, 224, 224, dtype=torch.float32),
                torch.ones(BS, CAMS, dtype=torch.float32),
            ),
        ),
        outputs=BatchedData(
            joint_target_positions=MaskableData(
                torch.randn(BS, PRED_HORIZON, JOINT_POSITION_DIM, dtype=torch.float32),
                torch.ones(BS, PRED_HORIZON, JOINT_POSITION_DIM, dtype=torch.float32),
            )
        ),
        output_predicition_mask=torch.ones(BS, PRED_HORIZON, dtype=torch.float32),
    )


@pytest.fixture
def sample_batch_full() -> BatchedTrainingSamples:
    return BatchedTrainingSamples(
        inputs=BatchedData(
            joint_positions=MaskableData(
                torch.randn(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                torch.ones(BS, JOINT_POSITION_DIM, dtype=torch.float32),
            ),
            joint_velocities=MaskableData(
                torch.randn(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                torch.ones(BS, JOINT_POSITION_DIM, dtype=torch.float32),
            ),
            joint_torques=MaskableData(
                torch.randn(BS, JOINT_POSITION_DIM, dtype=torch.float32),
                torch.ones(BS, JOINT_POSITION_DIM, dtype=torch.float32),
            ),
            rgb_images=MaskableData(
                torch.randn(BS, CAMS, 3, 224, 224, dtype=torch.float32),
                torch.ones(BS, CAMS, dtype=torch.float32),
            ),
            depth_images=MaskableData(
                torch.randn(BS, CAMS, 1, 224, 224, dtype=torch.float32),
                torch.ones(BS, CAMS, dtype=torch.float32),
            ),
            language_tokens=MaskableData(
                torch.randint(0, 1000, (BS, 512), dtype=torch.int64),
                torch.ones(BS, 512, dtype=torch.float32),
            ),
            end_effectors=MaskableData(
                torch.randn(BS, 1, dtype=torch.float32),
                torch.ones(BS, 1, dtype=torch.float32),
            ),
            point_clouds=MaskableData(
                torch.randn(BS, 1, 100, 3, dtype=torch.float32),
                torch.ones(BS, 1, dtype=torch.float32),
            ),
            poses=MaskableData(
                torch.randn(BS, 6, dtype=torch.float32),
                torch.ones(BS, 1, dtype=torch.float32),
            ),
            custom_data={
                "sensor_data": MaskableData(
                    torch.randn(BS, 1, dtype=torch.float32),
                    torch.ones(BS, 1, dtype=torch.float32),
                )
            },
        ),
        outputs=BatchedData(
            joint_target_positions=MaskableData(
                torch.randn(BS, PRED_HORIZON, JOINT_POSITION_DIM, dtype=torch.float32),
                torch.ones(BS, PRED_HORIZON, JOINT_POSITION_DIM, dtype=torch.float32),
            )
        ),
        output_predicition_mask=torch.ones(BS, PRED_HORIZON, dtype=torch.float32),
    )


@pytest.fixture
def sample_inference_batch() -> BatchedInferenceSamples:
    return BatchedInferenceSamples(
        joint_positions=MaskableData(
            torch.randn(BS, JOINT_POSITION_DIM, dtype=torch.float32),
            torch.ones(BS, JOINT_POSITION_DIM, dtype=torch.float32),
        ),
        joint_velocities=MaskableData(
            torch.randn(BS, JOINT_POSITION_DIM, dtype=torch.float32),
            torch.ones(BS, JOINT_POSITION_DIM, dtype=torch.float32),
        ),
        joint_torques=MaskableData(
            torch.randn(BS, JOINT_POSITION_DIM, dtype=torch.float32),
            torch.ones(BS, JOINT_POSITION_DIM, dtype=torch.float32),
        ),
        rgb_images=MaskableData(
            torch.randn(BS, CAMS, 3, 224, 224, dtype=torch.float32),
            torch.ones(BS, CAMS, dtype=torch.float32),
        ),
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
    # Use CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    assert isinstance(model, nn.Module)


def test_model_forward(
    model_init_description_partial: ModelInitDescription,
    model_config: dict,
    sample_inference_batch: BatchedInferenceSamples,
):
    model = CNNMLP(model_init_description_partial, **model_config)
    # Use CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    sample_inference_batch = sample_inference_batch.to(device)
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
    # Use CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    sample_batch = sample_batch.to(device)
    output: BatchedTrainingOutputs = model.training_step(sample_batch)

    # Compute loss
    loss = output.losses["mse_loss"]

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
    # Use CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    sample_batch_full = sample_batch_full.to(device)
    output: BatchedTrainingOutputs = model.training_step(sample_batch_full)

    # Compute loss
    loss = output.losses["mse_loss"]

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
    )
    if len(error_msg) > 0:
        raise RuntimeError(error_msg)
