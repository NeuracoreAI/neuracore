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
from neuracore.ml.algorithms.simple_vla.simple_vla import SimpleVLA
from neuracore.ml.core.ml_types import BatchedData
from neuracore.ml.utils.validate import run_validation

BS = 2
CAMS = 1
JOINT_POSITION_DIM = 32
OUTPUT_PRED_DIM = JOINT_POSITION_DIM
PRED_HORIZON = 10
LANGUAGE_MAX_LEN = 512  # Maximum length for language tokens
DEVICE = torch.device("cpu")


@pytest.fixture
def model_init_description() -> ModelInitDescription:
    dataset_description = DatasetDescription(
        joint_positions=DataItemStats(
            mean=np.zeros(JOINT_POSITION_DIM, dtype=float),
            std=np.ones(JOINT_POSITION_DIM, dtype=float),
            max_len=JOINT_POSITION_DIM,
        ),
        joint_target_positions=DataItemStats(
            mean=np.zeros(JOINT_POSITION_DIM, dtype=float),
            std=np.ones(JOINT_POSITION_DIM, dtype=float),
            max_len=JOINT_POSITION_DIM,
        ),
        joint_velocities=DataItemStats(
            mean=np.zeros(JOINT_POSITION_DIM, dtype=float),
            std=np.ones(JOINT_POSITION_DIM, dtype=float),
            max_len=JOINT_POSITION_DIM,
        ),
        joint_torques=DataItemStats(
            mean=np.zeros(JOINT_POSITION_DIM, dtype=float),
            std=np.ones(JOINT_POSITION_DIM, dtype=float),
            max_len=JOINT_POSITION_DIM,
        ),
        rgb_images=DataItemStats(
            max_len=CAMS,
        ),
        language=DataItemStats(max_len=LANGUAGE_MAX_LEN),
    )
    return ModelInitDescription(
        dataset_description=dataset_description,
        input_data_types=[
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.RGB_IMAGE,
            DataType.LANGUAGE,
        ],
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
            language_tokens=MaskableData(  # Add language tokens input
                torch.randint(0, 1000, (BS, LANGUAGE_MAX_LEN), dtype=torch.long),
                torch.ones(BS, LANGUAGE_MAX_LEN, dtype=torch.float32),
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
        language_tokens=MaskableData(  # Add language tokens for inference
            torch.randint(0, 1000, (BS, LANGUAGE_MAX_LEN), dtype=torch.long),
            torch.ones(BS, LANGUAGE_MAX_LEN, dtype=torch.float32),
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
    model_init_description: ModelInitDescription, model_config: dict
):
    model = SimpleVLA(model_init_description, **model_config)
    model = model.to(DEVICE)
    assert isinstance(model, nn.Module)


def test_model_forward(
    model_init_description: ModelInitDescription,
    model_config: dict,
    sample_inference_batch: BatchedInferenceSamples,
):
    model = SimpleVLA(model_init_description, **model_config)
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
    model_init_description: ModelInitDescription,
    model_config: dict,
    sample_batch: BatchedTrainingSamples,
):
    model = SimpleVLA(model_init_description, **model_config)
    model = model.to(DEVICE)
    sample_batch = sample_batch.to(DEVICE)
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


def test_run_validation(tmp_path: Path, mock_login):
    algorithm_dir = Path(inspect.getfile(SimpleVLA)).parent
    _, error_msg = run_validation(
        output_dir=tmp_path,
        algorithm_dir=algorithm_dir,
        port=random.randint(10000, 20000),
    )
    assert len(error_msg) == 0
