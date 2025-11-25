"""Tests for CQN (Coarse-to-Fine Q-Network) algorithm."""

import inspect
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from neuracore_types import (
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
from neuracore.ml.algorithms.cqn.cqn import CQN
from neuracore.ml.core.ml_types import BatchedData
from neuracore.ml.utils.device_utils import get_default_device
from neuracore.ml.utils.validate import run_validation

BS = 2
CAMS = 1
JOINT_POSITION_DIM = 7
OUTPUT_PRED_DIM = JOINT_POSITION_DIM
PRED_HORIZON = 4
DEVICE = get_default_device()

# Smaller model config for faster testing
CQN_TEST_ARGS = {
    "feature_dim": 32,
    "hidden_dim": 64,
    "levels": 2,
    "bins": 3,
    "atoms": 11,
}


@pytest.fixture
def model_init_description() -> ModelInitDescription:
    """Create model initialization description for CQN."""
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
def model_config() -> dict:
    """Return model configuration."""
    return CQN_TEST_ARGS


@pytest.fixture
def sample_batch() -> BatchedTrainingSamples:
    """Create a sample training batch."""
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
                # CQN expects uint8 images (0-255 range)
                torch.randint(0, 256, (BS, CAMS, 3, 224, 224), dtype=torch.float32),
                torch.ones(BS, CAMS, dtype=torch.float32),
            ),
        ),
        outputs=BatchedData(
            joint_target_positions=MaskableData(
                torch.randn(BS, PRED_HORIZON, JOINT_POSITION_DIM, dtype=torch.float32),
                torch.ones(BS, PRED_HORIZON, JOINT_POSITION_DIM, dtype=torch.float32),
            )
        ),
        output_prediction_mask=torch.ones(BS, PRED_HORIZON, dtype=torch.float32),
    )


@pytest.fixture
def sample_inference_batch() -> BatchedInferenceSamples:
    """Create a sample inference batch."""
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
            # CQN expects uint8 images (0-255 range)
            torch.randint(0, 256, (BS, CAMS, 3, 224, 224), dtype=torch.float32),
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
    model_init_description: ModelInitDescription, model_config: dict
):
    """Test that CQN model can be constructed."""
    model = CQN(model_init_description, **model_config)
    model = model.to(DEVICE)
    assert isinstance(model, nn.Module)


def test_model_forward(
    model_init_description: ModelInitDescription,
    model_config: dict,
    sample_inference_batch: BatchedInferenceSamples,
):
    """Test forward pass produces correct output shape."""
    model = CQN(model_init_description, **model_config)
    model = model.to(DEVICE)
    sample_inference_batch = sample_inference_batch.to(DEVICE)
    model.eval()
    with torch.no_grad():
        output = model(sample_inference_batch)
    assert isinstance(output, ModelPrediction)
    assert DataType.JOINT_TARGET_POSITIONS in output.outputs
    # CQN predicts single-step actions expanded to prediction horizon
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
    """Test backward pass computes gradients."""
    model = CQN(model_init_description, **model_config)
    model = model.to(DEVICE)
    sample_batch = sample_batch.to(DEVICE)
    output: BatchedTrainingOutputs = model.training_step(sample_batch)

    # Compute loss
    loss = output.losses["critic_loss"]

    # Perform backward pass
    loss.backward()

    # Check that gradients are computed for non-target network parameters
    # critic_target is updated via soft updates, not gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Skip target network parameters - they don't receive gradients
            is_target_param = "critic_target" in name

            if not is_target_param:
                assert param.grad is not None, f"Gradient for {name} is None"
                assert torch.isfinite(
                    param.grad
                ).all(), f"Gradient for {name} is not finite"
            elif param.grad is not None:
                # If target parameters do have gradients, they should be finite
                assert torch.isfinite(
                    param.grad
                ).all(), f"Gradient for {name} is not finite"


def test_model_optimizer_configuration(
    model_init_description: ModelInitDescription, model_config: dict
):
    """Test that optimizers are properly configured."""
    model = CQN(model_init_description, **model_config)
    model = model.to(DEVICE)
    optimizers = model.configure_optimizers()
    assert len(optimizers) == 2  # encoder_opt and critic_opt
    assert all(isinstance(opt, torch.optim.Optimizer) for opt in optimizers)


def test_supported_data_types():
    """Test that supported data types are correctly defined."""
    input_types = CQN.get_supported_input_data_types()
    output_types = CQN.get_supported_output_data_types()

    assert DataType.JOINT_POSITIONS in input_types
    assert DataType.RGB_IMAGE in input_types
    assert DataType.JOINT_TARGET_POSITIONS in output_types


def test_single_step_action_prediction(
    model_init_description: ModelInitDescription,
    model_config: dict,
    sample_inference_batch: BatchedInferenceSamples,
):
    """Test that CQN predicts single-step actions (same action for all timesteps)."""
    model = CQN(model_init_description, **model_config)
    model = model.to(DEVICE)
    sample_inference_batch = sample_inference_batch.to(DEVICE)
    model.eval()
    with torch.no_grad():
        output = model(sample_inference_batch)

    actions = output.outputs[DataType.JOINT_TARGET_POSITIONS]
    # Check that all timesteps have the same action (since CQN predicts single-step)
    for t in range(1, PRED_HORIZON):
        np.testing.assert_array_almost_equal(
            actions[:, 0, :], actions[:, t, :], decimal=5
        )


def test_run_validation(tmp_path: Path, mock_login):
    """Test validation run with CQN algorithm."""
    algorithm_dir = Path(inspect.getfile(CQN)).parent
    _, error_msg = run_validation(
        output_dir=tmp_path,
        algorithm_dir=algorithm_dir,
        port=random.randint(10000, 20000),
        algorithm_config=CQN_TEST_ARGS,
        device=DEVICE,
    )
    assert len(error_msg) == 0
