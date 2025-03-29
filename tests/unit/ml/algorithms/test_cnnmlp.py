import numpy as np
import pytest
import torch
import torch.nn as nn

from neuracore.core.nc_types import (
    DataItemStats,
    DatasetDescription,
    ModelInitDescription,
)
from neuracore.ml import (
    ActionMaskableData,
    BatchedInferenceOutputs,
    BatchedInferenceSamples,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    MaskableData,
)
from neuracore.ml.algorithms.cnnmlp.cnnmlp import CNNMLP

BS = 2
CAMS = 1
JOINT_POSITION_DIM = 32
ACTION_DIM = 7
PRED_HORIZON = 10


@pytest.fixture
def model_init_description() -> ModelInitDescription:
    dataset_description = DatasetDescription(
        actions=DataItemStats(
            mean=np.zeros(ACTION_DIM, dtype=float), std=np.ones(ACTION_DIM, dtype=float)
        ),
        joint_positions=DataItemStats(
            mean=np.zeros(JOINT_POSITION_DIM, dtype=float),
            std=np.ones(JOINT_POSITION_DIM, dtype=float),
        ),
        max_num_rgb_images=CAMS,
    )
    return ModelInitDescription(
        dataset_description=dataset_description,
        action_prediction_horizon=PRED_HORIZON,
    )


@pytest.fixture
def model_config() -> dict:
    return {}


@pytest.fixture
def sample_batch() -> BatchedTrainingSamples:
    return BatchedTrainingSamples(
        joint_positions=MaskableData(
            torch.randn(BS, JOINT_POSITION_DIM, dtype=torch.float32),
            torch.ones(BS, JOINT_POSITION_DIM, dtype=torch.float32),
        ),
        rgb_images=MaskableData(
            torch.randn(BS, CAMS, 3, 224, 224, dtype=torch.float32),
            torch.ones(BS, CAMS, dtype=torch.float32),
        ),
        actions=ActionMaskableData(
            torch.randn(BS, PRED_HORIZON, ACTION_DIM, dtype=torch.float32),
            torch.ones(BS, ACTION_DIM, dtype=torch.float32),
            torch.ones(BS, PRED_HORIZON, dtype=torch.float32),
        ),
    )


@pytest.fixture
def sample_inference_batch() -> BatchedTrainingSamples:
    return BatchedInferenceSamples(
        joint_positions=MaskableData(
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
    model_init_description: ModelInitDescription, model_config: dict
):
    model = CNNMLP(model_init_description, **model_config)
    assert isinstance(model, nn.Module)


def test_model_forward(
    model_init_description: ModelInitDescription,
    model_config: dict,
    sample_inference_batch: BatchedInferenceSamples,
):
    model = CNNMLP(model_init_description, **model_config)
    output = model(sample_inference_batch)
    assert isinstance(output, BatchedInferenceOutputs)
    assert output.action_predicitons.shape == (BS, PRED_HORIZON, ACTION_DIM)


def test_model_backward(
    model_init_description: ModelInitDescription,
    model_config: dict,
    sample_batch: BatchedTrainingSamples,
):
    model = CNNMLP(model_init_description, **model_config)
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
