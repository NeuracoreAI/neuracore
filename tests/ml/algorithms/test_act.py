import numpy as np
import pytest
import torch
import torch.nn as nn

from neuracore import BatchedTrainingOutputs, BatchedTrainingSamples, DatasetDescription
from neuracore.ml.algorithms.act.act import ACT

BS = 2
CAMS = 1
STATE_DIM = 32
ACTION_DIM = 7
PRED_HORIZON = 10


@pytest.fixture
def dataset_description() -> DatasetDescription:
    return DatasetDescription(
        max_num_cameras=CAMS,
        max_state_size=STATE_DIM,
        max_action_size=ACTION_DIM,
        action_mean=np.ones(ACTION_DIM),
        action_std=np.ones(ACTION_DIM),
        state_mean=np.ones(STATE_DIM),
        state_std=np.ones(STATE_DIM),
        action_prediction_horizon=PRED_HORIZON,
    )


@pytest.fixture
def model_config() -> dict:
    return {
        "hidden_dim": 256,
        "num_encoder_layers": 4,
        "num_decoder_layers": 7,
        "nheads": 8,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "lr": 1e-4,
        "lr_backbone": 1e-5,
        "weight_decay": 1e-4,
        "kl_weight": 0.5,
    }


@pytest.fixture
def sample_batch() -> BatchedTrainingSamples:
    return BatchedTrainingSamples(
        states=torch.randn(BS, STATE_DIM),
        states_mask=torch.ones(BS, STATE_DIM, dtype=torch.float32),
        camera_images=torch.randn(BS, CAMS, 3, 224, 224),
        camera_images_mask=torch.ones(BS, CAMS, dtype=torch.float32),
        actions=torch.randn(BS, PRED_HORIZON, ACTION_DIM),
        actions_mask=torch.ones(BS, ACTION_DIM, dtype=torch.float32),
        actions_sequence_mask=torch.ones(BS, PRED_HORIZON, dtype=torch.float32),
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
    dataset_description: DatasetDescription, model_config: dict
):
    model = ACT(dataset_description, **model_config)
    assert isinstance(model, nn.Module)


def test_model_forward(
    dataset_description: DatasetDescription,
    model_config: dict,
    sample_batch: BatchedTrainingSamples,
):
    model = ACT(dataset_description, **model_config)
    output = model(sample_batch)
    assert isinstance(output, BatchedTrainingOutputs)
    assert output.action_predicitons.shape == (BS, PRED_HORIZON, ACTION_DIM)
    assert len(output.losses.keys()) > 0


def test_model_backward(
    dataset_description: DatasetDescription,
    model_config: dict,
    sample_batch: BatchedTrainingSamples,
):
    model = ACT(dataset_description, **model_config)
    output: BatchedTrainingOutputs = model(sample_batch)

    # Compute loss
    loss = output.losses["l1_and_kl_loss"]

    # Perform backward pass
    loss.backward()

    # Check that gradients are computed
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
