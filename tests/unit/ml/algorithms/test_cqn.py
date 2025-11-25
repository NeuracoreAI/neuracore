"""Tests for CQN (Coarse-to-Fine Q-Network) algorithm."""

import inspect
import random
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
import torch.nn as nn
from neuracore_types import BatchedNCData, DataType, ModelInitDescription
from torch.utils.data import DataLoader

from neuracore.ml import BatchedInferenceInputs, BatchedTrainingSamples
from neuracore.ml.algorithms.cqn.cqn import CQN
from neuracore.ml.core.ml_types import BatchedTrainingOutputs
from neuracore.ml.datasets.pytorch_dummy_dataset import PytorchDummyDataset
from neuracore.ml.utils.device_utils import get_default_device
from neuracore.ml.utils.robot_data_spec_utils import extract_data_types
from neuracore.ml.utils.validate import run_validation

BS = 2
DEVICE = get_default_device()
OUTPUT_PREDICTION_HORIZON = 4

# Smaller model config for faster testing
CQN_TEST_ARGS = {
    "feature_dim": 32,
    "hidden_dim": 64,
    "levels": 2,
    "bins": 3,
    "atoms": 11,
}


@pytest.fixture
def pytorch_dummy_dataset() -> PytorchDummyDataset:
    """Create dummy dataset for CQN tests."""
    input_data_types = CQN.get_supported_input_data_types()
    output_data_types = CQN.get_supported_output_data_types()
    input_robot_data_spec = {
        "robot_1": {data_type: [] for data_type in input_data_types}
    }
    output_robot_data_spec = {
        "robot_1": {data_type: [] for data_type in output_data_types}
    }

    dataset = PytorchDummyDataset(
        num_samples=5,
        input_robot_data_spec=input_robot_data_spec,
        output_robot_data_spec=output_robot_data_spec,
        output_prediction_horizon=OUTPUT_PREDICTION_HORIZON,
    )
    return dataset


@pytest.fixture
def model_init_description(
    pytorch_dummy_dataset: PytorchDummyDataset,
) -> ModelInitDescription:
    """Create model initialization description for CQN."""
    input_data_types = extract_data_types(pytorch_dummy_dataset.input_robot_data_spec)
    output_data_types = extract_data_types(pytorch_dummy_dataset.output_robot_data_spec)
    return ModelInitDescription(
        input_data_types=input_data_types,
        output_data_types=output_data_types,
        dataset_statistics=pytorch_dummy_dataset.dataset_statistics,
        output_prediction_horizon=pytorch_dummy_dataset.output_prediction_horizon,
    )


@pytest.fixture
def model_config() -> dict:
    """Return model configuration."""
    return CQN_TEST_ARGS


@pytest.fixture
def sample_inference_batch(
    pytorch_dummy_dataset: PytorchDummyDataset,
) -> BatchedInferenceInputs:
    """Create a sample inference batch."""
    dataloader = DataLoader(
        pytorch_dummy_dataset,
        batch_size=BS,
        shuffle=True,
        collate_fn=pytorch_dummy_dataset.collate_fn,
    )
    sample = cast(BatchedTrainingSamples, next(iter(dataloader)))
    return BatchedInferenceInputs(
        inputs=sample.inputs,
        inputs_mask=sample.inputs_mask,
        batch_size=BS,
    )


@pytest.fixture
def sample_batch(
    pytorch_dummy_dataset: PytorchDummyDataset,
) -> BatchedTrainingSamples:
    """Create a sample training batch."""
    dataloader = DataLoader(
        pytorch_dummy_dataset,
        batch_size=BS,
        shuffle=True,
        collate_fn=pytorch_dummy_dataset.collate_fn,
    )
    sample = cast(BatchedTrainingSamples, next(iter(dataloader)))
    return sample


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
    sample_inference_batch: BatchedInferenceInputs,
):
    """Test forward pass produces correct output shape."""
    model = CQN(model_init_description, **model_config)
    model = model.to(DEVICE)
    sample_inference_batch = sample_inference_batch.to(DEVICE)
    model.eval()
    with torch.no_grad():
        output: dict[DataType, list[BatchedNCData]] = model(sample_inference_batch)
    assert isinstance(output, dict)
    assert DataType.JOINT_TARGET_POSITIONS in output

    # CQN predicts single-step actions expanded to prediction horizon
    tensors = output[DataType.JOINT_TARGET_POSITIONS]
    # Each element is BatchedNCData with shape [B, T, 1]
    assert len(tensors) > 0
    joint_preds = tensors[0].value  # (B, T, 1)
    assert joint_preds.shape[0] == BS
    assert joint_preds.shape[1] == OUTPUT_PREDICTION_HORIZON


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
    assert DataType.RGB_IMAGES in input_types
    assert DataType.JOINT_TARGET_POSITIONS in output_types


def test_single_step_action_prediction(
    model_init_description: ModelInitDescription,
    model_config: dict,
    sample_inference_batch: BatchedInferenceInputs,
):
    """Test that CQN predicts single-step actions (same action for all timesteps)."""
    model = CQN(model_init_description, **model_config)
    model = model.to(DEVICE)
    sample_inference_batch = sample_inference_batch.to(DEVICE)
    model.eval()
    with torch.no_grad():
        output: dict[DataType, list[BatchedNCData]] = model(sample_inference_batch)

    tensors = output[DataType.JOINT_TARGET_POSITIONS]
    # Reconstruct full joint tensor: (B, T, num_joints)
    first_tensor = tensors[0].value
    b_size, t_steps, _ = first_tensor.shape
    all_joints = np.concatenate(
        [t.value.cpu().numpy() for t in tensors],
        axis=-1,
    )  # (B, T, num_joints)

    # Check that all timesteps have the same action (since CQN predicts single-step)
    for t in range(1, t_steps):
        np.testing.assert_array_almost_equal(
            all_joints[:, 0, :],
            all_joints[:, t, :],
            decimal=5,
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
