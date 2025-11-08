import inspect
import os
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
    BatchedData,
    BatchedInferenceSamples,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    MaskableData,
)
from neuracore.ml.algorithms.pi0.pi0 import Pi0
from neuracore.ml.utils.validate import run_validation

BS = 1
CAMS = 2
JOINT_POSITION_DIM = 16
OUTPUT_PRED_DIM = JOINT_POSITION_DIM
PRED_HORIZON = 8
LANGUAGE_MAX_LEN = 128  # Maximum length for language tokens
# Use cpu because the model takes a lot of vram
DEVICE = torch.device("cpu")
SKIP_TEST = os.environ.get("CI", "false").lower() == "true"


PI_TINY_ARGS = {
    "vlm_expert_intermediate_size": 4,
    "vlm_expert_num_heads": 1,
    "vlm_expert_head_dim": 4,
    "action_expert_width": 16,
    "action_expert_intermediate_size": 4,
    "action_expert_num_heads": 1,
    "action_expert_head_dim": 4,
    "moe_depth": 1,
}


@pytest.fixture
def model_init_description() -> ModelInitDescription:
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
        device=DEVICE.type,
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
                torch.rand(BS, CAMS, 3, 224, 224, dtype=torch.float32),
                torch.ones(BS, CAMS, dtype=torch.float32),
            ),
            language_tokens=MaskableData(
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
        output_prediction_mask=torch.ones(BS, PRED_HORIZON, dtype=torch.float32),
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
            torch.rand(BS, CAMS, 3, 224, 224, dtype=torch.float32),
            torch.ones(BS, CAMS, dtype=torch.float32),
        ),
        language_tokens=MaskableData(
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


@pytest.mark.skipif(SKIP_TEST, reason="Skipping test in CI environment")
def test_model_construction(
    model_init_description: ModelInitDescription, model_config: dict
):
    model = Pi0(model_init_description, DEVICE, **PI_TINY_ARGS)
    model = model.to(DEVICE)
    assert isinstance(model, nn.Module)


@pytest.mark.skipif(SKIP_TEST, reason="Skipping test in CI environment")
def test_model_forward(
    model_init_description: ModelInitDescription,
    model_config: dict,
    sample_inference_batch: BatchedInferenceSamples,
):
    model = Pi0(model_init_description, DEVICE, **PI_TINY_ARGS)
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


@pytest.mark.skipif(SKIP_TEST, reason="Skipping test in CI environment")
def test_model_backward(
    model_init_description: ModelInitDescription,
    model_config: dict,
    sample_batch: BatchedTrainingSamples,
):
    model = Pi0(model_init_description, DEVICE, **PI_TINY_ARGS)
    model = model.to(DEVICE)
    sample_batch = sample_batch.to(DEVICE)
    output: BatchedTrainingOutputs = model.training_step(sample_batch)

    # Compute loss
    loss = output.losses["mse_loss"]

    # Perform backward pass
    loss.backward()

    # Check that gradients are computed for parameters that should have them
    for name, param in model.named_parameters():
        if param.requires_grad:
            # VLM parameters may not get gradients if they're not used in the
            # forward pass
            is_vlm_param = any(keyword in name.lower() for keyword in ["vlm", "vision"])

            if not is_vlm_param:
                # Non-VLM parameters should definitely have gradients
                assert (
                    param.grad is not None
                ), f"Non-VLM parameter {name} should have gradients"
                assert torch.isfinite(
                    param.grad
                ).all(), f"Parameter {name} has non-finite gradients"
            elif param.grad is not None:
                # If VLM parameters do have gradients, they should be finite
                assert torch.isfinite(
                    param.grad
                ).all(), f"Parameter {name} has non-finite gradients"


@pytest.mark.skipif(SKIP_TEST, reason="Skipping test in CI environment")
def test_run_validation(tmp_path: Path, mock_login):
    # Long timeout due to larger model run on CPU
    os.environ["NEURACORE_ENDPOINT_TIMEOUT"] = "120"
    algorithm_dir = Path(inspect.getfile(Pi0)).parent
    _, error_msg = run_validation(
        output_dir=tmp_path,
        algorithm_dir=algorithm_dir,
        port=random.randint(10000, 20000),
        skip_endpoint_check=False,
        algorithm_config=PI_TINY_ARGS,
        device=DEVICE,
    )
    assert len(error_msg) == 0
