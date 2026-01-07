import inspect
import os
import random
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from neuracore_types import BatchedNCData, DataType, ModelInitDescription
from torch import nn
from torch.utils.data import DataLoader

from neuracore.core.utils.robot_data_spec_utils import extract_data_types
from neuracore.ml import BatchedInferenceInputs, BatchedTrainingSamples
from neuracore.ml.algorithms.pi0.pi0 import Pi0
from neuracore.ml.core.ml_types import BatchedTrainingOutputs
from neuracore.ml.datasets.pytorch_dummy_dataset import PytorchDummyDataset
from neuracore.ml.utils.device_utils import get_default_device
from neuracore.ml.utils.validate import run_validation

BS = 2
DEVICE = get_default_device()
OUTPUT_PREDICTION_HORIZON = 5

# Use cpu because the model takes a lot of vram
DEVICE = torch.device("cpu")
SKIP_TEST = os.environ.get("CI", "false").lower() == "true"


PI0_TEST_ARGS: dict[str, Any] = {
    "paligemma_variant": "gemma_300m",
    "action_expert_variant": "gemma_300m",
    "use_pretrained_weights": False,
    "num_inference_steps": 2,
    "vlm_max_text_tokens": 8,
    "compile_model": False,
    "gradient_checkpointing": False,
}


class DummyPI0Policy(nn.Module):
    """Lightweight stand-in for PI0Policy to keep unit tests fast."""

    def __init__(self, config, **_kwargs: Any) -> None:
        super().__init__()
        self.config = config
        self.state_proj = nn.Linear(config.max_state_dim, config.max_action_dim)
        self.action_out_proj = nn.Linear(config.max_action_dim, config.max_action_dim)
        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing_enabled = False

    def compile_model_enable(self) -> None:
        return None

    def forward(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(actions)
        if time is None:
            time = torch.rand(actions.shape[0], device=actions.device)

        bsize, seq_len, dim = actions.shape
        state_emb = self.state_proj(state)[:, None, :].expand(bsize, seq_len, dim)
        v_t = self.action_out_proj(actions + state_emb)
        u_t = noise - actions
        return (u_t - v_t).pow(2)

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        bsize = state.shape[0]
        device = state.device
        if noise is None:
            noise = torch.zeros(
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
                device=device,
            )
        state_emb = self.state_proj(state)[:, None, :].expand(
            bsize, self.config.chunk_size, self.config.max_action_dim
        )
        return noise + state_emb


@pytest.fixture(autouse=True)
def _patch_pi0_policy(monkeypatch):
    import neuracore.ml.algorithms.pi0.pi0 as pi0_module

    monkeypatch.setattr(pi0_module, "PI0Policy", DummyPI0Policy)


@pytest.fixture
def pytorch_dummy_dataset() -> PytorchDummyDataset:
    input_data_types = Pi0.get_supported_input_data_types()
    output_data_types = Pi0.get_supported_output_data_types()
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
    input_data_types = extract_data_types(pytorch_dummy_dataset.input_robot_data_spec)
    output_data_types = extract_data_types(pytorch_dummy_dataset.output_robot_data_spec)
    return ModelInitDescription(
        input_data_types=input_data_types,
        output_data_types=output_data_types,
        dataset_statistics=pytorch_dummy_dataset.dataset_statistics,
        output_prediction_horizon=pytorch_dummy_dataset.output_prediction_horizon,
    )


@pytest.fixture
def sample_inference_batch(
    pytorch_dummy_dataset: PytorchDummyDataset,
) -> BatchedInferenceInputs:
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
def sample_training_batch(
    pytorch_dummy_dataset: PytorchDummyDataset,
) -> BatchedTrainingSamples:
    dataloader = DataLoader(
        pytorch_dummy_dataset,
        batch_size=BS,
        shuffle=True,
        collate_fn=pytorch_dummy_dataset.collate_fn,
    )
    sample = cast(BatchedTrainingSamples, next(iter(dataloader)))
    return sample


@pytest.mark.skipif(SKIP_TEST, reason="Skipping test in CI environment")
def test_model_construction(model_init_description: ModelInitDescription):
    model = Pi0(model_init_description, **PI0_TEST_ARGS)
    model = model.to(DEVICE)
    assert isinstance(model, nn.Module)


@pytest.mark.skipif(SKIP_TEST, reason="Skipping test in CI environment")
def test_model_forward(
    model_init_description: ModelInitDescription,
    sample_inference_batch: BatchedInferenceInputs,
):
    model = Pi0(model_init_description, **PI0_TEST_ARGS)
    model = model.to(DEVICE)
    sample_inference_batch = sample_inference_batch.to(DEVICE)
    output: dict[DataType, list[BatchedNCData]] = model(sample_inference_batch)
    assert isinstance(output, dict)
    for data_type, tensors in output.items():
        assert isinstance(data_type, DataType)
        assert isinstance(tensors, list)
        for tensor in tensors:
            assert isinstance(tensor, BatchedNCData)


@pytest.mark.skipif(SKIP_TEST, reason="Skipping test in CI environment")
def test_model_backward(
    model_init_description: ModelInitDescription,
    sample_training_batch: BatchedTrainingSamples,
):
    model = Pi0(model_init_description, **PI0_TEST_ARGS)
    model = model.to(DEVICE)
    sample_training_batch = sample_training_batch.to(DEVICE)
    output: BatchedTrainingOutputs = model.training_step(sample_training_batch)

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
def test_run_validation(tmp_path: Path, mock_login, monkeypatch):
    from neuracore.ml.algorithms.pi0.pi0 import Pi0
    from neuracore.ml.utils import validate as validate_module

    monkeypatch.setattr(validate_module.AlgorithmLoader, "load_model", lambda self: Pi0)

    # Long timeout due to larger model run on CPU
    os.environ["NEURACORE_ENDPOINT_TIMEOUT"] = "120"
    algorithm_dir = Path(inspect.getfile(Pi0)).parent
    _, error_msg = run_validation(
        output_dir=tmp_path,
        algorithm_dir=algorithm_dir,
        port=random.randint(10000, 20000),
        skip_endpoint_check=False,
        algorithm_config=PI0_TEST_ARGS,
        device=DEVICE,
    )
    if len(error_msg) > 0:
        raise RuntimeError(error_msg)
