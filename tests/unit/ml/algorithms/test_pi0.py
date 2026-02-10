import inspect
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from neuracore_types import BatchedNCData, DataType, ModelInitDescription
from torch import nn
from torch.utils.data import DataLoader

from neuracore.core.utils.robot_data_spec_utils import extract_data_types
from neuracore.ml import BatchedInferenceInputs, BatchedTrainingSamples
from neuracore.ml.core.ml_types import BatchedTrainingOutputs
from neuracore.ml.datasets.pytorch_dummy_dataset import PytorchDummyDataset
from neuracore.ml.utils.validate import run_validation

PI0_DEP_PACKAGES = ["transformers", "tokenizers", "huggingface-hub"]


# This is a workaround to avoid the issue of different versions of transformers being
# installed. For tests we will have pi0 use a separate subprocess and install the
# specific requirements.
def _resolve_pi0_requirements_file() -> Path:
    repo_root = Path(__file__).resolve().parents[4]

    requirements_file = repo_root / "neuracore/ml/algorithms/pi0/requirements.txt"

    if not requirements_file.exists():
        raise FileNotFoundError(
            f"Could not find pi0 requirements.txt at {requirements_file}"
        )
    return requirements_file


def _pip_uninstall_packages(packages: list[str]) -> None:
    # Best-effort cleanup to avoid mixed package files across version switches.
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", *packages],
        check=False,
    )


def install_pi0_requirements() -> None:
    """Install requirements.txt from the pi0 folder."""
    requirements_file = _resolve_pi0_requirements_file()
    _pip_uninstall_packages(PI0_DEP_PACKAGES)

    # Force a clean reinstall to avoid mixed transformers files from previous versions.
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--upgrade",
        "--force-reinstall",
        "--ignore-installed",
        "-r",
        str(requirements_file),
    ])


@pytest.fixture(scope="module", autouse=True)
def _pi0_env():
    install_pi0_requirements()


@pytest.fixture(scope="module")
def Pi0(_pi0_env):  # noqa: N802
    from neuracore.ml.algorithms.pi0.pi0 import Pi0 as Pi0Model

    return Pi0Model


BS = 1
OUTPUT_PREDICTION_HORIZON = 1

# Use cpu because the model takes a lot of vram
DEVICE = torch.device("cpu")


PI0_TEST_ARGS: dict[str, Any] = {
    "paligemma_variant": "gemma_tiny",
    "action_expert_variant": "gemma_tiny",
    "use_pretrained_weights": False,
    "num_inference_steps": 1,
    "vlm_max_text_tokens": 4,
    "compile_model": False,
    "gradient_checkpointing": False,
}


@pytest.fixture
def pytorch_dummy_dataset(Pi0) -> PytorchDummyDataset:  # noqa: N803
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


def test_model_construction(
    model_init_description: ModelInitDescription, Pi0
):  # noqa: N803
    model = Pi0(model_init_description, **PI0_TEST_ARGS)
    model = model.to(DEVICE)
    assert isinstance(model, nn.Module)


def test_model_forward(
    model_init_description: ModelInitDescription,
    sample_inference_batch: BatchedInferenceInputs,
    Pi0,  # noqa: N803
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


def test_model_backward(
    model_init_description: ModelInitDescription,
    sample_training_batch: BatchedTrainingSamples,
    Pi0,  # noqa: N803
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
            is_vlm_param = any(
                keyword in name.lower()
                for keyword in ["vlm", "vision", "paligemma", "language_model"]
            )

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


def test_run_validation(tmp_path: Path, mock_login, monkeypatch, Pi0):  # noqa: N803
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
