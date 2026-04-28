import inspect
import os
import random
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from neuracore_types import (
    BatchedNCData,
    CrossEmbodimentDescription,
    DataType,
    ModelInitDescription,
)
from ordered_set import OrderedSet
from torch import nn
from torch.utils.data import DataLoader

from neuracore.ml import BatchedInferenceInputs, BatchedTrainingSamples
from neuracore.ml.algorithms.groot.groot import Groot as GrootModel
from neuracore.ml.core.ml_types import BatchedTrainingOutputs
from neuracore.ml.datasets.pytorch_dummy_dataset import PytorchDummyDataset
from neuracore.ml.utils.algorithm_loader import AlgorithmLoader
from neuracore.ml.utils.validate import run_validation

BS = 1
OUTPUT_PREDICTION_HORIZON = 16

# Use cpu because the model takes a lot of vram
DEVICE = torch.device("cpu")

# Tiny architecture overrides for fast unit testing (no HuggingFace downloads)
GROOT_TEST_ARGS: dict[str, Any] = {
    "use_pretrained_weights": False,
    "use_tiny_vlm": True,
    "num_denoising_steps": 1,
    "num_dit_layers": 2,
    "num_attention_heads": 2,
    "attention_head_dim": 16,
    "dit_output_dim": 32,
    "backbone_embedding_dim": 64,
}

INPUT_PARAMS = [
    pytest.param(OrderedSet([data_type]), id=data_type.value)
    for data_type in GrootModel.get_supported_input_data_types()
]
OUTPUT_PARAMS = [
    # TODO: expand to all supported output data types in nightly CI.
    pytest.param(
        OrderedSet([DataType.JOINT_TARGET_POSITIONS]),
        id=DataType.JOINT_TARGET_POSITIONS.value,
    )
]


GROOT_ALGORITHM_DIR = (
    Path(__file__).resolve().parents[4] / "neuracore/ml/algorithms/groot"
)


@pytest.fixture(scope="module")
def Groot():  # noqa: N802
    # AlgorithmLoader.install_requirements tries uv pip first and falls back
    # to pip, and skips the install entirely if requirements are already
    # satisfied (avoiding unnecessary reinstalls that can break PyO3 deps).
    if not GROOT_ALGORITHM_DIR.exists():
        raise FileNotFoundError(
            f"Could not find groot algorithm dir at {GROOT_ALGORITHM_DIR}"
        )
    AlgorithmLoader(GROOT_ALGORITHM_DIR).install_requirements()

    from neuracore.ml.algorithms.groot.groot import Groot as GrootModel

    return GrootModel


@pytest.fixture
def pytorch_dummy_dataset(Groot) -> PytorchDummyDataset:  # noqa: N803
    input_data_types = Groot.get_supported_input_data_types()
    output_data_types = Groot.get_supported_output_data_types()
    input_cross_embodiment_description: CrossEmbodimentDescription = {
        "robot_1": {data_type: {} for data_type in input_data_types}
    }
    output_cross_embodiment_description: CrossEmbodimentDescription = {
        "robot_1": {data_type: {} for data_type in output_data_types}
    }

    dataset = PytorchDummyDataset(
        num_samples=5,
        input_cross_embodiment_description=input_cross_embodiment_description,
        output_cross_embodiment_description=output_cross_embodiment_description,
        output_prediction_horizon=OUTPUT_PREDICTION_HORIZON,
    )
    return dataset


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


@pytest.mark.parametrize("output_data_types", OUTPUT_PARAMS)
@pytest.mark.parametrize("input_data_types", INPUT_PARAMS)
def test_model_construction(
    input_data_types: OrderedSet[DataType],
    output_data_types: OrderedSet[DataType],
    pytorch_dummy_dataset: PytorchDummyDataset,
    Groot,  # noqa: N803
):  # noqa: N803
    model_init_description = ModelInitDescription(
        input_data_types=input_data_types,
        output_data_types=output_data_types,
        input_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["input"],
        output_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["output"],
        output_prediction_horizon=pytorch_dummy_dataset.output_prediction_horizon,
    )
    model = Groot(model_init_description, **GROOT_TEST_ARGS)
    model = model.to(DEVICE)
    assert isinstance(model, nn.Module)


@pytest.mark.parametrize("output_data_types", OUTPUT_PARAMS)
@pytest.mark.parametrize("input_data_types", INPUT_PARAMS)
def test_model_forward(
    sample_inference_batch: BatchedInferenceInputs,
    input_data_types: OrderedSet[DataType],
    output_data_types: OrderedSet[DataType],
    pytorch_dummy_dataset: PytorchDummyDataset,
    Groot,  # noqa: N803
):
    model_init_description = ModelInitDescription(
        input_data_types=input_data_types,
        output_data_types=output_data_types,
        input_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["input"],
        output_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["output"],
        output_prediction_horizon=pytorch_dummy_dataset.output_prediction_horizon,
    )
    model = Groot(model_init_description, **GROOT_TEST_ARGS)
    model = model.to(DEVICE)
    sample_inference_batch = sample_inference_batch.to(DEVICE)
    output: dict[DataType, list[BatchedNCData]] = model(sample_inference_batch)
    assert isinstance(output, dict)
    for data_type, tensors in output.items():
        assert isinstance(data_type, DataType)
        assert isinstance(tensors, list)
        for tensor in tensors:
            assert isinstance(tensor, BatchedNCData)
            assert data_type in output_data_types


@pytest.mark.parametrize("output_data_types", OUTPUT_PARAMS)
@pytest.mark.parametrize("input_data_types", INPUT_PARAMS)
def test_model_backward(
    sample_training_batch: BatchedTrainingSamples,
    input_data_types: OrderedSet[DataType],
    output_data_types: OrderedSet[DataType],
    pytorch_dummy_dataset: PytorchDummyDataset,
    Groot,  # noqa: N803
):
    model_init_description = ModelInitDescription(
        input_data_types=input_data_types,
        output_data_types=output_data_types,
        input_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["input"],
        output_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["output"],
        output_prediction_horizon=pytorch_dummy_dataset.output_prediction_horizon,
    )
    model = Groot(model_init_description, **GROOT_TEST_ARGS)
    model = model.to(DEVICE)
    sample_training_batch = sample_training_batch.to(DEVICE)
    output: BatchedTrainingOutputs = model.training_step(sample_training_batch)

    # Compute loss
    loss = output.losses["flow_matching_loss"]

    # Perform backward pass
    loss.backward()

    # Check that gradients are computed for parameters that should have them
    for name, param in model.named_parameters():
        if param.requires_grad:
            # VLM parameters may not get gradients if they're not used in the
            # forward pass
            is_vlm_param = any(
                keyword in name.lower()
                for keyword in ["vlm", "vision", "language_model"]
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


def test_run_validation(tmp_path: Path, mock_login, monkeypatch, Groot):  # noqa: N803
    from neuracore.ml.utils import validate as validate_module

    monkeypatch.setattr(
        validate_module.AlgorithmLoader, "load_model", lambda self: Groot
    )

    # Long timeout due to larger model run on CPU
    os.environ["NEURACORE_ENDPOINT_TIMEOUT"] = "120"
    algorithm_dir = Path(inspect.getfile(Groot)).parent
    _, error_msg = run_validation(
        output_dir=tmp_path,
        algorithm_dir=algorithm_dir,
        port=random.randint(10000, 20000),
        skip_endpoint_check=False,
        algorithm_config=GROOT_TEST_ARGS,
        device=DEVICE,
    )
    if len(error_msg) > 0:
        raise RuntimeError(error_msg)
