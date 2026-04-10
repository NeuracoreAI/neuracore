import inspect
import random
from pathlib import Path
from typing import cast

import pytest
import torch
from neuracore_types import BatchedNCData, DataType, ModelInitDescription
from ordered_set import OrderedSet
from torch import nn
from torch.utils.data import DataLoader

from neuracore.ml import BatchedInferenceInputs, BatchedTrainingSamples
from neuracore.ml.algorithms.diffusion_policy.diffusion_policy import DiffusionPolicy
from neuracore.ml.core.ml_types import BatchedTrainingOutputs
from neuracore.ml.datasets.pytorch_dummy_dataset import PytorchDummyDataset
from neuracore.ml.utils.validate import run_validation

BS = 2
DEVICE = torch.device("cpu")
OUTPUT_PREDICTION_HORIZON = 5

INPUT_PARAMS = [
    pytest.param(OrderedSet([data_type]), id=data_type.value)
    for data_type in DiffusionPolicy.get_supported_input_data_types()
]
OUTPUT_PARAMS = [
    # TODO: expand to all supported output data types in nightly CI.
    pytest.param(
        OrderedSet([DataType.JOINT_TARGET_POSITIONS]),
        id=DataType.JOINT_TARGET_POSITIONS.value,
    )
]


@pytest.fixture(scope="module")
def pytorch_dummy_dataset() -> PytorchDummyDataset:
    input_data_types = DiffusionPolicy.get_supported_input_data_types()
    output_data_types = DiffusionPolicy.get_supported_output_data_types()
    input_cross_embodiment_description = {
        "robot_1": {data_type: [] for data_type in input_data_types}
    }
    output_cross_embodiment_description = {
        "robot_1": {data_type: [] for data_type in output_data_types}
    }
    return PytorchDummyDataset(
        num_samples=5,
        input_cross_embodiment_description=input_cross_embodiment_description,
        output_cross_embodiment_description=output_cross_embodiment_description,
        output_prediction_horizon=OUTPUT_PREDICTION_HORIZON,
    )


@pytest.fixture
def model_config() -> dict:
    return {}


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
    return cast(BatchedTrainingSamples, next(iter(dataloader)))


@pytest.mark.parametrize("output_data_types", OUTPUT_PARAMS)
@pytest.mark.parametrize("input_data_types", INPUT_PARAMS)
def test_model_construction(
    input_data_types: OrderedSet[DataType],
    output_data_types: OrderedSet[DataType],
    pytorch_dummy_dataset: PytorchDummyDataset,
    model_config: dict,
):
    description = ModelInitDescription(
        input_data_types=input_data_types,
        output_data_types=output_data_types,
        input_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["input"],
        output_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["output"],
        output_prediction_horizon=pytorch_dummy_dataset.output_prediction_horizon,
    )
    model = DiffusionPolicy(description, **model_config)
    model = model.to(DEVICE)
    assert isinstance(model, nn.Module)


@pytest.mark.parametrize("output_data_types", OUTPUT_PARAMS)
@pytest.mark.parametrize("input_data_types", INPUT_PARAMS)
def test_model_forward(
    input_data_types: OrderedSet[DataType],
    output_data_types: OrderedSet[DataType],
    pytorch_dummy_dataset: PytorchDummyDataset,
    model_config: dict,
    sample_inference_batch: BatchedInferenceInputs,
):
    description = ModelInitDescription(
        input_data_types=input_data_types,
        output_data_types=output_data_types,
        input_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["input"],
        output_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["output"],
        output_prediction_horizon=pytorch_dummy_dataset.output_prediction_horizon,
    )
    model = DiffusionPolicy(description, **model_config)
    model = model.to(DEVICE)
    sample_inference_batch = sample_inference_batch.to(DEVICE)
    output: dict[DataType, list[BatchedNCData]] = model(sample_inference_batch)
    assert isinstance(output, dict)
    for data_type, tensors in output.items():
        assert isinstance(data_type, DataType)
        assert isinstance(tensors, list)
        for tensor in tensors:
            assert isinstance(tensor, BatchedNCData)


@pytest.mark.parametrize("output_data_types", OUTPUT_PARAMS)
@pytest.mark.parametrize("input_data_types", INPUT_PARAMS)
def test_model_backward(
    input_data_types: OrderedSet[DataType],
    output_data_types: OrderedSet[DataType],
    pytorch_dummy_dataset: PytorchDummyDataset,
    model_config: dict,
    sample_training_batch: BatchedTrainingSamples,
):
    description = ModelInitDescription(
        input_data_types=input_data_types,
        output_data_types=output_data_types,
        input_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["input"],
        output_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["output"],
        output_prediction_horizon=pytorch_dummy_dataset.output_prediction_horizon,
    )
    model = DiffusionPolicy(description, **model_config)
    model = model.to(DEVICE)
    sample_training_batch = sample_training_batch.to(DEVICE)
    output: BatchedTrainingOutputs = model.training_step(sample_training_batch)

    loss = output.losses["mse_loss"]
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert torch.isfinite(param.grad).all()


def test_run_validation(tmp_path: Path, mock_login):
    algorithm_dir = Path(inspect.getfile(DiffusionPolicy)).parent
    _, error_msg = run_validation(
        output_dir=tmp_path,
        algorithm_dir=algorithm_dir,
        port=random.randint(10000, 20000),
        device=DEVICE,
    )
    if len(error_msg) > 0:
        raise RuntimeError(error_msg)
