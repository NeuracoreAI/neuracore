import inspect
import random
from pathlib import Path
from typing import cast

import pytest
import torch
from neuracore_types import BatchedNCData, DataType, ModelInitDescription, RobotDataSpec
from torch import nn
from torch.utils.data import DataLoader

from neuracore.ml import BatchedInferenceInputs, BatchedTrainingSamples
from neuracore.ml.algorithms.cnnmlp.cnnmlp import CNNMLP
from neuracore.ml.core.ml_types import BatchedTrainingOutputs
from neuracore.ml.datasets.pytorch_dummy_dataset import PytorchDummyDataset
from neuracore.ml.utils.device_utils import get_default_device
from neuracore.ml.utils.robot_data_spec_utils import extract_data_types
from neuracore.ml.utils.validate import run_validation

BS = 2
DEVICE = get_default_device()
OUTPUT_PREDICTION_HORIZON = 5


@pytest.fixture
def pytorch_dummy_dataset() -> PytorchDummyDataset:
    input_data_types = CNNMLP.get_supported_input_data_types()
    output_data_types = CNNMLP.get_supported_output_data_types()
    input_robot_data_spec: RobotDataSpec = {
        "robot_1": {data_type: [] for data_type in input_data_types}
    }
    output_robot_data_spec: RobotDataSpec = {
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
def pytorch_dummy_dataset_no_proprio() -> PytorchDummyDataset:
    input_robot_data_spec: RobotDataSpec = {"robot_1": {DataType.RGB_IMAGES: []}}
    output_robot_data_spec: RobotDataSpec = {
        "robot_1": {
            data_type: [] for data_type in CNNMLP.get_supported_output_data_types()
        }
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
def model_init_description_no_proprio(
    pytorch_dummy_dataset_no_proprio: PytorchDummyDataset,
) -> ModelInitDescription:
    input_data_types = extract_data_types(
        pytorch_dummy_dataset_no_proprio.input_robot_data_spec
    )
    output_data_types = extract_data_types(
        pytorch_dummy_dataset_no_proprio.output_robot_data_spec
    )
    return ModelInitDescription(
        input_data_types=input_data_types,
        output_data_types=output_data_types,
        dataset_statistics=pytorch_dummy_dataset_no_proprio.dataset_statistics,
        output_prediction_horizon=pytorch_dummy_dataset_no_proprio.output_prediction_horizon,
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
def sample_inference_batch_no_proprio(
    pytorch_dummy_dataset_no_proprio: PytorchDummyDataset,
) -> BatchedInferenceInputs:
    dataloader = DataLoader(
        pytorch_dummy_dataset_no_proprio,
        batch_size=BS,
        shuffle=True,
        collate_fn=pytorch_dummy_dataset_no_proprio.collate_fn,
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


@pytest.fixture
def sample_training_batch_no_proprio(
    pytorch_dummy_dataset_no_proprio: PytorchDummyDataset,
) -> BatchedTrainingSamples:
    dataloader = DataLoader(
        pytorch_dummy_dataset_no_proprio,
        batch_size=BS,
        shuffle=True,
        collate_fn=pytorch_dummy_dataset_no_proprio.collate_fn,
    )
    sample = cast(BatchedTrainingSamples, next(iter(dataloader)))
    return sample


def test_model_construction(
    model_init_description: ModelInitDescription, model_config: dict
):
    model = CNNMLP(model_init_description, **model_config)
    model = model.to(DEVICE)
    assert isinstance(model, nn.Module)


def test_model_forward(
    model_init_description: ModelInitDescription,
    model_config: dict,
    sample_inference_batch: BatchedInferenceInputs,
):
    model = CNNMLP(model_init_description, **model_config)
    model = model.to(DEVICE)
    sample_inference_batch = sample_inference_batch.to(DEVICE)
    output: dict[DataType, list[BatchedNCData]] = model(sample_inference_batch)
    assert isinstance(output, dict)
    for data_type, tensors in output.items():
        assert isinstance(data_type, DataType)
        assert isinstance(tensors, list)
        for tensor in tensors:
            assert isinstance(tensor, BatchedNCData)


def test_model_forward_without_proprioception(
    model_init_description_no_proprio: ModelInitDescription,
    model_config: dict,
    sample_inference_batch_no_proprio: BatchedInferenceInputs,
):
    model = CNNMLP(model_init_description_no_proprio, **model_config)
    model = model.to(DEVICE)
    sample_inference_batch_no_proprio = sample_inference_batch_no_proprio.to(DEVICE)
    assert model.proprio_normalizer is None
    output: dict[DataType, list[BatchedNCData]] = model(
        sample_inference_batch_no_proprio
    )
    assert isinstance(output, dict)
    for data_type, tensors in output.items():
        assert isinstance(data_type, DataType)
        assert isinstance(tensors, list)
        for tensor in tensors:
            assert isinstance(tensor, BatchedNCData)


def test_model_backward(
    model_init_description: ModelInitDescription,
    model_config: dict,
    sample_training_batch: BatchedTrainingSamples,
):
    model = CNNMLP(model_init_description, **model_config)
    model = model.to(DEVICE)
    sample_training_batch = sample_training_batch.to(DEVICE)
    output: BatchedTrainingOutputs = model.training_step(sample_training_batch)

    # Compute loss
    loss = output.losses["l1_loss"]

    # Perform backward pass
    loss.backward()

    # Check that gradients are computed
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert torch.isfinite(param.grad).all()


def test_model_backward_without_proprioception(
    model_init_description_no_proprio: ModelInitDescription,
    model_config: dict,
    sample_training_batch_no_proprio: BatchedTrainingSamples,
):
    model = CNNMLP(model_init_description_no_proprio, **model_config)
    model = model.to(DEVICE)
    sample_training_batch_no_proprio = sample_training_batch_no_proprio.to(DEVICE)
    output: BatchedTrainingOutputs = model.training_step(
        sample_training_batch_no_proprio
    )

    loss = output.losses["l1_loss"]
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert torch.isfinite(param.grad).all()


def test_run_validation(tmp_path: Path, mock_login):
    algorithm_dir = Path(inspect.getfile(CNNMLP)).parent
    _, error_msg = run_validation(
        output_dir=tmp_path,
        algorithm_dir=algorithm_dir,
        port=random.randint(10000, 20000),
        device=DEVICE,
    )
    if len(error_msg) > 0:
        raise RuntimeError(error_msg)
