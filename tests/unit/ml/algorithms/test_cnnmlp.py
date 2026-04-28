import inspect
import random
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
import torch
from neuracore_types import (
    BatchedJointData,
    BatchedNCData,
    BatchedParallelGripperOpenAmountData,
    DataType,
    ModelInitDescription,
)
from ordered_set import OrderedSet
from torch import nn
from torch.utils.data import DataLoader

from neuracore.ml import BatchedInferenceInputs, BatchedTrainingSamples
from neuracore.ml.algorithms.cnnmlp.cnnmlp import CNNMLP
from neuracore.ml.core.ml_types import BatchedTrainingOutputs
from neuracore.ml.datasets.pytorch_dummy_dataset import PytorchDummyDataset
from neuracore.ml.utils.device_utils import get_default_device
from neuracore.ml.utils.validate import run_validation

BS = 2
DEVICE = get_default_device()
OUTPUT_PREDICTION_HORIZON = 5

INPUT_PARAMS = [
    pytest.param(OrderedSet([data_type]), id=data_type.value)
    for data_type in CNNMLP.get_supported_input_data_types()
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
    input_cross_embodiment_description = {
        "robot_1": {
            data_type: [] for data_type in CNNMLP.get_supported_input_data_types()
        }
    }
    output_cross_embodiment_description = {
        "robot_1": {
            data_type: [] for data_type in CNNMLP.get_supported_output_data_types()
        }
    }
    return PytorchDummyDataset(
        num_samples=5,
        input_cross_embodiment_description=input_cross_embodiment_description,
        output_cross_embodiment_description=output_cross_embodiment_description,
        output_prediction_horizon=OUTPUT_PREDICTION_HORIZON,
    )


@pytest.fixture
def pytorch_dummy_dataset_no_proprio() -> PytorchDummyDataset:
    input_cross_embodiment_description = {"robot_1": {DataType.RGB_IMAGES: []}}
    output_cross_embodiment_description = {
        "robot_1": {
            data_type: [] for data_type in CNNMLP.get_supported_output_data_types()
        }
    }
    return PytorchDummyDataset(
        num_samples=5,
        input_cross_embodiment_description=input_cross_embodiment_description,
        output_cross_embodiment_description=output_cross_embodiment_description,
        output_prediction_horizon=OUTPUT_PREDICTION_HORIZON,
    )


def _build_model_init_description(
    pytorch_dummy_dataset: PytorchDummyDataset,
    input_data_types: OrderedSet[DataType],
    output_data_types: OrderedSet[DataType],
) -> ModelInitDescription:
    return ModelInitDescription(
        input_data_types=input_data_types,
        output_data_types=output_data_types,
        input_dataset_statistics={
            data_type: deepcopy(
                pytorch_dummy_dataset.dataset_statistics["input"][data_type]
            )
            for data_type in input_data_types
        },
        output_dataset_statistics={
            data_type: deepcopy(
                pytorch_dummy_dataset.dataset_statistics["output"][data_type]
            )
            for data_type in output_data_types
        },
        output_prediction_horizon=pytorch_dummy_dataset.output_prediction_horizon,
    )


@pytest.fixture
def model_init_description(
    pytorch_dummy_dataset: PytorchDummyDataset,
) -> ModelInitDescription:
    return _build_model_init_description(
        pytorch_dummy_dataset,
        OrderedSet(CNNMLP.get_supported_input_data_types()),
        OrderedSet(CNNMLP.get_supported_output_data_types()),
    )


@pytest.fixture
def model_init_description_no_proprio(
    pytorch_dummy_dataset_no_proprio: PytorchDummyDataset,
) -> ModelInitDescription:
    return _build_model_init_description(
        pytorch_dummy_dataset_no_proprio,
        OrderedSet([DataType.RGB_IMAGES]),
        OrderedSet(CNNMLP.get_supported_output_data_types()),
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
    return cast(BatchedTrainingSamples, next(iter(dataloader)))


@pytest.mark.parametrize("output_data_types", OUTPUT_PARAMS)
@pytest.mark.parametrize("input_data_types", INPUT_PARAMS)
def test_model_construction(
    input_data_types: OrderedSet[DataType],
    output_data_types: OrderedSet[DataType],
    pytorch_dummy_dataset: PytorchDummyDataset,
    model_config: dict,
):
    model_init_description = _build_model_init_description(
        pytorch_dummy_dataset,
        input_data_types,
        output_data_types,
    )
    model = CNNMLP(model_init_description, **model_config)
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
    model_init_description = _build_model_init_description(
        pytorch_dummy_dataset,
        input_data_types,
        output_data_types,
    )
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

    expected_output_width = dict(model.output_layout)
    assert set(output.keys()) == set(model.ordered_output_data_types)
    for data_type, tensors in output.items():
        assert len(tensors) == expected_output_width[data_type]


@pytest.mark.parametrize("output_data_types", OUTPUT_PARAMS)
@pytest.mark.parametrize("input_data_types", INPUT_PARAMS)
def test_model_backward(
    input_data_types: OrderedSet[DataType],
    output_data_types: OrderedSet[DataType],
    pytorch_dummy_dataset: PytorchDummyDataset,
    model_config: dict,
    sample_training_batch: BatchedTrainingSamples,
):
    model_init_description = _build_model_init_description(
        pytorch_dummy_dataset,
        input_data_types,
        output_data_types,
    )
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


def test_model_supports_input_output_count_divergence(
    model_init_description: ModelInitDescription,
):
    input_stats = deepcopy(model_init_description.input_dataset_statistics)
    output_stats = {
        DataType.JOINT_TARGET_POSITIONS: deepcopy(
            model_init_description.output_dataset_statistics[
                DataType.JOINT_TARGET_POSITIONS
            ][:1]
        )
    }
    model_init = ModelInitDescription(
        input_data_types=model_init_description.input_data_types,
        output_data_types=OrderedSet([DataType.JOINT_TARGET_POSITIONS]),
        input_dataset_statistics=input_stats,
        output_dataset_statistics=output_stats,
        output_prediction_horizon=model_init_description.output_prediction_horizon,
    )

    model = CNNMLP(model_init)

    joint_encoder = cast(nn.Linear, model.encoders[DataType.JOINT_POSITIONS])
    assert joint_encoder.in_features == len(input_stats[DataType.JOINT_POSITIONS])
    assert model.max_output_size == 1
    assert model.output_layout == [(DataType.JOINT_TARGET_POSITIONS, 1)]


def test_training_step_uses_deterministic_output_order(
    model_init_description: ModelInitDescription,
    sample_training_batch: BatchedTrainingSamples,
    monkeypatch: pytest.MonkeyPatch,
):
    model = CNNMLP(model_init_description).to(DEVICE)
    batch = sample_training_batch.to(DEVICE)

    reversed_output_types = list(reversed(list(batch.outputs.keys())))
    reordered_batch = BatchedTrainingSamples(
        inputs=batch.inputs,
        inputs_mask=batch.inputs_mask,
        outputs={
            data_type: batch.outputs[data_type] for data_type in reversed_output_types
        },
        outputs_mask={
            data_type: batch.outputs_mask[data_type]
            for data_type in reversed_output_types
        },
        batch_size=batch.batch_size,
    )

    expected_targets = []
    for data_type in model.ordered_output_data_types:
        if data_type in [DataType.JOINT_TARGET_POSITIONS, DataType.JOINT_POSITIONS]:
            joints = cast(list[BatchedJointData], reordered_batch.outputs[data_type])
            expected_targets.extend([joint.value for joint in joints])
        else:
            grippers = cast(
                list[BatchedParallelGripperOpenAmountData],
                reordered_batch.outputs[data_type],
            )
            expected_targets.extend([gripper.open_amount for gripper in grippers])
    expected_action_data = torch.cat(expected_targets, dim=-1).view(
        reordered_batch.batch_size, model.output_prediction_horizon, -1
    )

    captured: dict[str, torch.Tensor] = {}

    def capture_normalize(tensor: torch.Tensor) -> torch.Tensor:
        captured["action_data"] = tensor.detach().clone()
        return tensor

    def fake_predict_action(_: BatchedInferenceInputs) -> torch.Tensor:
        return torch.zeros(
            (
                reordered_batch.batch_size,
                model.output_prediction_horizon,
                model.max_output_size,
            ),
            device=model.device,
            dtype=torch.float32,
        )

    monkeypatch.setattr(model.action_normalizer, "normalize", capture_normalize)
    monkeypatch.setattr(model, "_predict_action", fake_predict_action)

    model.training_step(reordered_batch)
    assert torch.allclose(captured["action_data"], expected_action_data)


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
