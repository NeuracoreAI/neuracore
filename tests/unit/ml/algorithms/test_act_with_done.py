import inspect
import os
import random
from pathlib import Path
from typing import cast

import pytest
import torch
from neuracore_types import (
    BatchedCustom1DData,
    BatchedNCData,
    CrossEmbodimentDescription,
    DataType,
    ModelInitDescription,
)
from ordered_set import OrderedSet
from torch import nn
from torch.utils.data import DataLoader

from neuracore.ml import BatchedInferenceInputs, BatchedTrainingSamples
from neuracore.ml.algorithms.act_with_done import ACTWithDone
from neuracore.ml.core.ml_types import BatchedTrainingOutputs
from neuracore.ml.datasets.pytorch_dummy_dataset import PytorchDummyDataset
from neuracore.ml.utils.device_utils import get_default_device
from neuracore.ml.utils.validate import run_validation

BS = 2
DEVICE = get_default_device()
OUTPUT_PREDICTION_HORIZON = 5

_CONTINUOUS_OUTPUT_TYPES = sorted(
    ACTWithDone.get_supported_output_data_types() - {DataType.CUSTOM_1D},
    key=lambda d: d.value,
)

INPUT_PARAMS = [
    pytest.param(
        OrderedSet([data_type]),
        id="".join(w.capitalize() for w in data_type.value.split("_")),
    )
    for data_type in sorted(
        ACTWithDone.get_supported_input_data_types(), key=lambda d: d.value
    )
]
OUTPUT_PARAMS = [
    pytest.param(
        OrderedSet([data_type]),
        id="".join(w.capitalize() for w in data_type.value.split("_")),
    )
    for data_type in _CONTINUOUS_OUTPUT_TYPES
] + [
    pytest.param(
        OrderedSet([DataType.JOINT_TARGET_POSITIONS, DataType.CUSTOM_1D]),
        id="JointTargetPositionsWithDone",
    ),
]


@pytest.fixture(scope="module")
def pytorch_dummy_dataset() -> PytorchDummyDataset:
    input_data_types = ACTWithDone.get_supported_input_data_types()
    output_data_types = ACTWithDone.get_supported_output_data_types()
    input_cross_embodiment_description: CrossEmbodimentDescription = {
        "robot_1": {data_type: {} for data_type in input_data_types}
    }
    output_cross_embodiment_description: CrossEmbodimentDescription = {
        "robot_1": {data_type: {} for data_type in output_data_types}
    }
    return PytorchDummyDataset(
        num_samples=5,
        input_cross_embodiment_description=input_cross_embodiment_description,
        output_cross_embodiment_description=output_cross_embodiment_description,
        output_prediction_horizon=OUTPUT_PREDICTION_HORIZON,
    )


ACT_WITH_DONE_TEST_ARGS: dict = {
    "hidden_dim": 64,
    "num_encoder_layers": 2,
    "num_decoder_layers": 1,
    "nheads": 4,
    "dim_feedforward": 128,
    "latent_dim": 32,
}


@pytest.fixture
def model_config() -> dict:
    return ACT_WITH_DONE_TEST_ARGS


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


def test_custom_1d_only_raises(
    pytorch_dummy_dataset: PytorchDummyDataset,
    model_config: dict,
):
    description = ModelInitDescription(
        input_data_types=OrderedSet([DataType.JOINT_POSITIONS]),
        output_data_types=OrderedSet([DataType.CUSTOM_1D]),
        input_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["input"],
        output_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["output"],
        output_prediction_horizon=pytorch_dummy_dataset.output_prediction_horizon,
    )
    with pytest.raises(ValueError, match="continuous action"):
        ACTWithDone(model_init_description=description, **model_config)


def test_custom_1d_requires_rgb(
    pytorch_dummy_dataset: PytorchDummyDataset,
    model_config: dict,
):
    description = ModelInitDescription(
        input_data_types=OrderedSet([DataType.JOINT_POSITIONS]),
        output_data_types=OrderedSet(
            [DataType.JOINT_TARGET_POSITIONS, DataType.CUSTOM_1D]
        ),
        input_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["input"],
        output_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["output"],
        output_prediction_horizon=pytorch_dummy_dataset.output_prediction_horizon,
    )
    with pytest.raises(ValueError, match="RGB_IMAGES"):
        ACTWithDone(model_init_description=description, **model_config)


@pytest.mark.parametrize("output_data_types", OUTPUT_PARAMS)
@pytest.mark.parametrize("input_data_types", INPUT_PARAMS)
def test_model_construction_forward_backward(
    input_data_types: OrderedSet[DataType],
    output_data_types: OrderedSet[DataType],
    pytorch_dummy_dataset: PytorchDummyDataset,
    model_config: dict,
    sample_inference_batch: BatchedInferenceInputs,
    sample_training_batch: BatchedTrainingSamples,
):
    if (
        DataType.CUSTOM_1D in output_data_types
        and DataType.RGB_IMAGES not in input_data_types
    ):
        pytest.skip("Done-flag head requires RGB images")

    description = ModelInitDescription(
        input_data_types=input_data_types,
        output_data_types=output_data_types,
        input_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["input"],
        output_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["output"],
        output_prediction_horizon=pytorch_dummy_dataset.output_prediction_horizon,
    )
    model = ACTWithDone(model_init_description=description, **model_config)
    model = model.to(DEVICE)
    assert isinstance(model, nn.Module)
    if DataType.CUSTOM_1D in output_data_types:
        assert model.done_head is not None
        assert model.done_head[0].in_features == model.vision_cond_dim

    sample_inference_batch = sample_inference_batch.to(DEVICE)
    output: dict[DataType, list[BatchedNCData]] = model(sample_inference_batch)
    assert isinstance(output, dict)
    for data_type, tensors in output.items():
        assert isinstance(data_type, DataType)
        assert isinstance(tensors, list)
        for tensor in tensors:
            assert isinstance(tensor, BatchedNCData)

    if DataType.CUSTOM_1D in output_data_types:
        assert DataType.CUSTOM_1D in output
        for item in output[DataType.CUSTOM_1D]:
            assert isinstance(item, BatchedCustom1DData)
            assert torch.all(item.data >= 0.0)
            assert torch.all(item.data <= 1.0)

    sample_training_batch = sample_training_batch.to(DEVICE)
    train_output: BatchedTrainingOutputs = model.training_step(sample_training_batch)

    loss = train_output.losses["l1_and_kl_loss"]
    if DataType.CUSTOM_1D in output_data_types:
        assert "done_bce_loss" in train_output.losses
        assert "done_bce_loss" in train_output.metrics
        loss = loss + train_output.losses["done_bce_loss"]
    else:
        assert "done_bce_loss" not in train_output.losses

    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert torch.isfinite(param.grad).all()


def test_run_validation(tmp_path: Path, mock_login):
    os.environ["NEURACORE_ENDPOINT_TIMEOUT"] = "60"
    algorithm_dir = Path(inspect.getfile(ACTWithDone)).parent
    _, error_msg = run_validation(
        output_dir=tmp_path,
        algorithm_dir=algorithm_dir,
        port=random.randint(10000, 20000),
        algorithm_config=ACT_WITH_DONE_TEST_ARGS,
        device=DEVICE,
    )
    if len(error_msg) > 0:
        raise RuntimeError(error_msg)
