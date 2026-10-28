import inspect
import os
import random
from pathlib import Path
from typing import cast

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
from neuracore.ml.core.ml_types import BatchedTrainingOutputs
from neuracore.ml.datasets.pytorch_dummy_dataset import PytorchDummyDataset
from neuracore.ml.utils.algorithm_loader import AlgorithmLoader
from neuracore.ml.utils.device_utils import get_default_device
from neuracore.ml.utils.validate import run_validation

DIFFUSION_POLICY_ALGORITHM_DIR = (
    Path(__file__).resolve().parents[4] / "neuracore/ml/algorithms/diffusion_policy"
)
AlgorithmLoader(DIFFUSION_POLICY_ALGORITHM_DIR).install_requirements()

from diffusers.schedulers.scheduling_ddim import DDIMScheduler  # noqa: E402

from neuracore.ml.algorithms.diffusion_policy.diffusion_policy import (  # noqa: E402
    BetaSchedule,
    DiffusionPolicy,
    LRSchedulerType,
    NoiseSchedulerType,
    PredictionType,
)

BS = 2
DEVICE = get_default_device()
OUTPUT_PREDICTION_HORIZON = 6

INPUT_PARAMS = [
    pytest.param(
        OrderedSet([data_type]),
        id="".join(w.capitalize() for w in data_type.value.split("_")),
    )
    for data_type in sorted(
        DiffusionPolicy.get_supported_input_data_types(), key=lambda d: d.value
    )
]
OUTPUT_PARAMS = [
    pytest.param(
        OrderedSet([data_type]),
        id="".join(w.capitalize() for w in data_type.value.split("_")),
    )
    for data_type in sorted(
        DiffusionPolicy.get_supported_output_data_types(), key=lambda d: d.value
    )
]


@pytest.fixture(scope="module")
def pytorch_dummy_dataset() -> PytorchDummyDataset:
    input_data_types = DiffusionPolicy.get_supported_input_data_types()
    output_data_types = DiffusionPolicy.get_supported_output_data_types()
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


DIFFUSION_POLICY_TEST_ARGS: dict = {
    "num_train_timesteps": 1,
    "num_inference_steps": 1,
    "hidden_dim": 64,
    "unet_n_groups": 4,
    "unet_down_dims": [128, 256],
}


@pytest.fixture
def model_config() -> dict:
    return DIFFUSION_POLICY_TEST_ARGS


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
def test_model_construction_forward_backward(
    input_data_types: OrderedSet[DataType],
    output_data_types: OrderedSet[DataType],
    pytorch_dummy_dataset: PytorchDummyDataset,
    model_config: dict,
    sample_inference_batch: BatchedInferenceInputs,
    sample_training_batch: BatchedTrainingSamples,
):
    description = ModelInitDescription(
        input_data_types=input_data_types,
        output_data_types=output_data_types,
        input_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["input"],
        output_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["output"],
        output_prediction_horizon=pytorch_dummy_dataset.output_prediction_horizon,
    )
    model = DiffusionPolicy(model_init_description=description, **model_config)
    model = model.to(DEVICE)
    assert isinstance(model, nn.Module)

    sample_inference_batch = sample_inference_batch.to(DEVICE)
    output: dict[DataType, list[BatchedNCData]] = model(sample_inference_batch)
    assert isinstance(output, dict)
    for data_type, tensors in output.items():
        assert isinstance(data_type, DataType)
        assert isinstance(tensors, list)
        for tensor in tensors:
            assert isinstance(tensor, BatchedNCData)

    sample_training_batch = sample_training_batch.to(DEVICE)
    output: BatchedTrainingOutputs = model.training_step(sample_training_batch)

    # Compute loss
    loss = output.losses["mse_loss"]

    # Perform backward pass
    loss.backward()

    # Check that gradients are computed
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert torch.isfinite(param.grad).all()


@pytest.mark.parametrize("output_data_types", OUTPUT_PARAMS)
@pytest.mark.parametrize("input_data_types", INPUT_PARAMS)
def test_flow_matching_forward_backward(
    input_data_types: OrderedSet[DataType],
    output_data_types: OrderedSet[DataType],
    pytorch_dummy_dataset: PytorchDummyDataset,
    model_config: dict,
    sample_inference_batch: BatchedInferenceInputs,
    sample_training_batch: BatchedTrainingSamples,
):
    """Flow matching constructs, infers, and backward passes.

    The diffusion process is already covered by
    test_model_construction_forward_backward (the default process_type).
    """
    description = ModelInitDescription(
        input_data_types=input_data_types,
        output_data_types=output_data_types,
        input_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["input"],
        output_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["output"],
        output_prediction_horizon=pytorch_dummy_dataset.output_prediction_horizon,
    )
    config = {**model_config, "process_type": "flow_matching"}
    model = DiffusionPolicy(model_init_description=description, **config)
    model = model.to(DEVICE)
    assert isinstance(model, nn.Module)

    sample_inference_batch = sample_inference_batch.to(DEVICE)
    output: dict[DataType, list[BatchedNCData]] = model(sample_inference_batch)
    assert isinstance(output, dict)
    for data_type, tensors in output.items():
        assert isinstance(data_type, DataType)
        assert isinstance(tensors, list)
        for tensor in tensors:
            assert isinstance(tensor, BatchedNCData)

    sample_training_batch = sample_training_batch.to(DEVICE)
    output: BatchedTrainingOutputs = model.training_step(sample_training_batch)

    # Compute loss
    loss = output.losses["mse_loss"]

    # Perform backward pass
    loss.backward()

    # Check that gradients are computed
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert torch.isfinite(param.grad).all()


@pytest.mark.parametrize(
    "choices",
    [
        {
            "noise_scheduler_type": "DDIM",
            "prediction_type": "sample",
            "beta_schedule": "linear",
            "lr_scheduler_type": "constant_with_warmup",
        },
        {
            "noise_scheduler_type": NoiseSchedulerType.DDIM,
            "prediction_type": PredictionType.SAMPLE,
            "beta_schedule": BetaSchedule.LINEAR,
            "lr_scheduler_type": LRSchedulerType.CONSTANT_WITH_WARMUP,
        },
    ],
    ids=["plain_strings", "enum_members"],
)
def test_fixed_choice_parameters_accept_plain_strings_and_members(
    pytorch_dummy_dataset: PytorchDummyDataset, model_config: dict, choices: dict
):
    """Training configs pass plain strings, which act like the enum members."""
    description = ModelInitDescription(
        input_data_types=OrderedSet([DataType.JOINT_POSITIONS]),
        output_data_types=OrderedSet([DataType.JOINT_TARGET_POSITIONS]),
        input_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["input"],
        output_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["output"],
        output_prediction_horizon=pytorch_dummy_dataset.output_prediction_horizon,
    )

    model = DiffusionPolicy(
        model_init_description=description, **{**model_config, **choices}
    )

    assert isinstance(model.noise_scheduler, DDIMScheduler)
    assert model.noise_scheduler.config.prediction_type == "sample"
    assert model.noise_scheduler.config.beta_schedule == "linear"
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    (scheduler,) = model.configure_schedulers([optimizer], num_training_steps=10)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)


def test_run_validation(tmp_path: Path, mock_login):
    os.environ["NEURACORE_ENDPOINT_TIMEOUT"] = "60"
    algorithm_dir = Path(inspect.getfile(DiffusionPolicy)).parent
    _, error_msg = run_validation(
        output_dir=tmp_path,
        algorithm_dir=algorithm_dir,
        port=random.randint(10000, 20000),
        device=DEVICE,
        algorithm_config=DIFFUSION_POLICY_TEST_ARGS,
    )
    if len(error_msg) > 0:
        raise RuntimeError(error_msg)
