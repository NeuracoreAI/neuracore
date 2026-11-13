import inspect
import random
from pathlib import Path
from typing import cast

import pytest
import torch
from neuracore_types import (
    BatchedDoneData,
    BatchedNCData,
    CrossEmbodimentDescription,
    DataType,
    ModelInitDescription,
)
from ordered_set import OrderedSet
from torch import nn
from torch.utils.data import DataLoader

from neuracore.ml import BatchedInferenceInputs, BatchedTrainingSamples
from neuracore.ml.algorithms.act.act import ACT
from neuracore.ml.core.ml_types import BatchedTrainingOutputs
from neuracore.ml.datasets.pytorch_dummy_dataset import PytorchDummyDataset
from neuracore.ml.utils.device_utils import get_default_device
from neuracore.ml.utils.validate import run_validation

BS = 2
DEVICE = get_default_device()
OUTPUT_PREDICTION_HORIZON = 5

INPUT_PARAMS = [
    pytest.param(
        OrderedSet([data_type]),
        id="".join(w.capitalize() for w in data_type.value.split("_")),
    )
    for data_type in sorted(ACT.get_supported_input_data_types(), key=lambda d: d.value)
]
OUTPUT_PARAMS = [
    pytest.param(
        OrderedSet([data_type]),
        id="".join(w.capitalize() for w in data_type.value.split("_")),
    )
    # DONE needs an action output next to it; it is covered by the done tests.
    for data_type in sorted(
        ACT.get_supported_output_data_types() - {DataType.DONE},
        key=lambda d: d.value,
    )
]


@pytest.fixture(scope="module")
def pytorch_dummy_dataset() -> PytorchDummyDataset:
    input_data_types = ACT.get_supported_input_data_types()
    output_data_types = ACT.get_supported_output_data_types()
    input_cross_embodiment_description: CrossEmbodimentDescription = {
        "robot_1": {data_type: {} for data_type in input_data_types}
    }
    output_cross_embodiment_description: CrossEmbodimentDescription = {
        "robot_1": {data_type: {} for data_type in output_data_types}
    }
    output_cross_embodiment_description["robot_1"][DataType.DONE] = {0: "done"}
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
def test_model_construction_forward_backward(
    input_data_types: OrderedSet[DataType],
    output_data_types: OrderedSet[DataType],
    pytorch_dummy_dataset: PytorchDummyDataset,
    model_config: dict,
    sample_inference_batch: BatchedInferenceInputs,
    sample_training_batch: BatchedTrainingSamples,
):
    model_init_description = ModelInitDescription(
        input_data_types=input_data_types,
        output_data_types=output_data_types,
        input_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["input"],
        output_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["output"],
        output_prediction_horizon=pytorch_dummy_dataset.output_prediction_horizon,
    )
    model = ACT(model_init_description=model_init_description, **model_config)
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
    loss = output.losses["l1_and_kl_loss"]

    # Perform backward pass
    loss.backward()

    # Check that gradients are computed
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert torch.isfinite(param.grad).all()


# --- Done head ---

SMALL_MODEL_CONFIG = {
    "hidden_dim": 64,
    "num_encoder_layers": 1,
    "nheads": 4,
    "dim_feedforward": 128,
    "latent_dim": 32,
}

DONE_INPUTS = OrderedSet([
    DataType.RGB_IMAGES,
    DataType.JOINT_POSITIONS,
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
])
DONE_OUTPUTS = OrderedSet([DataType.JOINT_TARGET_POSITIONS, DataType.DONE])


def _inference_batch(batch: BatchedTrainingSamples) -> BatchedInferenceInputs:
    return BatchedInferenceInputs(
        inputs=batch.inputs,
        inputs_mask=batch.inputs_mask,
        batch_size=batch.batch_size,
    )


def _model_init_description(
    dataset: PytorchDummyDataset,
    input_data_types: OrderedSet[DataType],
    output_data_types: OrderedSet[DataType],
) -> ModelInitDescription:
    return ModelInitDescription(
        input_data_types=input_data_types,
        output_data_types=output_data_types,
        input_dataset_statistics=dataset.dataset_statistics["input"],
        output_dataset_statistics=dataset.dataset_statistics["output"],
        output_prediction_horizon=dataset.output_prediction_horizon,
    )


def _labelled_done_batch(batch: BatchedTrainingSamples) -> BatchedTrainingSamples:
    """Mark every input and output as present and make one current step done.

    The dummy dataset leaves masks at zero, which would hide the done loss and
    zero the camera features the done head reads.
    """
    for data_type in batch.inputs_mask:
        batch.inputs_mask[data_type] = torch.ones_like(batch.inputs_mask[data_type])
    for data_type in batch.outputs_mask:
        batch.outputs_mask[data_type] = torch.ones_like(batch.outputs_mask[data_type])
    for item in cast(list[BatchedDoneData], batch.outputs[DataType.DONE]):
        item.done.zero_()
        item.done[0, 0, 0] = 1.0
    return batch


@pytest.mark.parametrize("done_use_proprio", [False, True])
@pytest.mark.parametrize("detach_done_encoder", [False, True])
def test_done_head_forward_backward(
    detach_done_encoder: bool,
    done_use_proprio: bool,
    pytorch_dummy_dataset: PytorchDummyDataset,
    sample_training_batch: BatchedTrainingSamples,
):
    model = ACT(
        model_init_description=_model_init_description(
            pytorch_dummy_dataset, DONE_INPUTS, DONE_OUTPUTS
        ),
        detach_done_encoder=detach_done_encoder,
        done_use_proprio=done_use_proprio,
        **SMALL_MODEL_CONFIG,
    ).to(DEVICE)
    assert model.done_head is not None
    assert (model.done_proprio_to_logits is not None) == done_use_proprio

    batch = _labelled_done_batch(sample_training_batch.to(DEVICE))

    model.eval()
    with torch.no_grad():
        output = model(_inference_batch(batch))
    done_outputs = output[DataType.DONE]
    assert len(done_outputs) == 1
    for done_output in done_outputs:
        assert isinstance(done_output, BatchedDoneData)
        assert done_output.done.shape == (BS, 1, 1)
        assert ((done_output.done >= 0) & (done_output.done <= 1)).all()

    model.train()
    training_output = model.training_step(batch)
    assert training_output.losses["done_bce_loss"] > 0
    for metric in ("done_accuracy", "done_precision", "done_recall"):
        assert 0.0 <= float(training_output.metrics[metric]) <= 1.0

    sum(training_output.losses.values()).backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert torch.isfinite(param.grad).all()


@pytest.mark.parametrize("detach_done_encoder", [False, True])
def test_detach_done_encoder_blocks_done_gradient_to_image_encoder(
    detach_done_encoder: bool,
    pytorch_dummy_dataset: PytorchDummyDataset,
    sample_training_batch: BatchedTrainingSamples,
):
    model = ACT(
        model_init_description=_model_init_description(
            pytorch_dummy_dataset, DONE_INPUTS, DONE_OUTPUTS
        ),
        detach_done_encoder=detach_done_encoder,
        **SMALL_MODEL_CONFIG,
    ).to(DEVICE)
    batch = _labelled_done_batch(sample_training_batch.to(DEVICE))

    model.training_step(batch).losses["done_bce_loss"].backward()

    encoder_grads = [
        param.grad for param in model.image_encoders.parameters() if param.requires_grad
    ]
    reached_encoder = any(
        grad is not None and grad.abs().sum() > 0 for grad in encoder_grads
    )
    assert reached_encoder != detach_done_encoder
    assert all(param.grad is not None for param in model.done_head.parameters())


@pytest.mark.parametrize("done_max_pos_weight", [10.0, 50.0, 200.0])
def test_done_pos_weight_is_capped(
    done_max_pos_weight: float,
    pytorch_dummy_dataset: PytorchDummyDataset,
):
    # The dummy dataset has no done frames, so (1 - p) / p hits the cap.
    model = ACT(
        model_init_description=_model_init_description(
            pytorch_dummy_dataset, DONE_INPUTS, DONE_OUTPUTS
        ),
        done_max_pos_weight=done_max_pos_weight,
        **SMALL_MODEL_CONFIG,
    )
    assert torch.allclose(
        model.done_pos_weight,
        torch.full_like(model.done_pos_weight, done_max_pos_weight),
    )


def test_done_requires_rgb_images(pytorch_dummy_dataset: PytorchDummyDataset):
    with pytest.raises(ValueError, match="RGB_IMAGES"):
        ACT(
            model_init_description=_model_init_description(
                pytorch_dummy_dataset,
                OrderedSet([DataType.JOINT_POSITIONS]),
                DONE_OUTPUTS,
            ),
            **SMALL_MODEL_CONFIG,
        )


def test_done_alone_is_not_a_valid_output(
    pytorch_dummy_dataset: PytorchDummyDataset,
):
    with pytest.raises(ValueError, match="continuous action"):
        ACT(
            model_init_description=_model_init_description(
                pytorch_dummy_dataset, DONE_INPUTS, OrderedSet([DataType.DONE])
            ),
            **SMALL_MODEL_CONFIG,
        )


def test_without_done_output_model_is_plain_act(
    pytorch_dummy_dataset: PytorchDummyDataset,
    sample_training_batch: BatchedTrainingSamples,
):
    # No done modules or buffers, so checkpoints from before the done head load.
    model = ACT(
        model_init_description=_model_init_description(
            pytorch_dummy_dataset,
            DONE_INPUTS,
            OrderedSet([DataType.JOINT_TARGET_POSITIONS]),
        ),
        **SMALL_MODEL_CONFIG,
    ).to(DEVICE)
    assert model.done_head is None
    assert not [key for key in model.state_dict() if "done" in key]

    batch = sample_training_batch.to(DEVICE)
    assert DataType.DONE not in model(_inference_batch(batch))
    training_output = model.training_step(batch)
    assert set(training_output.losses) == {"l1_and_kl_loss"}


def test_run_validation(tmp_path: Path, mock_login):
    algorithm_dir = Path(inspect.getfile(ACT)).parent
    _, error_msg = run_validation(
        output_dir=tmp_path,
        algorithm_dir=algorithm_dir,
        port=random.randint(10000, 20000),
        algorithm_config={
            "hidden_dim": 64,
            "num_encoder_layers": 2,
            "num_decoder_layers": 4,
            "nheads": 4,
        },
        device=DEVICE,
    )
    if len(error_msg) > 0:
        raise RuntimeError(error_msg)
