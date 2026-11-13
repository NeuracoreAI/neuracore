import inspect
import random
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
import torch
from neuracore_types import (
    BatchedDoneData,
    BatchedJointData,
    BatchedNCData,
    BatchedParallelGripperOpenAmountData,
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

CNNMLP_ALGORITHM_DIR = (
    Path(__file__).resolve().parents[4] / "neuracore/ml/algorithms/cnnmlp"
)
AlgorithmLoader(CNNMLP_ALGORITHM_DIR).install_requirements()

from neuracore.ml.algorithms.cnnmlp.cnnmlp import CNNMLP  # noqa: E402

BS = 2
DEVICE = get_default_device()
OUTPUT_PREDICTION_HORIZON = 5

INPUT_PARAMS = [
    pytest.param(
        OrderedSet([data_type]),
        id="".join(w.capitalize() for w in data_type.value.split("_")),
    )
    for data_type in sorted(
        CNNMLP.get_supported_input_data_types(), key=lambda d: d.value
    )
]
OUTPUT_PARAMS = [
    pytest.param(
        OrderedSet([data_type]),
        id="".join(w.capitalize() for w in data_type.value.split("_")),
    )
    # DONE needs an action output next to it; it is covered by the done tests.
    for data_type in sorted(
        CNNMLP.get_supported_output_data_types() - {DataType.DONE},
        key=lambda d: d.value,
    )
]


@pytest.fixture(scope="module")
def pytorch_dummy_dataset() -> PytorchDummyDataset:
    input_cross_embodiment_description: CrossEmbodimentDescription = {
        "robot_1": {
            data_type: {} for data_type in CNNMLP.get_supported_input_data_types()
        }
    }
    output_cross_embodiment_description: CrossEmbodimentDescription = {
        "robot_1": {
            data_type: {} for data_type in CNNMLP.get_supported_output_data_types()
        }
        # A robot logs exactly one done flag.
        | {DataType.DONE: {0: "done"}}
    }
    return PytorchDummyDataset(
        num_samples=5,
        input_cross_embodiment_description=input_cross_embodiment_description,
        output_cross_embodiment_description=output_cross_embodiment_description,
        output_prediction_horizon=OUTPUT_PREDICTION_HORIZON,
    )


@pytest.fixture
def pytorch_dummy_dataset_no_proprio() -> PytorchDummyDataset:
    input_cross_embodiment_description: CrossEmbodimentDescription = {
        "robot_1": {DataType.RGB_IMAGES: {}}
    }
    output_cross_embodiment_description: CrossEmbodimentDescription = {
        "robot_1": {
            data_type: {} for data_type in CNNMLP.get_supported_output_data_types()
        }
        # A robot logs exactly one done flag.
        | {DataType.DONE: {0: "done"}}
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
def test_model_construction_forward_backward(
    input_data_types: OrderedSet[DataType],
    output_data_types: OrderedSet[DataType],
    pytorch_dummy_dataset: PytorchDummyDataset,
    model_config: dict,
    sample_inference_batch: BatchedInferenceInputs,
    sample_training_batch: BatchedTrainingSamples,
):
    model_init_description = _build_model_init_description(
        pytorch_dummy_dataset,
        input_data_types,
        output_data_types,
    )
    model = CNNMLP(model_init_description=model_init_description, **model_config)
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
    expected_output_width = dict(model.output_layout)
    assert set(output.keys()) == set(model.ordered_output_data_types)
    for data_type, tensors in output.items():
        assert len(tensors) == expected_output_width[data_type]

    sample_training_batch = sample_training_batch.to(DEVICE)
    output: BatchedTrainingOutputs = model.training_step(sample_training_batch)
    loss = output.losses["l1_loss"]
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert torch.isfinite(param.grad).all()


def test_model_backward_without_proprioception(
    model_init_description_no_proprio: ModelInitDescription,
    model_config: dict,
    sample_training_batch_no_proprio: BatchedTrainingSamples,
):
    model = CNNMLP(
        model_init_description=model_init_description_no_proprio, **model_config
    )
    model = model.to(DEVICE)
    sample_training_batch_no_proprio = sample_training_batch_no_proprio.to(DEVICE)
    output: BatchedTrainingOutputs = model.training_step(
        sample_training_batch_no_proprio
    )

    # The outputs include DONE, so the done head needs its loss for gradients.
    loss = sum(output.losses.values())
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

    model = CNNMLP(model_init_description=model_init)

    joint_encoder = cast(nn.Linear, model.encoders[DataType.JOINT_POSITIONS])
    assert joint_encoder.in_features == len(input_stats[DataType.JOINT_POSITIONS])
    assert model.max_output_size == 1
    assert model.output_layout == [(DataType.JOINT_TARGET_POSITIONS, 1)]


def test_training_step_uses_deterministic_output_order(
    model_init_description: ModelInitDescription,
    sample_training_batch: BatchedTrainingSamples,
    monkeypatch: pytest.MonkeyPatch,
):
    model = CNNMLP(model_init_description=model_init_description).to(DEVICE)
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
    for data_type in model.action_output_data_types:
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

    def fake_predict_action(
        _: BatchedInferenceInputs,
    ) -> tuple[torch.Tensor, None]:
        action_preds = torch.zeros(
            (
                reordered_batch.batch_size,
                model.output_prediction_horizon,
                model.max_output_size,
            ),
            device=model.device,
            dtype=torch.float32,
        )
        return action_preds, None

    monkeypatch.setattr(model.action_normalizer, "normalize", capture_normalize)
    monkeypatch.setattr(model, "_predict_action", fake_predict_action)

    model.training_step(reordered_batch)
    assert torch.allclose(captured["action_data"], expected_action_data)


# --- Done head ---

SMALL_MODEL_CONFIG = {
    "hidden_dim": 64,
    "cnn_output_dim": 32,
    "num_layers": 2,
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
    model = CNNMLP(
        model_init_description=_build_model_init_description(
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
    model = CNNMLP(
        model_init_description=_build_model_init_description(
            pytorch_dummy_dataset, DONE_INPUTS, DONE_OUTPUTS
        ),
        detach_done_encoder=detach_done_encoder,
        **SMALL_MODEL_CONFIG,
    ).to(DEVICE)
    batch = _labelled_done_batch(sample_training_batch.to(DEVICE))

    model.training_step(batch).losses["done_bce_loss"].backward()

    encoder_grads = [
        param.grad
        for param in model.encoders[DataType.RGB_IMAGES].parameters()
        if param.requires_grad
    ]
    reached_encoder = any(
        grad is not None and grad.abs().sum() > 0 for grad in encoder_grads
    )
    assert reached_encoder != detach_done_encoder
    assert model.done_head is not None
    assert all(param.grad is not None for param in model.done_head.parameters())


@pytest.mark.parametrize("done_max_pos_weight", [10.0, 50.0, 200.0])
def test_done_pos_weight_is_capped(
    done_max_pos_weight: float,
    pytorch_dummy_dataset: PytorchDummyDataset,
):
    # The dummy dataset has no done frames, so (1 - p) / p hits the cap.
    model = CNNMLP(
        model_init_description=_build_model_init_description(
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
        CNNMLP(
            model_init_description=_build_model_init_description(
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
        CNNMLP(
            model_init_description=_build_model_init_description(
                pytorch_dummy_dataset, DONE_INPUTS, OrderedSet([DataType.DONE])
            ),
            **SMALL_MODEL_CONFIG,
        )


def test_without_done_output_model_is_plain_cnnmlp(
    pytorch_dummy_dataset: PytorchDummyDataset,
    sample_training_batch: BatchedTrainingSamples,
):
    # No done modules or buffers, so checkpoints from before the done head load.
    model = CNNMLP(
        model_init_description=_build_model_init_description(
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
    assert set(training_output.losses) == {"l1_loss"}
    assert not [key for key in training_output.metrics if "done" in key]


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
