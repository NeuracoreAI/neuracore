import inspect
import os
import random
from pathlib import Path
from typing import Any, cast

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
from neuracore.ml.core.ml_types import BatchedTrainingOutputs
from neuracore.ml.datasets.pytorch_dummy_dataset import PytorchDummyDataset
from neuracore.ml.utils.algorithm_loader import AlgorithmLoader
from neuracore.ml.utils.validate import run_validation

GROOT_ALGORITHM_DIR = (
    Path(__file__).resolve().parents[4] / "neuracore/ml/algorithms/groot"
)
# AlgorithmLoader.install_requirements tries uv pip first and falls back
# to pip, and skips the install entirely if requirements are already
# satisfied (avoiding unnecessary reinstalls that can break PyO3 deps).
if not GROOT_ALGORITHM_DIR.exists():
    raise FileNotFoundError(
        f"Could not find groot algorithm dir at {GROOT_ALGORITHM_DIR}"
    )
AlgorithmLoader(GROOT_ALGORITHM_DIR).install_requirements()

from neuracore.ml.algorithms.groot.groot import Groot as GrootModel  # noqa: E402

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
    "gradient_checkpointing": False,
}

INPUT_PARAMS = [
    pytest.param(
        OrderedSet([data_type]),
        id="".join(w.capitalize() for w in data_type.value.split("_")),
    )
    for data_type in sorted(
        GrootModel.get_supported_input_data_types(), key=lambda d: d.value
    )
]
OUTPUT_PARAMS = [
    pytest.param(
        OrderedSet([data_type]),
        id="".join(w.capitalize() for w in data_type.value.split("_")),
    )
    # DONE needs an action output next to it; it is covered by the done tests.
    for data_type in sorted(
        GrootModel.get_supported_output_data_types() - {DataType.DONE},
        key=lambda d: d.value,
    )
]


@pytest.fixture(scope="module")
def Groot():  # noqa: N802
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
    output_cross_embodiment_description["robot_1"][DataType.DONE] = {0: "done"}

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
def test_model_construction_forward_backward(
    input_data_types: OrderedSet[DataType],
    output_data_types: OrderedSet[DataType],
    pytorch_dummy_dataset: PytorchDummyDataset,
    sample_inference_batch: BatchedInferenceInputs,
    sample_training_batch: BatchedTrainingSamples,
    Groot,  # noqa: N803
):
    model_init_description = ModelInitDescription(
        input_data_types=input_data_types,
        output_data_types=output_data_types,
        input_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["input"],
        output_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["output"],
        output_prediction_horizon=pytorch_dummy_dataset.output_prediction_horizon,
    )
    model = Groot(model_init_description=model_init_description, **GROOT_TEST_ARGS)
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
            assert data_type in output_data_types

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


# --- Done head ---

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


def _done_model(dataset: PytorchDummyDataset, **kwargs: Any) -> GrootModel:
    return GrootModel(
        model_init_description=_model_init_description(
            dataset, DONE_INPUTS, DONE_OUTPUTS
        ),
        **GROOT_TEST_ARGS,
        **kwargs,
    ).to(DEVICE)


def _labelled_done_batch(batch: BatchedTrainingSamples) -> BatchedTrainingSamples:
    """Mark every input and output as present and make one current step done.

    The dummy dataset leaves masks at zero, which would hide the done loss and
    zero the joint and gripper state the done head can read.
    """
    for data_type in batch.inputs_mask:
        batch.inputs_mask[data_type] = torch.ones_like(batch.inputs_mask[data_type])
    for data_type in batch.outputs_mask:
        batch.outputs_mask[data_type] = torch.ones_like(batch.outputs_mask[data_type])
    for item in cast(list[BatchedDoneData], batch.outputs[DataType.DONE]):
        item.done.zero_()
        item.done[0, 0, 0] = 1.0
    return batch


def _is_vlm_param(name: str) -> bool:
    return any(
        keyword in name.lower() for keyword in ["vlm", "vision", "language_model"]
    )


@pytest.mark.parametrize("done_use_proprio", [False, True])
@pytest.mark.parametrize("detach_done_encoder", [False, True])
def test_done_head_forward_backward(
    detach_done_encoder: bool,
    done_use_proprio: bool,
    pytorch_dummy_dataset: PytorchDummyDataset,
    sample_training_batch: BatchedTrainingSamples,
):
    model = _done_model(
        pytorch_dummy_dataset,
        detach_done_encoder=detach_done_encoder,
        done_use_proprio=done_use_proprio,
    )
    assert model.done_head is not None
    assert (model.done_proprio_to_logits is not None) == done_use_proprio

    batch = _labelled_done_batch(sample_training_batch.to(DEVICE))

    model.eval()
    with torch.no_grad():
        output = model(_inference_batch(batch))
    assert DataType.JOINT_TARGET_POSITIONS in output
    done_outputs = output[DataType.DONE]
    assert len(done_outputs) == 1
    for done_output in done_outputs:
        assert isinstance(done_output, BatchedDoneData)
        assert done_output.done.shape == (BS, 1, 1)
        assert ((done_output.done >= 0) & (done_output.done <= 1)).all()

    model.train()
    training_output = model.training_step(batch)
    assert set(training_output.losses) == {"flow_matching_loss", "done_bce_loss"}
    assert training_output.losses["done_bce_loss"] > 0
    for metric in ("done_accuracy", "done_precision", "done_recall"):
        assert 0.0 <= float(training_output.metrics[metric]) <= 1.0

    sum(training_output.losses.values()).backward()
    for name, param in model.named_parameters():
        if param.requires_grad and not _is_vlm_param(name):
            assert param.grad is not None, f"Gradient for {name} is None"
            assert torch.isfinite(param.grad).all()


@pytest.mark.parametrize("detach_done_encoder", [False, True])
def test_detach_done_encoder_blocks_done_gradient_to_backbone(
    detach_done_encoder: bool,
    pytorch_dummy_dataset: PytorchDummyDataset,
    sample_training_batch: BatchedTrainingSamples,
):
    # Without configure_optimizers every backbone parameter is trainable.
    model = _done_model(
        pytorch_dummy_dataset,
        detach_done_encoder=detach_done_encoder,
        done_use_proprio=True,
    )
    batch = _labelled_done_batch(sample_training_batch.to(DEVICE))

    model.training_step(batch).losses["done_bce_loss"].backward()

    encoder_params = list(model.vlm_backbone.parameters()) + list(
        model.mlp_connector.parameters()
    )
    reached_encoder = any(
        param.grad is not None and param.grad.abs().sum() > 0
        for param in encoder_params
    )
    assert reached_encoder != detach_done_encoder
    # The done loss never reaches the action pathway.
    for module in (model.dit, model.action_encoder_mlp, model.proprio_mlp):
        assert all(param.grad is None for param in module.parameters())
    assert all(param.grad is not None for param in model.done_head.parameters())


def test_done_head_is_trained_at_base_lr_with_frozen_backbone(
    pytorch_dummy_dataset: PytorchDummyDataset,
    sample_training_batch: BatchedTrainingSamples,
):
    model = _done_model(pytorch_dummy_dataset, done_use_proprio=True)
    (optimizer,) = model.configure_optimizers()
    done_params = {
        id(param) for name, param in model.named_parameters() if "done" in name
    }
    assert done_params
    groups = [
        group
        for group in optimizer.param_groups
        if any(id(param) in done_params for param in group["params"])
    ]
    assert len(groups) == 1 and groups[0]["lr"] == model.lr
    assert {id(param) for param in groups[0]["params"]} == done_params
    for name, param in model.named_parameters():
        if "done" in name:
            assert param.requires_grad, name

    # The tiny VLM stays frozen, so no done-loss gradient can reach it.
    batch = _labelled_done_batch(sample_training_batch.to(DEVICE))
    model.training_step(batch).losses["done_bce_loss"].backward()
    assert all(param.grad is None for param in model.vlm_backbone.parameters())


def test_done_does_not_depend_on_action_noise_or_targets(
    pytorch_dummy_dataset: PytorchDummyDataset,
    sample_training_batch: BatchedTrainingSamples,
):
    model = _done_model(pytorch_dummy_dataset, done_use_proprio=True)
    model.eval()  # disable dropout so outputs are deterministic
    batch = _labelled_done_batch(sample_training_batch.to(DEVICE))

    with torch.no_grad():
        torch.manual_seed(0)
        first = model(_inference_batch(batch))
        torch.manual_seed(1)
        second = model(_inference_batch(batch))
    # Different seeds draw different denoising noise; done must not change.
    for a, b in zip(first[DataType.DONE], second[DataType.DONE]):
        assert torch.equal(a.done, b.done)

    with torch.no_grad():
        torch.manual_seed(0)
        loss_before = model.training_step(batch).metrics["done_bce_loss"]
        for item in batch.outputs[DataType.JOINT_TARGET_POSITIONS]:
            item.value.add_(5.0)
        torch.manual_seed(1)
        loss_after = model.training_step(batch).metrics["done_bce_loss"]
    assert torch.equal(loss_before, loss_after)


@pytest.mark.parametrize("done_max_pos_weight", [10.0, 50.0, 200.0])
def test_done_pos_weight_is_capped(
    done_max_pos_weight: float,
    pytorch_dummy_dataset: PytorchDummyDataset,
):
    # The dummy dataset has no done frames, so (1 - p) / p hits the cap.
    model = _done_model(pytorch_dummy_dataset, done_max_pos_weight=done_max_pos_weight)
    assert torch.allclose(
        model.done_pos_weight,
        torch.full_like(model.done_pos_weight, done_max_pos_weight),
    )


def test_done_requires_rgb_images(pytorch_dummy_dataset: PytorchDummyDataset):
    with pytest.raises(ValueError, match="RGB_IMAGES"):
        GrootModel(
            model_init_description=_model_init_description(
                pytorch_dummy_dataset,
                OrderedSet([DataType.JOINT_POSITIONS]),
                DONE_OUTPUTS,
            ),
            **GROOT_TEST_ARGS,
        )


def test_done_alone_is_not_a_valid_output(
    pytorch_dummy_dataset: PytorchDummyDataset,
):
    with pytest.raises(ValueError, match="continuous action"):
        GrootModel(
            model_init_description=_model_init_description(
                pytorch_dummy_dataset, DONE_INPUTS, OrderedSet([DataType.DONE])
            ),
            **GROOT_TEST_ARGS,
        )


def test_without_done_output_model_is_plain_groot(
    pytorch_dummy_dataset: PytorchDummyDataset,
    sample_training_batch: BatchedTrainingSamples,
):
    # No done modules or buffers, so checkpoints from before the done head load.
    model = GrootModel(
        model_init_description=_model_init_description(
            pytorch_dummy_dataset,
            DONE_INPUTS,
            OrderedSet([DataType.JOINT_TARGET_POSITIONS]),
        ),
        **GROOT_TEST_ARGS,
    ).to(DEVICE)
    assert model.done_head is None
    assert not [key for key in model.state_dict() if "done" in key]

    batch = sample_training_batch.to(DEVICE)
    assert DataType.DONE not in model(_inference_batch(batch))
    training_output = model.training_step(batch)
    assert set(training_output.losses) == {"flow_matching_loss"}
    assert set(training_output.metrics) == {"flow_matching_loss"}


@pytest.mark.slow
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
