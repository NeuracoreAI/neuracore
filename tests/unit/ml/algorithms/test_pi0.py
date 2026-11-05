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

PI0_ALGORITHM_DIR = Path(__file__).resolve().parents[4] / "neuracore/ml/algorithms/pi0"
AlgorithmLoader(PI0_ALGORITHM_DIR).install_requirements()

from neuracore.ml.algorithms.pi0.pi0 import Pi0 as Pi0Model  # noqa: E402

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
    "gradient_checkpointing": True,
    "dtype": "bfloat16",
}

INPUT_PARAMS = [
    pytest.param(
        OrderedSet([data_type]),
        id="".join(w.capitalize() for w in data_type.value.split("_")),
    )
    for data_type in sorted(
        Pi0Model.get_supported_input_data_types(), key=lambda d: d.value
    )
]
OUTPUT_PARAMS = [
    pytest.param(
        OrderedSet([data_type]),
        id="".join(w.capitalize() for w in data_type.value.split("_")),
    )
    # DONE needs an action output next to it; it is covered by the done tests.
    for data_type in sorted(
        Pi0Model.get_supported_output_data_types() - {DataType.DONE},
        key=lambda d: d.value,
    )
]


@pytest.fixture(scope="module")
def Pi0():  # noqa: N802
    from neuracore.ml.algorithms.pi0.pi0 import Pi0 as Pi0Model

    return Pi0Model


@pytest.fixture
def pytorch_dummy_dataset(Pi0) -> PytorchDummyDataset:  # noqa: N803
    input_data_types = Pi0.get_supported_input_data_types()
    output_data_types = Pi0.get_supported_output_data_types()
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
def test_model_construction_forward_backward(
    input_data_types: OrderedSet[DataType],
    output_data_types: OrderedSet[DataType],
    pytorch_dummy_dataset: PytorchDummyDataset,
    sample_inference_batch: BatchedInferenceInputs,
    sample_training_batch: BatchedTrainingSamples,
    Pi0,  # noqa: N803
):
    model_init_description = ModelInitDescription(
        input_data_types=input_data_types,
        output_data_types=output_data_types,
        input_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["input"],
        output_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["output"],
        output_prediction_horizon=pytorch_dummy_dataset.output_prediction_horizon,
    )
    model = Pi0(model_init_description=model_init_description, **PI0_TEST_ARGS)
    model = model.to(DEVICE)
    assert isinstance(model, nn.Module)

    sample_inference_batch = sample_inference_batch.to(DEVICE)
    inference_output: dict[DataType, list[BatchedNCData]] = model(
        sample_inference_batch
    )
    assert isinstance(inference_output, dict)
    for data_type, tensors in inference_output.items():
        assert isinstance(data_type, DataType)
        assert isinstance(tensors, list)
        for tensor in tensors:
            assert isinstance(tensor, BatchedNCData)
            assert data_type in output_data_types

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


def _vision_tower(model: Any) -> nn.Module:
    return model.model.paligemma_with_expert.paligemma.vision_tower


@pytest.mark.parametrize("done_use_proprio", [False, True])
@pytest.mark.parametrize("detach_done_encoder", [False, True])
def test_done_head_forward_backward(
    detach_done_encoder: bool,
    done_use_proprio: bool,
    pytorch_dummy_dataset: PytorchDummyDataset,
    sample_training_batch: BatchedTrainingSamples,
    Pi0,  # noqa: N803
):
    model = Pi0(
        model_init_description=_model_init_description(
            pytorch_dummy_dataset, DONE_INPUTS, DONE_OUTPUTS
        ),
        detach_done_encoder=detach_done_encoder,
        done_use_proprio=done_use_proprio,
        **PI0_TEST_ARGS,
    ).to(DEVICE)
    assert model.done_head is not None
    assert (model.done_proprio_to_logits is not None) == done_use_proprio

    batch = _labelled_done_batch(sample_training_batch.to(DEVICE))

    with torch.no_grad():
        output = model(_inference_batch(batch))
    assert DataType.JOINT_TARGET_POSITIONS in output
    done_outputs = output[DataType.DONE]
    assert len(done_outputs) == len(model.done_slot_dims)
    for done_output in done_outputs:
        assert isinstance(done_output, BatchedDoneData)
        assert done_output.done.shape == (BS, 1, 1)
        assert done_output.done.dtype == torch.float32
        assert ((done_output.done >= 0) & (done_output.done <= 1)).all()

    model.train()
    training_output = model.training_step(batch)
    assert training_output.losses["done_bce_loss"] > 0
    for metric in ("done_accuracy", "done_precision", "done_recall"):
        assert 0.0 <= float(training_output.metrics[metric]) <= 1.0

    sum(training_output.losses.values()).backward()
    for name, param in model.named_parameters():
        if "done" in name:
            assert param.grad is not None, f"Gradient for {name} is None"
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"{name} has non-finite grads"


@pytest.mark.parametrize("detach_done_encoder", [False, True])
def test_detach_done_encoder_blocks_done_gradient_to_vision_tower(
    detach_done_encoder: bool,
    pytorch_dummy_dataset: PytorchDummyDataset,
    sample_training_batch: BatchedTrainingSamples,
    Pi0,  # noqa: N803
):
    model = Pi0(
        model_init_description=_model_init_description(
            pytorch_dummy_dataset, DONE_INPUTS, DONE_OUTPUTS
        ),
        detach_done_encoder=detach_done_encoder,
        **PI0_TEST_ARGS,
    ).to(DEVICE)
    vision_params = [p for p in _vision_tower(model).parameters() if p.requires_grad]
    assert vision_params, "the vision tower should be trainable in this test"
    batch = _labelled_done_batch(sample_training_batch.to(DEVICE))

    model.train()
    model.training_step(batch).losses["done_bce_loss"].backward()

    reached_vision_tower = any(
        param.grad is not None and param.grad.abs().sum() > 0 for param in vision_params
    )
    assert reached_vision_tower != detach_done_encoder
    assert all(param.grad is not None for param in model.done_head.parameters())


def test_done_head_ignores_target_actions(
    pytorch_dummy_dataset: PytorchDummyDataset,
    sample_training_batch: BatchedTrainingSamples,
    Pi0,  # noqa: N803
):
    # The done head reads image embeddings only, so changing the action
    # targets (and the random flow noise and time) leaves the done loss alone.
    model = Pi0(
        model_init_description=_model_init_description(
            pytorch_dummy_dataset, DONE_INPUTS, DONE_OUTPUTS
        ),
        **PI0_TEST_ARGS,
    ).to(DEVICE)
    model.eval()  # dropout off
    batch = _labelled_done_batch(sample_training_batch.to(DEVICE))

    with torch.no_grad():
        first = model.training_step(batch)
        for item in batch.outputs[DataType.JOINT_TARGET_POSITIONS]:
            item.value.add_(10.0)
        second = model.training_step(batch)
    assert first.metrics["done_bce_loss"] == second.metrics["done_bce_loss"]
    assert first.metrics["mse_loss"] != second.metrics["mse_loss"]


@pytest.mark.parametrize("finetune_action_expert_only", [False, True])
def test_done_head_parameters_are_optimized(
    finetune_action_expert_only: bool,
    pytorch_dummy_dataset: PytorchDummyDataset,
    Pi0,  # noqa: N803
):
    model = Pi0(
        model_init_description=_model_init_description(
            pytorch_dummy_dataset, DONE_INPUTS, DONE_OUTPUTS
        ),
        finetune_action_expert_only=finetune_action_expert_only,
        **PI0_TEST_ARGS,
    )
    [optimizer] = model.configure_optimizers()
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    done_params = [p for name, p in model.named_parameters() if "done" in name]
    assert done_params
    assert all(id(p) in optimized for p in done_params)
    assert all(group["lr"] == model.optimizer_lr for group in optimizer.param_groups)


@pytest.mark.parametrize("done_max_pos_weight", [10.0, 50.0, 200.0])
def test_done_pos_weight_is_capped(
    done_max_pos_weight: float,
    pytorch_dummy_dataset: PytorchDummyDataset,
    Pi0,  # noqa: N803
):
    # The dummy dataset has no done frames, so (1 - p) / p hits the cap.
    model = Pi0(
        model_init_description=_model_init_description(
            pytorch_dummy_dataset, DONE_INPUTS, DONE_OUTPUTS
        ),
        done_max_pos_weight=done_max_pos_weight,
        **PI0_TEST_ARGS,
    )
    assert torch.allclose(
        model.done_pos_weight,
        torch.full_like(model.done_pos_weight, done_max_pos_weight),
    )


def test_done_requires_rgb_images(
    pytorch_dummy_dataset: PytorchDummyDataset,
    Pi0,  # noqa: N803
):
    with pytest.raises(ValueError, match="RGB_IMAGES"):
        Pi0(
            model_init_description=_model_init_description(
                pytorch_dummy_dataset,
                OrderedSet([DataType.JOINT_POSITIONS]),
                DONE_OUTPUTS,
            ),
            **PI0_TEST_ARGS,
        )


def test_done_alone_is_not_a_valid_output(
    pytorch_dummy_dataset: PytorchDummyDataset,
    Pi0,  # noqa: N803
):
    with pytest.raises(ValueError, match="continuous action"):
        Pi0(
            model_init_description=_model_init_description(
                pytorch_dummy_dataset, DONE_INPUTS, OrderedSet([DataType.DONE])
            ),
            **PI0_TEST_ARGS,
        )


def test_without_done_output_model_is_plain_pi0(
    pytorch_dummy_dataset: PytorchDummyDataset,
    sample_training_batch: BatchedTrainingSamples,
    Pi0,  # noqa: N803
):
    # No done modules or buffers, so checkpoints from before the done head load.
    model = Pi0(
        model_init_description=_model_init_description(
            pytorch_dummy_dataset,
            DONE_INPUTS,
            OrderedSet([DataType.JOINT_TARGET_POSITIONS]),
        ),
        **PI0_TEST_ARGS,
    ).to(DEVICE)
    assert model.done_head is None
    assert not [key for key in model.state_dict() if "done" in key]

    batch = sample_training_batch.to(DEVICE)
    assert DataType.DONE not in model(_inference_batch(batch))
    model.train()
    training_output = model.training_step(batch)
    assert set(training_output.losses) == {"mse_loss"}


@pytest.mark.slow
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
