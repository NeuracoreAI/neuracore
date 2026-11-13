import inspect
import os
import random
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from huggingface_hub import get_token
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

PI05_ALGORITHM_DIR = (
    Path(__file__).resolve().parents[4] / "neuracore/ml/algorithms/pi05"
)
AlgorithmLoader(PI05_ALGORITHM_DIR).install_requirements()

from neuracore.ml.algorithms.pi05.pi05 import Pi05 as Pi05Model  # noqa: E402

# Pi05 needs the gated paligemma tokenizer; skip in CI when HF auth is absent.
requires_hf_auth = pytest.mark.skipif(
    os.environ.get("CI", "false").lower() == "true" and get_token() is None,
    reason="CI without Hugging Face auth cannot access gated paligemma tokenizer",
)


@pytest.fixture(scope="module")
def Pi05():  # noqa: N802
    from neuracore.ml.algorithms.pi05.pi05 import Pi05 as Pi05Model

    return Pi05Model


BS = 1
OUTPUT_PREDICTION_HORIZON = 1

# Use cpu because the model takes a lot of vram
DEVICE = torch.device("cpu")


PI05_TEST_ARGS: dict[str, Any] = {
    "paligemma_variant": "gemma_tiny",
    "action_expert_variant": "gemma_tiny",
    "use_pretrained_weights": False,
    "num_inference_steps": 1,
    "vlm_max_text_tokens": 4,
    "compile_model": False,
    "gradient_checkpointing": True,
    "discrete_state_input": True,
    "dtype": "bfloat16",
}

INPUT_PARAMS = [
    pytest.param(
        OrderedSet([data_type]),
        id="".join(w.capitalize() for w in data_type.value.split("_")),
    )
    for data_type in sorted(
        Pi05Model.get_supported_input_data_types(), key=lambda d: d.value
    )
]
OUTPUT_PARAMS = [
    pytest.param(
        OrderedSet([data_type]),
        id="".join(w.capitalize() for w in data_type.value.split("_")),
    )
    # DONE needs an action output next to it; it is covered by the done tests.
    for data_type in sorted(
        Pi05Model.get_supported_output_data_types() - {DataType.DONE},
        key=lambda d: d.value,
    )
]


@pytest.fixture
def pytorch_dummy_dataset(Pi05) -> PytorchDummyDataset:  # noqa: N803
    input_data_types = Pi05.get_supported_input_data_types()
    output_data_types = Pi05.get_supported_output_data_types()
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


@requires_hf_auth
@pytest.mark.parametrize("output_data_types", OUTPUT_PARAMS)
@pytest.mark.parametrize("input_data_types", INPUT_PARAMS)
def test_model_construction_forward_backward(
    input_data_types: OrderedSet[DataType],
    output_data_types: OrderedSet[DataType],
    pytorch_dummy_dataset: PytorchDummyDataset,
    sample_inference_batch: BatchedInferenceInputs,
    sample_training_batch: BatchedTrainingSamples,
    Pi05,  # noqa: N803
):
    model_init_description = ModelInitDescription(
        input_data_types=input_data_types,
        output_data_types=output_data_types,
        input_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["input"],
        output_dataset_statistics=pytorch_dummy_dataset.dataset_statistics["output"],
        output_prediction_horizon=pytorch_dummy_dataset.output_prediction_horizon,
    )
    model = Pi05(model_init_description=model_init_description, **PI05_TEST_ARGS)
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


def _vision_tower(model: nn.Module) -> nn.Module:
    return model.model.paligemma_with_expert.paligemma.vision_tower


@requires_hf_auth
@pytest.mark.parametrize(
    ("detach_done_encoder", "done_use_proprio"),
    [(False, False), (True, True)],
    ids=["attached-vision", "detached-proprio"],
)
def test_done_head_forward_backward(
    detach_done_encoder: bool,
    done_use_proprio: bool,
    pytorch_dummy_dataset: PytorchDummyDataset,
    sample_training_batch: BatchedTrainingSamples,
    Pi05,  # noqa: N803
):
    # Two of the four detach x proprio combinations keep this fast: each flag
    # is exercised on and off, and the detach gradient test covers both modes
    # of detach_done_encoder separately.
    model = Pi05(
        model_init_description=_model_init_description(
            pytorch_dummy_dataset, DONE_INPUTS, DONE_OUTPUTS
        ),
        detach_done_encoder=detach_done_encoder,
        done_use_proprio=done_use_proprio,
        **PI05_TEST_ARGS,
    ).to(DEVICE)
    assert model.done_head is not None
    assert (model.done_proprio_to_logits is not None) == done_use_proprio

    batch = _labelled_done_batch(sample_training_batch.to(DEVICE))

    with torch.no_grad():
        output = model(_inference_batch(batch))
    done_outputs = output[DataType.DONE]
    assert len(done_outputs) == 1
    for done_output in done_outputs:
        assert isinstance(done_output, BatchedDoneData)
        assert done_output.done.shape == (BS, 1, 1)
        assert ((done_output.done >= 0) & (done_output.done <= 1)).all()
    assert DataType.JOINT_TARGET_POSITIONS in output

    model.train()
    training_output = model.training_step(batch)
    assert training_output.losses["done_bce_loss"] > 0
    for metric in ("done_accuracy", "done_precision", "done_recall"):
        assert 0.0 <= float(training_output.metrics[metric]) <= 1.0

    sum(training_output.losses.values()).backward()
    for module in model._done_modules():
        for name, param in module.named_parameters():
            assert param.grad is not None, f"Gradient for done {name} is None"
            assert torch.isfinite(param.grad).all()
    optimized = {id(p) for group in model.param_groups for p in group["params"]}
    for module in model._done_modules():
        assert all(id(param) in optimized for param in module.parameters())


@requires_hf_auth
@pytest.mark.parametrize("detach_done_encoder", [False, True])
def test_detach_done_encoder_blocks_done_gradient_to_vision_tower(
    detach_done_encoder: bool,
    pytorch_dummy_dataset: PytorchDummyDataset,
    sample_training_batch: BatchedTrainingSamples,
    Pi05,  # noqa: N803
):
    model = Pi05(
        model_init_description=_model_init_description(
            pytorch_dummy_dataset, DONE_INPUTS, DONE_OUTPUTS
        ),
        detach_done_encoder=detach_done_encoder,
        **PI05_TEST_ARGS,
    ).to(DEVICE)
    model.train()
    batch = _labelled_done_batch(sample_training_batch.to(DEVICE))

    model.training_step(batch).losses["done_bce_loss"].backward()

    tower_params = [p for p in _vision_tower(model).parameters() if p.requires_grad]
    assert tower_params
    reached_tower = any(
        param.grad is not None and param.grad.abs().sum() > 0 for param in tower_params
    )
    assert reached_tower != detach_done_encoder
    assert all(param.grad is not None for param in model.done_head.parameters())


@requires_hf_auth
@pytest.mark.parametrize("done_max_pos_weight", [10.0, 200.0])
def test_done_pos_weight_is_capped(
    done_max_pos_weight: float,
    pytorch_dummy_dataset: PytorchDummyDataset,
    Pi05,  # noqa: N803
):
    # The dummy dataset has no done frames, so (1 - p) / p hits the cap.
    model = Pi05(
        model_init_description=_model_init_description(
            pytorch_dummy_dataset, DONE_INPUTS, DONE_OUTPUTS
        ),
        done_max_pos_weight=done_max_pos_weight,
        **PI05_TEST_ARGS,
    )
    assert torch.allclose(
        model.done_pos_weight,
        torch.full_like(model.done_pos_weight, done_max_pos_weight),
    )


@requires_hf_auth
def test_done_requires_rgb_images(
    pytorch_dummy_dataset: PytorchDummyDataset,
    Pi05,  # noqa: N803
):
    with pytest.raises(ValueError, match="RGB_IMAGES"):
        Pi05(
            model_init_description=_model_init_description(
                pytorch_dummy_dataset,
                OrderedSet([DataType.JOINT_POSITIONS]),
                DONE_OUTPUTS,
            ),
            **PI05_TEST_ARGS,
        )


@requires_hf_auth
def test_done_alone_is_not_a_valid_output(
    pytorch_dummy_dataset: PytorchDummyDataset,
    Pi05,  # noqa: N803
):
    with pytest.raises(ValueError, match="continuous action"):
        Pi05(
            model_init_description=_model_init_description(
                pytorch_dummy_dataset, DONE_INPUTS, OrderedSet([DataType.DONE])
            ),
            **PI05_TEST_ARGS,
        )


@requires_hf_auth
def test_without_done_output_model_is_plain_pi05(
    pytorch_dummy_dataset: PytorchDummyDataset,
    sample_training_batch: BatchedTrainingSamples,
    Pi05,  # noqa: N803
):
    # No done modules or buffers, so checkpoints from before the done head load.
    model = Pi05(
        model_init_description=_model_init_description(
            pytorch_dummy_dataset,
            DONE_INPUTS,
            OrderedSet([DataType.JOINT_TARGET_POSITIONS]),
        ),
        **PI05_TEST_ARGS,
    ).to(DEVICE)
    assert model.done_head is None
    assert not model._done_modules()
    assert not [key for key in model.state_dict() if "done" in key]

    batch = sample_training_batch.to(DEVICE)
    assert DataType.DONE not in model(_inference_batch(batch))
    model.train()
    training_output = model.training_step(batch)
    assert set(training_output.losses) == {"mse_loss"}
    assert set(training_output.metrics) == {"mse_loss"}


@requires_hf_auth
@pytest.mark.slow
def test_run_validation(tmp_path: Path, mock_login, monkeypatch, Pi05):  # noqa: N803
    from neuracore.ml.utils import validate as validate_module

    monkeypatch.setattr(
        validate_module.AlgorithmLoader, "load_model", lambda self: Pi05
    )

    # Long timeout due to larger model run on CPU
    os.environ["NEURACORE_ENDPOINT_TIMEOUT"] = "120"
    algorithm_dir = Path(inspect.getfile(Pi05)).parent
    _, error_msg = run_validation(
        output_dir=tmp_path,
        algorithm_dir=algorithm_dir,
        port=random.randint(10000, 20000),
        skip_endpoint_check=False,
        algorithm_config=PI05_TEST_ARGS,
        device=DEVICE,
    )
    if len(error_msg) > 0:
        raise RuntimeError(error_msg)
