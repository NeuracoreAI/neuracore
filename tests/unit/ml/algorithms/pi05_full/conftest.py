"""Shared fixtures for pi05_full tests."""

from typing import cast

import pytest
from neuracore_types import DataType, ModelInitDescription
from torch.utils.data import DataLoader

from neuracore.core.utils.robot_data_spec_utils import extract_data_types
from neuracore.ml import BatchedInferenceInputs, BatchedTrainingSamples
from neuracore.ml.algorithms.pi05_full.utils import PI05FullConfig
from neuracore.ml.datasets.pytorch_dummy_dataset import PytorchDummyDataset

PI05_FULL_TEST_ARGS = {
    "paligemma_variant": "gemma_tiny",
    "action_expert_variant": "gemma_tiny",
    "use_pretrained_weights": False,
    "num_inference_steps": 1,
    "vlm_max_text_tokens": 4,
    "compile_model": False,
    "gradient_checkpointing": False,
    "discrete_state_input": True,
    "max_subtask_tokens": 8,
    "max_fast_tokens": 8,
    "max_decoding_steps": 4,
}

OUTPUT_PREDICTION_HORIZON = 1


@pytest.fixture
def tiny_pi05_full_config() -> PI05FullConfig:
    """Smallest possible config for fast tests.

    Uses gemma_tiny for both branches and reduces action dimensions and
    chunk sizes so a forward+backward fits comfortably in CPU memory.
    """
    return PI05FullConfig(
        paligemma_variant="gemma_tiny",
        action_expert_variant="gemma_tiny",
        chunk_size=8,
        max_state_dim=8,
        max_action_dim=8,
        num_inference_steps=2,
        max_subtask_tokens=8,
        max_fast_tokens=8,
        max_decoding_steps=4,
        device="cpu",
        dtype="float32",
        gradient_checkpointing=False,
    )


@pytest.fixture
def pi05_full_cls():
    """Lazy import to avoid module-load failures from breaking other tests."""
    from neuracore.ml.algorithms.pi05_full.pi05 import Pi05Full

    return Pi05Full


def _make_dummy_dataset(
    pi05_full_cls,
    *,
    include_subtask_in_inputs: bool = True,
    include_subtask_in_outputs: bool = True,
) -> PytorchDummyDataset:
    input_data_types = set(pi05_full_cls.get_supported_input_data_types())
    output_data_types = set(pi05_full_cls.get_supported_output_data_types())
    if not include_subtask_in_inputs:
        input_data_types.discard(DataType.SUBTASK_LANGUAGE)
    if not include_subtask_in_outputs:
        output_data_types.discard(DataType.SUBTASK_LANGUAGE)
    input_desc = {"robot_1": {dt: [] for dt in input_data_types}}
    output_desc = {"robot_1": {dt: [] for dt in output_data_types}}
    return PytorchDummyDataset(
        num_samples=2,
        input_cross_embodiment_description=input_desc,
        output_cross_embodiment_description=output_desc,
        output_prediction_horizon=OUTPUT_PREDICTION_HORIZON,
    )


def _make_model_init_description(dataset: PytorchDummyDataset) -> ModelInitDescription:
    return ModelInitDescription(
        input_data_types=extract_data_types(dataset.input_cross_embodiment_description),
        output_data_types=extract_data_types(
            dataset.output_cross_embodiment_description
        ),
        input_dataset_statistics=dataset.dataset_statistics["input"],
        output_dataset_statistics=dataset.dataset_statistics["output"],
        output_prediction_horizon=dataset.output_prediction_horizon,
    )


@pytest.fixture
def pi05_full_dataset(pi05_full_cls) -> PytorchDummyDataset:
    return _make_dummy_dataset(pi05_full_cls)


@pytest.fixture
def model_init_description_with_subtask(pi05_full_dataset) -> ModelInitDescription:
    return _make_model_init_description(pi05_full_dataset)


@pytest.fixture
def model_init_description_no_subtask_input(pi05_full_cls) -> ModelInitDescription:
    dataset = _make_dummy_dataset(pi05_full_cls, include_subtask_in_inputs=False)
    return _make_model_init_description(dataset)


@pytest.fixture
def model_init_description_no_subtask_output(pi05_full_cls) -> ModelInitDescription:
    dataset = _make_dummy_dataset(pi05_full_cls, include_subtask_in_outputs=False)
    return _make_model_init_description(dataset)


@pytest.fixture
def synthetic_training_batch(pi05_full_dataset) -> BatchedTrainingSamples:
    loader = DataLoader(
        pi05_full_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=pi05_full_dataset.collate_fn,
    )
    return cast(BatchedTrainingSamples, next(iter(loader)))


@pytest.fixture
def synthetic_inference_batch(synthetic_training_batch) -> BatchedInferenceInputs:
    return BatchedInferenceInputs(
        inputs=synthetic_training_batch.inputs,
        inputs_mask=synthetic_training_batch.inputs_mask,
        batch_size=synthetic_training_batch.batch_size,
    )
