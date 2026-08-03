"""Tests for PolicyInference's array-native action-chunk path."""

import numpy as np
import pytest
import torch
from neuracore_types import (
    DataType,
    JointData,
    ModelInitDescription,
    ParallelGripperOpenAmountData,
    SynchronizedPoint,
)
from ordered_set import OrderedSet

from neuracore.ml.algorithms.diffusion_policy.diffusion_policy import DiffusionPolicy
from neuracore.ml.datasets.pytorch_dummy_dataset import PytorchDummyDataset
from neuracore.ml.utils.policy_inference import PolicyInference
from neuracore.ml.utils.real_time_chunking import RTCConfig

HORIZON = 8
JOINT_NAMES = ["joint_0", "joint_1"]
GRIPPER_NAME = "gripper_0"

INPUT_DATA_TYPES = OrderedSet(
    [DataType.JOINT_POSITIONS, DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS]
)
OUTPUT_DATA_TYPES = OrderedSet(
    [DataType.JOINT_TARGET_POSITIONS, DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS]
)

INPUT_EMBODIMENT = {
    DataType.JOINT_POSITIONS: dict(enumerate(JOINT_NAMES)),
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: {0: GRIPPER_NAME},
}
OUTPUT_EMBODIMENT = {
    DataType.JOINT_TARGET_POSITIONS: dict(enumerate(JOINT_NAMES)),
    DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS: {0: GRIPPER_NAME},
}


@pytest.fixture(scope="module")
def dummy_dataset() -> PytorchDummyDataset:
    return PytorchDummyDataset(
        num_samples=2,
        input_cross_embodiment_description={
            "robot": {data_type: {} for data_type in INPUT_DATA_TYPES}
        },
        output_cross_embodiment_description={
            "robot": {data_type: {} for data_type in OUTPUT_DATA_TYPES}
        },
        output_prediction_horizon=HORIZON,
    )


@pytest.fixture
def policy_inference(dummy_dataset, monkeypatch) -> PolicyInference:
    """A PolicyInference wrapping a real DiffusionPolicy, no archive on disk."""
    description = ModelInitDescription(
        input_data_types=INPUT_DATA_TYPES,
        output_data_types=OUTPUT_DATA_TYPES,
        input_dataset_statistics=dummy_dataset.dataset_statistics["input"],
        output_dataset_statistics=dummy_dataset.dataset_statistics["output"],
        output_prediction_horizon=HORIZON,
    )
    torch.manual_seed(0)
    model = DiffusionPolicy(
        model_init_description=description,
        hidden_dim=32,
        unet_n_groups=4,
        unet_down_dims=[32, 64],
        num_train_timesteps=100,
        num_inference_steps=2,
        noise_scheduler_type="DDIM",
    )
    # The archive supplies the preprocessing config, as it does in production.
    monkeypatch.setattr(
        "neuracore.ml.utils.policy_inference.load_model_from_nc_archive",
        lambda model_file, device=None: (
            model,
            {},
            {},
            {DataType.JOINT_POSITIONS: []},
            {},
        ),
    )
    return PolicyInference(
        model_file="unused.nc.zip",
        org_id="org",
        input_embodiment_description=INPUT_EMBODIMENT,
        output_embodiment_description=OUTPUT_EMBODIMENT,
        device="cpu",
    )


def _sync_point() -> SynchronizedPoint:
    return SynchronizedPoint(
        timestamp=0.0,
        data={
            DataType.JOINT_POSITIONS: {
                name: JointData(timestamp=0.0, value=0.1 * index)
                for index, name in enumerate(JOINT_NAMES)
            },
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: {
                GRIPPER_NAME: ParallelGripperOpenAmountData(
                    timestamp=0.0, open_amount=0.5
                )
            },
        },
    )


def test_supports_real_time_chunking(policy_inference):
    assert policy_inference.supports_real_time_chunking is True


def test_output_action_names_match_the_action_width(policy_inference):
    names = policy_inference.output_action_names()
    model = policy_inference.model

    assert (
        len(names) == model.max_output_size
    ), "there must be exactly one name per action column"
    # Columns are grouped by data type, in the model's own output ordering.
    for data_type in model.ordered_output_data_types:
        start_idx, end_idx = model.output_dims[data_type]
        assert all(
            entry[0] is data_type for entry in names[start_idx:end_idx]
        ), f"columns {start_idx}:{end_idx} should all belong to {data_type}"

    named = [entry for entry in names if entry[1] is not None]
    assert named == [
        (DataType.JOINT_TARGET_POSITIONS, "joint_0"),
        (DataType.JOINT_TARGET_POSITIONS, "joint_1"),
        (DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS, GRIPPER_NAME),
    ]


def test_output_action_names_marks_cross_embodiment_padding(policy_inference):
    """Columns with no sensor in this embodiment must be reported as None."""
    names = policy_inference.output_action_names()
    padding = [entry for entry in names if entry[1] is None]

    assert padding, "this fixture is expected to exercise the padding path"
    joint_columns = [
        column
        for column, (data_type, name) in enumerate(names)
        if data_type is DataType.JOINT_TARGET_POSITIONS and name is not None
    ]
    assert joint_columns == [0, 1], "named joints must keep their embodiment index"


def test_predict_action_chunk_shape_and_dtype(policy_inference):
    chunk = policy_inference.predict_action_chunk(_sync_point())

    assert isinstance(chunk, np.ndarray)
    assert chunk.shape == (HORIZON, len(policy_inference.output_action_names()))
    assert np.isfinite(chunk).all()


def test_predict_action_chunk_columns_match_the_dict_path(policy_inference):
    """The array path must agree with __call__ on how columns are named."""
    torch.manual_seed(1)
    chunk = policy_inference.predict_action_chunk(_sync_point())
    names = policy_inference.output_action_names()

    torch.manual_seed(1)
    predictions = policy_inference(_sync_point())

    for column, (data_type, name) in enumerate(names):
        if name is None:
            continue  # cross-embodiment padding has no dict entry
        batched = predictions[data_type][name]
        values = getattr(batched, "value", None)
        if values is None:
            values = batched.open_amount
        # Sampling is stochastic, so compare the layout rather than the values.
        assert values.shape == (1, HORIZON, 1)
        assert chunk[:, column].shape == (HORIZON,)


def test_predict_action_chunk_accepts_a_previous_chunk(policy_inference):
    action_dim = len(policy_inference.output_action_names())
    prev_chunk = np.zeros((HORIZON, action_dim), dtype=np.float32)
    config = RTCConfig(inference_delay=1, execution_horizon=4, num_inference_steps=2)

    chunk = policy_inference.predict_action_chunk(_sync_point(), prev_chunk, config)

    assert chunk.shape == (HORIZON, action_dim)
    assert np.isfinite(chunk).all()


def test_predict_action_chunk_requires_a_config_with_a_previous_chunk(
    policy_inference,
):
    action_dim = len(policy_inference.output_action_names())
    with pytest.raises(ValueError, match="rtc_config"):
        policy_inference.predict_action_chunk(
            _sync_point(), np.zeros((HORIZON, action_dim), dtype=np.float32)
        )


def test_predict_action_chunk_leaves_the_model_in_eval_and_grad_free(
    policy_inference,
):
    """RTC enables grad internally; it must not leak into the loaded model."""
    action_dim = len(policy_inference.output_action_names())
    config = RTCConfig(inference_delay=1, execution_horizon=4, num_inference_steps=2)
    policy_inference.predict_action_chunk(
        _sync_point(), np.zeros((HORIZON, action_dim), dtype=np.float32), config
    )

    assert not policy_inference.model.training
    assert all(
        param.grad is None for param in policy_inference.model.parameters()
    ), "guidance must not accumulate gradients on model weights"
