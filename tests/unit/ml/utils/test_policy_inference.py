"""Tests for policy inference."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from neuracore_types import (
    BatchedJointData,
    BatchedNCData,
    CrossEmbodimentDescription,
    DataType,
    JointData,
    RGBCameraData,
    SynchronizedPoint,
)

from neuracore.core.video_encoding import (
    INFERENCE_CODEC_ENV_VAR,
    resolve_inference_codec_options,
)
from neuracore.ml.preprocessing.base import PreprocessingMethod
from neuracore.ml.preprocessing.methods import H264Match
from neuracore.ml.utils.policy_inference import PolicyInference


def _make_policy_inference(
    output_embodiment_description: CrossEmbodimentDescription | None = None,
) -> PolicyInference:
    policy_inference = PolicyInference.__new__(PolicyInference)
    policy_inference.output_embodiment_description = output_embodiment_description or {}
    policy_inference.input_embodiment_description = {}
    policy_inference.input_preprocessing_config = {}
    policy_inference.org_id = "test_org"
    policy_inference.job_id = None
    policy_inference.device = torch.device("cpu")
    policy_inference.model = None
    policy_inference.input_dataset_statistics = {}
    policy_inference.prediction_horizon = 1
    policy_inference._inference_codec_options = None
    policy_inference._codec_methods = {}
    policy_inference.model = SimpleNamespace(
        model_init_description=SimpleNamespace(
            input_data_types=[],
        )
    )
    return policy_inference


def _joint_prediction(value: float) -> BatchedJointData:
    return BatchedJointData(value=torch.full((1, 1, 1), value))


def _indexed_names(*names: str) -> dict[int, str]:
    """Build index-to-name mapping preserving declared order."""
    return dict(enumerate(names))


def test_assign_names_to_model_outputs_drops_extra_padded_tensors() -> None:
    policy_inference = _make_policy_inference({
        DataType.JOINT_TARGET_POSITIONS: _indexed_names("joint1", "joint2"),
    })

    first_prediction = _joint_prediction(0.1)
    second_prediction = _joint_prediction(0.2)
    extra_prediction = _joint_prediction(0.3)

    outputs = policy_inference._assign_names_to_model_outputs({
        DataType.JOINT_TARGET_POSITIONS: [
            first_prediction,
            second_prediction,
            extra_prediction,
        ]
    })

    assert list(outputs[DataType.JOINT_TARGET_POSITIONS].keys()) == [
        "joint1",
        "joint2",
    ]
    assert outputs[DataType.JOINT_TARGET_POSITIONS]["joint1"] is first_prediction
    assert outputs[DataType.JOINT_TARGET_POSITIONS]["joint2"] is second_prediction
    assert extra_prediction not in outputs[DataType.JOINT_TARGET_POSITIONS].values()


def test_assign_names_to_model_outputs_raises_for_missing_output_configuration() -> (
    None
):
    policy_inference = _make_policy_inference({
        DataType.JOINT_TARGET_POSITIONS: _indexed_names("joint1"),
    })

    with pytest.raises(
        ValueError,
        match="JOINT_POSITIONS not in output configuration.",
    ):
        policy_inference._assign_names_to_model_outputs({
            DataType.JOINT_POSITIONS: [_joint_prediction(0.1)],
        })


def test_assign_names_to_model_outputs_raises_for_short_sparse_output_tensor_list() -> (
    None
):
    policy_inference = _make_policy_inference({
        DataType.JOINT_TARGET_POSITIONS: {0: "joint1", 2: "joint3"},
    })

    with pytest.raises(
        ValueError,
        match="Expected at least 3, but got 2",
    ):
        policy_inference._assign_names_to_model_outputs({
            DataType.JOINT_TARGET_POSITIONS: [
                _joint_prediction(0.1),
                _joint_prediction(0.2),
            ],
        })


def test_assign_names_to_model_outputs_supports_sparse_indices() -> None:
    policy_inference = _make_policy_inference({
        DataType.JOINT_TARGET_POSITIONS: {0: "joint1", 2: "joint3"},
    })
    tensor_at_index_0 = _joint_prediction(0.1)
    tensor_at_index_1 = _joint_prediction(0.2)
    tensor_at_index_2 = _joint_prediction(0.3)

    outputs = policy_inference._assign_names_to_model_outputs({
        DataType.JOINT_TARGET_POSITIONS: [
            tensor_at_index_0,
            tensor_at_index_1,
            tensor_at_index_2,
        ],
    })

    assert outputs[DataType.JOINT_TARGET_POSITIONS]["joint1"] is tensor_at_index_0
    assert outputs[DataType.JOINT_TARGET_POSITIONS]["joint3"] is tensor_at_index_2
    assert tensor_at_index_1 not in outputs[DataType.JOINT_TARGET_POSITIONS].values()


def test_validate_input_sync_point_accepts_mixed_string_and_enum_data_types() -> None:
    policy_inference = _make_policy_inference()
    policy_inference.model = SimpleNamespace(
        model_init_description=SimpleNamespace(
            input_data_types=[DataType.JOINT_POSITIONS, DataType.JOINT_VELOCITIES.value]
        )
    )

    sync_point = SynchronizedPoint(
        timestamp=123.0,
        data={
            DataType.JOINT_POSITIONS: {
                "joint1": JointData(timestamp=123.0, value=0.1),
            },
            DataType.JOINT_VELOCITIES: {
                "joint1": JointData(timestamp=123.0, value=0.2),
            },
        },
    )

    policy_inference._validate_input_sync_point(sync_point)


def test_validate_input_sync_point_raises_when_required_data_type_missing() -> None:
    policy_inference = _make_policy_inference()
    policy_inference.model = SimpleNamespace(
        model_init_description=SimpleNamespace(
            input_data_types=[DataType.JOINT_POSITIONS, DataType.JOINT_VELOCITIES.value]
        )
    )
    sync_point = SynchronizedPoint(
        timestamp=123.0,
        data={
            DataType.JOINT_POSITIONS: {
                "joint1": JointData(timestamp=123.0, value=0.1),
            },
        },
    )

    with pytest.raises(
        ValueError,
        match="SynchronizedPoint is missing required data types: JOINT_VELOCITIES",
    ):
        policy_inference._validate_input_sync_point(sync_point)


def test_preprocess_builds_inputs_and_masks_for_multiple_data_types() -> None:
    policy_inference = _make_policy_inference()
    policy_inference.input_embodiment_description = {
        DataType.JOINT_POSITIONS: {0: "joint1"},
        DataType.JOINT_VELOCITIES: {0: "joint1"},
    }
    policy_inference.input_dataset_statistics = {
        DataType.JOINT_POSITIONS: [{"joint1": {}}, {"joint2": {}}],
        DataType.JOINT_VELOCITIES: [{"joint1": {}}],
    }
    sync_point = SynchronizedPoint(
        timestamp=123.0,
        data={
            DataType.JOINT_POSITIONS: {
                "joint1": JointData(timestamp=123.0, value=0.1),
            },
            DataType.JOINT_VELOCITIES: {
                "joint1": JointData(timestamp=123.0, value=0.2),
            },
        },
    )

    batch = policy_inference._preprocess(sync_point)

    assert batch.batch_size == 1
    assert set(batch.inputs.keys()) == {
        DataType.JOINT_POSITIONS,
        DataType.JOINT_VELOCITIES,
    }
    assert batch.inputs_mask[DataType.JOINT_POSITIONS].tolist() == [[1.0, 0.0]]
    assert batch.inputs_mask[DataType.JOINT_VELOCITIES].tolist() == [[1.0]]
    assert len(batch.inputs[DataType.JOINT_POSITIONS]) == 2
    assert len(batch.inputs[DataType.JOINT_VELOCITIES]) == 1


def test_preprocess_pads_missing_cross_embodiment_indices() -> None:
    policy_inference = _make_policy_inference()
    policy_inference.input_embodiment_description = {
        DataType.JOINT_POSITIONS: {
            0: "joint1",
            1: "joint2",
        },
    }
    policy_inference.input_dataset_statistics = {
        DataType.JOINT_POSITIONS: [{"joint1": {}}, {"joint2": {}}, {"joint3": {}}],
    }
    sync_point = SynchronizedPoint(
        timestamp=123.0,
        data={
            DataType.JOINT_POSITIONS: {
                "joint1": JointData(timestamp=123.0, value=0.1),
                "joint2": JointData(timestamp=123.0, value=0.2),
            },
        },
    )

    batch = policy_inference._preprocess(sync_point)

    assert batch.inputs_mask[DataType.JOINT_POSITIONS].tolist() == [[1.0, 1.0, 0.0]]
    assert len(batch.inputs[DataType.JOINT_POSITIONS]) == 3
    joint_values = torch.cat(
        [joint.value for joint in batch.inputs[DataType.JOINT_POSITIONS]], dim=-1
    )
    assert joint_values.shape[-1] == 3
    assert joint_values[0, -1, 2].item() == 0.0


def test_preprocess_raises_when_statistics_for_data_type_missing() -> None:
    policy_inference = _make_policy_inference()
    policy_inference.input_embodiment_description = {
        DataType.JOINT_POSITIONS: {0: "joint1"},
    }
    policy_inference.input_dataset_statistics = {}
    sync_point = SynchronizedPoint(
        timestamp=123.0,
        data={
            DataType.JOINT_POSITIONS: {
                "joint1": JointData(timestamp=123.0, value=0.1),
            },
        },
    )

    with pytest.raises(
        ValueError,
        match=("Model was not trained with input statistics for "),
    ):
        policy_inference._preprocess(sync_point)


def test_preprocess_raises_when_received_items_exceed_training_limit() -> None:
    policy_inference = _make_policy_inference()
    policy_inference.input_embodiment_description = {
        DataType.JOINT_POSITIONS: {0: "joint1", 1: "joint2"},
    }
    policy_inference.input_dataset_statistics = {
        DataType.JOINT_POSITIONS: [{"joint1": {}}],
    }
    sync_point = SynchronizedPoint(
        timestamp=123.0,
        data={
            DataType.JOINT_POSITIONS: {
                "joint1": JointData(timestamp=123.0, value=0.1),
                "joint2": JointData(timestamp=123.0, value=0.2),
            },
        },
    )

    with pytest.raises(
        ValueError,
        match="Received 2 items for data type",
    ):
        policy_inference._preprocess(sync_point)


def test_init_loads_embodiments_from_archive_when_robot_id_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = SimpleNamespace(
        eval=lambda: None,
        model_init_description=SimpleNamespace(
            input_dataset_statistics={},
            output_prediction_horizon=1,
        ),
    )
    monkeypatch.setattr(
        "neuracore.ml.utils.policy_inference.load_model_from_nc_archive",
        lambda model_file, device=None: (
            fake_model,
            {"robot-1": {"JOINT_POSITIONS": {"0": "joint1"}}},
            {"robot-1": {"JOINT_TARGET_POSITIONS": {"0": "joint1"}}},
            {"JOINT_POSITIONS": []},
            {"JOINT_TARGET_POSITIONS": []},
        ),
    )

    inference = PolicyInference(
        input_embodiment_description=None,
        output_embodiment_description=None,
        input_preprocessing_config=None,
        model_file=Path("dummy.nc.zip"),
        org_id="org",
        robot_id="robot-1",
    )

    assert DataType.JOINT_POSITIONS in inference.input_embodiment_description
    assert DataType.JOINT_TARGET_POSITIONS in inference.output_embodiment_description


def test_init_raises_when_no_descriptions_and_no_robot_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = SimpleNamespace(
        eval=lambda: None,
        model_init_description=SimpleNamespace(
            input_dataset_statistics={},
            output_prediction_horizon=1,
        ),
    )
    monkeypatch.setattr(
        "neuracore.ml.utils.policy_inference.load_model_from_nc_archive",
        lambda model_file, device=None: (fake_model, {}, {}, {}, {}),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Must provide both input_embodiment_description and "
            "output_embodiment_description"
        ),
    ):
        PolicyInference(
            input_embodiment_description=None,
            output_embodiment_description=None,
            input_preprocessing_config=None,
            model_file=Path("dummy.nc.zip"),
            org_id="org",
        )


class _RgbOnlyMethod(PreprocessingMethod):
    @staticmethod
    def allowed_data_types() -> frozenset[DataType]:
        return frozenset({DataType.RGB_IMAGES})

    def __call__(
        self, data: BatchedNCData
    ) -> BatchedNCData:  # pragma: no cover - behavior irrelevant for validation
        return data


def test_init_raises_when_input_preprocessing_method_is_not_allowed_for_data_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = SimpleNamespace(
        eval=lambda: None,
        model_init_description=SimpleNamespace(
            input_dataset_statistics={},
            output_prediction_horizon=1,
        ),
    )
    monkeypatch.setattr(
        "neuracore.ml.utils.policy_inference.load_model_from_nc_archive",
        lambda model_file, device=None: (
            fake_model,
            {"robot-1": {"JOINT_POSITIONS": {"0": "joint1"}}},
            {"robot-1": {"JOINT_TARGET_POSITIONS": {"0": "joint1"}}},
            {DataType.JOINT_POSITIONS: []},
            {"JOINT_TARGET_POSITIONS": []},
        ),
    )

    with pytest.raises(
        ValueError,
        match="is not allowed for data type JOINT_POSITIONS",
    ):
        PolicyInference(
            input_embodiment_description=None,
            output_embodiment_description=None,
            input_preprocessing_config={DataType.JOINT_POSITIONS: [_RgbOnlyMethod()]},
            model_file=Path("dummy.nc.zip"),
            org_id="org",
            robot_id="robot-1",
        )


def _rgb_camera_data() -> RGBCameraData:
    """A small even-dimensioned frame the codec matcher can encode."""
    return RGBCameraData(
        timestamp=123.0,
        frame=np.full((16, 16, 3), 120, dtype=np.uint8),
    )


def _preprocess_with_one_camera(
    policy_inference: PolicyInference, present: bool
) -> None:
    """Run _preprocess for one RGB camera, present or absent from the sync point."""
    policy_inference.input_embodiment_description = {
        DataType.RGB_IMAGES: _indexed_names("cam1"),
    }
    policy_inference.input_dataset_statistics = {DataType.RGB_IMAGES: [{}]}
    policy_inference.input_preprocessing_config = {DataType.RGB_IMAGES: []}
    data = {"cam1": _rgb_camera_data()} if present else {}
    policy_inference._preprocess(
        SynchronizedPoint(timestamp=123.0, data={DataType.RGB_IMAGES: data})
    )


@pytest.mark.parametrize("codec", [None, "h264_lossless", "not-a-codec"])
def test_no_codec_matching_unless_env_selects_a_lossy_codec(
    monkeypatch: pytest.MonkeyPatch, codec: str | None
) -> None:
    """Unset, lossless, and unrecognised values must all leave frames untouched."""
    monkeypatch.delenv(INFERENCE_CODEC_ENV_VAR, raising=False)
    if codec is not None:
        monkeypatch.setenv(INFERENCE_CODEC_ENV_VAR, codec)

    assert resolve_inference_codec_options() is None


def test_codec_matching_applies_per_camera_before_configured_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(INFERENCE_CODEC_ENV_VAR, "h264_medium")
    policy_inference = _make_policy_inference()
    policy_inference._inference_codec_options = resolve_inference_codec_options()

    first = policy_inference._codec_method(DataType.RGB_IMAGES, 0)
    second = policy_inference._codec_method(DataType.RGB_IMAGES, 1)

    assert isinstance(first, H264Match)
    # Each camera needs its own encoder: the codec carries inter-frame state.
    assert first is not second
    # Re-requesting a slot must reuse it, or the state resets every prediction.
    assert policy_inference._codec_method(DataType.RGB_IMAGES, 0) is first
    assert first.crf == "23"
    assert first.preset == "medium"


def test_codec_matching_never_applies_to_non_rgb_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Depth keeps lossless storage whatever codec was selected for recording."""
    monkeypatch.setenv(INFERENCE_CODEC_ENV_VAR, "h264_medium")
    policy_inference = _make_policy_inference()
    policy_inference._inference_codec_options = resolve_inference_codec_options()

    assert policy_inference._codec_method(DataType.DEPTH_IMAGES, 0) is None
    assert policy_inference._codec_method(DataType.JOINT_POSITIONS, 0) is None


def test_preprocess_degrades_present_cameras(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(INFERENCE_CODEC_ENV_VAR, "h264_medium")
    policy_inference = _make_policy_inference()
    policy_inference._inference_codec_options = resolve_inference_codec_options()

    _preprocess_with_one_camera(policy_inference, present=True)

    method = policy_inference._codec_methods[0]
    assert method._resolution == (16, 16), "camera frame was not put through the codec"


def test_preprocess_skips_codec_for_absent_cameras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-filled placeholder must not become the encoder's reference frame."""
    monkeypatch.setenv(INFERENCE_CODEC_ENV_VAR, "h264_medium")
    policy_inference = _make_policy_inference()
    policy_inference._inference_codec_options = resolve_inference_codec_options()

    _preprocess_with_one_camera(policy_inference, present=False)

    assert policy_inference._codec_methods == {}
