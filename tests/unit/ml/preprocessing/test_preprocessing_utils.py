"""Tests for resolve_preprocessing_config OmegaConf conversion."""

import json
import typing

import pytest
from neuracore_types import (
    ColorJitterConfig,
    DataType,
    PreprocessingMethodConfig,
    PreprocessingStage,
    ResizePadConfig,
    TrainingConfig,
)
from omegaconf import OmegaConf

from neuracore.ml.preprocessing.methods import ColorJitter, ResizePad
from neuracore.ml.utils.preprocessing import (
    PREPROCESSING_METHOD_CLASSES_BY_NAME,
    build_preprocessing_method,
    resolve_input_output_preprocessing,
    resolve_preprocessing_config,
)


def _make_config(size: list[int]) -> object:
    return OmegaConf.create({
        "RGB_IMAGES": [{
            "_target_": "neuracore.ml.preprocessing.methods.resize_pad.ResizePad",
            "size": size,
        }]
    })


def test_resolve_preprocessing_config_converts_listconfig_to_plain_list():
    pytest.importorskip("hydra")
    pytest.importorskip("torch")
    cfg = _make_config([224, 224])
    resolved = resolve_preprocessing_config(cfg)

    method = resolved[DataType.RGB_IMAGES][0]
    assert isinstance(method.size, (list, tuple)), type(method.size)
    assert not type(method.size).__name__ == "ListConfig"


def test_resolve_preprocessing_config_to_dict_is_json_serializable():
    pytest.importorskip("hydra")
    pytest.importorskip("torch")
    cfg = _make_config([224, 224])
    resolved = resolve_preprocessing_config(cfg)

    method = resolved[DataType.RGB_IMAGES][0]
    # This raised TypeError: Object of type ListConfig is not JSON serializable
    # before the fix.
    serialized = json.dumps(method.to_dict())
    assert '"size"' in serialized


def test_preprocessing_configuration_str_is_human_readable():
    pytest.importorskip("hydra")
    pytest.importorskip("torch")
    cfg = OmegaConf.create({
        "RGB_IMAGES": [{
            "_target_": "neuracore.ml.preprocessing.methods.resize_pad.ResizePad",
            "size": [224, 224],
        }],
        "DEPTH_IMAGES": [{
            "_target_": "neuracore.ml.preprocessing.methods.resize_pad.ResizePad",
            "size": [224, 224],
        }],
    })
    resolved = resolve_preprocessing_config(cfg)

    assert str(resolved) == (
        "PreprocessingConfiguration({\n"
        "  DEPTH_IMAGES: [ResizePad(size=[224, 224])]\n"
        "  RGB_IMAGES: [ResizePad(size=[224, 224])]\n"
        "})"
    )


def test_method_class_table_matches_types_union():
    union_members = typing.get_args(typing.get_args(PreprocessingMethodConfig)[0])
    method_names = {member.model_fields["method"].default for member in union_members}
    assert method_names == set(PREPROCESSING_METHOD_CLASSES_BY_NAME)


def test_build_preprocessing_method_passes_settings():
    resize_pad = build_preprocessing_method(ResizePadConfig(size=(128, 96)))
    assert isinstance(resize_pad, ResizePad)
    assert tuple(resize_pad.size) == (128, 96)


def test_archive_dict_from_built_method_loads_again():
    pytest.importorskip("hydra")
    built_method = build_preprocessing_method(ColorJitterConfig(brightness=0.3))
    reloaded = resolve_preprocessing_config(
        OmegaConf.create({"RGB_IMAGES": [built_method.to_dict()]})
    )
    reloaded_method = reloaded[DataType.RGB_IMAGES][0]
    assert isinstance(reloaded_method, ColorJitter)
    assert reloaded_method.brightness == 0.3


def test_default_preprocessing_splits_methods_by_stage():
    preprocessing = TrainingConfig().preprocessing
    training_input, training_output = resolve_input_output_preprocessing(
        preprocessing, stage=PreprocessingStage.TRAINING
    )
    inference_input, inference_output = resolve_input_output_preprocessing(
        preprocessing, stage=PreprocessingStage.INFERENCE
    )
    assert [type(method) for method in training_input[DataType.RGB_IMAGES]] == [
        ResizePad,
        ColorJitter,
        PREPROCESSING_METHOD_CLASSES_BY_NAME["gaussian_noise"],
    ]
    assert [type(method) for method in inference_input[DataType.RGB_IMAGES]] == [
        ResizePad
    ]
    for output_config in (training_output, inference_output):
        assert isinstance(output_config[DataType.DEPTH_IMAGES][0], ResizePad)


def test_data_type_without_methods_for_a_stage_is_left_out():
    preprocessing = TrainingConfig.model_validate({
        "preprocessing": {
            "input": {
                "RGB_IMAGES": [{"method": "resize_pad"}],
                "DEPTH_IMAGES": [{"method": "resize_pad", "stages": ["training"]}],
            },
            "output": {"RGB_IMAGES": [{"method": "resize_pad"}]},
        }
    }).preprocessing
    inference_input, _ = resolve_input_output_preprocessing(
        preprocessing, stage=PreprocessingStage.INFERENCE
    )
    assert set(inference_input) == {DataType.RGB_IMAGES}
