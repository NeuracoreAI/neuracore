"""Tests for resolve_preprocessing_config OmegaConf conversion."""

import json
from importlib.resources import files

import pytest
from neuracore_types import DataType
from omegaconf import OmegaConf

from neuracore.ml.utils.preprocessing import (
    resolve_input_output_preprocessing,
    resolve_preprocessing_config,
)

TRAINING_CONFIG_DIR = str(files("neuracore.ml") / "config")


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


def test_rgb_and_depth_preprocessing_overrides_reach_their_own_pipelines():
    hydra = pytest.importorskip("hydra")
    pytest.importorskip("torch")
    with hydra.initialize_config_dir(config_dir=TRAINING_CONFIG_DIR, version_base=None):
        cfg = hydra.compose(
            "config",
            overrides=[
                (
                    "++rgb_preprocessing={resize_pad:{size:[128,96]},"
                    "gaussian_noise:{std:0}}"
                ),
                "++depth_preprocessing={resize_pad:{size:[64,48]}}",
            ],
        )

    train_input, train_output = resolve_input_output_preprocessing(
        cfg.train_preprocessing, role_name="train_preprocessing"
    )
    inference_input, inference_output = resolve_input_output_preprocessing(
        cfg.inference_preprocessing, role_name="inference_preprocessing"
    )
    pipelines = {
        "train input": train_input,
        "train output": train_output,
        "inference input": inference_input,
        "inference output": inference_output,
    }

    assert {
        name: str(pipeline[DataType.RGB_IMAGES]) for name, pipeline in pipelines.items()
    } == {
        "train input": (
            "[ResizePad(size=[128, 96]), "
            "ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05), "
            "GaussianNoise(std=0, clip=True)]"
        ),
        "train output": "[ResizePad(size=[128, 96])]",
        "inference input": "[ResizePad(size=[128, 96])]",
        "inference output": "[ResizePad(size=[128, 96])]",
    }
    assert {
        name: str(pipeline[DataType.DEPTH_IMAGES])
        for name, pipeline in pipelines.items()
    } == dict.fromkeys(pipelines, "[ResizePad(size=[64, 48])]")
