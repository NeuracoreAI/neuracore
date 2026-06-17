"""Tests for resolve_preprocessing_config OmegaConf conversion."""

import json

import pytest
from neuracore_types import DataType
from omegaconf import OmegaConf

from neuracore.ml.utils.preprocessing import resolve_preprocessing_config


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
