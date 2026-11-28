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


def test_resolve_preprocessing_config_runs_geometric_first_in_the_written_order():
    resolved = resolve_preprocessing_config(
        OmegaConf.create({
            "RGB_IMAGES": {
                "photometric": [
                    {"gaussian_noise": {"std": 1.0}},
                    {"color_jitter": {"brightness": 0.1}},
                ],
                "geometric": [{"resize_pad": {"size": [64, 64]}}],
            }
        })
    )

    assert [method.name for method in resolved[DataType.RGB_IMAGES]] == [
        "resize_pad",
        "gaussian_noise",
        "color_jitter",
    ]


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        (
            {"geometric": [{"color_jitter": {}}]},
            "color_jitter' is photometric and cannot run under RGB_IMAGES.geometric",
        ),
        (
            {"photometric": [{"resize_pad": {}}]},
            "resize_pad' is geometric and cannot run under RGB_IMAGES.photometric",
        ),
        (
            {"photometric": [{"blur": {}}]},
            "Unknown preprocessing method 'blur' under RGB_IMAGES.photometric",
        ),
        (
            {"photometric": [{"color_jitter": {}, "gaussian_noise": {}}]},
            "exactly one key",
        ),
        ({"augment": [{"color_jitter": {}}]}, "Unknown preprocessing group"),
        (
            "resize_pad",
            "Preprocessing for RGB_IMAGES must map geometric and photometric",
        ),
        (
            {"geometric": {"resize_pad": {}}},
            "RGB_IMAGES.geometric must be a list of steps",
        ),
    ],
)
def test_resolve_preprocessing_config_rejects_malformed_misplaced_or_unknown_steps(
    groups: object, message: str
):
    with pytest.raises(ValueError, match=message):
        resolve_preprocessing_config({"RGB_IMAGES": groups})


def test_resolve_preprocessing_config_builds_a_step_without_params_from_defaults():
    resolved = resolve_preprocessing_config(
        OmegaConf.create({"RGB_IMAGES": {"geometric": [{"resize_pad": None}]}})
    )

    assert resolved[DataType.RGB_IMAGES][0].to_dict() == {
        "resize_pad": {"size": (224, 224)}
    }


def test_resolve_preprocessing_config_drops_data_types_with_no_steps():
    resolved = resolve_preprocessing_config(
        OmegaConf.create({
            "RGB_IMAGES": {"geometric": [{"resize_pad": {"size": [32, 32]}}]},
            "DEPTH_IMAGES": {"geometric": [], "photometric": []},
        })
    )

    assert list(resolved) == [DataType.RGB_IMAGES]


def test_resolve_preprocessing_config_sorts_a_legacy_target_list_geometric_first():
    resolved = resolve_preprocessing_config({
        "RGB_IMAGES": [
            {
                "_target_": (
                    "neuracore.ml.preprocessing.methods.gaussian_noise.GaussianNoise"
                ),
                "std": 2.0,
            },
            {
                "_target_": "neuracore.ml.preprocessing.methods.resize_pad.ResizePad",
                "size": [32, 32],
            },
        ]
    })

    assert [method.name for method in resolved[DataType.RGB_IMAGES]] == [
        "resize_pad",
        "gaussian_noise",
    ]


def test_preprocessing_configuration_round_trips_through_its_serialized_form():
    serialized = {
        "RGB_IMAGES": {
            "geometric": [{"resize_pad": {"size": [48, 64]}}],
            "photometric": [{"gaussian_noise": {"std": 5.0, "clip": True}}],
        }
    }

    resolved = resolve_preprocessing_config(serialized)

    assert json.loads(json.dumps(resolved.to_serializable())) == serialized
